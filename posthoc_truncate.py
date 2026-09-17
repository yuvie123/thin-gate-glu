"""
Experiment A: which of the three GLU projections tolerates rank reduction best?

For a pretrained model, take ONE projection type (gate_proj, up_proj or down_proj), replace it in
EVERY layer by its best rank-r approximation, and measure perplexity. Repeat for each type and rank.
The three matrices have the same shape, so equal rank means equal parameter savings.
No training is involved.

Two ways of choosing the rank-r approximation:
  plain     truncated SVD of the weight W (minimises ||W - W'||)
  --whiten  activation-aware: minimises ||(W - W') X|| on calibration inputs X, by taking the SVD of
            W S where S S^T = X^T X (Cholesky), then multiplying by S^-1. This is the standard
            "whitening" trick from the SVD-compression literature and is the stronger baseline.
            Calibration uses the wikitext-2 TRAIN split; evaluation uses the TEST split.

Run on the GPU machine:
    python posthoc_truncate.py --model HuggingFaceTB/SmolLM2-135M
    python posthoc_truncate.py --model Qwen/Qwen2.5-0.5B --whiten
Toy check (random tiny model, no downloads):
    python posthoc_truncate.py --smoke
"""

import argparse
import json
import os
import time

import torch

from hf_utils import MLP_TYPES, eval_text, load_model, mlp_linears, perplexity, tokenize_blocks


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4"])
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_eval_blocks", type=int, default=0, help="0 = use the full evaluation set")
    ap.add_argument("--rank_fracs", default="0.75,0.5,0.375,0.25,0.125,0.0625",
                    help="ranks as fractions of the full rank min(d_model, d_ff)")
    ap.add_argument("--types", default=",".join(MLP_TYPES))
    ap.add_argument("--whiten", action="store_true")
    ap.add_argument("--calib_blocks", type=int, default=32)
    ap.add_argument("--layers_per_pass", type=int, default=4, help="whitening: layers calibrated at once (memory)")
    ap.add_argument("--out_dir", default="results/posthoc")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


@torch.no_grad()
def collect_grams(model, modules, calib_blocks, batch_size):
    """Sum of x x^T over all calibration tokens, for the input of each given Linear module."""
    grams = {id(m): torch.zeros(m.in_features, m.in_features, device=m.weight.device, dtype=torch.float32)
             for m in modules}

    def hook(mod, inputs):
        x = inputs[0].reshape(-1, mod.in_features).float()
        grams[id(mod)] += x.T @ x

    handles = [m.register_forward_pre_hook(hook) for m in modules]
    device = next(model.parameters()).device
    for i in range(0, calib_blocks.size(0), batch_size):
        model(calib_blocks[i:i + batch_size].to(device))
    for h in handles:
        h.remove()
    return [grams[id(m)] for m in modules]


@torch.no_grad()
def factorize(weight, gram=None):
    """Returns (U, s, V) on the CPU such that the rank-r approximation is (U[:, :r] * s[:r]) @ V[:r]."""
    W = weight.float()
    if gram is None:
        U, s, Vh = torch.linalg.svd(W, full_matrices=False)
        return U.cpu(), s.cpu(), Vh.cpu()
    # Whitening. A small ridge keeps the Cholesky factorization numerically safe.
    ridge = 1e-5 * gram.diagonal().mean()
    eye = torch.eye(gram.size(0), device=gram.device)
    for attempt in range(6):
        try:
            S = torch.linalg.cholesky(gram + ridge * eye)
            break
        except torch.linalg.LinAlgError:
            ridge = ridge * 10
    else:
        raise RuntimeError("Cholesky failed even with a large ridge")
    U, s, Vh = torch.linalg.svd(W @ S, full_matrices=False)
    V = torch.linalg.solve_triangular(S, Vh, upper=False, left=False)   # Vh @ S^-1
    return U.cpu(), s.cpu(), V.cpu()


@torch.no_grad()
def run(model, eval_blocks, calib_blocks, args):
    rank_fracs = [float(f) for f in args.rank_fracs.split(",")]
    base_ppl = perplexity(model, eval_blocks, args.batch_size)
    print(f"baseline perplexity: {base_ppl:.3f}")
    records = []

    for proj_type in args.types.split(","):
        modules = [m for _, m in mlp_linears(model, proj_type)]
        originals = [m.weight.data.clone().cpu() for m in modules]

        t0 = time.time()
        factors = []
        step = args.layers_per_pass if args.whiten else len(modules)
        for start in range(0, len(modules), step):
            chunk = modules[start:start + step]
            grams = collect_grams(model, chunk, calib_blocks, args.batch_size) if args.whiten else [None] * len(chunk)
            for m, g in zip(chunk, grams):
                factors.append(factorize(m.weight.data, g))
            del grams
        print(f"{proj_type}: factorized {len(modules)} layers in {time.time() - t0:.0f}s")

        full_rank = min(modules[0].in_features, modules[0].out_features)
        n_in, n_out = modules[0].in_features, modules[0].out_features
        for frac in rank_fracs:
            r = max(1, int(round(frac * full_rank)))
            for m, (U, s, V) in zip(modules, factors):
                dev = m.weight.device
                approx = (U[:, :r].to(dev) * s[:r].to(dev)) @ V[:r].to(dev)
                m.weight.data.copy_(approx.to(m.weight.dtype))
            ppl = perplexity(model, eval_blocks, args.batch_size)
            rec = {"type": proj_type, "rank_frac": frac, "rank": r, "ppl": ppl,
                   # parameters of the factorized matrix relative to the dense one
                   "param_ratio": r * (n_in + n_out) / (n_in * n_out)}
            records.append(rec)
            print(f"  {proj_type}  rank {r:5d} ({frac:.3f} of full)  ppl {ppl:10.3f}  (x{ppl / base_ppl:.2f} baseline)")
        for m, w in zip(modules, originals):      # restore before moving to the next type
            m.weight.data.copy_(w.to(m.weight.device))
        del factors, originals

    check = perplexity(model, eval_blocks, args.batch_size)
    assert abs(check - base_ppl) < 1e-3 * base_ppl, "weights were not restored correctly"
    return base_ppl, records


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(0)
        cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=192, num_hidden_layers=2,
                          num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128)
        model = LlamaForCausalLM(cfg).to(device).eval()
        eval_blocks = torch.randint(0, 256, (8, 64))
        calib_blocks = torch.randint(0, 256, (8, 64))
        args.model, args.out_dir = "smoke-random-llama", "results/smoke"
    else:
        model, tok = load_model(args.model, device=device)
        eval_blocks = tokenize_blocks(tok, eval_text(args.dataset, "test"), args.seq_len, args.max_eval_blocks or None)
        calib_blocks = None
        if args.whiten:
            calib_blocks = tokenize_blocks(tok, eval_text("wikitext2", "train"), args.seq_len, args.calib_blocks)
    print(f"model {args.model}: {eval_blocks.size(0)} evaluation windows of {eval_blocks.size(1)} tokens")

    base_ppl, records = run(model, eval_blocks, calib_blocks, args)

    sample = mlp_linears(model, "gate_proj")[0][1]
    result = {
        "model": args.model, "dataset": args.dataset, "seq_len": args.seq_len,
        "method": "whiten" if args.whiten else "plain",
        "d_model": sample.in_features, "d_ff": sample.out_features,
        "n_layers": len(mlp_linears(model, "gate_proj")),
        "n_params": sum(p.numel() for p in model.parameters()),
        "base_ppl": base_ppl, "records": records,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.model.replace('/', '__')}__{args.dataset}__{result['method']}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()

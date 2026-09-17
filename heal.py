"""
Experiment C ("healing"): convert a pretrained model and recover quality by training ONLY the new factors.

1. Measure the perplexity of the pretrained model.
2. Replace one projection type (gate_proj / up_proj / down_proj) in every layer by a rank-r factorization
   initialised from a (optionally whitened) SVD. Measure perplexity again.
3. Freeze everything else and train just the factors on FineWeb-Edu text. Measure perplexity again.

Because only the small factors are trained, the optimizer state is tiny and this fits on an 8 GB GPU.
Compare `--type gate_proj` against `--type up_proj` and `--type down_proj` at the same rank.

    python heal.py --model HuggingFaceTB/SmolLM2-135M --type gate_proj --rank_frac 0.25
    python heal.py --smoke
"""

import argparse
import json
import math
import os
import time

import torch

from hf_utils import eval_text, mlp_linears, perplexity, tokenize_blocks
from model import LowRankLinear
from posthoc_truncate import collect_grams, factorize
from train import pick_amp_dtype


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--type", default="gate_proj", choices=["gate_proj", "up_proj", "down_proj"])
    ap.add_argument("--rank_frac", type=float, default=0.25)
    ap.add_argument("--whiten", action="store_true", help="activation-aware SVD init (see posthoc_truncate.py)")
    ap.add_argument("--tokens", type=float, default=20e6)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--micro_bs", type=int, default=2)
    ap.add_argument("--batch_seqs", type=int, default=16, help="sequences per optimizer step")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--grad_ckpt", action="store_true", help="trade speed for memory")
    ap.add_argument("--amp_dtype", default="auto", choices=["auto", "bfloat16", "float16"],
                    help="autocast precision; auto = bfloat16 where native, float16 with loss scaling on older GPUs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results/heal")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def packed_stream(tok, seq_len, micro_bs, seed):
    """Endless stream of [micro_bs, seq_len] token batches from FineWeb-Edu (streamed, nothing stored on disk)."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    buf, need = [], seq_len * micro_bs
    for doc in ds:
        buf.extend(tok(doc["text"]).input_ids + [tok.eos_token_id])
        while len(buf) >= need:
            yield torch.tensor(buf[:need]).view(micro_bs, seq_len)
            buf = buf[need:]


def random_stream(vocab, seq_len, micro_bs):
    while True:
        yield torch.randint(0, vocab, (micro_bs, seq_len))


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    if args.smoke:
        from transformers import LlamaConfig, LlamaForCausalLM
        cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=192, num_hidden_layers=2,
                          num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128)
        model = LlamaForCausalLM(cfg).to(device)
        args.seq_len, args.tokens, args.warmup_steps, args.out_dir = 64, 64 * 16 * 6, 2, "results/smoke"
        args.model = "smoke-random-llama"
        eval_blocks = torch.randint(0, 256, (8, 64))
        calib_blocks = torch.randint(0, 256, (8, 64))
        stream = random_stream(256, args.seq_len, args.micro_bs)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        # float32 master weights + bfloat16 autocast: simple and stable for fine-tuning small models
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
        eval_blocks = tokenize_blocks(tok, eval_text("wikitext2", "test"), args.seq_len)
        calib_blocks = tokenize_blocks(tok, eval_text("wikitext2", "train"), args.seq_len, 32) if args.whiten else None
        stream = packed_stream(tok, args.seq_len, args.micro_bs, args.seed)

    amp_dtype = pick_amp_dtype(device, args.amp_dtype)
    autocast = torch.autocast(device_type=device, dtype=amp_dtype)
    scaler = torch.amp.GradScaler(device, enabled=(amp_dtype == torch.float16))   # a no-op with bfloat16
    model.eval()
    with autocast:
        ppl_base = perplexity(model, eval_blocks)
    print(f"pretrained perplexity: {ppl_base:.3f}")

    # --- replace the chosen projection in every layer by a rank-r factorization ---
    targets = mlp_linears(model, args.type)
    rank = max(1, int(round(args.rank_frac * min(targets[0][1].in_features, targets[0][1].out_features))))
    modules = [m for _, m in targets]
    removed = sum(m.weight.numel() for m in modules)
    with torch.no_grad(), autocast:
        grams = collect_grams(model, modules, calib_blocks, args.micro_bs) if args.whiten else [None] * len(modules)
    for (name, m), g in zip(targets, grams):
        U, s, V = factorize(m.weight.data, g)
        new = LowRankLinear(m.in_features, m.out_features, rank)
        root = s[:rank].sqrt()
        new.A.weight.data = (root[:, None] * V[:rank]).contiguous()
        new.B.weight.data = (U[:, :rank] * root[None, :]).contiguous()
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, args.type, new.to(device))
    del grams

    for p in model.parameters():
        p.requires_grad = False
    factors = [p for n, p in model.named_parameters() if f"mlp.{args.type}." in n]
    for p in factors:
        p.requires_grad = True
    trainable = sum(p.numel() for p in factors)
    print(f"{args.type}: rank {rank}, {removed:,} dense parameters -> {trainable:,} trainable factor parameters")

    with autocast:
        ppl_trunc = perplexity(model, eval_blocks)
    print(f"after truncation: {ppl_trunc:.3f}")

    # --- train only the factors ---
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    opt = torch.optim.AdamW(factors, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    accum = args.batch_seqs // args.micro_bs
    total_steps = max(1, int(args.tokens // (args.batch_seqs * args.seq_len)))
    curve, t0 = [], time.time()
    model.train()
    for step in range(total_steps):
        if step < args.warmup_steps:
            lr = args.lr * (step + 1) / args.warmup_steps
        else:
            lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps))))
        for group in opt.param_groups:
            group["lr"] = lr
        loss_sum = 0.0
        for _ in range(accum):
            x = next(stream).to(device)
            with autocast:
                loss = model(input_ids=x, labels=x).loss
            scaler.scale(loss / accum).backward()
            loss_sum += loss.item() / accum
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(factors, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if step % 20 == 0 or step + 1 == total_steps:
            curve.append({"step": step, "loss": loss_sum})
            print(f"step {step}/{total_steps}  loss {loss_sum:.4f}  ({(time.time() - t0) / 60:.1f} min)")

    model.eval()
    with autocast:
        ppl_healed = perplexity(model, eval_blocks)
    print(f"after healing: {ppl_healed:.3f}   (pretrained {ppl_base:.3f}, truncated {ppl_trunc:.3f})")

    result = {
        "model": args.model, "type": args.type, "rank": rank, "rank_frac": args.rank_frac,
        "init": "whiten" if args.whiten else "plain", "args": vars(args),
        "dense_params_removed": removed, "trainable_params": trainable,
        "ppl_pretrained": ppl_base, "ppl_truncated": ppl_trunc, "ppl_healed": ppl_healed,
        "train_tokens": total_steps * args.batch_seqs * args.seq_len,
        "wall_seconds": time.time() - t0, "curve": curve,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.model.replace('/', '__')}__{args.type}__r{args.rank_frac:g}__{result['init']}__s{args.seed}"
    path = os.path.join(args.out_dir, tag + ".json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()

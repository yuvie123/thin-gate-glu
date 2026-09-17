"""
Experiment B: pretrain one small GPT from scratch and log everything to results/<name>.json.

Examples (run on the GPU machine):
    python train.py --name S_dense_s0 --size S --seed 0
    python train.py --name S_thin_gate_r96_s0 --size S --gate_rank 96 --seed 0
Toy check on any machine (random tokens, tiny model, < 1 minute on CPU):
    python train.py --smoke

You normally don't call this by hand: `python grid.py` builds the commands for every arm.
"""

import argparse
import json
import math
import os
import time

import torch

from data import TokenData
from model import GPT, GPTConfig, config_dict, count_params, flops_per_token

# name: (n_layer, n_head, d_model, d_ff)   d_ff ~ 8/3 * d_model, rounded to a multiple of 64
SIZES = {
    "S": (6, 6, 384, 1024),     # ~10.6M non-embedding parameters
    "M": (8, 8, 512, 1344),     # ~24.9M
    "L": (12, 12, 768, 2048),   # ~85M (stretch goal on an 8 GB GPU)
}


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None, help="run name; results go to results/<name>.json")
    ap.add_argument("--size", default="S", choices=list(SIZES))
    ap.add_argument("--d_ff", type=int, default=0, help="override the MLP width (0 = size default)")
    ap.add_argument("--gate_rank", type=int, default=0)
    ap.add_argument("--up_rank", type=int, default=0)
    ap.add_argument("--down_rank", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens", type=float, default=300e6, help="total training tokens")
    ap.add_argument("--batch_tokens", type=int, default=65536, help="tokens per optimizer step")
    ap.add_argument("--micro_bs", type=int, default=4,
                    help="sequences per forward pass. 4 is safe on 8 GB; try 8 for speed if nvidia-smi shows room. "
                         "Changing it does not change the optimizer batch or the data order.")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_seqs", type=int, default=512, help="validation sequences (x seq_len tokens), fixed for all runs")
    ap.add_argument("--data_dir", default="data/fineweb10B")
    ap.add_argument("--out_dir", default="results/train")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny model + random tokens; checks the code runs")
    return ap.parse_args()


def lr_at(step, total_steps, args):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * cosine)


@torch.no_grad()
def evaluate(model, data, args, device, autocast):
    model.eval()
    losses = []
    for x, y in data.val_batches(args.micro_bs, args.eval_seqs // args.micro_bs):
        with autocast:
            _, loss = model(x.to(device), y.to(device))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    args = get_args()
    n_layer, n_head, d_model, d_ff = SIZES[args.size]
    if args.smoke:
        n_layer, n_head, d_model, d_ff = 2, 2, 64, 192
        args.seq_len, args.micro_bs, args.batch_tokens = 64, 4, 64 * 8
        args.tokens, args.warmup_steps, args.eval_every, args.eval_seqs = 64 * 8 * 20, 5, 10, 16
        args.name = args.name or "smoke"
        args.out_dir = "results/smoke"
        args.gate_rank = args.gate_rank or 16
    assert args.name, "give the run a --name"
    d_ff = args.d_ff or d_ff

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = GPTConfig(
        vocab_size=512 if args.smoke else 50304, seq_len=args.seq_len,
        n_layer=n_layer, n_head=n_head, d_model=d_model, d_ff=d_ff,
        gate_rank=args.gate_rank, up_rank=args.up_rank, down_rank=args.down_rank,
    )
    model = GPT(cfg).to(device)
    params = count_params(model)
    print(f"[{args.name}] params: {params}")

    data = TokenData(args.data_dir, args.seq_len, seed=args.seed,
                     synthetic_vocab=cfg.vocab_size if args.smoke else None)

    # Weight decay on matrices only (not on norms / embeddings), the usual GPT recipe.
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "embed" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 or "embed" in n]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=(device == "cuda"),
    )

    autocast = torch.autocast(device_type=device, dtype=torch.bfloat16)
    step_model = model
    if device == "cuda" and not args.no_compile:
        step_model = torch.compile(model)

    assert args.batch_tokens % (args.micro_bs * args.seq_len) == 0, "batch_tokens must be divisible by micro_bs*seq_len"
    accum = args.batch_tokens // (args.micro_bs * args.seq_len)
    total_steps = int(args.tokens // args.batch_tokens)
    print(f"[{args.name}] {total_steps} steps x {args.batch_tokens} tokens, grad accumulation = {accum}, device = {device}")

    log = {"train": [], "val": []}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    timed_tokens, t_timed = 0, None

    for step in range(total_steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, total_steps, args)
        loss_sum = 0.0
        for _ in range(accum):
            x, y = data.train_batch(args.micro_bs)
            with autocast:
                _, loss = step_model(x.to(device, non_blocking=True), y.to(device, non_blocking=True))
            (loss / accum).backward()
            loss_sum += loss.item() / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if not math.isfinite(loss_sum):
            raise RuntimeError(f"loss became {loss_sum} at step {step}; lower --lr")

        # throughput: skip the first 20 steps (compile + warm-up) before timing
        if step == 20:
            if device == "cuda":
                torch.cuda.synchronize()
            t_timed, timed_tokens = time.time(), 0
        elif step > 20:
            timed_tokens += args.batch_tokens

        if step % 50 == 0:
            log["train"].append({"step": step, "loss": loss_sum})
            print(f"[{args.name}] step {step}/{total_steps}  loss {loss_sum:.4f}  lr {opt.param_groups[0]['lr']:.2e}")
        if (step + 1) % args.eval_every == 0 or step + 1 == total_steps:
            val = evaluate(step_model, data, args, device, autocast)
            log["val"].append({"step": step + 1, "loss": val})
            print(f"[{args.name}] step {step + 1}  VAL loss {val:.4f}")

    if device == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t_start
    tok_per_s = timed_tokens / (time.time() - t_timed) if t_timed and timed_tokens else None

    result = {
        "name": args.name,
        "args": vars(args),
        "config": config_dict(cfg),
        "params": params,
        "flops_per_token": flops_per_token(model),
        "final_val_loss": log["val"][-1]["loss"],
        "tokens_per_second": tok_per_s,
        "wall_seconds": wall,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "log": log,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[{args.name}] done: final val loss {result['final_val_loss']:.4f}, "
          f"{tok_per_s and round(tok_per_s)} tok/s, {wall / 60:.1f} min -> {path}")


if __name__ == "__main__":
    main()

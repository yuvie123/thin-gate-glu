"""
Measured speed and memory of each arm (so the paper reports real numbers, not just FLOP counts).

For every arm of a size: training throughput (forward+backward tokens/s), inference prefill throughput,
and peak memory. Low-rank layers replace one large matmul by two small ones, which is not always
faster on a real GPU -- report whatever this prints, even if it is unflattering.

    python bench.py --size S            # on the GPU machine
    python bench.py --smoke             # toy check
"""

import argparse
import json
import os
import time

import torch

from grid import arms_for
from model import GPT, GPTConfig, count_params, flops_per_token
from train import SIZES, pick_amp_dtype


def timed(fn, device, warmup, iters):
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="S", choices=list(SIZES))
    ap.add_argument("--micro_bs", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--amp_dtype", default="auto", choices=["auto", "bfloat16", "float16"])
    ap.add_argument("--out_dir", default="results/bench")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    L, H, d, F = SIZES[args.size]
    vocab = 50304
    if args.smoke:
        args.micro_bs, args.seq_len, args.iters, args.out_dir, vocab = 2, 32, 2, "results/smoke", 512
    amp_dtype = pick_amp_dtype(device, args.amp_dtype)
    autocast = torch.autocast(device_type=device, dtype=amp_dtype)
    rows = []

    for name, arm in arms_for(args.size, (2, 4, 8)).items():
        cfg = GPTConfig(vocab_size=vocab, seq_len=args.seq_len, n_layer=2 if args.smoke else L, n_head=H,
                        d_model=d, d_ff=arm.get("d_ff", F), gate_rank=arm.get("gate_rank", 0),
                        up_rank=arm.get("up_rank", 0), down_rank=arm.get("down_rank", 0))
        model = GPT(cfg).to(device)
        run = torch.compile(model) if (device == "cuda" and not args.no_compile) else model
        x = torch.randint(0, vocab, (args.micro_bs, args.seq_len), device=device)

        def train_step():
            with autocast:
                _, loss = run(x, x)
            loss.backward()
            model.zero_grad(set_to_none=True)

        @torch.no_grad()
        def prefill():
            with autocast:
                run(x)

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        tokens = args.micro_bs * args.seq_len
        t_train = timed(train_step, device, warmup=5, iters=args.iters)
        peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None
        t_prefill = timed(prefill, device, warmup=5, iters=args.iters)
        row = {"arm": name, **count_params(model), "flops_per_token": flops_per_token(model),
               "train_tokens_per_s": tokens / t_train, "prefill_tokens_per_s": tokens / t_prefill,
               "peak_train_memory_gb": peak}
        rows.append(row)
        print(f"{name:18s} mlp params {row['mlp']:>11,}  train {row['train_tokens_per_s']:>9.0f} tok/s  "
              f"prefill {row['prefill_tokens_per_s']:>9.0f} tok/s  peak {peak if peak is None else round(peak, 2)} GB")
        del model, run

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"bench_{args.size}.json")
    with open(path, "w") as f:
        json.dump({"size": args.size, "micro_bs": args.micro_bs, "seq_len": args.seq_len,
                   "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                   "amp_dtype": str(amp_dtype).replace("torch.", ""), "rows": rows}, f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()

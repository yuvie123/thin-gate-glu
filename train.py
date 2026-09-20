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
import dataclasses
import json
import math
import os
import time

import numpy as np
import torch

from data import TokenData
from posthoc_truncate import collect_grams
from model import GPT, GPTConfig, LowRankLinear, MonarchLinear, config_dict, count_params, flops_per_token

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
    # screening variants (grid.py `screen`); every default reproduces the original arms exactly
    ap.add_argument("--gate_groups", type=int, default=1, help="g > 1: one gate value per g hidden units")
    ap.add_argument("--up_groups", type=int, default=1)
    ap.add_argument("--down_groups", type=int, default=1)
    ap.add_argument("--gate_monarch", type=int, default=0, help="nb > 0: Monarch gate with nb blocks")
    ap.add_argument("--lowrank_init", default="balanced", choices=["balanced", "spectral"])
    ap.add_argument("--bottleneck", default="linear", choices=["linear", "silu", "norm_silu"])
    ap.add_argument("--factor_wd", default="same", choices=["same", "half", "none"],
                    help="weight decay on low-rank / Monarch factors relative to --weight_decay")
    ap.add_argument("--thin_at", type=float, default=0.0,
                    help="warm start: train a DENSE gate for this fraction of the steps, then replace it by a "
                         "rank --gate_rank factorization (whitened SVD) and continue. 0 = off")
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
    ap.add_argument("--amp_dtype", default="auto", choices=["auto", "bfloat16", "float16"],
                    help="autocast precision. auto = bfloat16 on the CPU and on GPUs that support it natively "
                         "(Ampere or newer); float16 with loss scaling on older GPUs such as the T4")
    ap.add_argument("--smoke", action="store_true", help="tiny model + random tokens; checks the code runs")
    return ap.parse_args()


def pick_amp_dtype(device, choice="auto"):
    """Autocast precision: bfloat16 where it is native (the CPU, Ampere-or-newer GPUs), float16 on older
    GPUs such as the T4. float16 needs loss scaling; callers enable a GradScaler exactly when this returns it."""
    if choice != "auto":
        return getattr(torch, choice)
    old_gpu = device == "cuda" and torch.cuda.get_device_capability()[0] < 8
    return torch.float16 if old_gpu else torch.bfloat16


def lr_at(step, total_steps, args):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * cosine)


def calibration_blocks(data, n_seqs, seed):
    """n_seqs training windows drawn with a private RNG (the data's own RNG, and so the training order, is
    untouched). Used to whiten the SVD when a dense gate is thinned mid-training."""
    rng = np.random.default_rng(seed)
    T, rows = data.seq_len, []
    for _ in range(n_seqs):
        shard = data.train[rng.integers(len(data.train))]
        s = rng.integers(0, len(shard) - T - 1)
        rows.append(np.asarray(shard[s:s + T]))
    return torch.from_numpy(np.stack(rows).astype(np.int64))


def thin_gates(model, opt, rank, grams, factor_wd):
    """Replace every dense gate by its rank-r factorization (whitened SVD when a gram is given, plain SVD for
    None) and move the optimizer over: the dense weights leave their group and lose their Adam state, every
    other parameter keeps its state, and the new factors join as a fresh group. The same optimizer object is
    kept so the GradScaler and the learning-rate loop carry on unchanged."""
    from posthoc_truncate import factorize
    device = next(model.parameters()).device
    old = [b.mlp.gate.weight for b in model.blocks]
    new_params = []
    for b, g in zip(model.blocks, grams):
        U, s, V = factorize(b.mlp.gate.weight.data, g)
        b.mlp.gate = LowRankLinear.from_factors(U, s, V, rank).to(device)
        new_params += list(b.mlp.gate.parameters())
    old_ids = {id(p) for p in old}
    for group in opt.param_groups:
        group["params"] = [p for p in group["params"] if id(p) not in old_ids]
    for p in old:
        opt.state.pop(p, None)
    opt.add_param_group({"params": new_params, "weight_decay": factor_wd})
    return new_params


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
        if not (args.gate_rank or args.gate_groups > 1 or args.gate_monarch):
            args.gate_rank = 16                     # the smoke run exercises the low-rank path by default
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
        gate_groups=args.gate_groups, up_groups=args.up_groups, down_groups=args.down_groups,
        gate_monarch=args.gate_monarch, lowrank_init=args.lowrank_init, bottleneck=args.bottleneck,
    )
    if args.thin_at > 0:
        assert 0 < args.thin_at < 1 and args.gate_rank > 0, "--thin_at needs 0 < f < 1 and a --gate_rank target"
        model = GPT(dataclasses.replace(cfg, gate_rank=0)).to(device)      # dense gate for the warm-up
    else:
        model = GPT(cfg).to(device)
    params_initial = count_params(model)
    flops_initial = flops_per_token(model)
    print(f"[{args.name}] params: {params_initial}")

    data = TokenData(args.data_dir, args.seq_len, seed=args.seed,
                     synthetic_vocab=cfg.vocab_size if args.smoke else None)

    # Weight decay on matrices only (not on norms / embeddings), the usual GPT recipe. The factors of a
    # low-rank or Monarch projection form a third group so --factor_wd can scale their decay.
    factor_wd = args.weight_decay * {"same": 1.0, "half": 0.5, "none": 0.0}[args.factor_wd]
    factor_ids = {id(p) for m in model.modules() if isinstance(m, (LowRankLinear, MonarchLinear))
                  for p in m.parameters() if p.dim() >= 2}
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "embed" not in n and id(p) not in factor_ids]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 or "embed" in n]
    factors = [p for p in model.parameters() if id(p) in factor_ids]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0},
         {"params": factors, "weight_decay": factor_wd}],
        lr=args.lr, betas=(0.9, 0.95), fused=(device == "cuda"),
    )

    amp_dtype = pick_amp_dtype(device, args.amp_dtype)
    autocast = torch.autocast(device_type=device, dtype=amp_dtype)
    # float16 has a narrow exponent range, so its gradients need loss scaling. With bfloat16 the scaler is
    # disabled and every call below passes straight through, so that path is unchanged.
    scaler = torch.amp.GradScaler(device, enabled=(amp_dtype == torch.float16))
    step_model = model
    if device == "cuda" and not args.no_compile:
        step_model = torch.compile(model)

    assert args.batch_tokens % (args.micro_bs * args.seq_len) == 0, "batch_tokens must be divisible by micro_bs*seq_len"
    accum = args.batch_tokens // (args.micro_bs * args.seq_len)
    total_steps = int(args.tokens // args.batch_tokens)
    print(f"[{args.name}] {total_steps} steps x {args.batch_tokens} tokens, grad accumulation = {accum}, device = {device}")

    thin_step = int(args.thin_at * total_steps) if args.thin_at > 0 else None
    if thin_step is not None:
        assert thin_step > args.warmup_steps, "thin the gate after the learning-rate warm-up"
        # Calibration windows for the whitening come from their own RNG, so the training stream is untouched.
        calib_blocks = calibration_blocks(data, 8 * args.micro_bs, seed=args.seed + 10_000)

    log = {"train": [], "val": []}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    timed_tokens, t_timed = 0, None

    for step in range(total_steps):
        if thin_step is not None and step == thin_step:
            val_before = evaluate(step_model, data, args, device, autocast)
            model.eval()
            with torch.no_grad(), autocast:
                grams = collect_grams(model, [b.mlp.gate for b in model.blocks], calib_blocks, args.micro_bs)
            model.train()
            thin_gates(model, opt, args.gate_rank, grams, factor_wd)
            model.cfg = cfg
            if step_model is not model:
                torch._dynamo.reset()
                step_model = torch.compile(model)
            val_after = evaluate(step_model, data, args, device, autocast)
            log["thin_swap"] = {"step": step, "val_before": val_before, "val_after": val_after,
                                "params_after": count_params(model)}
            print(f"[{args.name}] thinned gates at step {step} to rank {args.gate_rank}: "
                  f"VAL {val_before:.4f} -> {val_after:.4f}, params {log['thin_swap']['params_after']}")
        for group in opt.param_groups:
            group["lr"] = lr_at(step, total_steps, args)
        loss_sum = 0.0
        for _ in range(accum):
            x, y = data.train_batch(args.micro_bs)
            with autocast:
                _, loss = step_model(x.to(device, non_blocking=True), y.to(device, non_blocking=True))
            scaler.scale(loss / accum).backward()
            loss_sum += loss.item() / accum
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
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
        "params": count_params(model),                  # the final model (differs from the start only for --thin_at)
        "flops_per_token": flops_per_token(model),
        "params_initial": params_initial,
        "flops_per_token_initial": flops_initial,
        "final_val_loss": log["val"][-1]["loss"],
        "tokens_per_second": tok_per_s,
        "wall_seconds": wall,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
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

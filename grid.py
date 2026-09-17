"""
Builds (and optionally runs) every training run of Experiment B.

    python grid.py --stage pilot            # print the commands for a stage
    python grid.py --stage pilot --run      # run them one after another (skips finished runs)
    python grid.py --table                  # print the parameter count of every arm, no training

Stages follow the day-by-day plan:
    pilot   Day 1-2: size S, seed 0: dense, shrunk, thin-gate r=d/4; then dense seed 1 (noise floor)
    lr      optional: learning-rate check for dense vs thin-gate at size S
    main_S  Day 4-5: all arms at size S, seeds 0-2
    main_M  Day 4-5: the five key arms at size M, seeds 0-1
    big_L   stretch: dense vs thin-gate at size L, one seed

Arms (r = rank, d = d_model, F = default d_ff):
    dense            standard SwiGLU
    thin_gate_rK     gate factorized to rank d/K, up/down dense                  <- the method
    thin_up_rK       control: only the up-projection factorized
    thin_down_rK     control: only the down-projection factorized
    shrunk_rK        dense SwiGLU with smaller d_ff, same parameter count as thin_gate_rK
    all_lowrank_rK   all three factorized at one rank, same parameter count as thin_gate_rK
    reinvest_rK      thin gate + wider d_ff, same parameter count as dense
"""

import argparse
import os
import subprocess
import sys

from train import SIZES

TOKENS = {"S": 300e6, "M": 500e6, "L": 1000e6}


def round_to(x, m):
    return max(m, int(round(x / m)) * m)


def arms_for(size, divisors):
    _, _, d, F = SIZES[size]
    arms = {"dense": {}}
    for K in divisors:
        r = d // K
        thin_params = 2 * d * F + r * (d + F)                 # MLP params per layer with a thin gate
        arms[f"thin_gate_r{K}"] = {"gate_rank": r}
        arms[f"thin_up_r{K}"] = {"up_rank": r}
        arms[f"thin_down_r{K}"] = {"down_rank": r}
        arms[f"shrunk_r{K}"] = {"d_ff": round_to(thin_params / (3 * d), 8)}
        R = round_to(thin_params / (3 * (d + F)), 8)
        arms[f"all_lowrank_r{K}"] = {"gate_rank": R, "up_rank": R, "down_rank": R}
        arms[f"reinvest_r{K}"] = {"gate_rank": r, "d_ff": round_to((3 * d * F - r * d) / (2 * d + r), 8)}
    return arms


def mlp_params(size, arm):
    _, _, d, F = SIZES[size]
    F = arm.get("d_ff", F)
    total = 0
    for key in ("gate_rank", "up_rank", "down_rank"):
        r = arm.get(key, 0)
        total += r * (d + F) if r else d * F
    return total


def runs_for(stage):
    """Returns a list of (name, size, arm_kwargs, seed, extra_args)."""
    out = []

    def add(size, arm_names, seeds, divisors=(2, 4, 8), extra=None, tag=""):
        arms = arms_for(size, divisors)
        for seed in seeds:
            for a in arm_names:
                out.append((f"{size}_{a}{tag}_s{seed}", size, arms[a], seed, extra or {}))

    if stage == "pilot":
        add("S", ["dense", "shrunk_r4", "thin_gate_r4"], [0])
        add("S", ["dense"], [1])
        add("S", ["thin_up_r4", "thin_down_r4"], [0])
    elif stage == "lr":
        for lr in (5e-4, 2e-3):
            add("S", ["dense", "thin_gate_r4"], [0], extra={"lr": lr}, tag=f"_lr{lr:g}")
    elif stage == "main_S":
        key = ["dense", "thin_gate_r4", "thin_up_r4", "thin_down_r4", "shrunk_r4"]
        rest = ["thin_gate_r2", "thin_gate_r8", "all_lowrank_r4", "reinvest_r4", "shrunk_r8", "shrunk_r2"]
        add("S", key, [0, 1, 2])          # most important arms first, so partial grids are still usable
        add("S", rest, [0, 1, 2])
    elif stage == "main_M":
        add("M", ["dense", "thin_gate_r4", "thin_up_r4", "thin_down_r4", "shrunk_r4"], [0, 1])
        add("M", ["reinvest_r4", "thin_gate_r8"], [0, 1])
    elif stage == "big_L":
        add("L", ["dense", "thin_gate_r4"], [0], extra={"micro_bs": 4})
    else:
        sys.exit(f"unknown stage {stage}")
    return out


def command(name, size, arm, seed, extra):
    parts = [sys.executable, "train.py", "--name", name, "--size", size, "--seed", str(seed),
             "--tokens", str(TOKENS[size])]
    for k, v in {**arm, **extra}.items():
        parts += [f"--{k}", str(v)]
    return parts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="pilot")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--out_dir", default="results/train")
    args = ap.parse_args()

    if args.table:
        for size in SIZES:
            base = mlp_params(size, {})
            print(f"\nsize {size}: MLP parameters per layer (dense = {base:,})")
            for name, arm in arms_for(size, (2, 4, 8)).items():
                p = mlp_params(size, arm)
                print(f"  {name:18s} {p:>10,}  ({100 * p / base:5.1f}% of dense)  {arm}")
        sys.exit()

    todo = runs_for(args.stage)
    for name, size, arm, seed, extra in todo:
        done = os.path.exists(os.path.join(args.out_dir, f"{name}.json"))
        cmd = command(name, size, arm, seed, extra)
        print(("[done] " if done else "[todo] ") + " ".join(cmd[1:]))
        if args.run and not done:
            # a failed run (e.g. out of memory) stops the grid so the error is visible
            subprocess.run(cmd, check=True)

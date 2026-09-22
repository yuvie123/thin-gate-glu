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
    screen  size S, seed 0: variants that might beat shrunk_r4 at the same parameter count (see screen_arms)
    screen2 size S, seeds 0-2: gates that are cheap in a different way (tied, thin+tied, shared) plus a
            starved-budget pair; every run carries a reference curve for the early-kill rule (see screen2_arms)

Arms (r = rank, d = d_model, F = default d_ff):
    dense            standard SwiGLU
    thin_gate_rK     gate factorized to rank d/K, up/down dense                  <- the method
    thin_up_rK       control: only the up-projection factorized
    thin_down_rK     control: only the down-projection factorized
    shrunk_rK        dense SwiGLU with smaller d_ff, same parameter count as thin_gate_rK
    all_lowrank_rK   all three factorized at one rank, same parameter count as thin_gate_rK
    reinvest_rK      thin gate + wider d_ff, same parameter count as dense

Screening arms (all matched to thin_gate_r4 / shrunk_r4):
    thin_gate_r4_{spectral,nowd,halfwd,spectral_nowd}   the same thin gate with a different factor init / decay
    grouped_{gate,up,down}_g4   one value per 4 hidden units for that projection, d_ff widened to compensate
    monarch_gate_b4             the gate is a Monarch (block-structured, full-rank) matrix with 4 blocks
    monarch_gate_b2_narrow      2 blocks, d_ff narrowed to compensate
    bottleneck_{,norm_}gate_r4  thin gate with a silu (and RMSNorm) between the two factors
    warm_gate_r4_f{10,25}       dense gate for the first 10% / 25% of steps, then thinned to rank d/4 (not a
                                matched from-scratch arm: it costs a little more training compute)

Screen 2 (matched to shrunk_r4 unless noted):
    tied_gate                   the gate IS the up-projection: h = silu(z) * z, zero gate parameters, d_ff 1200
    tied_gate_relu              same with relu: h = relu(z)^2 (Primer's squared ReLU)
    thin_tied_gate_r4           rank d/4 gate plus a per-unit scale times the up pre-activation, d_ff 1024
    shared_gate                 one dense gate matrix for all layers, d_ff widened to 1104
    shrunk_w400 / tied_gate_w600   the starved-budget pair (460,800 params/layer each)
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


def screen_arms(size, K=4):
    """Variants at the parameter count of thin_gate_rK (= shrunk_rK), size S seed 0 first."""
    _, _, d, F = SIZES[size]
    r = d // K
    thin_params = 2 * d * F + r * (d + F)
    Fg = round_to(thin_params / (d * (2 + 1 / K)), 8)            # grouped: d*F/K + 2*d*F = thin_params
    Fg -= Fg % K                                                 # the groups must divide d_ff
    F2 = round_to((thin_params - d * d // 2) / (2.5 * d), 8)     # Monarch nb=2: d*d/2 + d*F/2 + 2*d*F
    F2 -= F2 % 2
    return {
        f"monarch_gate_b{K}": {"gate_monarch": K},
        f"grouped_gate_g{K}": {"d_ff": Fg, "gate_groups": K},
        f"thin_gate_r{K}_spectral_nowd": {"gate_rank": r, "lowrank_init": "spectral", "factor_wd": "none"},
        f"warm_gate_r{K}_f25": {"gate_rank": r, "thin_at": 0.25},
        f"bottleneck_gate_r{K}": {"gate_rank": r, "bottleneck": "silu"},
        f"thin_gate_r{K}_nowd": {"gate_rank": r, "factor_wd": "none"},
        f"thin_gate_r{K}_spectral": {"gate_rank": r, "lowrank_init": "spectral"},
        f"warm_gate_r{K}_f10": {"gate_rank": r, "thin_at": 0.1},
        f"bottleneck_norm_gate_r{K}": {"gate_rank": r, "bottleneck": "norm_silu"},
        f"grouped_up_g{K}": {"d_ff": Fg, "up_groups": K},
        f"grouped_down_g{K}": {"d_ff": Fg, "down_groups": K},
        f"thin_gate_r{K}_halfwd": {"gate_rank": r, "factor_wd": "half"},
        "monarch_gate_b2_narrow": {"gate_monarch": 2, "d_ff": F2},
    }


def screen2_arms(size, K=4):
    """Gates that are cheap in a different way than low rank, at the parameter count of shrunk_rK, plus a
    starved-budget pair. Priority order; every arm runs at seeds 0-2 with the early-kill rule."""
    L, _, d, F = SIZES[size]
    r = d // K
    thin_params = 2 * d * F + r * (d + F)
    Ft = round_to(thin_params / (2 * d), 8)                      # tied: no gate parameters at all
    Fs = round_to(thin_params / (d * (2 + 1 / L)), 8)            # shared: one gate matrix per L layers
    Fs -= Fs % 8
    return {
        "tied_gate": {"d_ff": Ft, "gate_tie": "up"},
        "tied_gate_relu": {"d_ff": Ft, "gate_tie": "up", "gate_act": "relu"},
        f"thin_tied_gate_r{K}": {"gate_rank": r, "gate_tie": "up"},
        "shared_gate": {"d_ff": Fs, "gate_shared": 1},
        "shrunk_w400": {"d_ff": 400},                             # starved budget: 3 * d * 400
        "tied_gate_w600": {"d_ff": 600, "gate_tie": "up"},        # the same budget, gate tied, 1.5x wider
    }


def mlp_params(size, arm):
    """MLP parameters per layer for an arm dict (the keys are train.py flags)."""
    L, _, d, F = SIZES[size]
    F = arm.get("d_ff", F)
    total = 0
    for proj, (n_in, n_out) in (("gate", (d, F)), ("up", (d, F)), ("down", (F, d))):
        r, g, nb = arm.get(f"{proj}_rank", 0), arm.get(f"{proj}_groups", 1), arm.get(f"{proj}_monarch", 0)
        if proj == "gate" and arm.get("gate_tie") == "up":
            total += r * (n_in + n_out) + n_out if r else 0      # optional rank-r term plus a per-unit scale
        elif proj == "gate" and arm.get("gate_shared"):
            assert (n_in * n_out) % L == 0, "shared gate: d * d_ff must divide by the layer count"
            total += n_in * n_out // L                           # one matrix, charged evenly to the layers
        elif r:
            total += r * (n_in + n_out) + (r if arm.get("bottleneck") == "norm_silu" else 0)
        elif nb:
            total += n_in * n_in // nb + n_in * n_out // nb          # two block-diagonal factors
        else:
            total += n_in * n_out // g
    return total


def runs_for(stage):
    """Returns a list of (name, size, arm_kwargs, seed, extra_args)."""
    out = []

    def add(size, arm_names, seeds, divisors=(2, 4, 8), extra=None, tag="", arms=None):
        arms = arms or arms_for(size, divisors)
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
    elif stage == "screen":
        arms = screen_arms("S")
        add("S", list(arms), [0], arms=arms)      # priority order: the launch deadline drops the tail
    elif stage == "screen2":
        arms = screen2_arms("S")
        for seed in (0, 1, 2):                    # round 1 = every arm at seed 0, then seeds 1 and 2
            for a in arms:
                ref = None if a == "shrunk_w400" else ("S_shrunk_w400" if a == "tied_gate_w600" else "S_shrunk_r4")
                extra = {"ref_json": f"results/train/{ref}_s{seed}.json"} if ref else {}
                out.append((f"S_{a}_s{seed}", "S", arms[a], seed, extra))
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
            for name, arm in {**arms_for(size, (2, 4, 8)), **screen_arms(size), **screen2_arms(size)}.items():
                p = mlp_params(size, arm)
                print(f"  {name:26s} {p:>10,}  ({100 * p / base:5.1f}% of dense)  {arm}")
        sys.exit()

    todo = runs_for(args.stage)
    for name, size, arm, seed, extra in todo:
        done = os.path.exists(os.path.join(args.out_dir, f"{name}.json"))
        cmd = command(name, size, arm, seed, extra)
        print(("[done] " if done else "[todo] ") + " ".join(cmd[1:]))
        if args.run and not done:
            # a failed run (e.g. out of memory) stops the grid so the error is visible
            subprocess.run(cmd, check=True)

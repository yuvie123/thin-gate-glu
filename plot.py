"""
Turns results/*.json into the paper's figures and tables. Never type a number into the paper by hand:
rerun this script and \\input the generated .tex files.

    python plot.py                       # reads results/, writes paper/figures/*.pdf and paper/tables/*.tex
    python plot.py --results_dir results/smoke --out_dir /tmp/plots    # toy check

Colours are a colour-blind-checked categorical set, and every series also has its own marker and line
style so the figures survive greyscale printing. gate = blue, up = orange, down = green, everywhere.
"""

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {   # entity -> (colour, marker, linestyle, label)
    "gate_proj": ("#2a78d6", "o", "-", "gate"),
    "up_proj": ("#eb6834", "s", "--", "up"),
    "down_proj": ("#1baf7a", "^", ":", "down"),
}
NEUTRAL, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "pdf.fonttype": 42,   # embed TrueType fonts (required by most conferences)
})


# ---------------------------------------------------------------- Experiment A
def plot_posthoc(results, method, out_dir, xkey="rank_frac"):
    """xkey: "rank_frac" (rank / full rank, same for every model) or "param_ratio" (parameters of the
    factorized projection / dense, which differs between model families because d_ff/d differs)."""
    results = [r for r in results if r["method"] == method]
    if not results:
        return
    results.sort(key=lambda r: r["n_params"])
    n = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(1.75 * cols + 0.4, 1.6 * rows + 0.5), squeeze=False, sharex=True)
    for ax, res in zip(axes.flat, results):
        for ptype, (color, marker, ls, label) in STYLE.items():
            pts = sorted((r[xkey], r["ppl"] / res["base_ppl"]) for r in res["records"] if r["type"] == ptype)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, marker=marker, ls=ls, lw=1.5, ms=4, label=label, clip_on=False)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, which="major", axis="y")
        ax.set_title(f"{res['model'].split('/')[-1]}", color=INK)
        if xkey == "rank_frac":
            ax.set_xticks([1 / 16, 1 / 8, 1 / 4, 1 / 2])
            ax.set_xticklabels(["1/16", "1/8", "1/4", "1/2"])
        else:
            ax.set_xticks([1 / 8, 1 / 4, 1 / 2, 1])
            ax.set_xticklabels(["1/8", "1/4", "1/2", "1"])
            ax.axvline(1.0, color=NEUTRAL, lw=0.6, ls="-")      # dense parameter count
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("rank / full rank" if xkey == "rank_frac" else "projection parameters / dense")
    for ax in axes[:, 0]:
        ax.set_ylabel("perplexity / baseline")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="truncated projection", loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    suffix = "" if xkey == "rank_frac" else "_params"
    path = os.path.join(out_dir, "figures", f"posthoc_{method}{suffix}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)

    if xkey != "rank_frac":
        return
    # table: perplexity at each rank
    fracs = sorted({r["rank_frac"] for res in results for r in res["records"]}, reverse=True)
    lines = ["\\begin{tabular}{ll" + "r" * len(fracs) + "}", "\\toprule",
             "Model & Proj. & " + " & ".join(f"$r/d={f:g}$" for f in fracs) + " \\\\", "\\midrule"]
    for res in results:
        for ptype in STYLE:
            vals = {r["rank_frac"]: r["ppl"] for r in res["records"] if r["type"] == ptype}
            cells = " & ".join(f"{vals[f]:.1f}" if f in vals else "--" for f in fracs)
            short = res["model"].split("/")[-1].replace("_", "\\_")      # LaTeX: "_" is a subscript
            first = f"{short} ({res['base_ppl']:.1f})" if ptype == "gate_proj" else ""
            lines.append(f"{first} & {STYLE[ptype][3]} & {cells} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    path = os.path.join(out_dir, "tables", f"posthoc_{method}.tex")
    open(path, "w").write("\n".join(lines) + "\n")
    print("wrote", path)


# ---------------------------------------------------------------- Experiment B
def arm_of(name):            # "S_thin_gate_r4_s0" -> ("S", "thin_gate_r4")
    if "_" not in name:
        return "?", name
    size, rest = name.split("_", 1)
    return size, rest.rsplit("_s", 1)[0]


def arm_color(arm):
    if arm.startswith(("thin_gate", "reinvest", "monarch_gate", "grouped_gate", "bottleneck", "warm_gate")):
        return STYLE["gate_proj"][0]
    if arm.startswith(("thin_up", "grouped_up")):
        return STYLE["up_proj"][0]
    if arm.startswith(("thin_down", "grouped_down")):
        return STYLE["down_proj"][0]
    return NEUTRAL


def plot_training(runs, out_dir):
    runs = [r for r in runs if "_lr" not in r["name"]]
    if not runs:
        return
    by = defaultdict(lambda: defaultdict(list))
    for r in runs:
        size, arm = arm_of(r["name"])
        by[size][arm].append(r)

    lines = ["\\begin{tabular}{llrrrr}", "\\toprule",
             "Size & Arm & MLP params & Val. loss (mean $\\pm$ std) & Seeds & Tokens/s \\\\", "\\midrule"]
    fig, axes = plt.subplots(1, len(by), figsize=(3.2 * len(by), 0.24 * max(len(a) for a in by.values()) + 1.0),
                             squeeze=False)
    for ax, (size, arms) in zip(axes[0], sorted(by.items(), key=lambda kv: "SML?".find(kv[0]))):
        order = sorted(arms, key=lambda a: statistics.mean(r["final_val_loss"] for r in arms[a]))
        for i, arm in enumerate(order):
            losses = [r["final_val_loss"] for r in arms[arm]]
            mean = statistics.mean(losses)
            std = statistics.stdev(losses) if len(losses) > 1 else 0.0
            ax.errorbar(mean, i, xerr=std, color=arm_color(arm), marker="o", ms=5, lw=1.5, capsize=2, zorder=3)
            ax.scatter(losses, [i] * len(losses), s=8, color=INK, alpha=0.5, zorder=4, linewidths=0)
            speed = statistics.mean(r["tokens_per_second"] or 0 for r in arms[arm])
            std_txt = f"{std:.4f}" if len(losses) > 1 else "n/a"
            lines.append(f"{size} & {arm.replace('_', ' ')} & {arms[arm][0]['params']['mlp']:,} & "
                         f"{mean:.4f} $\\pm$ {std_txt} & {len(losses)} & {speed:,.0f} \\\\")
        if "dense" in arms:     # reference line + seed-noise band of the dense baseline
            d = [r["final_val_loss"] for r in arms["dense"]]
            ax.axvline(statistics.mean(d), color=MUTED, lw=0.8, zorder=1)
            if len(d) > 1:
                ax.axvspan(min(d), max(d), color=NEUTRAL, alpha=0.18, lw=0, zorder=0)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([a.replace("_", " ") for a in order])
        ax.invert_yaxis()
        ax.set_xlabel("validation loss (lower is better)")
        ax.set_title(f"size {size}  (band = range of dense seeds)", color=INK)
        ax.grid(True, axis="x")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    fig.tight_layout()
    path = os.path.join(out_dir, "figures", "training.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)
    path = os.path.join(out_dir, "tables", "training.tex")
    open(path, "w").write("\n".join(lines) + "\n")
    print("wrote", path)


# ---------------------------------------------------------------- Experiment C
def table_heal(results, out_dir):
    if not results:
        return
    lines = ["\\begin{tabular}{lllrrrr}", "\\toprule",
             "Model & Proj. & Init & Pretrained & Truncated & Healed & Gap recovered \\\\", "\\midrule"]
    for r in sorted(results, key=lambda r: (r["model"], r["rank_frac"], r["type"], r["init"])):
        gap = r["ppl_truncated"] - r["ppl_pretrained"]
        rec = 100 * (r["ppl_truncated"] - r["ppl_healed"]) / gap if gap > 0 else float("nan")
        lines.append(f"{r['model'].split('/')[-1]} ($r/d={r['rank_frac']:g}$) & {STYLE[r['type']][3]} & {r['init']} & "
                     f"{r['ppl_pretrained']:.2f} & {r['ppl_truncated']:.2f} & {r['ppl_healed']:.2f} & {rec:.0f}\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path = os.path.join(out_dir, "tables", "heal.tex")
    open(path, "w").write("\n".join(lines) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="paper")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "tables"), exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.results_dir, "**", "*.json"), recursive=True))
    skip = [d for d in ("smoke", "scratch") if d not in args.results_dir]      # toy / throwaway runs never reach the paper
    found = [json.load(open(p)) for p in paths if not any(os.sep + d + os.sep in p for d in skip)]
    posthoc = [r for r in found if "records" in r]
    training = [r for r in found if "final_val_loss" in r]
    heal = [r for r in found if "ppl_healed" in r]
    print(f"found {len(posthoc)} post-hoc, {len(training)} training, {len(heal)} healing result files")
    for method in ("plain", "whiten"):
        plot_posthoc(posthoc, method, args.out_dir)
        plot_posthoc(posthoc, method, args.out_dir, xkey="param_ratio")
    plot_training(training, args.out_dir)
    table_heal(heal, args.out_dir)

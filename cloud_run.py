"""
Runs the experiments on a free or rented cloud GPU (Kaggle, Colab) and packs the small result files for download.
The notebook built by make_cloud_notebook.py calls this; it also works from any shell.

    python cloud_run.py check         # GPU, library versions, tests.py, smoke runs
    python cloud_run.py posthoc       # Exp. A: every model, whitened then plain, one queue per GPU
    python cloud_run.py throughput    # Exp. B: 20M-token speed check, projects the cost of one run
    python cloud_run.py pilot         # Exp. B: the pilot stage of grid.py, runs spread over the GPUs
    python cloud_run.py grid --stage main_S     # Exp. B: any other stage of grid.py
    python cloud_run.py heal          # Exp. C: gate vs up vs down at the same rank, on the two small models
    python cloud_run.py bench         # measured tokens/s and memory of every arm
    python cloud_run.py pack          # zip results/ and logs/ for download

Add --dry to print what would run without running it. Finished runs are skipped, so a second call resumes.

Precision. Free cloud GPUs are usually T4s, which have no native bfloat16. Exp. A therefore runs in float32
on EVERY model here, so that the whole study has one precision; the dtype is recorded in each JSON. Exp. B
falls back to float16 with loss scaling (see train.py --amp_dtype). Never pool Exp. B runs from different
precisions or GPUs in one table: one table, one recipe.
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import zipfile

# Same models, same order (smallest first) as run_posthoc.sh. The number is the evaluation batch size:
# it does not change the perplexity, only the memory needed. 2 keeps the 1B+ models inside 16 GB in float32.
MODELS = [
    ("HuggingFaceTB/SmolLM2-135M", 4),
    ("HuggingFaceTB/SmolLM2-360M", 4),
    ("Qwen/Qwen2.5-0.5B", 4),
    ("TinyLlama/TinyLlama_v1.1", 2),
    ("meta-llama/Llama-3.2-1B", 2),      # gated: needs an HF token with the licence accepted, else skipped
    ("allenai/OLMo-2-0425-1B", 2),
    ("Qwen/Qwen2.5-1.5B", 2),
    ("HuggingFaceTB/SmolLM2-1.7B", 2),
]
GATED = {"meta-llama/Llama-3.2-1B"}
PACK_DIRS = ["results/posthoc", "results/train", "results/heal", "results/bench", "results/scratch", "logs"]
HEAL_MODELS = ["HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M"]      # as in README, Day 4-5
ZIP_NAME = "thin_gate_results.zip"
SESSION_START = os.path.join("logs", "_session_start.txt")   # written by `check`; --max_hours counts from it

print_lock = threading.Lock()
pack_lock = threading.Lock()
DEADLINE = None     # epoch seconds; set by --max_hours. After it, no new job starts (the running ones finish).


def say(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with print_lock:
        print(line, flush=True)
        os.makedirs("logs", exist_ok=True)
        with open(os.path.join("logs", "_runner.log"), "a") as f:      # so the zip tells the whole story
            print(line, file=f)


def out_dir():
    """Where the platform keeps files for download after the session ends."""
    for d in ("/kaggle/working", "/content"):
        if os.path.isdir(d):
            return d
    return os.getcwd()


def gpu_info():
    import torch
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    caps = [torch.cuda.get_device_capability(i) for i in range(n)]
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    return n, caps, names


def pack():
    with pack_lock:
        path = os.path.join(out_dir(), ZIP_NAME)
        tmp = path + ".tmp"
        count = 0
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for d in PACK_DIRS:
                for root, _, files in os.walk(d):
                    for f in sorted(files):
                        if f.endswith((".json", ".log", ".txt")):
                            full = os.path.join(root, f)
                            z.write(full, full.replace(os.sep, "/"))
                            count += 1
        os.replace(tmp, path)
        return path, count


def worker(gpu, units, dry):
    """One worker per GPU. Each takes the next unit of work until none are left, so the GPUs stay balanced."""
    while True:
        try:
            jobs = units.get_nowait()
        except queue.Empty:
            return
        run_jobs(gpu, jobs, dry)


def run_jobs(gpu, jobs, dry):
    """jobs: list of (label, command, expected_json). Runs them one after another on one GPU."""
    for label, cmd, expected in jobs:
        if os.path.exists(expected):
            say(f"gpu{gpu} [done]   {label}")
            continue
        base, sep, seed = expected[:-5].rpartition("_s")
        if sep and seed.isdigit() and seed != "0" and killed_run(f"{base}_s0.json"):
            # a variant whose seed 0 was stopped by the reference rule does not get more seeds (no file is
            # written, so the decision can be revisited by hand later)
            say(f"gpu{gpu} [skip]   {label}: seed 0 was killed by the reference rule")
            continue
        if dry:
            say(f"gpu{gpu} [would]  {label}: {' '.join(cmd[1:])}")
            continue
        if DEADLINE is not None and time.time() > DEADLINE:
            # a Kaggle commit that overruns its 12 h is killed and its output is not saved, so a job that
            # cannot finish in time is left for the next session instead of started
            say(f"gpu{gpu} [skip]   {label}: past the launch deadline, left for the next session")
            continue
        log_path = os.path.join("logs", label + ".log")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
        say(f"gpu{gpu} [start]  {label}")
        t0 = time.time()
        with open(log_path, "w") as log:
            code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        mins = (time.time() - t0) / 60
        if code == 0 and os.path.exists(expected):
            say(f"gpu{gpu} [ok]     {label}  ({mins:.0f} min)  {summarize(expected)}")
        else:
            # keep going: one model that fails (out of memory, gated, download error) must not cost the others
            say(f"gpu{gpu} [FAILED] {label}  (exit {code}, {mins:.0f} min). Last lines of {log_path}:")
            with open(log_path) as log:
                for line in log.readlines()[-8:]:
                    say("      " + line.rstrip())
        pack()


def killed_run(path):
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return "killed" in json.load(f)


def summarize(path):
    with open(path) as f:
        r = json.load(f)
    if "killed" in r:
        k = r["killed"]
        return f"KILLED at step {k['step']}: val {k['val']:.4f} vs reference {k['ref_val']:.4f} (+{k['margin']:.3f} allowed)"
    if "base_ppl" in r:
        note = ""
        if not 3 < r["base_ppl"] < 100:
            note = "   <-- STOP: a baseline outside the tens means the evaluation is broken for this model"
        return f"baseline perplexity {r['base_ppl']:.3f}, {len(r['records'])} settings, {r['dtype']} on {r['device']}{note}"
    if "final_val_loss" in r:
        tps = r.get("tokens_per_second")
        return f"final val loss {r['final_val_loss']:.4f}, {round(tps) if tps else '?'} tok/s, {r.get('amp_dtype')}"
    return ""


def run_units(unit_list, dry):
    """unit_list: list of job lists. A unit stays on one GPU; units are handed out in order, first come first served."""
    os.makedirs("logs", exist_ok=True)
    n, _, _ = gpu_info()
    units = queue.Queue()
    for u in unit_list:
        units.put(u)
    threads = [threading.Thread(target=worker, args=(g, units, dry)) for g in range(max(1, n))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if not dry:
        path, count = pack()
        say(f"packed {count} files into {path}")


# ---------------------------------------------------------------------------------------------- subcommands
def cmd_check(args):
    import torch
    import transformers
    import datasets
    n, caps, names = gpu_info()
    say(f"torch {torch.__version__} | transformers {transformers.__version__} | datasets {datasets.__version__}")
    if n == 0:
        say("NO GPU VISIBLE. On Kaggle: Settings -> Accelerator -> GPU T4 x2. On Colab: Runtime -> Change runtime type.")
    for i in range(n):
        gb = torch.cuda.get_device_properties(i).total_memory / 1e9
        native = "native bfloat16" if caps[i][0] >= 8 else "no native bfloat16 -> float32 (Exp. A) / float16 (Exp. B)"
        say(f"gpu{i}: {names[i]}, {gb:.1f} GB, compute capability {caps[i][0]}.{caps[i][1]}, {native}")
        if caps[i][0] < 7:
            say(f"gpu{i}: too old for torch.compile; Exp. B will run with --no_compile (slower). Prefer a T4.")
    if args.dry:
        return
    os.makedirs("logs", exist_ok=True)
    with open(SESSION_START, "w") as f:                 # every later step's --max_hours counts from here
        f.write(str(time.time()))
    smoke = [sys.executable, "train.py", "--smoke"]
    for label, cmd in [("tests.py", [sys.executable, "tests.py"]),
                       ("posthoc smoke", [sys.executable, "posthoc_truncate.py", "--smoke"]),
                       ("train smoke", smoke),
                       # every screening code path, compiled on the GPU, before any real run starts
                       ("smoke monarch", smoke + ["--name", "smoke_monarch", "--gate_monarch", "2"]),
                       ("smoke grouped", smoke + ["--name", "smoke_grouped", "--gate_groups", "2", "--gate_rank", "0"]),
                       ("smoke spectral", smoke + ["--name", "smoke_spectral", "--lowrank_init", "spectral", "--factor_wd", "none"]),
                       ("smoke bottleneck", smoke + ["--name", "smoke_bottleneck", "--bottleneck", "norm_silu"]),
                       ("smoke warm start", smoke + ["--name", "smoke_warm", "--thin_at", "0.5"]),
                       ("smoke tied", smoke + ["--name", "smoke_tied", "--gate_tie", "up"]),
                       ("smoke tied relu", smoke + ["--name", "smoke_tied_relu", "--gate_tie", "up", "--gate_act", "relu"]),
                       ("smoke thin tied", smoke + ["--name", "smoke_thin_tied", "--gate_tie", "up", "--gate_rank", "16"]),
                       ("smoke shared", smoke + ["--name", "smoke_shared", "--gate_shared", "1"]),
                       ("smoke pair tied", smoke + ["--name", "smoke_pair", "--gate_tie", "pair", "--gate_act", "relu"]),
                       ("smoke kill rule", smoke + ["--name", "smoke_killed", "--ref_json", "results/smoke/smoke.json",
                                                    "--kill_steps", "10:-10", "--expect_killed"])]:
        r = subprocess.run(cmd, capture_output=True, text=True)
        tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
        say(f"{label}: {'ok' if r.returncode == 0 else 'FAILED'}   {tail[0]}")
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:])
            sys.exit(f"{label} failed; do not trust any number from this machine until it passes")


def cmd_posthoc(args):
    has_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    wanted = [m for m in MODELS if not args.models or m[0] in args.models.split(",")]
    units = []
    for model, batch in wanted:
        if model in GATED and not has_token:
            say(f"skipping {model}: gated model and no HF_TOKEN set (optional; see the notebook)")
            continue
        tag = model.replace("/", "__")
        unit = []
        for method in ("whiten", "plain"):          # whitened first: it is the figure the go/no-go rests on
            cmd = [sys.executable, "posthoc_truncate.py", "--model", model, "--dtype", "float32",
                   "--batch_size", str(batch)] + (["--whiten"] if method == "whiten" else [])
            expected = os.path.join("results", "posthoc", f"{tag}__wikitext2__{method}.json")
            unit.append((f"posthoc__{tag}__{method}", cmd, expected))
        units.append(unit)
    if not args.dry:
        # fetch the evaluation text once, so two queues never race to download the same dataset
        from hf_utils import eval_text
        eval_text("wikitext2", "test")
        eval_text("wikitext2", "train")
    run_units(units, args.dry)


def ensure_data(dry):
    if os.path.exists(os.path.join("data", "fineweb10B", "fineweb_val_000000.bin")):
        return
    say("downloading the pretraining tokens (about 1.4 GB, once)")
    if not dry:
        subprocess.run([sys.executable, "data.py", "--num_train_shards", "6"], check=True)


def compile_flag():
    n, caps, _ = gpu_info()
    return ["--no_compile"] if n and caps[0][0] < 7 else []


def cmd_throughput(args):
    from grid import TOKENS
    ensure_data(args.dry)
    expected = os.path.join("results", "scratch", "throughput_check.json")
    cmd = [sys.executable, "train.py", "--name", "throughput_check", "--size", "S", "--tokens", "20e6",
           "--out_dir", "results/scratch"] + compile_flag()
    run_units([[("throughput_check", cmd, expected)]], args.dry)
    if args.dry or not os.path.exists(expected):
        return
    with open(expected) as f:
        tps = json.load(f)["tokens_per_second"]
    hours = TOKENS["S"] / tps / 3600
    n, _, _ = gpu_info()
    say(f"{tps:,.0f} tokens/s -> one size-S run of {TOKENS['S'] / 1e6:.0f}M tokens takes about {hours:.1f} h")
    say(f"the 6 pilot runs on {max(1, n)} GPU(s) would take about {hours * -(-6 // max(1, n)):.1f} h")
    if hours > 1.5:
        say("That is over the ~1.5 h budget. STOP here and lower TOKENS['S'] in grid.py BEFORE any pilot run:")
        say("every arm and seed must use the same token budget, so it cannot change once a grid has started.")


def cmd_grid(args):
    from grid import command, runs_for
    ensure_data(args.dry)
    units = []
    for name, size, arm, seed, extra in runs_for(args.stage):
        cmd = command(name, size, arm, seed, extra) + compile_flag()
        units.append([(f"train__{name}", cmd, os.path.join("results", "train", f"{name}.json"))])
    run_units(units, args.dry)


def cmd_heal(args):
    units = []
    for model in HEAL_MODELS:
        tag = model.replace("/", "__")
        unit = []
        for ptype in ("gate_proj", "up_proj", "down_proj"):      # same rank, same data order, same seed
            cmd = [sys.executable, "heal.py", "--model", model, "--type", ptype, "--rank_frac", "0.25", "--whiten"]
            expected = os.path.join("results", "heal", f"{tag}__{ptype}__r0.25__whiten__s0.json")
            unit.append((f"heal__{tag}__{ptype}", cmd, expected))
        units.append(unit)
    run_units(units, args.dry)


def cmd_bench(args):
    units = [[(f"bench_{size}", [sys.executable, "bench.py", "--size", size] + compile_flag(),
               os.path.join("results", "bench", f"bench_{size}.json"))] for size in ("S", "M")]
    run_units(units, args.dry)


def cmd_pack(args):
    path, count = pack()
    say(f"packed {count} files into {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["check", "posthoc", "throughput", "pilot", "grid", "heal", "bench", "pack"])
    ap.add_argument("--models", default="", help="posthoc: comma-separated subset of the model list")
    ap.add_argument("--stage", default="pilot", help="grid: which grid.py stage to run")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--max_hours", type=float, default=0.0,
                    help="stop starting new jobs once this many hours minus --job_hours have passed (0 = no cap). "
                         "Use 11 on Kaggle so a 12 h commit finishes and keeps its output.")
    ap.add_argument("--job_hours", type=float, default=1.7,
                    help="how long one job may take; the last job starts this long before --max_hours")
    args = ap.parse_args()
    if args.max_hours > 0:
        # The budget counts from the session's check step (so several steps in one notebook share one clock),
        # or from now when this command runs on its own.
        start = time.time()
        if os.path.exists(SESSION_START):
            with open(SESSION_START) as f:
                start = float(f.read())
        elapsed = (time.time() - start) / 3600
        DEADLINE = start + (args.max_hours - args.job_hours) * 3600
        say(f"launch deadline: no new job after {args.max_hours - args.job_hours:.1f} h of the session "
            f"({elapsed:.1f} h used so far; --max_hours {args.max_hours:g}, --job_hours {args.job_hours:g}); "
            f"running jobs finish")
    if args.what == "pilot":
        args.stage = "pilot"
    {"check": cmd_check, "posthoc": cmd_posthoc, "throughput": cmd_throughput, "pilot": cmd_grid,
     "grid": cmd_grid, "heal": cmd_heal, "bench": cmd_bench, "pack": cmd_pack}[args.what](args)

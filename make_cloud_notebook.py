"""
Builds cloud_notebook.ipynb: one self-contained notebook that carries a copy of the experiment code, so the
experiments can run on a free cloud GPU (Kaggle, Colab) without giving anyone access to the private repo.

    python make_cloud_notebook.py                          # session 1: speed check + Exp. A
    python make_cloud_notebook.py --run pilot              # a later session: Exp. B pilot
    python make_cloud_notebook.py --run main_S,heal,bench  # steps run in the order given

Kaggle sessions start empty and stop after 12 hours. So every finished result file found under results/ on
this machine is packed INTO the notebook, and the runner skips whatever is already done. The loop is: run,
download the zip, unzip it into this folder, rebuild the notebook, upload, run again.

Re-run this after ANY change to the code, and upload the fresh notebook: the copy inside an old notebook does
not update itself. The notebook is git-ignored because it is generated; this script and cloud_run.py are the
source. Standard library only, so it also runs on the laptop.
"""

import argparse
import base64
import io
import json
import os
import subprocess
import zipfile

FILES = ["hf_utils.py", "posthoc_truncate.py", "model.py", "data.py", "train.py", "grid.py", "heal.py",
         "bench.py", "tests.py", "cloud_run.py"]
OUT = "cloud_notebook.ipynb"
CARRY_DIRS = ["results/posthoc", "results/train", "results/heal", "results/bench"]   # never smoke or scratch
STEPS = {   # name -> (heading, command, note)
    "throughput": ("Speed check for Experiment B", "python cloud_run.py throughput",
                   "Trains the smallest model on 20M tokens and says how long one real run would take. This "
                   "number decides the token budget, which must be fixed BEFORE any grid run starts."),
    "posthoc": ("Experiment A: which projection tolerates rank reduction?", "python cloud_run.py posthoc",
                "Every model, whitened SVD then plain SVD, smallest first. Look for SmolLM2-135M baseline "
                "perplexity close to 17.46, the value measured on the CPU."),
    "pilot": ("Experiment B: pilot runs", "python cloud_run.py pilot",
              "Dense (two seeds, which gives the noise floor), shrunk, thin-gate, thin-up, thin-down at size S."),
    "main_S": ("Experiment B: all arms at size S, three seeds", "python cloud_run.py grid --stage main_S --max_hours 11", ""),
    "main_M": ("Experiment B: key arms at size M, two seeds", "python cloud_run.py grid --stage main_M --max_hours 11", ""),
    "screen": ("Experiment B: screen of gate variants at the shrunk_r4 budget (size S, seed 0)",
               "python cloud_run.py grid --stage screen --max_hours 11",
               "Thirteen variants in priority order (grid.py screen_arms); the launch deadline drops the tail. "
               "Compare each final loss with S_shrunk_r4_s0 = 4.1043 and S_dense_s0 = 4.0867."),
    "screen2": ("Experiment B: screen 2, gates that are cheap in a different way (size S, seeds 0-2)",
                "python cloud_run.py grid --stage screen2 --max_hours 11",
                "Tied, tied-relu, thin+tied and shared gates at the shrunk_r4 budget, plus shrunk 400 vs tied 600 "
                "(grid.py screen2_arms). Every run compares itself with the same-seed shrunk reference at steps "
                "2000 and 3000 and stops early if it is more than 0.015 / 0.010 behind (train.py --ref_json). References: "
                "S_shrunk_r4 s0/s1/s2 = 4.1043 / 4.1043 / 4.1124."),
    "screen3": ("Experiment B: screen 3, controls for the relu-tied gate (size S)",
                "python cloud_run.py grid --stage screen3 --max_hours 11",
                "Context-thresholded self-gating h = relu(z + BAx) z at rank d/4 (three seeds) and at ranks d/8 and d/2, "
                "its zero-init and affine variants, the controls shrunk_relu_r4 (three seeds) and thin_gate_relu_r4 (relu "
                "low-rank gate without the tie), then one-seed diagnostics (grid.py screen3_arms). All at shrunk_r4's "
                "parameters unless noted. References: S_shrunk_r4 s0/s1/s2 = 4.1043 / 4.1043 / 4.1124, "
                "S_tied_gate_relu s0/s1/s2 = 4.0814 / 4.0757 / 4.0860, S_shrunk_w400 s0/s1/s2 = 4.1602 / 4.1535 / 4.1608."),
    "lr": ("Experiment B: learning-rate check", "python cloud_run.py grid --stage lr --max_hours 11", ""),
    "heal": ("Experiment C: truncate, then train only the new factors", "python cloud_run.py heal", ""),
    "bench": ("Measured speed and memory of every arm", "python cloud_run.py bench", ""),
}


def git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def carried_results():
    """Finished result files on this machine, zipped and base64-encoded, plus their names."""
    names, buf = [], io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for d in CARRY_DIRS:
            for root, _, files in os.walk(d):
                for f in sorted(files):
                    if f.endswith(".json"):
                        full = os.path.join(root, f)
                        arc = full.replace(os.sep, "/")
                        z.write(full, arc)
                        names.append(arc)
    return names, base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="throughput,posthoc", help="steps, in order: " + ", ".join(STEPS))
    args = ap.parse_args()
    plan = [x.strip() for x in args.run.split(",") if x.strip()]
    for step in plan:
        assert step in STEPS, f"unknown step {step!r}; choose from {list(STEPS)}"
    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    dirty = bool(git("status", "--porcelain", "--", *FILES))
    stamp = commit + (" plus uncommitted changes" if dirty else "")

    cells = [md(f"""
# Thin-gate GLU: experiments on a cloud GPU

Code snapshot: `{stamp}`. This notebook carries its own copy of the code; nothing is cloned.

**Before running (Kaggle):** Settings -> Accelerator -> **GPU T4 x2**; Settings -> Internet -> **On**
(needs a phone-verified account); keep the notebook **private**. Then *Save Version -> Save & Run All
(Commit)*: it runs in the background for up to 12 hours even with the browser closed. When it finishes, open
the version's **Output** tab and download `thin_gate_results.zip`.

**Colab:** Runtime -> Change runtime type -> T4 GPU, then Runtime -> Run all. Keep the tab open; download
`/content/thin_gate_results.zip` from the file browser at the end. Colab has one GPU and shorter sessions, so
it will not finish everything in one go. Unzip what it produced, rebuild the notebook, and run again.

**What comes back:** small JSON result files and logs, zipped. Unzip into the project folder so the files
land in `results/posthoc/` (and `results/train/` for the pilot), then commit them. No weights, no data.

**Precision:** Exp. A runs in float32 on every model (a T4 has no native bfloat16), so the whole study has
one precision. Each JSON records its dtype and GPU.
"""),
             code("""
# 1. Libraries. Kaggle and Colab already have PyTorch; only upgrade transformers if it is too old for the code.
import importlib.metadata as md, subprocess, sys
def too_old(pkg, minimum):
    try:
        have = tuple(int(x) for x in md.version(pkg).split(".")[:2])
    except Exception:
        return True
    return have < minimum
if too_old("transformers", (4, 56)):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.56", "accelerate"])
if too_old("datasets", (2, 14)):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "datasets"])
print("transformers", md.version("transformers"), "| datasets", md.version("datasets"), "| torch", md.version("torch"))
"""),
             code("""
# 2. A working folder OUTSIDE the download area, so the 1.4 GB of training tokens never end up in the output.
import os
WORK = "/tmp/thin_gate" if os.path.isdir("/tmp") else os.path.abspath("thin_gate")
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)
print("working in", os.getcwd())
"""),
             md("## 3. The code (written to disk exactly as it is in the repository)")]

    for name in FILES:
        with open(name, encoding="utf-8") as f:
            source = f.read().replace("\r\n", "\n")
        cells.append(code(f"%%writefile {name}\n{source}"))

    cells += [
        md("""
## 4. Optional: gated model
`meta-llama/Llama-3.2-1B` needs a Hugging Face token whose account has accepted the model licence. On Kaggle:
Add-ons -> Secrets -> add `HF_TOKEN`. Without it that one model is skipped and everything else still runs.
"""),
        code("""
import os
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle secrets")
except Exception:
    print("no HF_TOKEN secret: the gated Llama model will be skipped (fine)")
"""),
    ]

    names, blob = carried_results()
    if names:
        listing = "\n".join(f"#   {n}" for n in names)
        cells += [
            md(f"## 5. Results from earlier sessions ({len(names)} files)\n"
               "Cloud sessions start empty. These finished results were packed into the notebook when it was built, "
               "so the runner skips them instead of computing them again."),
            code("import base64, io, zipfile\n"
                 f"{listing}\n"
                 f"CARRIED = \"{blob}\"\n"
                 "zipfile.ZipFile(io.BytesIO(base64.b64decode(CARRIED))).extractall(\".\")\n"
                 f"print(\"restored {len(names)} finished result files\")"),
        ]
    else:
        cells.append(md("## 5. Results from earlier sessions\nNone yet: this is the first session."))

    cells += [
        md("## 6. Check the machine: GPU, tests, smoke runs. Stops here if anything fails."),
        code("!python cloud_run.py check"),
        md("## 7. This session's work\n"
           "Planned steps, in order: **" + ", ".join(plan) + "**. After every finished job the zip is refreshed, so "
           "a session that is cut off still leaves usable results, and grid steps stop starting new runs after "
           "about 9 h so the 12 h commit finishes (a killed commit keeps no output). All of Experiment B runs on "
           "this kind of GPU: "
           "runs made here in float16 must never be pooled with runs made elsewhere in bfloat16."),
    ]
    for step in plan:
        heading, command, note = STEPS[step]
        cells += [md(f"### {heading}" + (f"\n{note}" if note else "")), code("!" + command)]

    cells += [
        md("## 8. Pack the results for download"),
        code("""
!python cloud_run.py pack
import os, zipfile
for d in ("/kaggle/working", "/content", "."):
    p = os.path.join(d, "thin_gate_results.zip")
    if os.path.exists(p):
        print(p)
        for n in zipfile.ZipFile(p).namelist():
            print("   ", n)
        break
"""),
    ]

    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"wrote {OUT}: {len(cells)} cells, code snapshot {stamp}")
    if dirty:
        print("note: some code files have uncommitted changes; commit first if you want the snapshot to be exact")


if __name__ == "__main__":
    main()

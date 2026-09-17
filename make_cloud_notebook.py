"""
Builds cloud_notebook.ipynb: one self-contained notebook that carries a copy of the experiment code, so the
experiments can run on a free cloud GPU (Kaggle, Colab) without giving anyone access to the private repo.

    python make_cloud_notebook.py

Re-run this after ANY change to the code, and upload the fresh notebook: the copy inside an old notebook does
not update itself. The notebook is git-ignored because it is generated; this script and cloud_run.py are the
source. Standard library only, so it also runs on the laptop.
"""

import json
import subprocess

FILES = ["hf_utils.py", "posthoc_truncate.py", "model.py", "data.py", "train.py", "grid.py", "heal.py",
         "bench.py", "tests.py", "cloud_run.py"]
OUT = "cloud_notebook.ipynb"


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


def main():
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
it will not finish all models in one go; run it again and finished models are skipped only within a session.

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
        md("## 5. Check the machine: GPU, tests, smoke runs. Stops here if anything fails."),
        code("!python cloud_run.py check"),
        md("""
## 6. Experiment A: which projection tolerates rank reduction?
Every model, whitened SVD then plain SVD, smallest model first, one queue per GPU. Expect roughly 7 to 8
hours on two T4s. After each model the zip is refreshed, so an interrupted run still leaves usable results.
The first line to look for: SmolLM2-135M baseline perplexity close to 17.46 (what the CPU measured).
"""),
        code("!python cloud_run.py posthoc"),
        md("""
## 7. Experiment B pilot (off by default)
Set `RUN_PILOT = True` only after deciding that ALL of Experiment B runs on this kind of GPU: pilot runs
made here in float16 must never be pooled with runs made elsewhere in bfloat16. The speed check runs first and
says how long one run takes; if that is over about 1.5 hours, lower `TOKENS["S"]` in `grid.py`, rebuild this
notebook, and start again, because the token budget cannot change once a grid has started.
"""),
        code("""
RUN_PILOT = False
if RUN_PILOT:
    get_ipython().system("python cloud_run.py throughput")
    get_ipython().system("python cloud_run.py pilot")
else:
    print("pilot skipped (RUN_PILOT = False)")
"""),
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

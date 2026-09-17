# Run log

Every run launched, every failure and every decision, with dates. Times are EDT.

## 2026-09-17 (Day 0)

**Environment audit on the GPU PC. Setup is blocked; no experiment has run.**

- GPU is visible from Windows: RTX 3070, 8192 MiB, driver 560.94. `nvidia-smi` works.
- **WSL2 is not installed.** `wsl --version`, `wsl --status` and `wsl -l -v` all print only the
  usage stub and exit 1, which is what `wsl.exe` does when the Windows feature is disabled and
  no distribution exists. So there is no Linux environment, no `.venv` and no `tmux` yet.
- The hardware supports it: virtualization is enabled in firmware and SLAT is present.
  184 GB free on C:. The hypervisor is not currently running, which is expected while the
  feature is off.
- Python on the Windows side is 3.8 only, plus the Store alias stubs. That is too old for a
  current PyTorch, so even a native-Windows fallback needs a newer Python installed first.
- `setup_pc.sh` still has never been executed. It is Linux-only (`apt-get`) and deliberately
  refuses to run from `/mnt/c`, so it cannot run until a distribution exists and the repo is
  cloned under `~/`.
- Consequence: `tests.py` on this machine, Exp. A, the throughput check and the pilot grid are
  all blocked. `results/` is still empty and no number of any kind exists yet.
- Decision pending, and it needs an elevated shell and a reboot, so it is the author's call:
  install WSL2 + Ubuntu as SETUP_PC.md describes, or fall back to native Windows with a fresh
  Python. `train.py` and `bench.py` both accept `--no_compile`, so the fallback needs no code
  change, but it gives up `tmux` and makes the measured tokens/s non-comparable to a Linux run.

### 2026-09-17, evening: CPU fallback so Exp. A can start before WSL2 exists

Decision: rather than idle until WSL2 is installed, run Exp. A on the CPU. Exp. A is
training-free (SVD plus a perplexity evaluation), so it does not need the GPU. Exp. B, Exp. C
and `bench.py` still do and remain blocked.

Environment built on the Windows side (interim, not the machine of record):
Python 3.12.10 in `.venv`, torch 2.14.0+cpu, transformers 5.17.0, datasets 5.0.1, numpy 2.5.3,
8 threads. `lm-eval` deliberately not installed; it is only needed for the optional Day 4-5
zero-shot evaluations and is the dependency most likely to fight `transformers`.

- `python tests.py`: **all 6 pass** on this machine.
- **Bug found and fixed** (`hf_utils.py`): `load_dataset("wikitext", ...)` fails under
  datasets>=4, which rejects a bare legacy dataset name and demands `namespace/name`. Changed to
  `Salesforce/wikitext`, verified on the Hub as carrying the `wikitext-2-raw-v1` config. This was
  blocking Exp. A completely and would have blocked it on the GPU too, so it is not a CPU artifact.
- **Change** (`posthoc_truncate.py`): the dtype is now chosen from the device, bfloat16 on CUDA and
  float32 on CPU, because Zen 2 has no native bfloat16 and emulates it far too slowly to evaluate.
  The dtype is now also recorded in the results JSON. CUDA behaviour is unchanged. `tests.py` rerun
  and still 6/6. Consequence to remember: CPU fp32 perplexities are not bit-identical to GPU bf16
  ones, so the two must not be mixed inside one figure.
- **Baseline sanity check: perplexity 20.074** for SmolLM2-135M on wikitext-2 (on 8 windows of
  1024 tokens, not the full split). Plausible tens for a 135M model, so the evaluation is sound
  and downstream numbers can be trusted. README's stop condition is not triggered.
- Plain SVD on `gate_proj`, all 30 layers, 8 windows, degradation is smoothly monotonic:
  rank 570 (0.99) x1.01, 518 (0.90) x1.11, 432 (0.75) x2.37, 288 (0.50) x81.6, 144 (0.25) x3531.
  Monotonic, so this is the method's real behaviour and not a bug. Plain SVD is simply destructive
  at low rank, which is consistent with why the activation-aware variant exists at all.
- Whitened SVD on the same ranks is far better: 432 x1.10 and 288 x1.63, against x2.37 and x81.6
  plain. Confirms whitening is the stronger baseline and the one the go/no-go should rest on.
- Measured cost: about 1.9 s per 1024-token window on 8 CPU threads, and the full wikitext-2 test
  split is 297 windows, so roughly 9.5 min per evaluation and about 3 h per 19-config sweep.
- **Launched**: full Exp. A on SmolLM2-135M, both variants in sequence, full test split, all three
  projection types, default rank fractions. Logs in `results/logs/`. Expected around 6 h total.
  Only the 135M model is CPU-feasible; 360M would be roughly 8 h per variant and 1.7B far beyond
  reach, so the larger models in the abstract still need the GPU.

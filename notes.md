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

#!/usr/bin/env bash
# One-shot setup of the GPU machine. Run INSIDE Ubuntu (WSL2), from the cloned repo:
#     bash setup_pc.sh                 # everything
#     bash setup_pc.sh --tailscale     # also install Tailscale SSH so the Mac can log in remotely
# Safe to re-run: finished steps are skipped or are quick.
set -euo pipefail
cd "$(dirname "$0")"
step() { echo; echo "==================== $* ===================="; }

step "1/6  Is the GPU visible from Linux?"
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi failed. Fix: update the NVIDIA driver in WINDOWS (not in Linux), then run"
  echo "'wsl --shutdown' in PowerShell, reopen Ubuntu and run this script again."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
case "$PWD" in
  /mnt/*) echo "This repo is on the Windows drive ($PWD), which is very slow from Linux."
          echo "Move it:  mv \"$PWD\" ~/thin-gate && cd ~/thin-gate && bash setup_pc.sh"; exit 1;;
esac

step "2/6  System packages (asks for your Ubuntu password)"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip python3-dev git tmux build-essential curl

step "3/6  Python environment + PyTorch (a ~2-3 GB download the first time)"
[[ -d .venv ]] || python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch
pip install -q -r requirements.txt

step "4/6  PyTorch can use the GPU in bfloat16?"
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0),
      "|", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", bool(torch.isfinite((x @ x).float().norm())))
EOF

step "5/6  Correctness tests and toy runs"
python tests.py
python train.py --smoke | tail -1
python posthoc_truncate.py --smoke --whiten | tail -1
python heal.py --smoke | tail -1
rm -rf results/smoke

step "6/6  Training data (~1.4 GB) and a short real GPU run to measure speed"
python data.py --num_train_shards 6
python train.py --name gpu_check --size S --tokens 5e6 --eval_every 1000 --out_dir results/scratch | tail -3
python - <<'EOF'
import json
r = json.load(open("results/scratch/gpu_check.json"))
tps = r["tokens_per_second"]
print(f"\nmeasured {tps:,.0f} tokens/s, peak memory {r['peak_memory_gb']:.1f} GB")
print(f"=> one size-S run (300M tokens) takes about {300e6 / tps / 3600:.1f} h; "
      f"one size-M run (500M tokens) is roughly 2-3x that per token.")
print("If a size-S run is over ~1.5 h, lower TOKENS['S'] in grid.py BEFORE starting the grid.")
EOF

if [[ "${1:-}" == "--tailscale" ]]; then
  step "extra  Tailscale SSH (lets the Mac log in to this machine)"
  command -v tailscale >/dev/null 2>&1 || curl -fsSL https://tailscale.com/install.sh | sh
  sudo tailscale up --ssh
  echo "This machine's Tailscale name/IP:"; tailscale status | head -1
fi

cat <<'EOF'

All done. Next:
  1. Windows: Settings -> System -> Power -> Sleep: Never.  Pause Windows Update for a week.
  2. Open a new terminal, then:
         cd ~/thin-gate && tmux new -s main
     and follow the Day 1 commands in README.md.
EOF

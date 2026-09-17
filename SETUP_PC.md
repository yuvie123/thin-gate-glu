# Setting up the GPU machine (Ryzen 7 + RTX 3070, Windows)

Do this once, on Day 0. Budget about an hour. Everything heavy runs here; the MacBook only edits and writes.

## 1. WSL2 + Ubuntu (Linux inside Windows)
PyTorch's compiler (`torch.compile`) is unreliable on native Windows, so we use Linux via WSL2.

1. Update the **Windows** NVIDIA driver (GeForce Experience / nvidia.com). Do **not** install an NVIDIA driver inside Linux; WSL2 uses the Windows one.
2. Open PowerShell **as Administrator**: `wsl --install -d Ubuntu-24.04`, reboot, then create a username and password when Ubuntu opens.
3. In the Ubuntu terminal, check that the GPU is visible: `nvidia-smi` (you should see "GeForce RTX 3070").

Keep all project files inside the Linux home folder (`~/`), **not** under `/mnt/c/...`; the Windows drive is very slow from Linux.

## 2. Python environment
```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git tmux build-essential
git clone <your PRIVATE repo url> ~/thin-gate && cd ~/thin-gate
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch                      # Linux wheels ship with CUDA
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16); print((x @ x).float().norm())"
```
The last line must print `True` and `NVIDIA GeForce RTX 3070`. If `cuda.is_available()` is `False`, stop and fix that first.

Every new terminal: `cd ~/thin-gate && source .venv/bin/activate`.

## 3. Check the code before trusting it
```bash
python tests.py                 # 6 correctness tests, a few seconds
python train.py --smoke         # toy training run
python posthoc_truncate.py --smoke
```

## 4. Remote access from the Mac (optional)
SSH from the Mac. Inside Ubuntu: `sudo apt install -y openssh-server && sudo service ssh start`; easiest networking is Tailscale on both machines. Then `ssh <user>@<pc-name>` from the Mac.

## 5. Long runs: use tmux so closing the window doesn't kill them
```bash
tmux new -s runs          # start a session
python grid.py --stage pilot --run
# detach: press Ctrl+b, then d.    Re-attach later: tmux attach -t runs
```
In a second tmux window (`Ctrl+b` then `c`), `watch -n 5 nvidia-smi` shows GPU use and memory.

## 6. Keep the PC awake
Windows Settings -> System -> Power: Sleep = **Never** while plugged in. Settings -> Windows Update -> pause updates for 1 week (an automatic reboot at 3 am kills an overnight run). Runs are resumable at the *grid* level (`grid.py` skips finished runs), but a run that is interrupted halfway starts over.

## 7. Hugging Face
`pip install -U huggingface_hub && hf auth login` (free account, a "read" token). Only needed for gated models such as `meta-llama/Llama-3.2-1B`: request access on its model page; if it isn't approved in time, just skip that model.

## 8. Out of memory?
`torch.OutOfMemoryError` means the batch doesn't fit in 8 GB. Close games/browsers using the GPU, then:
- `train.py`: lower `--micro_bs` (4 -> 2). The optimizer batch and the data order stay the same; it just takes more, smaller forward passes. If `nvidia-smi` shows plenty of free memory at 4, try 8 for speed.
- `posthoc_truncate.py`: `--batch_size 2` or `1`; for `--whiten` on models >= 1B also `--layers_per_pass 2`.
- `heal.py`: `--micro_bs 1 --grad_ckpt`.

## 9. Getting results back to the Mac
Results are small JSON files in `results/`. `git add results paper/figures paper/tables && git commit -m "results" && git push`, then `git pull` on the Mac. Model weights, datasets and checkpoints are git-ignored and stay on the PC.

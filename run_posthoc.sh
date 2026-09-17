#!/usr/bin/env bash
# Experiment A on every model, smallest first. Finished models are skipped. Run on the GPU machine.
#   bash run_posthoc.sh            plain SVD
#   bash run_posthoc.sh --whiten   activation-aware SVD (the stronger baseline; run this too)
set -e
MODELS=(
  HuggingFaceTB/SmolLM2-135M
  HuggingFaceTB/SmolLM2-360M
  Qwen/Qwen2.5-0.5B
  TinyLlama/TinyLlama_v1.1
  meta-llama/Llama-3.2-1B      # gated: accept the licence on huggingface.co and run `hf auth login` first
  allenai/OLMo-2-0425-1B
  Qwen/Qwen2.5-1.5B
  HuggingFaceTB/SmolLM2-1.7B
)
METHOD=plain; [[ "$1" == "--whiten" ]] && METHOD=whiten
for M in "${MODELS[@]}"; do
  OUT="results/posthoc/${M//\//__}__wikitext2__${METHOD}.json"
  if [[ -f "$OUT" ]]; then echo "[done] $M"; continue; fi
  python posthoc_truncate.py --model "$M" "$@" || echo "[FAILED] $M (continuing)"
done

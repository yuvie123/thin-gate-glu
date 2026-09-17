#!/usr/bin/env bash
# Experiment C: same rank, three projection types, so the comparison is apples to apples.
#   bash run_heal.sh HuggingFaceTB/SmolLM2-135M
set -e
MODEL=${1:-HuggingFaceTB/SmolLM2-135M}
for TYPE in gate_proj up_proj down_proj; do
  python heal.py --model "$MODEL" --type "$TYPE" --rank_frac 0.25 --whiten "${@:2}"
done

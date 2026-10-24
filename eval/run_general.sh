#!/usr/bin/env bash
# General-capability retention check.
set -euo pipefail

MODEL=${1:?usage: run_general.sh <model_path_or_hf_id>}
OUT="outputs/lm_eval/$(basename "$MODEL")"
LIMIT_ARG=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARG=(--limit "$LIMIT")
fi

lm_eval \
  --model vllm \
  --model_args "pretrained=$MODEL,tensor_parallel_size=2,dtype=float16,gpu_memory_utilization=0.8" \
  --tasks ifeval,gsm8k,mmlu \
  --batch_size 8 \
  ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"} \
  --output_path "$OUT"

# Smoke: LIMIT=50 bash eval/run_general.sh <model>

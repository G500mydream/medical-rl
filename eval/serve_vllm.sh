#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?usage: serve_vllm.sh <model_path_or_hf_id> [port]}
PORT=${2:-8000}

vllm serve "$MODEL" \
  --served-model-name med \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --port "$PORT"

# If CUDA graph capture OOMs on 3090:
#   add --enforce-eager

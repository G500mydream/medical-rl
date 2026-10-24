#!/usr/bin/env bash
# One-time environment setup. Linux + conda + CUDA 12.4 wheels.
set -euo pipefail

ENV_NAME=${ENV_NAME:-medrl}
PY=${PY:-3.10}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda create -n "$ENV_NAME" python="$PY" -y || true
  conda activate "$ENV_NAME"
else
  echo "[warn] conda not found; installing into the current Python"
fi

pip install -U pip wheel setuptools
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# flash-attn source builds are slow; swap in a prebuilt wheel for your exact
# torch/cuda combo if you have one.
pip install flash-attn==2.7.4.post1 --no-build-isolation || \
  echo "[warn] flash-attn failed to build; vLLM will fall back to another attention backend"

mkdir -p third_party
cd third_party

[[ -d verl ]] || git clone --depth 1 https://github.com/volcengine/verl.git
pip install -e verl

[[ -d LLaMA-Factory ]] || git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
pip install -e LLaMA-Factory

[[ -d MedEvalKit ]] || git clone --depth 1 https://github.com/alibaba-damo-academy/MedEvalKit.git
pip install -e MedEvalKit || echo "[warn] MedEvalKit install failed; see its README"

cd "$ROOT"
echo "[done] next: conda activate $ENV_NAME && python scripts/check_env.py"

#!/usr/bin/env bash
# Thin stage runner. Teachers are trained separately (see teachers/README.md).
set -euo pipefail

cd "$(dirname "$0")/.."
STAGE=${1:-all}

run_sft() {
  echo "== SFT =="
  make sft
}

run_grpo() {
  echo "== GRPO =="
  bash rl/run_grpo.sh
}

run_mopd() {
  echo "== MOPD =="
  bash rl/mopd/run_mopd.sh
}

run_eval() {
  echo "== EVAL =="
  local model=${MODEL:-outputs/qwen3-4b-mopd}
  local input=${INPUT:-data/processed/healthbench_1k.jsonl}
  python eval/score_rubric.py generate \
    --model "$model" --input "$input" \
    --output outputs/eval_pred.jsonl --tp 2
  python eval/score_rubric.py score \
    --input outputs/eval_pred.jsonl --output outputs/eval_scored.jsonl
}

case "$STAGE" in
  sft) run_sft ;;
  grpo) run_grpo ;;
  mopd) run_mopd ;;
  eval) run_eval ;;
  all) run_sft; run_grpo; run_eval ;;
  *)
    echo "usage: $0 [sft|grpo|mopd|eval|all]" >&2
    echo "note: train teachers before mopd (see teachers/README.md)" >&2
    exit 1
    ;;
esac

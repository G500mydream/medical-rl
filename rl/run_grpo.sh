#!/usr/bin/env bash
# GRPO launch script tuned for 8x RTX 3090 (24GB each).
# Adjust keys to match the verl release you pin; see README.
set -euo pipefail

cd "$(dirname "$0")/.."
[ -f configs/paths.env ] && source configs/paths.env

: "${BASE_MODEL:=Qwen/Qwen3-4B-Instruct-2507}"
: "${DATA_DIR:=data/processed}"
: "${GRPO_OUT:=outputs/qwen3-4b-grpo}"
: "${TEACHER_DIR:=outputs/teachers}"
: "${JUDGE_MODEL:=gpt-4.1-mini}"
: "${JUDGE_CACHE:=.cache/judge}"

TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$TRAIN_FILE}"
REWARD_PY="${REWARD_PY:-rl/rewards/medical_reward.py}"

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "[error] missing $TRAIN_FILE" >&2
  echo "[hint]  python data/make_sample_parquet.py  # smoke data" >&2
  exit 1
fi

export JUDGE_MODEL JUDGE_CACHE
export RUBRIC_DROPOUT="${RUBRIC_DROPOUT:-0.3}"
export LENGTH_PENALTY="${LENGTH_PENALTY:-0.0005}"
export SOFT_MAX_TOKENS="${SOFT_MAX_TOKENS:-800}"
export REWARD_MODE="${REWARD_MODE:-train}"

# Optional: train a LoRA teacher on the shared base instead of the full-param student.
EXTRA_ARGS=()
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo-qwen3-4b}"
if [[ -n "${TEACHER_NAME:-}" ]]; then
  EXTRA_ARGS+=(
    actor_rollout_ref.model.lora_rank=32
    actor_rollout_ref.model.lora_alpha=64
  )
  GRPO_OUT="$TEACHER_DIR/$TEACHER_NAME"
  EXPERIMENT_NAME="teacher-$TEACHER_NAME"
fi

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=8 \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=true \
  actor_rollout_ref.model.path="$BASE_MODEL" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.ref.fsdp_config.param_offload=false \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  reward_model.enable=false \
  custom_reward_function.path="$REWARD_PY" \
  custom_reward_function.name=compute_score \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.project_name="${WANDB_PROJECT:-medical-sft-grpo-mopd}" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$GRPO_OUT" \
  trainer.save_freq=100 \
  trainer.test_freq=-1 \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

# 3090 OOM fallbacks, in order:
#   1. actor_rollout_ref.rollout.gpu_memory_utilization=0.45
#   2. data.max_response_length=768
#   3. actor_rollout_ref.rollout.tensor_model_parallel_size=4
#   4. switch actor to LoRA (lora_rank=32) at 8B

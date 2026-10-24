#!/usr/bin/env bash
# MOPD entry point (skeleton).
#
# Stock verl has no `mopd` estimator. You need to:
#   1. train the three teacher adapters (see teachers/README.md);
#   2. port rl/mopd/teacher_pool.py into your verl actor worker;
#   3. register algorithm.adv_estimator=mopd (or add the distill loss directly).
#
# Until then this script only checks inputs and prints the intended command.
set -euo pipefail

cd "$(dirname "$0")/../.."
[ -f configs/paths.env ] && source configs/paths.env

: "${BASE_MODEL:=Qwen/Qwen3-4B-Instruct-2507}"
: "${TEACHER_DIR:=outputs/teachers}"
: "${MOPD_OUT:=outputs/qwen3-4b-mopd}"

if ! compgen -G "$TEACHER_DIR/*/adapter_config.json" > /dev/null; then
  echo "[error] no teacher adapters under $TEACHER_DIR" >&2
  echo "[hint]  see teachers/README.md for the three training configs" >&2
  exit 1
fi

echo "[info] teachers:"
for d in "$TEACHER_DIR"/*/; do
  [[ -f "$d/adapter_config.json" ]] && echo "  - $d"
done

cat <<EOF
[intended command]

python -m verl.trainer.main_ppo \\
  algorithm.adv_estimator=mopd \\
  algorithm.mopd.teacher_dir=$TEACHER_DIR \\
  algorithm.mopd.lambda=0.1 \\
  algorithm.mopd.topk=50 \\
  algorithm.mopd.direction=forward \\
  actor_rollout_ref.model.path=$BASE_MODEL \\
  actor_rollout_ref.rollout.name=vllm \\
  actor_rollout_ref.rollout.n=8 \\
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \\
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \\
  custom_reward_function.path=rl/rewards/medical_reward.py \\
  custom_reward_function.name=compute_score \\
  trainer.n_gpus_per_node=8 \\
  trainer.nnodes=1 \\
  trainer.default_local_dir=$MOPD_OUT
EOF

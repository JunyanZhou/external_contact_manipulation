#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export PYTHONPATH="$project_dir${ISAACLAB_PATH:+:$ISAACLAB_PATH/source/isaaclab}${PYTHONPATH:+:$PYTHONPATH}"
export MEMPATH="${MEMPATH:-$project_dir/dataset}"
export OMNI_KIT_ACCEPT_EULA=Y
export ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE="${ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE:-1}"
exec "${PYTHON:-python}" scripts/train.py \
  "task=${TASK_NAME:-G1/G1_table_push_task}" +exp=train \
  algo.direct_policy=true algo.train_estimators=false algo.symmetry_augmentation=false \
  "wandb.mode=${WANDB_MODE:-disabled}" "wandb.project=${WANDB_PROJECT:-table-push-task}" "$@"

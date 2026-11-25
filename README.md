# G1 supported manipulation

Training and evaluation code for four G1 tasks, with shared control code and ground-only baselines.

| Task | Configuration in `cfg/task/G1/` |
| --- | --- |
| Table-supported foot push | `G1_table_push_goal_student.yaml` |
| Wall-supported box push | `G1_wall_box_push_goal_student.yaml`, `G1_wall_box_right_foot_push_goal_student.yaml` |
| Cabinet-supported drawer pull | `G1_drawer_pull_goal_student.yaml`, `G1_two_drawer_dynamic_pull.yaml` |
| Spring pedal | `G1_spring_pedal_goal_student.yaml` |

The student takes five Cartesian targets and proprioception. Teachers use privileged observations during training.

## Setup and data

This source export does not contain datasets, checkpoints or videos. Local `outputs/` and `artifacts/` records are excluded from Git. Reported results refer to earlier experiments, not a fresh run of this checkout.

Training requires a compatible Isaac Lab / Isaac Sim environment. `setup.py` does not install the simulator. Restore or regenerate the task scene assets and motion datasets before running; evaluation and warm starts also require the corresponding checkpoints. The complete pipeline has not been rerun from this export.

Run these Bash commands from the repository root in the configured environment:

```bash
git lfs install
git lfs pull
python -m pip install -e .
export MEMPATH="$PWD/dataset"
python scripts/train.py task=G1/G1_table_push_goal_student +exp=train
```

## Layout

- `active_adaptation/`: environments, robot assets, PPO and teacher/student modules.
- `cfg/`: task and training configurations.
- `scripts/`: training, reference generation, asset builders and evaluation.
- `deploy/mujoco_whole_body/g1/`: shared robot model and meshes.
- `tests/`: focused unit tests.

Shared modules retain compatibility branches needed by the retained tasks. The direct task-policy launcher is distinct from the goal-conditioned teacher/student pipeline.

## Distribution

USD and STL assets are configured for Git LFS. A project-wide license has not been established; existing file-level copyright and license notices remain applicable. Robot asset provenance and redistribution permissions still require confirmation before public release.

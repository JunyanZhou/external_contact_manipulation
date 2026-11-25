"""Run the same five-target stream and Isaac scene with a supported actor and a ground-only actor."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "scripts/table_push/eval_task_policy.py"
PAIR = ROOT / "scripts/table_push/check_five_target_pair.py"
TASKS = {
    "wall_foot_box": {
        "checkpoint": ROOT / "outputs/wall-box-right-foot-push/checkpoints/student_finetune_128.pt",
        "dataset": "wall_box_right_foot_goal_v2", "motion_id": 0,
        "goals": ROOT / "artifacts/wall_box_push/right_foot_waypoints_v2/waypoints/right_parallel_x31_y11_v0.npz",
        "extra": ["--wall-box-close-camera"],
    },
    "two_drawer_pull": {
        "checkpoint": ROOT / "artifacts/drawer_task/two_drawer_dynamic_training/checkpoints/student_selected.pt",
        "dataset": "two_drawer_dynamic_multi", "motion_id": 0,
        "goals": None, "steps": 105, "extra": ["--explicit-goals"],
    },
    "table_foot_block": {
        "checkpoint": ROOT / "outputs/table-push-goal/student_finetune_final.pt",
        "dataset": "table_push_goal_v3_interp", "motion_id": 2,
        "goals": ROOT / "artifacts/table_push/task_dataset_goal_v3_interp/waypoints/push_x40_y31_v0.npz",
        "extra": [],
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(MEMPATH=str(ROOT / "dataset"), OMNI_KIT_ACCEPT_EULA="Y",
               ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE="1",
               PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT), str(Path(env["ISAACLAB_PATH"]) / "source/isaaclab") if env.get("ISAACLAB_PATH") else "", env.get("PYTHONPATH", "")])),
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    report = {}
    for name in args.tasks:
        spec = TASKS[name]
        steps = (len(np.load(spec["goals"])["goal_xyz_world"]) if spec["goals"]
                 else spec["steps"])
        rows = {}
        for role in ("supported", "baseline"):
            result = args.output / f"{name}_{role}.json"
            cmd = [sys.executable, str(EVAL), "--checkpoint",
                   str(spec["checkpoint"] if role == "supported" else args.baseline),
                   "--dataset", spec["dataset"], "--split", "val", "--motion-id",
                   str(spec["motion_id"]), "--steps", str(steps), "--seed", "0",
                   "--output", str(result), *spec["extra"]]
            if spec["goals"]:
                cmd.extend(["--external-goals-npz", str(spec["goals"])])
            if role == "baseline":
                cmd.extend(["--matched-scene-checkpoint", str(spec["checkpoint"]),
                            "--video", str(args.output / f"{name}_baseline_1080p.mp4"),
                            "--video-label", f"Ground-only baseline | {name}",
                            "--video-width", "1920", "--video-height", "1080",
                            "--video-crf", "20", "--compact-video-overlay"])
            print(name, role, "steps", steps, flush=True)
            with (args.output / f"{name}_{role}.log").open("w") as log:
                status = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, check=False).returncode
            if status or not result.exists():
                raise RuntimeError(f"{name} {role} failed: {args.output / f'{name}_{role}.log'}")
            rows[role] = json.loads(result.read_text())
        pair = args.output / f"{name}_pair.json"
        with (args.output / f"{name}_pair.log").open("w") as log:
            status = subprocess.run([sys.executable, str(PAIR),
                                     str(args.output / f"{name}_supported.json"),
                                     str(args.output / f"{name}_baseline.json"),
                                     "--output", str(pair)], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT).returncode
        if status:
            raise RuntimeError(f"Input mismatch for {name}: {args.output / f'{name}_pair.log'}")
        report[name] = {"same_initial_state_and_goals": True,
                        "supported_success": rows["supported"].get("task_success"),
                        "baseline_success": rows["baseline"].get("task_success"),
                        "supported_first_fall_step": rows["supported"].get("first_fall_step"),
                        "baseline_first_fall_step": rows["baseline"].get("first_fall_step"),
                        "supported_steps": rows["supported"].get("steps"),
                        "baseline_steps": rows["baseline"].get("steps"),
                        "video": str(args.output / f"{name}_baseline_1080p.mp4")}
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({name: report[name]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

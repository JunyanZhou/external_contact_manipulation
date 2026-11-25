"""Evaluate a ground-only five-target actor on all held-out randomized reach clips."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = json.loads((ROOT / "dataset/ground_random_reach_baseline/val/source_files.json").read_text())
    env = os.environ.copy()
    env.update(MEMPATH=str(ROOT / "dataset"), OMNI_KIT_ACCEPT_EULA="Y",
               ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE="1",
               PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT), str(Path(env["ISAACLAB_PATH"]) / "source/isaaclab") if env.get("ISAACLAB_PATH") else "", env.get("PYTHONPATH", "")])),
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    rows = []
    for i, source in enumerate(sources):
        motion = np.load(ROOT / "artifacts/ground_random_reach_baseline/references" / source)
        steps = len(motion["root_pos"])
        result = args.output / f"val_{i:02d}.json"
        cmd = [sys.executable, str(ROOT / "scripts/table_push/eval_task_policy.py"),
               "--checkpoint", str(args.checkpoint), "--dataset", "ground_random_reach_baseline",
               "--split", "val", "--motion-id", str(i), "--explicit-goals",
               "--steps", str(steps), "--seed", "0", "--output", str(result)]
        if args.video and i == 0:
            cmd.extend(["--video", str(args.output / "ground_student_reach_1080p.mp4"),
                        "--video-label", "Ground-only student | in-place stepping and reach",
                        "--video-width", "1920", "--video-height", "1080",
                        "--video-crf", "20", "--compact-video-overlay"])
        with (args.output / f"val_{i:02d}.log").open("w") as log:
            status = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT).returncode
        if status or not result.exists():
            raise RuntimeError(args.output / f"val_{i:02d}.log")
        data = json.loads(result.read_text())
        row = {"motion_id": i, "steps": data["steps"],
               "ground_reach_success": data["ground_reach_success"],
               "first_instability_step": data["first_instability_step"],
               "left_lift_tracking_fraction": data["left_lift_tracking_fraction"],
               "right_lift_tracking_fraction": data["right_lift_tracking_fraction"],
               "mean_root_error_m": data["mean_root_error_m"],
               "mean_left_hand_error_m": data["mean_left_hand_error_m"],
               "mean_right_hand_error_m": data["mean_right_hand_error_m"],
               "policy_obs_only": data["policy_obs_only"]}
        rows.append(row)
        (args.output / "summary.json").write_text(json.dumps({
            "checkpoint": str(args.checkpoint), "successes": sum(r["ground_reach_success"] for r in rows),
            "completed": len(rows), "rows": rows}, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

"""Evaluate every held-out five-xyz motion with identical external goals."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--matched-scene-checkpoint", type=Path)
    p.add_argument("--block-init-x", type=float, help="Use the same box starting x for all rollouts")
    p.add_argument("--dataset", required=True)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stop-on-fall", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = json.loads((ROOT / "dataset" / args.dataset / args.split / "source_files.json").read_text())
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
               MEMPATH=str(ROOT / "dataset"), OMNI_KIT_ACCEPT_EULA="Y",
               ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE="1",
               PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT), str(Path(env["ISAACLAB_PATH"]) / "source/isaaclab") if env.get("ISAACLAB_PATH") else "", env.get("PYTHONPATH", "")])))
    rows = []
    for seed in args.seeds:
        for motion_id, source in enumerate(sources):
            name = Path(source).stem
            goal_file = args.artifacts / "waypoints" / f"{name}.npz"
            with np.load(goal_file) as wp:
                steps = len(wp["goal_xyz_world"])
            result = args.output / f"{name}_seed{seed}.json"
            log = result.with_suffix(".log")
            cmd = [sys.executable, str(ROOT / "scripts/table_push/eval_task_policy.py"),
                   "--checkpoint", str(args.checkpoint.resolve()),
                   "--dataset", args.dataset, "--split", args.split,
                   "--motion-id", str(motion_id), "--external-goals-npz", str(goal_file.resolve()),
                   "--steps", str(steps), "--seed", str(seed), "--output", str(result.resolve())]
            if args.matched_scene_checkpoint:
                cmd.extend(("--matched-scene-checkpoint", str(args.matched_scene_checkpoint.resolve())))
            if args.block_init_x is not None:
                cmd.extend(("--block-init-x", str(args.block_init_x)))
            if args.stop_on_fall:
                cmd.append("--stop-on-fall")
            print("EVAL",name,"seed",seed,"steps",steps,flush=True)
            with log.open("w") as stream:
                completed = subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if completed.returncode or not result.exists():
                raise RuntimeError(f"Evaluation failed: {log}")
            data = json.loads(result.read_text())
            row = {"name": name, "seed": seed, "motion_id": motion_id,
                   "result": str(result), "steps": data["steps"], "requested_steps": steps,
                   "task_success": data["task_success"],
                   "dual_hand_success": data.get("dual_hand_success"),
                   "task_success_strict_5cm": data.get("task_success_strict_5cm"),
                   "first_fall_step": data["first_fall_step"],
                   "block_lateral_displacement_m": data["block_lateral_displacement_m"],
                   "block_parallel_displacement_m": data.get("block_parallel_displacement_m"),
                   "block_normal_displacement_m": data.get("block_normal_displacement_m"),
                   "block_max_tilt_rad": data.get("block_max_tilt_rad"),
                   "left_hand_box_contact_fraction": data.get("left_hand_box_contact_fraction"),
                   "right_hand_box_contact_fraction": data.get("right_hand_box_contact_fraction"),
                   "max_forward_lean_rad": data.get("max_forward_lean_rad"),
                   "max_foot_lift_m": data.get("max_foot_lift_m"),
                   "target_max_foot_lift_m": data.get("target_max_foot_lift_m"),
                   "right_wall_support_fraction": data["right_wall_support_fraction"],
                   "right_support_fraction": data["right_support_fraction"],
                   "left_support_fraction": data.get("left_support_fraction"),
                   "right_foot_box_contact_fraction": data.get("right_foot_box_contact_fraction"),
                   "right_foot_box_contact_frames": data.get("right_foot_box_contact_frames"),
                   "max_left_foot_drift_m": data.get("max_left_foot_drift_m"),
                   "mean_right_foot_error_m": data.get("mean_right_foot_error_m"),
                   "insert_lift_from_settled_m": data.get("insert_lift_from_settled_m"),
                   "crate_planar_drift_m": data.get("crate_planar_drift_m"),
                   "left_foot_crate_support_fraction": data.get("left_foot_crate_support_fraction"),
                   "left_wall_contact_fraction": data.get("left_wall_contact_fraction"),
                   "mean_foot_error_m": data["mean_foot_error_m"],
                   "external_goals_sha256": data["external_goals_sha256"],
                   "initial_root_xyz_m": data["initial_root_xyz_m"],
                   "initial_root_quat_wxyz": data["initial_root_quat_wxyz"],
                   "initial_joint_pos_rad": data["initial_joint_pos_rad"],
                   "block_initial_xyz_m": data.get("block_initial_xyz_m"),
                   "crate_initial_xyz_m": data.get("crate_initial_xyz_m"),
                   "insert_initial_xyz_m": data.get("insert_initial_xyz_m")}
            rows.append(row)
            (args.output / "summary.json").write_text(json.dumps({
                "checkpoint": str(args.checkpoint.resolve()),
                "matched_scene_checkpoint": str(args.matched_scene_checkpoint.resolve()) if args.matched_scene_checkpoint else None,
                "completed": len(rows), "successes": sum(r["task_success"] for r in rows),
                "rows": rows},indent=2)+"\n")
            print(" RESULT",row["task_success"],"shift",row["block_lateral_displacement_m"],flush=True)


if __name__ == "__main__":
    main()

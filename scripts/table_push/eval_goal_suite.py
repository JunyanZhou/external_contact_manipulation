"""Run every held-out five-target table-push clip and summarize Isaac results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SUITES = {
    "table_push_goal_v1_full": "task_dataset_goal_v1",
    "table_push_goal_v2_long": "task_dataset_goal_v2_long",
    "table_push_goal_v3_interp": "task_dataset_goal_v3_interp",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--matched-scene-checkpoint", type=Path,
                   help="Use this checkpoint's task scene while keeping the tested policy")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stop-on-fall", action="store_true")
    p.add_argument("--stop-on-instability", action="store_true")
    p.add_argument("--datasets", nargs="+", choices=SUITES, default=list(SUITES))
    args = p.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
               MEMPATH=str(ROOT / "dataset"), OMNI_KIT_ACCEPT_EULA="Y",
               ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE="1",
               PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT), str(Path(env["ISAACLAB_PATH"]) / "source/isaaclab") if env.get("ISAACLAB_PATH") else "", env.get("PYTHONPATH", "")])))
    rows = []
    for dataset in args.datasets:
        artifact = ROOT / "artifacts/table_push" / SUITES[dataset]
        files = json.loads((ROOT / "dataset" / dataset / "val/source_files.json").read_text())
        for motion_id, source in enumerate(files):
            name = Path(source).stem
            waypoints = artifact / "waypoints" / f"{name}.npz"
            with np.load(waypoints) as data:
                steps = len(data["goal_xyz_world"])
            result = output / f"{dataset}_{name}_seed{args.seed}.json"
            log = result.with_suffix(".log")
            cmd = [sys.executable, str(ROOT / "scripts/table_push/eval_task_policy.py"),
                   "--checkpoint", str(checkpoint), "--dataset", dataset,
                   "--split", "val", "--motion-id", str(motion_id),
                   "--external-goals-npz", str(waypoints), "--steps", str(steps),
                   "--seed", str(args.seed), "--output", str(result)]
            if args.matched_scene_checkpoint is not None:
                cmd.extend(["--matched-scene-checkpoint", str(args.matched_scene_checkpoint.resolve())])
            if args.stop_on_fall:
                cmd.append("--stop-on-fall")
            if args.stop_on_instability:
                cmd.append("--stop-on-instability")
            print(f"EVAL {dataset} {name} seed={args.seed} steps={steps}", flush=True)
            with log.open("w") as stream:
                completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream,
                                           stderr=subprocess.STDOUT, check=False)
            if completed.returncode != 0 or not result.exists():
                raise RuntimeError(f"Evaluation failed ({completed.returncode}): {log}")
            record = json.loads(result.read_text())
            entry = {"dataset": dataset, "motion_id": motion_id, "name": name,
                     "seed": args.seed, "steps": steps, "result": str(result)}
            entry["actual_steps"] = record["steps"]
            entry["first_fall_step"] = record.get("first_fall_step")
            entry["first_instability_step"] = record.get("first_instability_step")
            entry["initial_root_xyz_m"] = record.get("initial_root_xyz_m")
            entry["external_goals_sha256"] = record.get("external_goals_sha256")
            entry.update({k: record[k] for k in (
                "task_success", "mean_foot_error_m", "right_support_fraction",
                "both_hands_support_fraction", "min_root_height_m", "max_root_tilt_rad",
                "block_lateral_displacement_m", "block_table_edge_clearance_m",
                "block_max_tilt_rad", "episode_done_count")})
            rows.append(entry)
            print(f"  success={entry['task_success']} foot={entry['mean_foot_error_m']:.3f}m "
                  f"edge={entry['block_table_edge_clearance_m']:.3f}m", flush=True)
            (output / "summary.json").write_text(json.dumps({"checkpoint": str(checkpoint),
                "seed": args.seed, "completed": len(rows), "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Verify a supported student and ground-only baseline used identical rollout inputs."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("supported", type=Path)
    p.add_argument("baseline", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--supported-summary", type=Path,
                   help="Task-specific success summary for the supported rollout")
    p.add_argument("--baseline-summary", type=Path,
                   help="Task-specific success summary for the baseline rollout")
    args = p.parse_args()
    a = json.loads(args.supported.read_text())
    b = json.loads(args.baseline.read_text())
    a_success = (json.loads(args.supported_summary.read_text())["task_success"]
                 if args.supported_summary else a["task_success"])
    b_success = (json.loads(args.baseline_summary.read_text())["task_success"]
                 if args.baseline_summary else b["task_success"])
    required = ["external_goals_sha256", "initial_root_xyz_m",
                "initial_root_quat_wxyz", "initial_joint_pos_rad",
                "block_initial_xyz_m", "crate_initial_xyz_m", "insert_initial_xyz_m",
                "drawer_initial_position_m", "drawer_asset_sha256",
                "pedal_initial_position_m", "pedal_asset_sha256"]
    if "completed" in a and "completed" in b:
        aa = {(r["name"], r["seed"]): r for r in a["rows"]}
        bb = {(r["name"], r["seed"]): r for r in b["rows"]}
        if aa.keys() != bb.keys():
            raise ValueError("Suite motion/seed sets differ")
        pairs = []
        for key in sorted(aa):
            checks = {field: aa[key].get(field) == bb[key].get(field) for field in required}
            pairs.append({"name": key[0], "seed": key[1], "checks": checks,
                          "all_match": all(checks.values()),
                          "supported_success": aa[key]["task_success"],
                          "baseline_success": bb[key]["task_success"]})
        report = {"supported": str(args.supported), "baseline": str(args.baseline),
                  "pairs": pairs, "all_match": all(r["all_match"] for r in pairs)}
    else:
        checks = {key: a.get(key) == b.get(key) for key in required}
        for key in ("dataset", "split", "motion_id", "seed", "robot_name",
                    "door_object_reference", "door_object_initial_xyz_m",
                    "door_object_shelf_xyz_m"):
            if key in a or key in b:
                checks[key] = a.get(key) == b.get(key)
        goal_fields = ("reference_t", "left_hand_goal_xyz_m", "right_hand_goal_xyz_m",
                       "target_foot_xyz_m", "right_sole_goal_xyz_m")
        checks["per_frame_targets"] = (len(a.get("rows", [])) == len(b.get("rows", []))
                                        and len(a.get("rows", [])) > 0
                                        and all(all(ar.get(key) == br.get(key) for key in goal_fields)
                                                for ar, br in zip(a["rows"], b["rows"])))
        report = {"supported": str(args.supported), "baseline": str(args.baseline),
                  "checks": checks, "all_match": all(checks.values()),
                  "supported_success": a_success, "baseline_success": b_success}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    if not report["all_match"]:
        raise SystemExit("Initial state or external goal mismatch")


if __name__ == "__main__":
    main()

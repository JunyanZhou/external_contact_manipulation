"""One-factor-at-a-time block sweep for the fixed table-push side trajectory."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = dict(mass=.08, friction=.1, height=.16, x=.4, y=-.02)
CASES = {
    "mass": [.02, .2, .5, 1., 2., 5., 10.],
    "friction": [.2, .3, .4, .5, .7, 1.],
    "height": [.04, .08, .12, .2, .24, .28],
    "x": [.32, .36, .38, .39, .41, .42, .44, .48],
    "y": [-.1, -.06, 0., .04, .06, .07, .08],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--groups", nargs="+", choices=tuple(CASES), default=list(CASES))
    p.add_argument("--cases", nargs="*", help="Only run these case labels, e.g. mass_2")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = p.parse_args()
    destination = ROOT / "artifacts/table_push/block_robustness"
    destination.mkdir(parents=True, exist_ok=True)
    python = Path(os.environ.get("PYTHON", sys.executable))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), str(Path(env["ISAACLAB_PATH"]) / "source/isaaclab") if env.get("ISAACLAB_PATH") else "", env.get("PYTHONPATH", "")]))
    env["OMNI_KIT_ACCEPT_EULA"] = "Y"
    rows = []
    for group in args.groups:
        for value in CASES[group]:
            case = dict(BASE)
            case[group] = value
            label = f"{group}_{value:g}".replace("-", "n").replace(".", "p")
            if args.cases and label not in args.cases:
                continue
            for seed in args.seeds:
                run_case(destination, python, env, case, group, value, label, seed, rows)
    suffix = "_selected" if args.cases else ""
    (destination / f"summary{suffix}.json").write_text(json.dumps(rows, indent=2) + "\n")


def run_case(destination, python, env, case, group, value, label, seed, rows):
    label = label if seed == 0 else f"{label}_seed{seed}"
    output = destination / f"{label}.json"
    log = destination / f"{label}.log"
    if not output.exists():
        command = [str(python), str(ROOT / "scripts/table_push/eval_task_policy.py"),
            "--checkpoint", "artifacts/table_push/checkpoints/task_v3_27m.pt",
            "--dataset", "table_push_v6_hand08", "--split", "train", "--motion-id", "7",
            "--external-target-npz", "artifacts/table_push/target_side26.npz",
            "--table-width", ".22", "--block", "--block-size-x", ".06",
            "--block-size-y", ".08", "--steps", "870", "--seed", str(seed),
            "--block-mass", str(case["mass"]), "--block-friction", str(case["friction"]),
            "--block-height", str(case["height"]), "--block-x", str(case["x"]),
            "--block-y", str(case["y"]), "--output", str(output)]
        print("RUN", label, case, flush=True)
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT)
        if result.returncode:
            print("ERROR", label, result.returncode, flush=True)
            return
    report = json.loads(output.read_text())
    stable = (report["min_root_height_m"] > .6 and report["max_root_tilt_rad"] < .35
              and report["right_support_fraction"] > .95
              and report["both_hands_support_fraction"] > .9)
    def tilt(quaternion):
        _, qx, qy, _ = quaternion
        return math.acos(max(-1., min(1., 1. - 2. * (qx * qx + qy * qy))))
    block_tilt = max(tilt(sample["block_quat_wxyz"]) for sample in report["rows"])
    upright = block_tilt < .35 and abs(report["block_final_xyz_m"][2] - case["height"] / 2) < .02
    cleared = report["block_table_edge_clearance_m"] > 0.
    initially_under_table = abs(case["y"]) + .04 <= .11 + 1e-8
    row = dict(group=group, value=value, seed=seed, stable=stable,
        upright=upright, max_block_tilt_rad=block_tilt, cleared=cleared,
        initially_under_table=initially_under_table,
        success=stable and upright and cleared and initially_under_table,
        lateral_m=report["block_lateral_displacement_m"],
        edge_clearance_m=report["block_table_edge_clearance_m"],
        error_m=report["mean_foot_error_m"],
        root_height_m=report["min_root_height_m"],
        hand_support_fraction=report["both_hands_support_fraction"],
        right_support_fraction=report["right_support_fraction"],
        block_final_xyz_m=report["block_final_xyz_m"],
        block_final_quat_wxyz=report["block_final_quat_wxyz"],
        report=str(output.relative_to(ROOT)))
    rows.append(row)
    (destination / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("RESULT", label, "success", row["success"], "upright", upright,
          "dy", round(row["lateral_m"], 3),
          "edge", round(row["edge_clearance_m"], 3),
          "error", round(row["error_m"], 3), flush=True)

if __name__ == "__main__":
    main()

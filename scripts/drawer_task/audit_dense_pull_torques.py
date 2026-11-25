"""Audit every sampled drawer reference frame with Pinocchio inverse dynamics."""

import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from optimize_handle_engaged_pull import ROOT, allocate, load_usd, add_frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    model, names, meshes, _ = load_usd(ROOT / "active_adaptation/assets/G1/g1_flat_fullcol.usd")
    sole_z = min(v[:, 2].min() for n, v, _ in meshes if n == "left_ankle_roll_link")
    ids = {n: model.getFrameId(n) for n in ("left_hand_mimic", "right_hand_mimic")}
    for side in ("left", "right"):
        ids[side + "_sole"] = add_frame(model, side + "_dense_sole",
            side + "_ankle_roll_link", np.array([0., 0., sole_z]))
    with np.load(args.motion) as motion:
        qs = np.column_stack((motion["root_pos"], motion["root_rot"], motion["dof_pos"]))
        fps = float(motion["fps"])
    data = model.createData()
    pin.framesForwardKinematics(model, data, qs[0])
    hand_x0 = float(data.oMf[ids["right_hand_mimic"]].translation[0])
    depth = []
    for q in qs:
        pin.framesForwardKinematics(model, data, q)
        depth.append(max(0., hand_x0 - float(data.oMf[ids["right_hand_mimic"]].translation[0])))
    depth = np.asarray(depth)
    speed = np.gradient(depth) * fps
    accel = np.gradient(speed) * fps
    v = np.zeros((len(qs), model.nv)); a = np.zeros_like(v)
    for i in range(1, len(qs)):
        v[i] = pin.difference(model, qs[i - 1], qs[i]) * fps
    v[0] = v[1]
    a[1:] = np.diff(v, axis=0) * fps
    a[0] = a[1]
    rows = []
    for i, q in enumerate(qs):
        row = allocate(model, data, ids, q, v[i], a[i], depth[i], speed[i], accel[i])
        rows.append({"frame": i, "depth_m": float(depth[i]), **row})
    report = {"frames_checked": len(rows), "joint_names": names,
        "feasible_frames": sum(row["feasible"] for row in rows),
        "max_effort_ratio": max((row.get("max_effort_ratio", 0.) for row in rows), default=0.),
        "rows": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print({k: v for k, v in report.items() if k != "rows" and k != "joint_names"})


if __name__ == "__main__":
    main()

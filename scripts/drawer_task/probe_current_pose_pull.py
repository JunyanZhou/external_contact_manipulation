"""Probe a lower-drawer pull starting from the approved leaning stance.

This generates IK references only. Physical replay must establish contact and
move the drawer without prescribed slider motion to count as success.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin
import crocoddyl as croc
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/references"))
from generate_wall_support import load_usd, add_frame
from generate_foot_reach_dataset import body_names_from_loader, quintic_segment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--depth", type=float, default=.10)
    args = p.parse_args()
    model, names, meshes, _ = load_usd(ROOT / "active_adaptation/assets/G1/g1_flat_fullcol.usd")
    with np.load(args.start) as f:
        start = np.r_[f["root_pos"][0], f["root_rot"][0], f["dof_pos"][0]].astype(float)
    sole_z = min(v[:, 2].min() for n, v, _ in meshes if n == "left_ankle_roll_link")
    ids = {n: model.getFrameId(n) for n in ("left_hand_mimic", "right_hand_mimic", "torso_link")}
    for side in ("left", "right"):
        ids[side + "_sole"] = add_frame(model, side + "_probe_sole",
            side + "_ankle_roll_link", np.array([0., 0., sole_z]))
    data = model.createData()
    pin.framesForwardKinematics(model, data, start)
    feet = {side: data.oMf[ids[side + "_sole"]].translation.copy()
            for side in ("left", "right")}
    left = data.oMf[ids["left_hand_mimic"]].translation.copy()
    right_start = data.oMf[ids["right_hand_mimic"]].translation.copy()
    torso_R = data.oMf[ids["torso_link"]].rotation.copy()
    state = croc.StateMultibody(model)
    collector = croc.DataCollectorMultibody(data)
    lower = np.r_[[-.25, -.12, .58], model.lowerPositionLimit[7:] + .012]
    upper = np.r_[[.14, .12, .84], model.upperPositionLimit[7:] - .012]

    def solve(previous, right):
        targets = [("left_sole", feet["left"]),
                   ("right_sole", feet["right"]),
                   ("left_hand_mimic", left),
                   ("right_hand_mimic", right)]
        residuals = [croc.ResidualModelFrameTranslation(state, ids[n], t, 0)
                     for n, t in targets]
        residual_data = [r.createData(collector) for r in residuals]

        def error(x):
            q = start.copy(); q[:3] = x[:3]; q[7:] = x[3:]
            pin.framesForwardKinematics(model, data, q)
            state_x = np.r_[q, np.zeros(model.nv)]
            for r, rd in zip(residuals, residual_data):
                r.calc(rd, state_x, np.zeros(0))
            return np.concatenate([
                *[w * rd.r for w, rd in zip((90., 90., 70., 75.), residual_data)],
                15. * pin.log3(data.oMf[ids["left_hand_mimic"]].rotation),
                15. * pin.log3(torso_R.T @ data.oMf[ids["torso_link"]].rotation),
                10. * (x[:3] - start[:3]),
                .08 * (x[3:] - previous[7:])])

        x0 = np.clip(np.r_[previous[:3], previous[7:]], lower + 1e-5, upper - 1e-5)
        result = least_squares(error, x0, bounds=(lower, upper), max_nfev=250,
                               ftol=1e-7, xtol=1e-7, gtol=1e-7)
        q = start.copy(); q[:3] = result.x[:3]; q[7:] = result.x[3:]
        pin.framesForwardKinematics(model, data, q)
        errors = {n: float(np.linalg.norm(data.oMf[ids[n]].translation - target))
                  for n, target in targets}
        return q, errors

    # Pass above the lower handle before descending behind its pull bar.
    waypoints = [
        (right_start, 0.),
        (np.array([.18, -.055, .55]), .7),
        (np.array([.18, -.055, .50]), .7),
        (np.array([.29, -.055, .50]), .6),
        (np.array([.29, -.055, .45]), .6),
        (np.array([.29 - args.depth, -.055, .45]), 2.0),
    ]
    poses = [start]
    errors = []
    for target, _ in waypoints[1:]:
        q, err = solve(poses[-1], target)
        poses.append(q); errors.append(err)
    chunks = []
    for (a, b), (_, seconds) in zip(zip(poses[:-1], poses[1:]), waypoints[1:]):
        joints, _, _ = quintic_segment(a[7:], b[7:], seconds, 50)
        roots, _, _ = quintic_segment(a[:3], b[:3], seconds, 50)
        qs = np.tile(start, (len(joints) - 1, 1))
        qs[:, :3] = roots[:-1]; qs[:, 7:] = joints[:-1]
        chunks.append(qs)
        chunks.append(np.tile(b, (15, 1)))
    chunks.append(np.tile(poses[-1], (25, 1)))
    qs = np.concatenate(chunks)
    body_names = body_names_from_loader()
    body_ids = [model.getFrameId(n) for n in body_names]
    world_pos = []; world_rot = []
    max_contact_error = 0.
    for q in qs:
        pin.framesForwardKinematics(model, data, q)
        world_pos.append([data.oMf[f].translation.copy() for f in body_ids])
        world_rot.append([data.oMf[f].rotation.copy() for f in body_ids])
        max_contact_error = max(max_contact_error,
            *(float(np.linalg.norm(data.oMf[ids[side + "_sole"]].translation - feet[side]))
              for side in ("left", "right")),
            float(np.linalg.norm(data.oMf[ids["left_hand_mimic"]].translation - left)))
    world_pos = np.asarray(world_pos); world_rot = np.asarray(world_rot)
    root_R = Rotation.from_quat(start[3:7]).as_matrix()
    local_pos = (world_pos - qs[:, None, :3]) @ root_R
    local_rot = np.einsum("ij,tbjk->tbik", root_R.T, world_rot)
    local_quat = Rotation.from_matrix(local_rot.reshape(-1, 3, 3)).as_quat().reshape(len(qs), -1, 4)
    v = np.zeros((len(qs), model.nv))
    for i in range(1, len(qs)):
        v[i] = pin.difference(model, qs[i-1], qs[i]) * 50.
    a = np.zeros_like(v); a[1:] = np.diff(v, axis=0) * 50.
    dest = args.output / "motions/train/current_pose_pull.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, fps=np.array(50), joint_names=np.array(names),
        body_names=np.array(body_names), root_pos=qs[:, :3].astype("f4"),
        root_rot=qs[:, 3:7].astype("f4"), dof_pos=qs[:, 7:].astype("f4"),
        local_body_pos=local_pos.astype("f4"), local_body_rot=local_quat.astype("f4"),
        joint_vel=v[:, 6:].astype("f4"), joint_acc=a[:, 6:].astype("f4"))
    manifest = {"fps": 50, "joint_names": names, "body_names": body_names,
        "clips": [{"name": "current_pose_pull", "split": "train",
                   "file": "motions/train/current_pose_pull.npz", "frames": len(qs),
                   "pull_depth_m": args.depth, "waypoints": [p.tolist() for p, _ in waypoints],
                   "waypoint_errors_m": errors,
                   "max_fixed_contact_position_error_m": max_contact_error}]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"frames": len(qs), "waypoint_errors_m": errors,
        "max_fixed_contact_position_error_m": max_contact_error}, indent=2))


if __name__ == "__main__":
    main()

"""Build a static two-drawer pose for visual review (kinematics only)."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/references"))
from generate_wall_support import add_frame, load_usd
from generate_foot_reach_dataset import body_names_from_loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--torso-ground-angle", type=float, default=60.)
    p.add_argument("--feet-back-offset", type=float, default=0.,
                   help="Move both feet away from the drawer along negative world X (m)")
    p.add_argument("--root-x-goal", type=float, default=None,
                   help="Keep the pelvis forward of the feet to increase leg lean (m)")
    p.add_argument("--palm-align-weight", type=float, default=0.,
                   help="Align left palm block with the vertical drawer front")
    args = p.parse_args()
    model, joint_names, meshes, _ = load_usd(ROOT / "active_adaptation/assets/G1/g1_flat_fullcol.usd")
    q0 = np.load(ROOT / "artifacts/table_push/static_poses_hand08.npz")["q_pinocchio"][0].copy()
    sole_z = min(v[:, 2].min() for name, v, _ in meshes if name == "left_ankle_roll_link")
    ids = {name: model.getFrameId(name) for name in
           ("left_hand_mimic", "right_hand_mimic", "torso_link", "head_mimic")}
    for side in ("left", "right"):
        ids[f"{side}_sole"] = add_frame(model, f"lean_{side}_sole",
                                         f"{side}_ankle_roll_link", np.array([0., 0., sole_z]))
    d = model.createData()
    pin.framesForwardKinematics(model, d, q0)
    torso_R0 = d.oMf[ids["torso_link"]].rotation.copy()
    foot_R0 = {side: d.oMf[ids[f"{side}_sole"]].rotation.copy()
               for side in ("left", "right")}
    foot_goal = {"left": np.array([.135, .19, .0]),
                 "right": np.array([-.155, -.18, .0])}
    for goal in foot_goal.values():
        goal[0] -= args.feet_back_offset
    hand_goal = {"left": np.array([.35, .185, .82]),
                 "right": np.array([.18, -.13, .61])}
    torso_R_goal = Rotation.from_euler("y", np.deg2rad(90. - args.torso_ground_angle)).as_matrix() @ torso_R0
    root_goal = np.array([args.root_x_goal if args.root_x_goal is not None else
                          -.025 - .65 * args.feet_back_offset, .0, .70])
    lower = np.r_[[-.30, -.10, .60], model.lowerPositionLimit[7:] + .012]
    upper = np.r_[[.14, .10, .84], model.upperPositionLimit[7:] - .012]
    left_knee = joint_names.index("left_knee_joint")
    right_knee = joint_names.index("right_knee_joint")

    def evaluate(x):
        q = q0.copy(); q[:3] = x[:3]; q[7:] = x[3:]
        pin.framesForwardKinematics(model, d, q)
        parts = []
        for side in ("left", "right"):
            frame = ids[f"{side}_sole"]
            parts.append(100. * (d.oMf[frame].translation - foot_goal[side]))
            parts.append(8. * pin.log3(foot_R0[side].T @ d.oMf[frame].rotation))
        parts += [32. * (d.oMf[ids["left_hand_mimic"]].translation - hand_goal["left"]),
                  12. * (d.oMf[ids["right_hand_mimic"]].translation - hand_goal["right"]),
                  args.palm_align_weight * pin.log3(
                      d.oMf[ids["left_hand_mimic"]].rotation),
                  20. * pin.log3(torso_R_goal.T @ d.oMf[ids["torso_link"]].rotation),
                  (15. if args.root_x_goal is not None else 8.) * (x[:3] - root_goal),
                  2. * np.array([q[7 + left_knee] - .95, q[7 + right_knee] - .38]),
                  .04 * (x[3:] - q0[7:])]
        return np.concatenate(parts)

    x0 = np.r_[q0[:3], q0[7:]]
    result = least_squares(evaluate, np.clip(x0, lower + 1e-5, upper - 1e-5),
                           bounds=(lower, upper), max_nfev=350,
                           ftol=1e-8, xtol=1e-8, gtol=1e-8)
    q = q0.copy(); q[:3] = result.x[:3]; q[7:] = result.x[3:]
    pin.framesForwardKinematics(model, d, q)
    torso_z = d.oMf[ids["torso_link"]].rotation[:, 2]
    torso_ground_angle = np.rad2deg(np.arcsin(np.clip(torso_z[2], -1, 1)))
    com = pin.centerOfMass(model, d, q).copy()
    metrics = {"requested_torso_ground_angle_deg": args.torso_ground_angle,
               "actual_torso_ground_angle_deg": float(torso_ground_angle),
               "root_xyz_m": q[:3].tolist(),
               "center_of_mass_xyz_m": com.tolist(),
               "com_ahead_of_front_sole_m": float(com[0] - max(
                   d.oMf[ids[f"{side}_sole"]].translation[0] for side in ("left", "right"))),
               "knee_rad": {"left": float(q[7 + left_knee]),
                            "right": float(q[7 + right_knee])},
               "left_palm_normal_world": d.oMf[ids["left_hand_mimic"]].rotation[:, 0].tolist(),
               "targets_m": {"left_hand": hand_goal["left"].tolist(),
                             "right_hand": hand_goal["right"].tolist(),
                             "left_sole": foot_goal["left"].tolist(),
                             "right_sole": foot_goal["right"].tolist()},
               "actual_m": {name: d.oMf[ids[name]].translation.tolist()
                            for name in ("left_hand_mimic", "right_hand_mimic", "left_sole", "right_sole")},
               "ik_success": bool(result.success),
               "ik_cost": float(result.cost),
               "preview_only": True}
    metrics["feet_back_offset_m"] = args.feet_back_offset
    print(json.dumps(metrics, indent=2), flush=True)
    body_names = body_names_from_loader()
    body_ids = [model.getFrameId(name) for name in body_names]
    root_R = Rotation.from_quat(q[3:7]).as_matrix()
    pos = np.array([d.oMf[f].translation.copy() for f in body_ids])
    rot = np.array([d.oMf[f].rotation.copy() for f in body_ids])
    local_pos = (pos - q[:3]) @ root_R
    local_rot = np.einsum("ij,bjk->bik", root_R.T, rot)
    local_quat = Rotation.from_matrix(local_rot).as_quat()
    frames = 60
    path = args.output / "motions" / "train" / "lean_pose_60deg.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, fps=np.array(50), joint_names=np.array(joint_names),
        body_names=np.array(body_names), root_pos=np.tile(q[:3], (frames, 1)).astype("f4"),
        root_rot=np.tile(q[3:7], (frames, 1)).astype("f4"),
        dof_pos=np.tile(q[7:], (frames, 1)).astype("f4"),
        local_body_pos=np.tile(local_pos, (frames, 1, 1)).astype("f4"),
        local_body_rot=np.tile(local_quat, (frames, 1, 1)).astype("f4"),
        joint_vel=np.zeros((frames, model.nv - 6), dtype="f4"),
        joint_acc=np.zeros((frames, model.nv - 6), dtype="f4"))
    (args.output / "manifest.json").write_text(json.dumps({"fps": 50,
        "joint_names": joint_names, "body_names": body_names,
        "clips": [{"name": "lean_pose_60deg", "split": "train",
                   "file": str(path.relative_to(args.output)), "frames": frames}]}, indent=2) + "\n")
    (args.output / "pose_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()

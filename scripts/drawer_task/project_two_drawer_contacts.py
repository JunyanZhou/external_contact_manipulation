"""Project interpolated drawer motions onto fixed foot and palm contacts.

This corrects contact geometry before an inverse-dynamics audit.  It does not
pretend that projection alone produces a dynamically tracked trajectory.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/references"))
from generate_wall_support import add_frame, load_usd
from generate_foot_reach_dataset import body_names_from_loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    model, names, meshes, _ = load_usd(ROOT / "active_adaptation/assets/G1/g1_flat_fullcol.usd")
    sole_z = min(v[:, 2].min() for n, v, _ in meshes if n == "left_ankle_roll_link")
    ids = {}
    for side in ("left", "right"):
        ids[f"{side}_sole"] = add_frame(model, f"drawer_{side}_sole",
                                        f"{side}_ankle_roll_link", np.array([0., 0., sole_z]))
    ids["left_hand"] = model.getFrameId("left_hand_mimic")
    data = model.createData()
    body_names = body_names_from_loader()
    body_ids = [model.getFrameId(name) for name in body_names]
    manifest = json.loads((args.source / "manifest.json").read_text())
    report = []
    for clip in manifest["clips"]:
        with np.load(args.source / clip["file"]) as f:
            q_ref = np.column_stack((f["root_pos"], f["root_rot"], f["dof_pos"])).astype(float)
            fps = int(f["fps"])
        q_out = q_ref.copy()
        pin.framesForwardKinematics(model, data, q_ref[0])
        anchors = {side: data.oMf[ids[f"{side}_sole"]].translation.copy()
                   for side in ("left", "right")}
        hand_start = int(round(clip.get("left_reach_seconds", 4.0) * fps)) + 25
        pin.framesForwardKinematics(model, data, q_ref[hand_start])
        hand_anchor = data.oMf[ids["left_hand"]].translation.copy()
        worst = 0.
        for i, q0 in enumerate(q_ref):
            q = q0.copy()
            targets = [(ids["left_sole"], anchors["left"]),
                       (ids["right_sole"], anchors["right"])]
            if i >= hand_start:
                targets.append((ids["left_hand"], hand_anchor))
            for _ in range(12):
                pin.computeJointJacobians(model, data, q)
                pin.updateFramePlacements(model, data)
                error = np.concatenate([data.oMf[frame].translation - goal
                                        for frame, goal in targets])
                if np.max(np.abs(error)) < 1e-5:
                    break
                jac = np.vstack([pin.getFrameJacobian(model, data, frame,
                              pin.LOCAL_WORLD_ALIGNED)[:3] for frame, _ in targets])
                cols = np.r_[np.arange(3), np.arange(6, model.nv)]
                j = jac[:, cols]
                dq = -j.T @ np.linalg.solve(j @ j.T + 1e-7 * np.eye(len(error)), error)
                velocity = np.zeros(model.nv)
                velocity[cols] = dq
                q = pin.integrate(model, q, velocity)
            q_out[i] = q
            worst = max(worst, float(np.max(np.abs(error))))
        positions = []; rotations = []
        for q in q_out:
            pin.framesForwardKinematics(model, data, q)
            positions.append([data.oMf[f].translation.copy() for f in body_ids])
            rotations.append([data.oMf[f].rotation.copy() for f in body_ids])
        positions = np.asarray(positions); rotations = np.asarray(rotations)
        root_R = Rotation.from_quat(q_out[0, 3:7]).as_matrix()
        local_pos = (positions - q_out[:, None, :3]) @ root_R
        local_rot = np.einsum("ij,tbjk->tbik", root_R.T, rotations)
        local_quat = Rotation.from_matrix(local_rot.reshape(-1, 3, 3)).as_quat().reshape(len(q_out), -1, 4)
        vel = np.zeros((len(q_out), model.nv))
        for i in range(1, len(q_out)):
            vel[i] = pin.difference(model, q_out[i-1], q_out[i]) * fps
        acc = np.zeros_like(vel)
        acc[1:] = np.diff(vel, axis=0) * fps
        path = args.output / clip["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, fps=np.array(fps), joint_names=np.array(names),
            body_names=np.array(body_names), root_pos=q_out[:, :3].astype("f4"),
            root_rot=q_out[:, 3:7].astype("f4"), dof_pos=q_out[:, 7:].astype("f4"),
            local_body_pos=local_pos.astype("f4"), local_body_rot=local_quat.astype("f4"),
            joint_vel=vel[:, 6:].astype("f4"), joint_acc=acc[:, 6:].astype("f4"))
        report.append({"name": clip["name"], "max_contact_coordinate_error_m": worst,
                       "max_joint_correction_rad": float(np.max(np.abs(q_out[:, 7:] - q_ref[:, 7:]))),
                       "max_root_correction_m": float(np.max(np.abs(q_out[:, :3] - q_ref[:, :3])))})
        print(report[-1], flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "contact_projection_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Generate conservative bilateral-table-contact left-foot reach/push clips."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/references"))
from generate_wall_support import add_frame, load_usd  # noqa: E402
from generate_foot_reach_dataset import body_names_from_loader, quintic_segment  # noqa: E402
from scripts.table_push.generate_static_pose import static_contact_lp  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose", type=Path, default=ROOT / "artifacts/table_push/static_poses_small_fixed.npz")
    p.add_argument("--pose-index", type=int, default=1,
                   help="Static support pose index used as the trajectory anchor")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/table_push/task_dataset_v3")
    p.add_argument("--seconds", type=float, default=2.5)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--push-grid", action="store_true",
                   help="Generate fixed-block push variants with endpoint-held-out validation")
    p.add_argument("--push-reaches-cm", type=int, nargs="+", default=[38, 39, 40, 41, 42])
    p.add_argument("--push-sides-cm", type=int, nargs="+", default=[20, 22, 24, 26])
    args = p.parse_args()
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Choose a new output directory")
    model, names, meshes, _ = load_usd(ROOT / "active_adaptation/assets/G1/g1_flat_fullcol.usd")
    pose_file = np.load(args.pose)
    q0 = pose_file["q_pinocchio"][args.pose_index].copy()
    sole_offset = min(v[:, 2].min() for n, v, _ in meshes if n == "left_ankle_roll_link")
    ids = {name: model.getFrameId(name) for name in
           ("left_hand_mimic", "right_hand_mimic", "torso_link", "left_ankle_roll_link")}
    ids["left_sole"] = add_frame(model, "table_task_left_sole", "left_ankle_roll_link", np.array([0, 0, sole_offset]))
    ids["right_sole"] = add_frame(model, "table_task_right_sole", "right_ankle_roll_link", np.array([0, 0, sole_offset]))
    data = model.createData()
    pin.framesForwardKinematics(model, data, q0)
    anchor = data.oMf[ids["left_sole"]].translation.copy()
    fixed = {name: data.oMf[ids[name]].translation.copy() for name in
             ("left_hand_mimic", "right_hand_mimic", "right_sole")}
    torso_R = data.oMf[ids["torso_link"]].rotation.copy()
    body_names = body_names_from_loader()
    body_ids = [model.getFrameId(n) for n in body_names]
    lower = np.r_[[-.15, -.15, .67], model.lowerPositionLimit[7:] + .02]
    upper = np.r_[[.18, .13, .9], model.upperPositionLimit[7:] - .02]
    qids = np.arange(7, model.nq)
    base_x = np.r_[q0[:3], q0[7:]]
    # Start with a foot lift, then reach under the front edge and sweep sideways.
    # Paths are deliberately graded by horizontal reach.
    plans = [
        ("lift03", [[.00, .00, .03]]),
        ("reach10", [[.00, .00, .04], [.10, .00, .04], [.10, -.04, .04]]),
        ("reach18", [[.00, .00, .05], [.18, .00, .05], [.18, -.06, .05]]),
        ("reach25", [[.00, .00, .06], [.25, .00, .06], [.25, -.08, .06]]),
        ("reach30", [[.00, .00, .06], [.30, .00, .06], [.30, -.10, .06]]),
        ("reach35", [[.00, .00, .07], [.35, .00, .07], [.35, -.12, .07]]),
        ("reach40", [[.00, .00, .07], [.40, .00, .07], [.40, -.14, .07]]),
        ("reach40_side18", [[.00, .00, .07], [.40, .00, .07], [.40, -.18, .07]]),
        ("reach40_side22", [[.00, .00, .07], [.40, .00, .07], [.40, -.22, .07]]),
    ]
    if args.push_grid:
        plans = []
        for reach_cm in args.push_reaches_cm:
            for side_cm in args.push_sides_cm:
                split = "val" if ((reach_cm in (39, 41) and side_cm in (22, 26, 30))
                                  or (reach_cm == 40 and side_cm in (29, 31))) else "train"
                for variant, seconds in enumerate((2.5, 3.0)):
                    name = f"push_x{reach_cm}_y{side_cm}_v{variant}"
                    offsets = [[.00, .00, .07], [reach_cm/100, .00, .07],
                               [reach_cm/100, -side_cm/100, .07]]
                    plans.append((name, offsets, seconds, split))
    else:
        plans = [(name, offsets, args.seconds, "val" if name == "reach18" else "train")
                 for name, offsets in plans]
    clips, rejected = [], []
    for name, offsets, seconds, split in plans:
        waypoints = [q0]
        for offset in offsets:
            target = anchor + np.asarray(offset)
            guess = waypoints[-1]
            x_init = np.r_[guess[:3], guess[7:]]

            def residual(x):
                q = q0.copy()
                q[:3], q[7:] = x[:3], x[3:]
                pin.framesForwardKinematics(model, data, q)
                terms = [18 * (data.oMf[ids["left_sole"]].translation - target)]
                terms += [22 * (data.oMf[ids[n]].translation - point) for n, point in fixed.items()]
                terms += [2 * pin.log3(torso_R.T @ data.oMf[ids["torso_link"]].rotation),
                          1.5 * (x[:3] - q0[:3]), .10 * (x[3:] - base_x[3:])]
                return np.concatenate(terms)

            solved = least_squares(residual, x_init, bounds=(lower, upper), max_nfev=250,
                                   ftol=1e-8, xtol=1e-8, gtol=1e-8)
            q = q0.copy()
            q[:3], q[7:] = solved.x[:3], solved.x[3:]
            pin.framesForwardKinematics(model, data, q)
            error = np.linalg.norm(data.oMf[ids["left_sole"]].translation - target)
            support_error = max(np.linalg.norm(data.oMf[ids[n]].translation - point)
                                for n, point in fixed.items())
            statics = static_contact_lp(model, q, ids)
            print(name, offset, "foot_error", error, "support_error", support_error,
                  "contact_lp", statics["feasible"], flush=True)
            if error > .012 or support_error > .012 or not statics["feasible"]:
                rejected.append({"name": name, "offset": offset, "foot_error": error,
                                 "support_error": support_error, "contact_lp": statics})
                break
            waypoints.append(q)
        else:
            path = waypoints + waypoints[-2::-1]
            chunks = []
            for a, b in zip(path[:-1], path[1:]):
                joints, vel, acc = quintic_segment(a[7:], b[7:], seconds, args.fps)
                roots, root_vel, root_acc = quintic_segment(a[:3], b[:3], seconds, args.fps)
                qs = np.tile(q0, (len(joints)-1, 1))
                qs[:, :3], qs[:, 7:] = roots[:-1], joints[:-1]
                vs = np.zeros((len(qs), model.nv))
                aa = np.zeros_like(vs)
                vs[:, :3], vs[:, 6:] = root_vel[:-1], vel[:-1]
                aa[:, :3], aa[:, 6:] = root_acc[:-1], acc[:-1]
                chunks.append((qs, vs, aa))
                chunks.append((np.tile(b, (20, 1)), np.zeros((20, model.nv)),
                               np.zeros((20, model.nv))))
            qs, vs, aa = (np.concatenate([c[k] for c in chunks]) for k in range(3))
            world_pos, world_rot, foot_target = [], [], []
            bad = None
            for i, (q, v, a) in enumerate(zip(qs, vs, aa)):
                pin.framesForwardKinematics(model, data, q)
                sole = data.oMf[ids["left_sole"]].translation.copy()
                if sole[2] < -.004:
                    bad = (i, "ground_intersection")
                    break
                contact = static_contact_lp(model, q, ids, v, a)
                if i % 5 == 0 and not contact["feasible"]:
                    bad = (i, "dynamic_contact_lp")
                    break
                world_pos.append([data.oMf[f].translation.copy() for f in body_ids])
                world_rot.append([data.oMf[f].rotation.copy() for f in body_ids])
                foot_target.append(Rotation.from_quat(q[3:7]).inv().apply(sole-q[:3]))
            if bad:
                rejected.append({"name": name, "frame": bad[0], "reason": bad[1]})
                print("REJECT", name, bad, flush=True)
                continue
            wp, wr = np.asarray(world_pos), np.asarray(world_rot)
            root_R = Rotation.from_quat(q0[3:7]).as_matrix()
            local_pos = (wp - qs[:, None, :3]) @ root_R
            local_rot = np.einsum("ij,tbjk->tbik", root_R.T, wr)
            body_quat = Rotation.from_matrix(local_rot.reshape(-1, 3, 3)).as_quat().reshape(len(qs), -1, 4)
            dest = args.output / "motions" / split / f"{name}.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, fps=np.array(args.fps), joint_names=np.array(names),
                body_names=np.array(body_names), root_pos=qs[:, :3].astype("f4"),
                root_rot=qs[:, 3:7].astype("f4"), dof_pos=qs[:, 7:].astype("f4"),
                local_body_pos=local_pos.astype("f4"), local_body_rot=body_quat.astype("f4"),
                target_foot_pos_b=np.asarray(foot_target, dtype="f4"),
                joint_vel=vs[:, 6:].astype("f4"), joint_acc=aa[:, 6:].astype("f4"))
            clips.append({"name": name, "split": split, "file": str(dest.relative_to(args.output)),
                          "frames": len(qs), "max_offset_m": offsets[-1], "segment_seconds": seconds})
            print("ACCEPT", name, len(qs), flush=True)
    manifest = {"fps": args.fps, "pose_file": str(args.pose), "joint_names": names,
                "body_names": body_names, "sole_offset": [0, 0, sole_offset],
                "clips": clips, "rejected": rejected,
                "note": "Ideal contact inverse dynamics screen; Isaac closed-loop remains required."}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not clips:
        raise RuntimeError("No dynamically screened trajectory was accepted")


if __name__ == "__main__":
    main()

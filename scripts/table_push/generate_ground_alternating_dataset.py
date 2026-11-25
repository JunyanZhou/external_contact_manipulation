"""Generate small alternating-foot G1 references with ground as the only support."""
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
from scripts.references.generate_wall_support import add_frame, load_usd
from scripts.references.generate_foot_reach_dataset import body_names_from_loader, quintic_segment


def stand(model, names, meshes):
    q = pin.neutral(model)
    q[:3] = (0., 0., .772)
    values = {
        "hip_pitch_joint": -.28, "knee_joint": .5, "ankle_pitch_joint": -.23,
        "elbow_joint": .87, "shoulder_pitch_joint": .35,
        "left_shoulder_roll_joint": .16, "right_shoulder_roll_joint": -.16,
    }
    for i, name in enumerate(names):
        for pattern, value in values.items():
            if name == pattern or (not pattern.startswith(("left_", "right_")) and name.endswith(pattern)):
                q[7+i] = value
    sole_z = min(v[:, 2].min() for n, v, _ in meshes if n == "left_ankle_roll_link")
    ids = {side: add_frame(model, f"ground_{side}_sole", f"{side}_ankle_roll_link",
                           np.array([0., 0., sole_z])) for side in ("left", "right")}
    ids["torso"] = model.getFrameId("torso_link")
    ids["left_hand"] = model.getFrameId("left_hand_mimic")
    ids["right_hand"] = model.getFrameId("right_hand_mimic")
    return q, ids, sole_z


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--random-clips", type=int, default=0,
                        help="Generate seeded, modest hand/foot reach variations for an in-place baseline")
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--walk-step", type=float, default=0.,
                        help="Forward displacement per alternating-foot cycle")
    args = parser.parse_args()
    if (args.output / "manifest.json").exists():
        raise FileExistsError(args.output / "manifest.json")
    model, names, meshes, _ = load_usd(ROOT / "active_adaptation/assets/G1/g1_flat_fullcol.usd")
    q0, ids, sole_z = stand(model, names, meshes)
    data = model.createData()
    pin.framesForwardKinematics(model, data, q0)
    feet0 = {side: data.oMf[ids[side]].translation.copy() for side in ("left", "right")}
    hands0 = {side: data.oMf[ids[f"{side}_hand"]].translation.copy()
              for side in ("left", "right")}
    torso_R = data.oMf[ids["torso"]].rotation.copy()
    body_names = body_names_from_loader()
    body_ids = [model.getFrameId(name) for name in body_names]
    lower = np.r_[[-.12, -.12, .67], model.lowerPositionLimit[7:] + .025]
    upper = np.r_[[.12, .12, .88], model.upperPositionLimit[7:] - .025]

    def solve(guess, root_y, root_x, side, lift, forward, left_hand_reach, right_hand_reach,
              left_offset=0., right_offset=0.):
        foot_targets = {s: feet0[s].copy() for s in ("left", "right")}
        foot_targets["left"][0] += left_offset
        foot_targets["right"][0] += right_offset
        if side:
            foot_targets[side] += (forward, 0., lift)
        target_root = q0[:3].copy()
        target_root[1] = root_y
        target_root[0] = root_x
        hand_targets = {"left": hands0["left"] + np.array([left_hand_reach + root_x, .01, .025 if left_hand_reach > .05 else 0.]),
                        "right": hands0["right"] + np.array([right_hand_reach + root_x, -.01, .025 if right_hand_reach > .05 else 0.])}
        def residual(x):
            q = q0.copy()
            q[:3], q[7:] = x[:3], x[3:]
            pin.framesForwardKinematics(model, data, q)
            feet = np.concatenate([42 * (data.oMf[ids[s]].translation - foot_targets[s])
                                   for s in ("left", "right")])
            hands = np.concatenate([16 * (data.oMf[ids[f"{s}_hand"]].translation - hand_targets[s])
                                    for s in ("left", "right")])
            return np.r_[feet, hands, 13 * (q[:3] - target_root),
                         5 * pin.log3(torso_R.T @ data.oMf[ids["torso"]].rotation),
                         .15 * (q[7:] - q0[7:])]
        x0 = np.r_[guess[:3], guess[7:]]
        result = least_squares(residual, x0, bounds=(lower, upper), max_nfev=150,
                               ftol=1e-7, xtol=1e-7, gtol=1e-7)
        q = q0.copy()
        q[:3], q[7:] = result.x[:3], result.x[3:]
        pin.framesForwardKinematics(model, data, q)
        error = max(np.linalg.norm(data.oMf[ids[s]].translation-foot_targets[s])
                    for s in ("left", "right"))
        if error > .008:
            raise RuntimeError(f"Ground foot IK error {error:.4f}: {side=} {lift=} {forward=}")
        hand_error = max(np.linalg.norm(data.oMf[ids[f"{s}_hand"]].translation-hand_targets[s])
                         for s in ("left", "right"))
        if hand_error > .025:
            raise RuntimeError(f"Hand IK error {hand_error:.4f}: {side=} {lift=}")
        return q

    clips = []
    rejected = []
    # Modest lifts and short steps. Every clip alternates left and right.
    if args.random_clips:
        rng = np.random.default_rng(args.seed)
        plans = [(float(rng.uniform(.012, .028)), float(rng.uniform(-.008, .012)),
                  float(rng.uniform(.85, 1.25)), float(rng.uniform(.035, .09)),
                  float(rng.uniform(.035, .09))) for _ in range(args.random_clips)]
    else:
        plans = [(lift, forward, duration, .08, .08) for lift in (.012, .02, .03)
                 for forward in (.0, .02, .04) for duration in (1.2, 1.6)]
    for plan_id, (lift, forward, duration, hand_left, hand_right) in enumerate(plans):
        name = (f"inplace_random_{plan_id:03d}" if args.random_clips else
                f"alternate_l{round(lift*1000):02d}_x{round(forward*100):02d}_t{round(duration*10):02d}")
        split = ("val" if plan_id % 5 == 0 else "train") if args.random_clips else (
            "val" if (lift, forward) in ((.02, .02), (.03, .04)) else "train")
        try:
            # Shift COM toward the stance foot before each lift.
            root_approach = .005 if args.random_clips else .015
            root_swing = .008 if args.random_clips else .025
            spec = [(0., 0., None, 0., 0., 0., 0.),
                    (-.11, root_approach, None, 0., 0., .025, .025),
                    (-.11, root_swing, "left", lift, forward, hand_left, .04),
                    (-.11, root_approach, None, 0., 0., .025, .025),
                    (0., 0., None, 0., 0., 0., 0.),
                    (.11, root_approach, None, 0., 0., .025, .025),
                    (.11, root_swing, "right", lift, forward, .04, hand_right),
                    (.11, root_approach, None, 0., 0., .025, .025),
                    (0., 0., None, 0., 0., 0., 0.)]
            if args.walk_step:
                s = args.walk_step
                spec = [(0., 0., None, 0., 0., 0., 0., 0., 0.),
                        (-.11, 0., None, 0., 0., .025, .025, 0., 0.),
                        (-.11, .4*s, "left", lift, s, hand_left, .04, 0., 0.),
                        (-.11, .5*s, None, 0., 0., .025, .025, s, 0.),
                        (0., .5*s, None, 0., 0., 0., 0., s, 0.),
                        (.11, .5*s, None, 0., 0., .025, .025, s, 0.),
                        (.11, .9*s, "right", lift, s, .04, hand_right, s, 0.),
                        (.11, s, None, 0., 0., .025, .025, s, s),
                        (0., s, None, 0., 0., 0., 0., s, s)]
            waypoints = [q0]
            for row in spec[1:]:
                waypoints.append(solve(waypoints[-1], *row))
            parts = []
            for a, b in zip(waypoints[:-1], waypoints[1:]):
                joint, jvel, jacc = quintic_segment(a[7:], b[7:], duration, args.fps)
                root, rvel, racc = quintic_segment(a[:3], b[:3], duration, args.fps)
                qs = np.tile(q0, (len(joint)-1, 1))
                qs[:, :3], qs[:, 7:] = root[:-1], joint[:-1]
                vs = np.zeros((len(qs), model.nv))
                aa = np.zeros_like(vs)
                vs[:, :3], vs[:, 6:] = rvel[:-1], jvel[:-1]
                aa[:, :3], aa[:, 6:] = racc[:-1], jacc[:-1]
                parts.append((qs, vs, aa))
                parts.append((np.tile(b, (12, 1)), np.zeros((12, model.nv)),
                              np.zeros((12, model.nv))))
            qs, vs, aa = (np.concatenate([p[i] for p in parts]) for i in range(3))
            wp, wr = [], []
            for q in qs:
                pin.framesForwardKinematics(model, data, q)
                wp.append([data.oMf[f].translation.copy() for f in body_ids])
                wr.append([data.oMf[f].rotation.copy() for f in body_ids])
            wp, wr = np.asarray(wp), np.asarray(wr)
            root_R = Rotation.from_quat(q0[3:7]).as_matrix()
            local_pos = (wp-qs[:, None, :3]) @ root_R
            local_rot = np.einsum("ij,tbjk->tbik", root_R.T, wr)
            body_quat = Rotation.from_matrix(local_rot.reshape(-1, 3, 3)).as_quat().reshape(len(qs), -1, 4)
            destination = args.output / "motions" / split / f"{name}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, fps=np.array(args.fps), joint_names=np.array(names),
                body_names=np.array(body_names), root_pos=qs[:, :3].astype("f4"),
                root_rot=qs[:, 3:7].astype("f4"), dof_pos=qs[:, 7:].astype("f4"),
                local_body_pos=local_pos.astype("f4"), local_body_rot=body_quat.astype("f4"),
                joint_vel=vs[:, 6:].astype("f4"), joint_acc=aa[:, 6:].astype("f4"))
            clips.append({"name": name, "split": split, "file": str(destination.relative_to(args.output)),
                          "frames": len(qs), "lift_m": lift, "forward_m": forward,
                          "segment_seconds": duration, "walk_step_m": args.walk_step,
                          "left_hand_reach_m": hand_left, "right_hand_reach_m": hand_right})
            print("ACCEPT", name, len(qs), flush=True)
        except Exception as exc:
            rejected.append({"name": name, "reason": str(exc)})
            print("REJECT", name, exc, flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"fps": args.fps, "pose": "G1 default standing pose at root z=.772m",
                "joint_names": names, "body_names": body_names,
                "sole_offset": [0, 0, sole_z], "clips": clips, "rejected": rejected,
                "note": "Ground-only alternating foot lift; no external support in training."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not clips:
        raise RuntimeError("No clips generated")


if __name__ == "__main__":
    main()

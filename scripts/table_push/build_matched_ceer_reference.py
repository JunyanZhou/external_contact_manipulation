"""Fit the task's external sole targets to CEER's joint-reference input."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/references"))
from generate_wall_support import add_frame, load_usd  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    motion = np.load(args.motion)
    target = np.load(args.target)["target_foot_pos_b"]
    if len(target) != len(motion["dof_pos"]):
        raise ValueError("Target and motion must have the same frame count")
    model, names, meshes, _ = load_usd(ROOT / "active_adaptation/assets/G1/g1_flat_fullcol.usd")
    if names != list(motion["joint_names"]):
        raise ValueError("Motion joint order differs from the robot model")
    sole_offset = min(v[:, 2].min() for n, v, _ in meshes if n == "left_ankle_roll_link")
    sole = add_frame(model, "matched_ceer_left_sole", "left_ankle_roll_link",
                     np.array([0., 0., sole_offset]))
    data = model.createData()
    left_ids = [names.index(n) for n in names if n.startswith("left_") and
                any(part in n for part in ("hip_", "knee_", "ankle_"))]
    if len(left_ids) != 6:
        raise ValueError("Expected six left-leg joints")
    qids = np.asarray(left_ids) + 7
    lower = model.lowerPositionLimit[qids] + .001
    upper = model.upperPositionLimit[qids] - .001
    solved = motion["dof_pos"].astype(np.float64).copy()
    errors = np.empty(len(target))
    for frame in range(len(target)):
        q = np.r_[motion["root_pos"][frame], motion["root_rot"][frame], solved[frame]]
        root_rotation = Rotation.from_quat(q[3:7])
        world_target = q[:3] + root_rotation.apply(target[frame])
        original = q[qids].copy()

        def residual(x):
            q[qids] = x
            pin.framesForwardKinematics(model, data, q)
            return np.r_[100. * (data.oMf[sole].translation - world_target), .05 * (x - original)]

        fit = least_squares(residual, np.clip(original, lower, upper), bounds=(lower, upper),
                            max_nfev=40, ftol=1e-7, xtol=1e-7, gtol=1e-7)
        q[qids] = fit.x
        solved[frame] = q[7:]
        pin.framesForwardKinematics(model, data, q)
        errors[frame] = np.linalg.norm(data.oMf[sole].translation - world_target)
    if errors.max() > .01:
        raise RuntimeError(f"IK misses the supplied sole target by {errors.max():.4f} m")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, joint_names=np.asarray(names), dof_pos=solved.astype("f4"),
                        root_pos=motion["root_pos"], root_rot=motion["root_rot"],
                        target_foot_pos_b=target.astype("f4"), fit_error_m=errors.astype("f4"))
    print(f"frames={len(target)} mean_ik_error_m={errors.mean():.6f} max_ik_error_m={errors.max():.6f}")


if __name__ == "__main__":
    main()

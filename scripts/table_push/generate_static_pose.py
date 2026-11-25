"""Generate initial whole-body table-support poses from the G1 USD.

Kinematic candidates only. Contact statics and Isaac closed-loop tests are separate gates.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares, linprog

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/references"))
from generate_wall_support import add_frame, load_usd  # noqa: E402


def static_contact_lp(model, q, ids, v=None, a=None):
    """Screen right-foot + two upward table contacts for static balance."""
    data = model.createData()
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    gravity = (pin.computeGeneralizedGravity(model, data, q) if v is None
               else pin.rnea(model, data, q, v, a))
    jf = pin.getFrameJacobian(model, data, ids["right_sole"], pin.LOCAL_WORLD_ALIGNED)
    jl = pin.getFrameJacobian(model, data, ids["left_hand_mimic"], pin.LOCAL_WORLD_ALIGNED)[:3]
    jr = pin.getFrameJacobian(model, data, ids["right_hand_mimic"], pin.LOCAL_WORLD_ALIGNED)[:3]
    jt = np.column_stack((jf.T, jl.T, jr.T))
    rows, bounds = [], []

    def row(entries, limit=0.):
        v = np.zeros(12)
        for i, value in entries.items():
            v[i] = value
        rows.append(v)
        bounds.append(limit)

    for start, mu in ((0, .7), (6, .6), (9, .6)):
        for sx in (-1., 1.):
            for sy in (-1., 1.):
                row({start: sx, start + 1: sy, start + 2: -mu})
    # Conservative ground contact moment limits around the ankle.
    for sign in (-1., 1.):
        row({3: sign, 2: -.04})
        row({5: sign, 2: -.025})
    row({4: -1., 2: -.11})
    row({4: 1., 2: -.045})
    effort = .85 * model.effortLimit[6:]
    a = np.vstack((rows, jt[6:], -jt[6:]))
    b = np.r_[bounds, gravity[6:] + effort, effort - gravity[6:]]
    objective = np.zeros(12)
    objective[[8, 11]] = 1.  # Prefer modest hand loading when feasible.
    answer = linprog(objective, A_ub=a, b_ub=b, A_eq=jt[:6], b_eq=gravity[:6],
                     bounds=[(None, None), (None, None), (20., None),
                             (None, None), (None, None), (None, None),
                             (None, None), (None, None), (10., None),
                             (None, None), (None, None), (10., None)], method="highs")
    if not answer.success:
        return {"feasible": False, "reason": answer.message}
    torque = gravity[6:] - jt[6:] @ answer.x
    return {"feasible": True, "forces_world": answer.x.tolist(),
            "max_torque_fraction": float(np.max(np.abs(torque) / model.effortLimit[6:])),
            "balance_residual": float(np.max(np.abs(jt[:6] @ answer.x - gravity[:6])))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/table_push/static_poses.npz")
    parser.add_argument("--table-top", type=float, default=0.84)
    parser.add_argument("--root-weight", type=float, default=2.0)
    parser.add_argument("--root-shift", type=float, default=0.06)
    parser.add_argument("--hand-clearance", type=float, default=0.04)
    parser.add_argument("--root-forward", type=float, default=0.0)
    parser.add_argument("--lifts", type=float, nargs="+", default=[0., 0.06, 0.12])
    parser.add_argument("--right-foot-y", type=float, default=-0.119)
    parser.add_argument("--left-foot-y", type=float, default=0.119)
    parser.add_argument("--hand-y", type=float, default=0.15)
    args = parser.parse_args()
    model, names, meshes, fk_error = load_usd(ROOT / "active_adaptation/assets/G1/g1_flat_fullcol.usd")
    q0 = pin.neutral(model)
    q0[:3] = [0., 0., 0.79]
    sole_offset = min(v[:, 2].min() for n, v, _ in meshes if n == "left_ankle_roll_link")
    ids = {name: model.getFrameId(name) for name in (
        "left_hand_mimic", "right_hand_mimic", "left_ankle_roll_link", "right_ankle_roll_link", "torso_link")}
    ids["right_sole"] = add_frame(model, "table_push_right_sole", "right_ankle_roll_link",
                                  np.array([0., 0., sole_offset]))
    # Pinocchio data must be allocated after all frames are added to the model.
    data = model.createData()
    hand_z = args.table_top + args.hand_clearance
    targets = {
        "left_hand_mimic": np.array([0.42, args.hand_y, hand_z]),
        "right_hand_mimic": np.array([0.42, -args.hand_y, hand_z]),
        "right_ankle_roll_link": np.array([0., args.right_foot_y, -sole_offset]),
    }
    lower = np.r_[[-0.15, -0.12, 0.68], model.lowerPositionLimit[7:] + 0.02]
    upper = np.r_[[0.16, 0.12, 0.88], model.upperPositionLimit[7:] - 0.02]
    x0 = np.r_[q0[:3], q0[7:]]
    x0 = np.clip(x0, lower + 1e-4, upper - 1e-4)
    poses = []
    results = []
    for lift in args.lifts:
        targets["left_ankle_roll_link"] = np.array([0.01, args.left_foot_y, lift - sole_offset])
        desired_root = q0[:3].copy()
        desired_root[1] = -min(lift / 0.12 * args.root_shift, args.root_shift)
        desired_root[0] = args.root_forward

        def residual(x):
            q = q0.copy()
            q[:3], q[7:] = x[:3], x[3:]
            pin.framesForwardKinematics(model, data, q)
            errors = [12. * (data.oMf[ids[name]].translation - target)
                      for name, target in targets.items()]
            torso_rotation = pin.log3(data.oMf[ids["torso_link"]].rotation)
            return np.r_[*errors, 2. * torso_rotation, args.root_weight * (x[:3] - desired_root),
                         0.12 * (x[3:] - x0[3:])]

        solved = least_squares(residual, x0, bounds=(lower, upper), max_nfev=350,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)
        q = q0.copy()
        q[:3], q[7:] = solved.x[:3], solved.x[3:]
        pin.framesForwardKinematics(model, data, q)
        errors = {name: float(np.linalg.norm(data.oMf[ids[name]].translation - target))
                  for name, target in targets.items()}
        result = dict(lift_m=lift, converged=bool(solved.success),
                      max_target_error_m=max(errors.values()), target_errors_m=errors,
                      root_xyz_m=q[:3].tolist(), iterations=int(solved.nfev))
        result["static_contact_lp"] = static_contact_lp(model, q, ids)
        results.append(result)
        poses.append(q)
        x0 = solved.x
        print(result, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, q_pinocchio=np.asarray(poses), joint_names=np.asarray(names),
             table_top_m=args.table_top, hand_clearance_m=args.hand_clearance,
             report_json=json.dumps(results), fk_error_m=fk_error)
    if any(not x["converged"] or x["max_target_error_m"] > 0.015
           or (x["lift_m"] > 0 and not x["static_contact_lp"]["feasible"]) for x in results):
        raise RuntimeError("One or more static candidates failed the IK/contact-statics gate")


if __name__ == "__main__":
    main()

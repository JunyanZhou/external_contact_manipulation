"""Audit a drawer reference with whole-body contact inverse dynamics.

The audit is a necessary feasibility test, not a substitute for physical replay.
All forces are forces exerted *on the robot* in world coordinates.  The right
hand's +X force is paired with a -X force on the lower drawer.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/references"))
from generate_wall_support import add_frame, load_usd


def solve_forces(model, data, frame_ids, q, v, a, lower_depth, lower_speed,
                 lower_accel, left_active, right_active):
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    h = pin.rnea(model, data, q, v, a)
    jac = [pin.getFrameJacobian(model, data, frame_ids[name], pin.LOCAL_WORLD_ALIGNED)
           for name in ("left_sole", "right_sole", "left_hand_mimic", "right_hand_mimic")]
    # The two sole wrenches have force and moment; the hands are point forces.
    J = np.column_stack((jac[0].T, jac[1].T, jac[2][:3].T, jac[3][:3].T))
    assert J.shape == (model.nv, 18)
    eq = J[:6]
    rhs = h[:6]
    rows = []
    bounds = [(None, None)] * 18
    # Conservative rectangular friction pyramid and centre-of-pressure limits.
    for foot in range(2):
        k = 6 * foot
        bounds[k + 2] = (0., 800.)
        for direction in (0, 1):
            for sign in (-1., 1.):
                r = np.zeros(18); r[k + direction] = sign; r[k + 2] = -1.8
                rows.append(r)
        for moment, arm in ((3, .025), (4, .045), (5, .015)):
            for sign in (-1., 1.):
                r = np.zeros(18); r[k + moment] = sign; r[k + 2] = -arm
                rows.append(r)
    # Upper drawer front supports the left palm through its -X normal.
    if left_active:
        bounds[12] = (-80., -2.)
        for tangential in (13, 14):
            for sign in (-1., 1.):
                r = np.zeros(18); r[tangential] = sign; r[12] = 2.
                rows.append(r)
    else:
        bounds[12:15] = [(0., 0.)] * 3
    if right_active:
        # Required drawer pull = inertial + damping + spring force.  The lower
        # joint is driven toward closed; positive depth means opening (-X).
        pull = max(0., 5. * lower_accel + 35. * lower_speed + 55. * lower_depth)
        # The drawer is a one-axis mass-spring-damper.  A larger horizontal
        # contact force would produce a different slide trajectory, so it is
        # not admissible here.  Only ±0.5 N numerical/model tolerance remains.
        bounds[15] = (max(0., pull - .5), pull + .5)
        for tangential in (16, 17):
            for sign in (-1., 1.):
                r = np.zeros(18); r[tangential] = sign; r[15] = -1.25
                rows.append(r)
    else:
        bounds[15:18] = [(0., 0.)] * 3
    effort = .85 * model.effortLimit[6:]
    A = np.vstack((rows, J[6:], -J[6:]))
    b = np.r_[np.zeros(len(rows)), h[6:] + effort, effort - h[6:]]
    # Select a modest force solution without claiming a contact optimum.
    objective = np.zeros(18)
    objective[2] = objective[8] = .01
    objective[12] = -.001
    objective[15] = .001
    res = linprog(objective, A_ub=A, b_ub=b, A_eq=eq, b_eq=rhs,
                  bounds=bounds, method="highs")
    out = {"feasible": bool(res.success), "reason": res.message,
           "required_pull_N": float(max(0., 5. * lower_accel + 35. * lower_speed + 55. * lower_depth))}
    if res.success:
        torque = h[6:] - J[6:] @ res.x
        out.update(left_hand_force_N=res.x[12:15].tolist(),
                   right_hand_force_N=res.x[15:18].tolist(),
                   left_foot_force_N=res.x[:3].tolist(),
                   right_foot_force_N=res.x[6:9].tolist(),
                   max_effort_ratio=float(np.max(np.abs(torque) / model.effortLimit[6:])),
                   joint_torque_Nm=torque.tolist(),
                   base_residual_N=float(np.max(np.abs(eq @ res.x - rhs))))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--pull-depth", type=float, help="Planned lower-drawer opening in metres")
    args = p.parse_args()
    if args.pull_depth is None:
        depth_by_name = {"preview_00": .10, "preview_01": .12,
                         "preview_02": .14, "preview_03": .16}
        args.pull_depth = depth_by_name.get(args.motion.stem)
        if args.pull_depth is None:
            p.error("--pull-depth is required for an unknown clip name")
    model, joint_names, meshes, _ = load_usd(ROOT / "active_adaptation/assets/G1/g1_flat_fullcol.usd")
    sole_z = min(v[:, 2].min() for n, v, _ in meshes if n == "left_ankle_roll_link")
    ids = {n: model.getFrameId(n) for n in ("left_hand_mimic", "right_hand_mimic")}
    for side in ("left", "right"):
        ids[f"{side}_sole"] = add_frame(model, f"drawer_{side}_sole",
                                         f"{side}_ankle_roll_link", np.array([0., 0., sole_z]))
    d = model.createData()
    with np.load(args.motion) as motion:
        q = np.column_stack((motion["root_pos"], motion["root_rot"], motion["dof_pos"]))
        fps = int(motion["fps"])
    dt = 1. / fps
    v = np.zeros((len(q), model.nv))
    for i in range(1, len(q)):
        v[i] = pin.difference(model, q[i - 1], q[i]) / dt
    a = np.zeros_like(v)
    a[1:] = np.diff(v, axis=0) / dt
    # Matches the reference generator: the final 100 frames are the pull phase.
    pull_start = len(q) - 125
    depth = np.zeros(len(q))
    for i in range(pull_start, min(len(q), pull_start + 100)):
        t = (i - pull_start) / 100.
        depth[i] = args.pull_depth * (10*t**3 - 15*t**4 + 6*t**5)
    depth[pull_start + 100:] = args.pull_depth
    speed = np.gradient(depth, dt)
    accel = np.gradient(speed, dt)
    rows = []
    for i in range(0, len(q), args.stride):
        pin.framesForwardKinematics(model, d, q[i])
        left_xyz = d.oMf[ids["left_hand_mimic"]].translation
        # Upper front face begins at x=0.3855 m.  The hand's collision
        # envelope first reaches it when its reference centre is near 0.34 m.
        left_active = bool(left_xyz[0] >= .34 and .65 < left_xyz[2] < .99)
        right_active = i >= pull_start
        result = solve_forces(model, d, ids, q[i], v[i], a[i], depth[i], speed[i],
                              accel[i], left_active, right_active)
        rows.append({"frame": i, "time_s": i * dt, "phase": "pull" if right_active else
                     ("left_support" if left_active else "free"), **result})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"motion": str(args.motion), "fps": fps,
        "joint_names": joint_names,
        "planned_pull_depth_m": args.pull_depth,
        "assumptions": "Forces on robot; both feet fixed, upper left-hand friction 2.0, lower handle friction 1.25; lower slide m=5 kg, spring=55 N/m, damping=35 Ns/m; 85% actuator effort limits. Necessary inverse-dynamics test only.",
        "frames_checked": len(rows), "feasible_frames": sum(r["feasible"] for r in rows),
        "rows": rows}, indent=2) + "\n")
    print(json.dumps({"frames_checked": len(rows),
        "feasible_frames": sum(r["feasible"] for r in rows),
        "first_infeasible": next((r for r in rows if not r["feasible"]), None)}, indent=2))


if __name__ == "__main__":
    main()

"""Contact-constrained drawer trajectory optimization with Pinocchio dynamics.

Optimizes pelvis placement and pull time. At each collocation knot, whole-body
IK holds both soles and the left palm while the right palm follows the lower
handle. A linear program then allocates contact wrenches and minimizes peak
normalized joint effort subject to floating-base inverse dynamics, friction,
sole pressure-center limits, and the drawer mass/spring/damper equation.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares, linprog, minimize
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/references"))
from generate_wall_support import load_usd, add_frame
from generate_foot_reach_dataset import body_names_from_loader, quintic_segment


def smooth(u):
    return u**3 * (10. + u * (-15. + 6. * u))


def allocate(model, data, ids, q, v, a, drawer_depth, drawer_speed, drawer_accel):
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    h = pin.rnea(model, data, q, v, a).copy()
    jac = [pin.getFrameJacobian(model, data, ids[n], pin.LOCAL_WORLD_ALIGNED)
           for n in ("left_sole", "right_sole", "left_hand_mimic", "right_hand_mimic")]
    J = np.column_stack((jac[0].T, jac[1].T, jac[2][:3].T, jac[3][:3].T))
    n = 19  # 18 wrench components and one peak effort fraction
    rows = []; rhs = []

    def row(entries, limit=0.):
        r = np.zeros(n)
        for i, value in entries:
            r[i] = value
        rows.append(r); rhs.append(limit)

    for foot in range(2):
        k = 6 * foot
        for tangent in (0, 1):
            for sign in (-1., 1.):
                row(((k + tangent, sign), (k + 2, -1.8)))
        for moment, arm in ((3, .025), (4, .045), (5, .015)):
            for sign in (-1., 1.):
                row(((k + moment, sign), (k + 2, -arm)))
    for tangent in (13, 14):
        for sign in (-1., 1.):
            row(((tangent, sign), (12, 2.)))
    for tangent in (16, 17):
        for sign in (-1., 1.):
            row(((tangent, sign), (15, -1.25)))
    limit = model.effortLimit[6:]
    for j in range(len(limit)):
        for sign in (-1., 1.):
            row([*( (i, -sign * J[6 + j, i]) for i in range(18)),
                 (18, -limit[j])], -sign * h[6 + j])
    pull = max(0., 5. * drawer_accel + 35. * drawer_speed + 55. * drawer_depth)
    bounds = [(None, None)] * 19
    bounds[2] = bounds[8] = (0., 800.)
    bounds[12] = (-80., -2.)
    bounds[15] = (max(0., pull - .5), pull + .5)
    bounds[18] = (0., 1.5)
    eq = np.column_stack((J[:6], np.zeros(6)))
    objective = np.zeros(n); objective[18] = 1.
    result = linprog(objective, A_ub=np.asarray(rows), b_ub=np.asarray(rhs),
        A_eq=eq, b_eq=h[:6], bounds=bounds, method="highs")
    if not result.success:
        return {"feasible": False, "reason": result.message, "required_pull_N": pull}
    force = result.x[:18]
    tau = h[6:] - J[6:] @ force
    return {"feasible": True, "required_pull_N": pull,
        "max_effort_ratio": float(np.max(np.abs(tau) / limit)),
        "left_hand_force_N": force[12:15].tolist(),
        "right_hand_force_N": force[15:18].tolist(),
        "left_foot_wrench": force[:6].tolist(),
        "right_foot_wrench": force[6:12].tolist(),
        "joint_torque_Nm": tau.tolist(),
        "floating_base_residual": float(np.max(np.abs(J[:6] @ force - h[:6])))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--depth", type=float, default=.10)
    p.add_argument("--handle-hand-x", type=float, default=.29,
                   help="Right-hand centre X at the closed drawer handle (m)")
    p.add_argument("--fixed-parameters", type=float, nargs=3,
                   metavar=("ROOT_X", "ROOT_Z", "DURATION"),
                   help="Evaluate one prescribed outer solution for geometry diagnostics")
    p.add_argument("--foot-forward-shift", type=float, default=0.,
                   help="Move both planted sole targets toward the cabinet (m)")
    p.add_argument("--knots", type=int, default=11)
    p.add_argument("--max-evals", type=int, default=30)
    args = p.parse_args()
    model, names, meshes, _ = load_usd(ROOT / "active_adaptation/assets/G1/g1_flat_fullcol.usd")
    with np.load(args.start) as f:
        q_seed = np.r_[f["root_pos"][0], f["root_rot"][0], f["dof_pos"][0]].astype(float)
    sole_z = min(v[:, 2].min() for n, v, _ in meshes if n == "left_ankle_roll_link")
    ids = {n: model.getFrameId(n) for n in ("left_hand_mimic", "right_hand_mimic", "torso_link")}
    for side in ("left", "right"):
        ids[side + "_sole"] = add_frame(model, side + "_dyn_sole",
            side + "_ankle_roll_link", np.array([0., 0., sole_z]))
    data = model.createData()
    pin.framesForwardKinematics(model, data, q_seed)
    feet = {s: data.oMf[ids[s + "_sole"]].translation.copy() for s in ("left", "right")}
    for target in feet.values():
        target[0] += args.foot_forward_shift
    left = data.oMf[ids["left_hand_mimic"]].translation.copy()
    torso_R = data.oMf[ids["torso_link"]].rotation.copy()
    lower = np.r_[[-.24, -.12, .58], model.lowerPositionLimit[7:] + .012]
    upper = np.r_[[.14, .12, .84], model.upperPositionLimit[7:] - .012]
    candidates = []
    us = np.linspace(0., 1., args.knots)

    def ik(previous, right, root_x, root_z):
        goals = (("left_sole", feet["left"]), ("right_sole", feet["right"]),
                 ("left_hand_mimic", left), ("right_hand_mimic", right))
        root_goal = np.array([root_x, q_seed[1], root_z])

        def residual(x):
            q = q_seed.copy(); q[:3] = x[:3]; q[7:] = x[3:]
            pin.framesForwardKinematics(model, data, q)
            return np.concatenate([
                *[w * (data.oMf[ids[n]].translation - target)
                  for w, (n, target) in zip((100., 100., 100., 100.), goals)],
                12. * pin.log3(data.oMf[ids["left_hand_mimic"]].rotation),
                10. * pin.log3(torso_R.T @ data.oMf[ids["torso_link"]].rotation),
                18. * (x[:3] - root_goal),
                .05 * (x[3:] - previous[7:])])

        x0 = np.clip(np.r_[previous[:3], previous[7:]], lower + 1e-5, upper - 1e-5)
        result = least_squares(residual, x0, bounds=(lower, upper),
            max_nfev=150, ftol=1e-6, xtol=1e-6, gtol=1e-6)
        q = q_seed.copy(); q[:3] = result.x[:3]; q[7:] = result.x[3:]
        pin.framesForwardKinematics(model, data, q)
        error = max(np.linalg.norm(data.oMf[ids[n]].translation - target)
                    for n, target in goals)
        return q, float(error)

    def evaluate(parameters, retain=False):
        root_x, root_z, duration = parameters
        if not (-.14 <= root_x <= .08 and .68 <= root_z <= .78 and 1.5 <= duration <= 6.):
            return 5. + np.linalg.norm(parameters)
        qs = []; errors = []
        previous = q_seed
        for u in us:
            right = np.array([args.handle_hand_x - args.depth * smooth(u), -.055, .45])
            q, error = ik(previous, right, root_x, root_z)
            qs.append(q); errors.append(error); previous = q
        qs = np.asarray(qs)
        if max(errors) > .006:
            score = 2. + 20. * max(errors)
            if retain: return score, qs, errors, []
            return score
        dt = duration / (len(us) - 1)
        v = np.zeros((len(qs), model.nv)); a = np.zeros_like(v)
        for i in range(1, len(qs)):
            v[i] = pin.difference(model, qs[i - 1], qs[i]) / dt
        v[0] = v[1]; v[-1] = np.zeros(model.nv)
        a[1:] = np.diff(v, axis=0) / dt
        a[0] = a[1]
        # Quintic drawer motion derivatives, consistent with the IK targets.
        depth = args.depth * smooth(us)
        speed = np.gradient(depth, dt)
        accel = np.gradient(speed, dt)
        rows = []
        for i, q in enumerate(qs):
            row = allocate(model, data, ids, q, v[i], a[i], depth[i], speed[i], accel[i])
            rows.append(row)
        infeasible = sum(not r["feasible"] for r in rows)
        if infeasible:
            score = 2. + .2 * infeasible + 10. * max(errors)
        else:
            ratios = np.array([r["max_effort_ratio"] for r in rows])
            normals = np.array([-r["left_hand_force_N"][0] for r in rows])
            score = (float(ratios.max()) + .08 * float(ratios.mean())
                     + .0003 * float(normals.max())
                     + 10. * max(0., .65 - float(qs[:, 2].min()))
                     + .01 * max(0., float(normals.max()) - 70.))
        candidates.append({"root_x_goal_m": float(root_x), "root_z_goal_m": float(root_z),
            "pull_duration_s": float(duration), "score": float(score),
            "feasible_knots": len(rows) - infeasible, "knots": len(rows),
            "max_ik_error_m": max(errors),
            "peak_effort_ratio": max((r.get("max_effort_ratio", 0.) for r in rows), default=0.)})
        candidates[-1]["min_pelvis_height_m"] = float(qs[:, 2].min())
        print("CANDIDATE", candidates[-1], flush=True)
        if retain: return score, qs, errors, rows
        return score

    # Bounded derivative-free search; every trial includes full RNEA and wrench LP.
    if args.fixed_parameters is None:
        result = minimize(evaluate, x0=np.array([-.07, .72, 3.0]),
            method="Powell", bounds=[(-.14, .08), (.68, .78), (1.5, 6.)],
            options={"maxfev": args.max_evals, "xtol": .015, "ftol": .01})
        optimizer_success = bool(result.success)
        optimizer_message = result.message
    else:
        evaluate(args.fixed_parameters)
        optimizer_success = True
        optimizer_message = "Fixed parameters evaluated"
    best = min(candidates, key=lambda c: c["score"])
    params = [best["root_x_goal_m"], best["root_z_goal_m"], best["pull_duration_s"]]
    score, knots, errors, rows = evaluate(params, retain=True)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"solver": "Powell outer trajectory search; Pinocchio IK/RNEA; HiGHS contact-wrench LP",
        "drawer_model": {"mass_kg": 5., "spring_N_per_m": 55., "damping_Ns_per_m": 35.},
        "foot_forward_shift_m": args.foot_forward_shift,
        "handle_hand_x_m": args.handle_hand_x,
        "contact_assumptions": "Both feet fixed 6D, left palm fixed 3D, right palm tracks moving lower handle 3D; point hand forces; foot friction 1.8; upper palm friction 2.0; pull force follows drawer mass-spring-damper. Necessary dynamics test only.",
        "optimized_parameters": best, "optimizer_success": optimizer_success,
        "optimizer_message": optimizer_message, "candidates": candidates,
        "knots": [{"u": float(u), "root_xyz_m": q[:3].tolist(),
                   "ik_error_m": error, **row}
                  for u, q, error, row in zip(us, knots, errors, rows)]}
    (args.output / "optimization_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if best["feasible_knots"] != args.knots:
        print("No fully force-feasible trajectory", flush=True)
        return
    # Smooth reference for downstream physical replay. The torque knots remain
    # in the report; no drawer slider state is prescribed in physical replay.
    fps = 50
    pieces = []
    for qa, qb in zip(knots[:-1], knots[1:]):
        seconds = best["pull_duration_s"] / (len(knots) - 1)
        joints, _, _ = quintic_segment(qa[7:], qb[7:], seconds, fps)
        roots, _, _ = quintic_segment(qa[:3], qb[:3], seconds, fps)
        q = np.tile(qa, (len(joints) - 1, 1))
        q[:, :3] = roots[:-1]; q[:, 7:] = joints[:-1]
        pieces.append(q)
    pieces.append(np.tile(knots[-1], (25, 1)))
    qs = np.concatenate(pieces)
    body_names = body_names_from_loader()
    body_ids = [model.getFrameId(n) for n in body_names]
    root_R = Rotation.from_quat(qs[0, 3:7]).as_matrix()
    world_pos = []; world_rot = []
    for q in qs:
        pin.framesForwardKinematics(model, data, q)
        world_pos.append([data.oMf[f].translation.copy() for f in body_ids])
        world_rot.append([data.oMf[f].rotation.copy() for f in body_ids])
    world_pos = np.asarray(world_pos); world_rot = np.asarray(world_rot)
    local_pos = (world_pos - qs[:, None, :3]) @ root_R
    local_rot = np.einsum("ij,tbjk->tbik", root_R.T, world_rot)
    local_quat = Rotation.from_matrix(local_rot.reshape(-1, 3, 3)).as_quat().reshape(len(qs), -1, 4)
    v = np.zeros((len(qs), model.nv)); a = np.zeros_like(v)
    for i in range(1, len(qs)):
        v[i] = pin.difference(model, qs[i - 1], qs[i]) * fps
    a[1:] = np.diff(v, axis=0) * fps
    motion = args.output / "motions/train/optimized_pull.npz"
    motion.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(motion, fps=np.array(fps), joint_names=np.array(names),
        body_names=np.array(body_names), root_pos=qs[:, :3].astype("f4"),
        root_rot=qs[:, 3:7].astype("f4"), dof_pos=qs[:, 7:].astype("f4"),
        local_body_pos=local_pos.astype("f4"), local_body_rot=local_quat.astype("f4"),
        joint_vel=v[:, 6:].astype("f4"), joint_acc=a[:, 6:].astype("f4"))
    manifest = {"fps": fps, "joint_names": names, "body_names": body_names,
        "clips": [{"name": "optimized_pull", "split": "train",
                   "file": "motions/train/optimized_pull.npz", "frames": len(qs),
                   "pull_depth_m": args.depth, "right_hand_initially_on_handle": True}]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("BEST", best, "FRAMES", len(qs), flush=True)


if __name__ == "__main__":
    main()

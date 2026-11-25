"""Generate and audit a flat-foot, two-hand, alternating-step box-push reference."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import crocoddyl as croc
import pinocchio as pin
from scipy.optimize import least_squares, linprog
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/references"))
from generate_wall_support import add_frame, load_usd, Retraction, Posture
from generate_foot_reach_dataset import body_names_from_loader, quintic_segment


def allocate(model, data, ids, q, v, a, active, minimum_push,
             foot_mu=1.6, hand_mu=.7, hand_patch_radius=0., foot_patch_radius=0.,
             maximum_push=None):
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    tau = pin.rnea(model, data, q, v, a)
    names = [n for n in ("left_sole", "right_sole", "left_hand_mimic", "right_hand_mimic") if n in active]
    jacobians = [pin.getFrameJacobian(model, data, ids[n], pin.LOCAL_WORLD_ALIGNED) for n in names]
    widths = [5 if ("sole" in name and foot_patch_radius) or
              ("hand" in name and hand_patch_radius) else 3 for name in names]
    nvar = sum(widths)
    J = np.column_stack([jac[:3].T if width == 3 else
                         np.column_stack((jac[:3].T, jac[3:5].T if "sole" in name else jac[4:6].T))
                         for name, jac, width in zip(names, jacobians, widths)])
    rows = []
    bounds = []
    offsets = np.cumsum([0]+widths[:-1])
    for name, off, width in zip(names, offsets, widths):
        if "sole" in name:
            bounds += [(None, None), (None, None), (20, None)]
            for axis in (0, 1):
                for sign in (-1., 1.):
                    row = np.zeros(nvar); row[off + axis] = sign; row[off+2] = -foot_mu; rows.append(row)
            if width == 5:
                bounds += [(None, None), (None, None)]
                for axis in (3, 4):
                    for sign in (-1., 1.):
                        row = np.zeros(nvar); row[off+axis] = sign
                        row[off+2] = -foot_patch_radius; rows.append(row)
        else:
            bounds += [(-250, -minimum_push / 2), (None, None), (None, None)]
            for axis in (1, 2):
                for sign in (-1., 1.):
                    row = np.zeros(nvar); row[off+axis] = sign; row[off] = hand_mu; rows.append(row)
            if width == 5:
                bounds += [(-250, 250), (-250, 250)]
                for axis in (3, 4):
                    for sign in (-1., 1.):
                        row = np.zeros(nvar); row[off+axis] = sign
                        row[off] = hand_patch_radius; rows.append(row)
    effort = .85 * model.effortLimit[6:]
    # Among dynamically feasible contact loads, use the smallest hand push
    # meeting the box's sliding threshold. An arbitrary LP vertex can produce
    # much larger forces than the prescribed slow box motion could absorb.
    objective = np.zeros(nvar)
    for name, off in zip(names, offsets):
        if "hand" in name:
            objective[off] = -1.
    if maximum_push is not None and any("hand" in name for name in names):
        cap = np.zeros(nvar)
        for name, off in zip(names, offsets):
            if "hand" in name:
                cap[off] = -1.
        rows.append(cap)
    result = linprog(objective, A_ub=np.vstack((rows, J[6:], -J[6:])),
                     b_ub=np.r_[np.zeros(len(rows) - (maximum_push is not None and any("hand" in name for name in names))),
                                *([maximum_push] if maximum_push is not None and any("hand" in name for name in names) else []),
                                tau[6:]+effort, effort-tau[6:]],
                     A_eq=J[:6], b_eq=tau[:6], bounds=bounds, method="highs")
    if not result.success:
        return {"feasible": False, "reason": result.message}
    forces = [result.x[off:off+3] for off in offsets]
    moments = {name: result.x[off+3:off+5].tolist()
               for name, off, width in zip(names, offsets, widths) if width == 5}
    residual_tau = tau[6:] - J[6:] @ result.x
    return {"feasible": True, "contact_forces_N": dict(zip(names, [f.tolist() for f in forces])),
            "contact_patch_moments_Nm": moments,
            "max_effort_ratio": float(np.max(np.abs(residual_tau) / model.effortLimit[6:])),
            "joint_torque_Nm": residual_tau.tolist(),
            "push_N": float(-sum(forces[j][0] for j, n in enumerate(names) if "hand" in n))}


def refine_waypoints_crocoddyl(model, ids, nodes):
    """Solve each contact pose with Crocoddyl DDP, then audit full dynamics later.

    Retraction controls are configuration displacements, not joint torques.
    Pinocchio RNEA and contact-force LP below screen the interpolated motion.
    """
    state = croc.StateMultibody(model)
    data = model.createData()
    refined = [nodes[0].copy()]
    reports = [{"node": 0, "converged": True, "iterations": 0}]
    for index, target_q in enumerate(nodes[1:], 1):
        pin.framesForwardKinematics(model, data, target_q)
        costs = croc.CostModelSum(state, 0)
        for name in ("left_sole", "right_sole", "left_hand_mimic", "right_hand_mimic"):
            target_pos = data.oMf[ids[name]].translation.copy()
            residual = croc.ResidualModelFrameTranslation(state, ids[name], target_pos, 0)
            costs.addCost(name+"_position", croc.CostModelResidual(state, residual),
                          1e6 if "sole" in name else 1e5)
            if "sole" in name and target_pos[2] < .008:
                residual = croc.ResidualModelFrameRotation(
                    state, ids[name], data.oMf[ids[name]].rotation.copy(), 0)
                costs.addCost(name+"_rotation", croc.CostModelResidual(state, residual), 1e4)
        costs.addCost("target_posture", croc.CostModelResidual(
            state, croc.ResidualModelState(state, np.r_[target_q, np.zeros(model.nv)], 0)), 10.)
        x0 = np.r_[refined[-1], np.zeros(model.nv)]
        solver = croc.SolverDDP(croc.ShootingProblem(
            x0, [Retraction(state)], Posture(state, costs)))
        solver.th_stop = 1e-7
        converged = bool(solver.solve([x0.copy(), x0.copy()], [np.zeros(model.nv)], 250))
        q = np.asarray(solver.xs[-1][:model.nq]).copy()
        refined.append(q)
        reports.append({"node": index, "converged": converged,
                        "iterations": solver.iter, "cost": float(solver.cost),
                        "configuration_distance": float(np.linalg.norm(pin.difference(model, target_q, q)))})
    return refined, reports


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--push-m", type=float, default=.12)
    p.add_argument("--mass-kg", type=float, default=20.)
    p.add_argument("--box-friction", type=float, default=.5)
    p.add_argument("--foot-mu", type=float, default=1.6)
    p.add_argument("--hand-mu", type=float, default=.7)
    p.add_argument("--hand-patch-radius", type=float, default=0.)
    p.add_argument("--foot-patch-radius", type=float, default=0.)
    p.add_argument("--crocoddyl-waypoints", action="store_true")
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--transition-seconds", type=float, default=.55)
    p.add_argument("--swing-lift-m", type=float, default=.025)
    p.add_argument("--left-step-m", type=float, default=.055)
    p.add_argument("--final-extra-m", type=float, default=None)
    p.add_argument("--start-upright", action="store_true")
    p.add_argument("--no-step", action="store_true", help="First curriculum stage: both soles remain planted")
    p.add_argument("--step-order", choices=("left-first", "right-first"), default="left-first")
    p.add_argument("--right-swing-root-y", type=float, default=.08)
    p.add_argument("--hold-right-swing-root-y", action="store_true")
    p.add_argument("--left-swing-root-y", type=float, default=-.065)
    p.add_argument("--left-transfer-root-y", type=float, default=None)
    p.add_argument("--right-swing-pitch-dx", type=float, default=.020)
    p.add_argument("--right-swing-root-dx", type=float, default=.025)
    p.add_argument("--right-step-lateral-m", type=float, default=0.,
                   help="Move the right foot toward the center during its first swing")
    p.add_argument("--root-y-weight-swing", type=float, default=300.)
    p.add_argument("--hand-y-shift-factor", type=float, default=.5)
    p.add_argument("--base-root-pitch-deg", type=float)
    p.add_argument("--torso-world-pitch-deg", type=float)
    p.add_argument("--torso-weight", type=float, default=4.)
    p.add_argument("--swing-hand-weight", type=float, default=110.)
    p.add_argument("--swing-torso-weight", type=float)
    p.add_argument("--hand-initial-forward-m", type=float, default=0.)
    p.add_argument("--hand-initial-down-m", type=float, default=0.)
    p.add_argument("--feet-initial-back-m", type=float, default=0.)
    p.add_argument("--seed-pose", type=Path, default=ROOT / "artifacts/heavy_box_push/static_staggered_pose/pose.npz")
    args = p.parse_args()
    model, joint_names, meshes, _ = load_usd(ROOT / "active_adaptation/assets/G1/g1_flat_fullcol.usd")
    q0 = np.load(args.seed_pose)["q_pinocchio"].copy()
    rightload_path = ROOT / "artifacts/heavy_box_push/static_staggered_pose_crouch/pose.npz"
    rightload_q = np.load(rightload_path)["q_pinocchio"].copy() if rightload_path.exists() else None
    sole_z = min(v[:, 2].min() for n, v, _ in meshes if n == "left_ankle_roll_link")
    ids = {n: model.getFrameId(n) for n in ("left_hand_mimic", "right_hand_mimic", "torso_link")}
    for side in ("left", "right"):
        ids[f"{side}_sole"] = add_frame(model, f"dynamic_{side}_sole", f"{side}_ankle_roll_link", np.array([0., 0., sole_z]))
    data = model.createData()
    pin.framesForwardKinematics(model, data, q0)
    feet0 = {n: data.oMf[ids[n]].translation.copy() for n in ("left_sole", "right_sole")}
    hands0 = {n: data.oMf[ids[n]].translation.copy() for n in ("left_hand_mimic", "right_hand_mimic")}
    foot_R0 = {n: data.oMf[ids[n]].rotation.copy() for n in feet0}
    original_torso_pitch = Rotation.from_matrix(data.oMf[ids["torso_link"]].rotation).as_euler("xyz")[1]
    original_root_pitch = Rotation.from_quat(q0[3:7]).as_euler("xyz")[1]
    torso_R0 = (Rotation.from_euler("y", args.torso_world_pitch_deg, degrees=True).as_matrix()
                if args.torso_world_pitch_deg is not None else data.oMf[ids["torso_link"]].rotation.copy())
    pitch0 = (np.deg2rad(args.base_root_pitch_deg) if args.base_root_pitch_deg is not None else
              original_root_pitch)
    lower = np.r_[[-.20, -.10, .60, -.02], model.lowerPositionLimit[7:]+.02]
    upper = np.r_[[.50, .10, .93, .85], model.upperPositionLimit[7:]-.02]
    knee_q = {s: 7 + joint_names.index(f"{s}_knee_joint") for s in ("left", "right")}
    hip_q = {s: 7 + joint_names.index(f"{s}_hip_pitch_joint") for s in ("left", "right")}

    # Start in the validated lean. Each thrust spans a planted phase; the
    # opposite foot swings only after the box has advanced.
    dx = args.push_m / 3
    final_extra = dx if args.final_extra_m is None else args.final_extra_m
    stages = [
        (0., 0., 0., 0., 0., 0., 0.),
        (dx, .035, .025, 0., 0., 0., 0.),
        (dx, .040, .025, 0., args.swing_lift_m, 0., 0.),
        (dx, .045, .025, args.left_step_m, 0., 0., 0.),
        (2*dx, .085, .050, args.left_step_m, 0., 0., 0.),
        (2*dx, .090, .050, args.left_step_m, 0., 0., args.swing_lift_m),
        (2*dx, .090, .050, args.left_step_m, 0., 0., args.swing_lift_m),
        (2*dx, .070, .050, args.left_step_m, 0., 0., 0.),
        (2*dx + final_extra, .080 if args.final_extra_m is not None else .130,
         .055 if args.final_extra_m is not None else .075, args.left_step_m, 0., 0., 0.),
    ]
    if args.step_order == "right-first":
        transfer_y = args.left_swing_root_y if args.left_transfer_root_y is None else args.left_transfer_root_y
        # Start from a left-loaded stance so the right foot can swing first.
        # Each push is separated from a swing; both hands stay on the box.
        stages = [
            (0., 0., 0., 0., 0., 0., 0.),
            (dx, .025, .020, 0., 0., 0., 0.),
            (dx, args.right_swing_root_dx, args.right_swing_pitch_dx,
             0., 0., -.07, args.swing_lift_m),
            (dx, .030, .020, 0., 0., args.left_step_m, 0., args.right_swing_root_y)
            if args.hold_right_swing_root_y else
            (dx, .030, .020, 0., 0., args.left_step_m, 0.),
            (2*dx, .060, .045, 0., 0., args.left_step_m, 0., transfer_y),
            (2*dx, .060, .045, 0., args.swing_lift_m, args.left_step_m, 0., args.left_swing_root_y),
            (2*dx, .100, .045, args.left_step_m, 0., args.left_step_m, 0., args.left_swing_root_y),
            (2*dx+final_extra, .100, .065, args.left_step_m, 0., args.left_step_m, 0., args.left_swing_root_y),
        ]
    if args.no_step:
        stages = [(0., 0., 0., 0., 0., 0., 0.),
                  (args.push_m*.5, args.push_m*.3, .025, 0., 0., 0., 0.),
                  (args.push_m, args.push_m*.6, .050, 0., 0., 0., 0.)]
    if args.start_upright:
        stages = [(-.16, -.13, -.58, 0., 0., 0., 0.),
                  (-.025, -.09, -.48, 0., 0., 0., 0.)] + stages
    def solve(seed, stage, flatten_swing=False):
        hand_dx, root_dx, pitch_dx, left_dx, left_z, right_dx, right_z = stage[:7]
        swing = {"left_sole": left_z > .01, "right_sole": right_z > .01}
        root_y_goal = (stage[7] if len(stage) > 7 else
                       args.right_swing_root_y if swing["right_sole"] else
                       args.left_swing_root_y if swing["left_sole"] else q0[1])
        hand_y_shift = args.hand_y_shift_factor * (root_y_goal - q0[1])
        right_y_shift = (args.right_step_lateral_m
                         if args.step_order == "right-first" and (right_z > .01 or right_dx > 0.)
                         else 0.)
        targets = {"left_sole": feet0["left_sole"] + [left_dx-args.feet_initial_back_m, 0., left_z-feet0["left_sole"][2]],
                   "right_sole": feet0["right_sole"] + [right_dx-args.feet_initial_back_m, right_y_shift, right_z-feet0["right_sole"][2]],
                   "left_hand_mimic": hands0["left_hand_mimic"] + [args.hand_initial_forward_m + hand_dx, hand_y_shift, -args.hand_initial_down_m],
                   "right_hand_mimic": hands0["right_hand_mimic"] + [args.hand_initial_forward_m + hand_dx, hand_y_shift, -args.hand_initial_down_m]}
        def residual(x):
            q = q0.copy(); q[:3] = x[:3]
            q[3:7] = Rotation.from_euler("y", x[3]).as_quat(); q[7:] = x[4:]
            pin.framesForwardKinematics(model, data, q)
            return np.r_[*[(np.array([50000., 50000., 50000.]) if swing[n] else
                            np.array([10000., 10000., 1000000.]))
                           *(data.oMf[ids[n]].translation-targets[n]) for n in feet0],
                         *[(600 if not swing[n] else 45 if flatten_swing else 2)
                           *pin.log3(foot_R0[n].T@data.oMf[ids[n]].rotation) for n in feet0],
                         *[(args.swing_hand_weight if any(swing.values()) else 240)*(data.oMf[ids[n]].translation-targets[n]) for n in hands0],
                         (args.swing_torso_weight if any(swing.values()) and args.swing_torso_weight is not None
                          else args.torso_weight)*pin.log3(torso_R0.T@data.oMf[ids["torso_link"]].rotation),
                         (4 if any(swing.values()) else 10)*(x[0]-q0[0]-root_dx),
                         (args.root_y_weight_swing if any(swing.values()) else 600 if len(stage) > 7 else 20)
                         *(x[1]-root_y_goal), 20*(x[3]-pitch0-pitch_dx),
                         .04*(x[4:]-seed[7:])]
        x0 = np.r_[seed[:3], Rotation.from_quat(seed[3:7]).as_euler("xyz")[1], seed[7:]]
        starts = [x0]
        if args.base_root_pitch_deg is not None and args.torso_world_pitch_deg is not None:
            target_torso_pitch = np.deg2rad(args.torso_world_pitch_deg)
            for root_pitch in (pitch0, original_root_pitch):
                for root_shift in (-.05, 0., .05):
                    trial = x0.copy()
                    trial[0] += root_shift
                    trial[3] = root_pitch
                    trial[4 + joint_names.index("waist_pitch_joint")] += target_torso_pitch - original_torso_pitch - (root_pitch - original_root_pitch)
                    for side in ("left", "right"):
                        trial[4 + joint_names.index(f"{side}_hip_pitch_joint")] -= root_pitch - original_root_pitch
                        trial[4 + joint_names.index(f"{side}_shoulder_pitch_joint")] -= target_torso_pitch - original_torso_pitch
                    starts.append(trial)
        if rightload_q is not None and root_y_goal < -.04:
            starts.append(np.r_[rightload_q[:3],
                                Rotation.from_quat(rightload_q[3:7]).as_euler("xyz")[1],
                                rightload_q[7:]])
        swing_side = "left" if left_z > .01 else "right" if right_z > .01 else None
        if swing_side:
            for knee_bias, hip_bias in ((.3, .2), (.6, .3), (.6, -.2)):
                trial = x0.copy()
                trial[4 + knee_q[swing_side] - 7] += knee_bias
                trial[4 + hip_q[swing_side] - 7] += hip_bias
                starts.append(trial)
        else:
            starts.append(np.r_[q0[:3], pitch0, q0[7:]])
        sols = [least_squares(residual, np.clip(start, lower+1e-5, upper-1e-5), bounds=(lower, upper),
                              max_nfev=320, ftol=1e-7, xtol=1e-7, gtol=1e-7) for start in starts]
        sol = min(sols, key=lambda s: np.linalg.norm(residual(s.x)))
        q = q0.copy(); q[:3] = sol.x[:3]; q[3:7] = Rotation.from_euler("y", sol.x[3]).as_quat(); q[7:] = sol.x[4:]
        pin.framesForwardKinematics(model, data, q)
        errors = {n: float(np.linalg.norm(data.oMf[ids[n]].translation-targets[n])) for n in targets}
        flat_error = {n: float(np.linalg.norm(pin.log3(foot_R0[n].T@data.oMf[ids[n]].rotation))) for n in feet0}
        return q, {"contact_error_m": errors, "sole_rotation_error_rad": flat_error, "ik_success": bool(sol.success)}

    start, start_report = solve(q0, stages[0])
    nodes = [start]
    node_reports = [start_report]
    for index, stage in enumerate(stages[1:], 1):
        q, report = solve(nodes[-1], stage, flatten_swing=(index == 6 and not args.start_upright))
        nodes.append(q); node_reports.append(report)
        print("node", len(nodes)-1, report, flush=True)
    crocoddyl_reports = None
    if args.crocoddyl_waypoints:
        nodes, crocoddyl_reports = refine_waypoints_crocoddyl(model, ids, nodes)
        print("Crocoddyl waypoint DDP", crocoddyl_reports, flush=True)
    fps = 50
    chunks = []
    for i, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
        duration = (args.seconds if args.no_step else
                    (1.0 if i == 0 else 1.4 if i == 1 else args.seconds if i in (2, 5, 9) else .4 if i == 7 else args.transition_seconds)
                    if args.start_upright else (args.seconds if i in (0, 3, 7) else .4 if i == 5 else args.transition_seconds))
        joints, _, _ = quintic_segment(a[7:], b[7:], duration, fps)
        roots, _, _ = quintic_segment(a[:3], b[:3], duration, fps)
        pa = Rotation.from_quat(a[3:7]).as_euler("xyz")[1]
        pb = Rotation.from_quat(b[3:7]).as_euler("xyz")[1]
        pitches, _, _ = quintic_segment(np.array([pa]), np.array([pb]), duration, fps)
        qs = np.tile(a, (len(joints)-1, 1))
        qs[:, :3] = roots[:-1]; qs[:, 7:] = joints[:-1]
        qs[:, 3:7] = np.stack([Rotation.from_euler("y", float(p)).as_quat() for p in pitches[:-1, 0]])
        chunks.append(qs)
        chunks.append(np.tile(b, (8, 1)))
    qs = np.concatenate(chunks)
    vs = np.zeros((len(qs), model.nv)); acc = np.zeros_like(vs)
    for i in range(1, len(qs)):
        vs[i] = pin.difference(model, qs[i-1], qs[i])*fps
    vs[0] = vs[1]; acc[1:] = np.diff(vs, axis=0)*fps; acc[0] = acc[1]
    body_names = body_names_from_loader(); body_ids = [model.getFrameId(n) for n in body_names]
    local_pos = np.empty((len(qs), len(body_ids), 3), dtype="f4")
    local_rot = np.empty((len(qs), len(body_ids), 4), dtype="f4")
    audits = []; max_sole_z = 0.; max_sole_pitch = 0.; prev_hand_x = None
    actual_feet = []; actual_hand_x = []
    for i, q in enumerate(qs):
        pin.framesForwardKinematics(model, data, q)
        root_R = Rotation.from_quat(q[3:7]).as_matrix()
        for j, fid in enumerate(body_ids):
            pose = data.oMf[fid]
            local_pos[i, j] = root_R.T@(pose.translation-q[:3])
            local_rot[i, j] = Rotation.from_matrix(root_R.T@pose.rotation).as_quat()
        feet = {n: data.oMf[ids[n]].translation.copy() for n in feet0}
        actual_feet.append([feet[n].copy() for n in feet0])
        max_sole_z = max([max_sole_z] + [abs(p[2]) for p in feet.values() if p[2] < .015])
        max_sole_pitch = max([max_sole_pitch] + [abs(Rotation.from_matrix(data.oMf[ids[n]].rotation).as_euler("xyz")[1]) for n in feet0 if feet[n][2] < .015])
        active = [n for n in feet0 if feet[n][2] < .008]
        if min(data.oMf[ids[n]].translation[0] for n in hands0) >= min(p[0] for p in hands0.values()) + args.hand_initial_forward_m - .03:
            active += list(hands0)
        # Demand enough force to start box sliding only while the hands advance.
        hand_x = np.mean([data.oMf[ids[n]].translation[0] for n in hands0])
        actual_hand_x.append(float(hand_x))
        hand_speed = 0. if prev_hand_x is None else (hand_x-prev_hand_x)*fps
        prev_hand_x = hand_x
        required_push = (args.mass_kg * args.box_friction * 9.81 + 5.
                         if hand_speed > .005 else 10.)
        row = allocate(model, data, ids, q, vs[i], acc[i], active,
                       minimum_push=required_push, foot_mu=args.foot_mu,
                       hand_mu=args.hand_mu, hand_patch_radius=args.hand_patch_radius,
                       foot_patch_radius=args.foot_patch_radius)
        audits.append({"minimum_push_N": required_push, "hand_speed_m_s": hand_speed, **row})
    out = args.output
    path = out / "motions/train/flat_foot_push.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, fps=np.array(fps), joint_names=np.array(joint_names), body_names=np.array(body_names),
                        root_pos=qs[:, :3].astype("f4"), root_rot=qs[:, 3:7].astype("f4"),
                        dof_pos=qs[:, 7:].astype("f4"), local_body_pos=local_pos,
                        local_body_rot=local_rot, joint_vel=vs[:, 6:].astype("f4"), joint_acc=acc[:, 6:].astype("f4"))
    manifest = {"fps": fps, "kind": "flat_foot_box_push_dynamic_screen", "step_order": args.step_order,
                "joint_names": joint_names,
                "body_names": body_names, "sole_offset": [0., 0., float(sole_z)],
                "contact_model": {"foot_mu": args.foot_mu, "hand_mu": args.hand_mu,
                                  "hand_patch_radius_m": args.hand_patch_radius,
                                  "foot_patch_radius_m": args.foot_patch_radius},
                "box": {"center_m": [float(actual_hand_x[0]+(.16 if args.start_upright else 0.)+.45), 0., .45],
                        "size_m": [.9, .7, .9], "mass_kg": args.mass_kg, "friction": args.box_friction},
                "clips": [{"name": "flat_foot_push", "split": "train", "file": "motions/train/flat_foot_push.npz",
                           "frames": len(qs), "push_cm": (args.push_m if args.no_step else 2*dx+final_extra)*100}]}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    summary = {"frames": len(qs), "frames_checked": len(qs), "joint_names": joint_names,
               "crocoddyl_waypoints": crocoddyl_reports,
               "sliding_minimum_push_N": args.mass_kg * args.box_friction * 9.81 + 5.,
               "feasible_frames": sum(r["feasible"] for r in audits),
               "max_effort_ratio": max((r.get("max_effort_ratio", 0) for r in audits), default=0),
               "max_grounded_sole_height_m": max_sole_z, "max_grounded_sole_pitch_deg": float(np.rad2deg(max_sole_pitch)),
               "nodes": node_reports, "rows": [{"frame": i, **r} for i, r in enumerate(audits)],
               "first_failed_frames": [{"frame": i, **r} for i, r in enumerate(audits) if not r["feasible"]][:12]}
    actual_feet = np.asarray(actual_feet)
    swing_counts = np.sum(actual_feet[:, :, 2] > .008, axis=0)
    foot_forward = actual_feet[-1, :, 0] - actual_feet[0, :, 0]
    hand_forward = actual_hand_x[-1] - actual_hand_x[0]
    summary["actual_motion"] = {
        "left_swing_frames": int(swing_counts[0]),
        "right_swing_frames": int(swing_counts[1]),
        "left_max_sole_height_m": float(actual_feet[:, 0, 2].max()),
        "right_max_sole_height_m": float(actual_feet[:, 1, 2].max()),
        "left_foot_forward_m": float(foot_forward[0]),
        "right_foot_forward_m": float(foot_forward[1]),
        "hands_forward_m": float(hand_forward),
        "root_min_height_m": float(qs[:, 2].min()),
    }
    summary["offline_alternating_screen_pass"] = bool(
        summary["feasible_frames"] == len(qs)
        and np.all(swing_counts >= 20)
        and np.all(foot_forward >= .01)
        and hand_forward >= .02
        and qs[:, 2].min() > .65)
    (out/"dynamic_audit.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("nodes", "rows", "first_failed_frames")}, indent=2))


if __name__ == "__main__":
    main()

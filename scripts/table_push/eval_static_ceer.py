"""Run frozen CEER on three static table-support poses in Isaac Lab."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE", "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--table-center-z", type=float, default=0.85)
    parser.add_argument("--controller", choices=("native_ceer", "ceer", "joint_pd"), default="native_ceer")
    parser.add_argument("--pose-path", type=Path, default=ROOT / "artifacts/table_push/static_poses.npz")
    parser.add_argument("--initial-pose-path", type=Path)
    parser.add_argument("--ramp-steps", type=int, default=0)
    parser.add_argument("--leg-command-gain", type=float, default=1.0)
    parser.add_argument("--future-reference", action="store_true")
    parser.add_argument("--root-velocity-reference", action="store_true")
    parser.add_argument("--root-velocity-gain", type=float, default=1.0)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--video-poses", type=int, nargs="+")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/table_push/static_ceer.json")
    args = parser.parse_args()
    import hydra
    import numpy as np
    import torch
    from tensordict import TensorDict
    from omegaconf import OmegaConf
    from isaaclab.app import AppLauncher

    with hydra.initialize_config_dir(config_dir=str(ROOT / "cfg"), version_base=None):
        cfg = hydra.compose(config_name="train", overrides=["task=G1/G1_table_push_static", "algo=ceer_residual"])
    OmegaConf.set_struct(cfg, False)
    cfg.task.command.pose_path = str(args.pose_path.resolve())
    cfg.task.action.ceer_checkpoint = str((ROOT / ".cache/ceer/checkpoint_final.pt").resolve())
    cfg.task.objects[0].pos[2] = args.table_center_z
    cfg.task.max_episode_length = max(args.steps + 10, 1000)
    render_video = args.video_dir is not None
    cfg.app = {"headless": True, "enable_cameras": render_video}
    if render_video:
        cfg.task.viewer.resolution = [960, 540]
        cfg.task.viewer.eye = [1.45, 2.45, 0.6]
        cfg.task.viewer.lookat = [0.22, 0.0, -0.24]
        args.video_dir.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(headless=True, device="cuda:0", enable_cameras=render_video).app
    try:
        from active_adaptation.envs import SimpleEnv
        if render_video:
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw, ImageFont
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)

        env = SimpleEnv(cfg.task)
        env.eval()
        act, cmd, robot = env.action_manager, env.command_manager, env.robot
        if args.initial_pose_path is not None:
            initial_data = np.load(args.initial_pose_path)
            if list(initial_data["joint_names"]) != cmd.names:
                raise ValueError("Initial and reference joint order differ")
            cmd.poses[0] = torch.as_tensor(initial_data["q_pinocchio"][0].copy(),
                                           device=env.device, dtype=torch.float32)
        if not all(not p.requires_grad for p in act.low_policy.parameters()):
            raise RuntimeError("CEER weights are not frozen")
        if args.controller == "native_ceer":
            from scripts.export_whole_body_student import build_policy_state, build_student_rollout_from_state
            from active_adaptation.utils.ceer_residual import TARGET_JOINT_NAMES

            checkpoint = torch.load(ROOT / ".cache/ceer/checkpoint_final.pt", map_location="cpu",
                                    weights_only=False)
            state = build_policy_state(checkpoint)
            native = build_student_rollout_from_state(state)
            native.load_state_dict(state, strict=True)
            native.to(env.device).eval().requires_grad_(False)
            normalization = act.low_policy

            class NativeCEERAdapter(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.model = native
                    self.register_buffer("loc", normalization.policy_loc.clone())
                    self.register_buffer("scale", normalization.policy_scale.clone())
                    self.register_buffer("joint_loc", normalization.joint_loc.clone())
                    self.register_buffer("joint_scale", normalization.joint_scale.clone())
                    self.last_predicted_joint_raw = None

                def forward(self, obs, unused_target):
                    normalized = (obs - self.loc) / self.scale
                    prediction = self.model.adapt_joint(normalized)
                    self.last_predicted_joint_raw = prediction * self.joint_scale + self.joint_loc
                    return self.model(normalized)

            act.low_policy = NativeCEERAdapter().to(env.device).eval().requires_grad_(False)
            native_left_knee_index = TARGET_JOINT_NAMES.index("left_knee_joint")
        if args.controller == "joint_pd":
            # Diagnostic only: identical motor, reset, and scene; bypass CEER's actor.
            act.low_policy = lambda obs, target: (act.reference.q - act.nominal) / act.action_scaling
        if act.action_dim != 29:
            raise RuntimeError("Expected 29-dimensional zero residual")
        hand_sensor = env.scene["ee_contact_forces"]
        table_hand_sensors = [env.scene["eval_left_hand_wall_force"],
                              env.scene["eval_right_hand_wall_force"]]
        foot_sensor = env.scene["contact_forces"]
        left_foot_id = robot.body_names.index("left_ankle_roll_link")
        right_foot_id = robot.body_names.index("right_ankle_roll_link")
        body_masses = robot.root_physx_view.get_masses()[0].to(env.device)
        total_mass = body_masses.sum()
        pose_data = np.load(cfg.task.command.pose_path)
        reference_poses = pose_data["q_pinocchio"]
        if "report_json" in pose_data.files:
            pose_lifts = [float(row["lift_m"]) for row in json.loads(str(pose_data["report_json"]))]
        elif "lift_targets_m" in pose_data.files:
            pose_lifts = [float(x) for x in pose_data["lift_targets_m"]]
        else:
            pose_lifts = [float("nan")] * len(reference_poses)
        joint_target = torch.tensor(reference_poses[:, 7:].copy(),
                                    device=env.device, dtype=torch.float32)
        root_targets = torch.tensor(reference_poses[:, :3].copy(), device=env.device,
                                    dtype=torch.float32)
        target_ids = [cmd.names.index(name) for name in act.joint_names]
        left_knee_index = act.joint_names.index("left_knee_joint")
        left_leg_indices = [i for i, name in enumerate(act.joint_names)
                            if name.startswith("left_") and any(x in name for x in
                                ("hip_", "knee_", "ankle_"))]
        reports = []
        pose_indices = args.video_poses if args.video_poses is not None else range(len(joint_target))
        for pose_index in pose_indices:
            if not 0 <= pose_index < len(joint_target):
                raise ValueError(f"Pose index {pose_index} is outside the reference file")
            writer = None
            if render_video:
                label = f"target_{round(100 * pose_lifts[pose_index]):03d}cm"
                if args.leg_command_gain != 1.0:
                    label += f"_leg_gain_{args.leg_command_gain:g}"
                video_path = args.video_dir / f"{label}.mp4"
                writer = imageio.get_writer(str(video_path), fps=25, codec="libx264",
                                            macro_block_size=1, ffmpeg_params=["-crf", "24",
                                            "-pix_fmt", "yuv420p"])
            cmd.pose_index = 0 if args.ramp_steps else pose_index
            td = env.reset()
            env.episode_length_buf.zero_()
            initial = cmd.poses[0, 7:][target_ids].unsqueeze(0)
            goal = joint_target[pose_index, target_ids].unsqueeze(0)
            if args.leg_command_gain != 1.0:
                goal = goal.clone()
                goal[:, left_leg_indices] = (initial[:, left_leg_indices] + args.leg_command_gain
                    * (goal[:, left_leg_indices] - initial[:, left_leg_indices])).clamp(
                    act.reference.lower[:, left_leg_indices], act.reference.upper[:, left_leg_indices])
            hand_start = robot.data.body_pos_w[0, [robot.body_names.index("left_hand_mimic"),
                                                    robot.body_names.index("right_hand_mimic")]].clone()
            hand_start_xyz = hand_start.tolist()
            rows = []
            for step in range(args.steps):
                phase = min(1., (step + 1) / args.ramp_steps) if args.ramp_steps else 1.
                blend = 6. * phase**5 - 15. * phase**4 + 10. * phase**3
                reference = initial + blend * (goal - initial)
                if args.future_reference:
                    future = []
                    for offset in (0, 2, 4, 8, 16):
                        future_phase = min(1., (step + offset + 1) / args.ramp_steps) if args.ramp_steps else 1.
                        future_blend = (6. * future_phase**5 - 15. * future_phase**4
                                        + 10. * future_phase**3)
                        future.append(initial + future_blend * (goal - initial))
                    act.joint_future_override = torch.stack(future, dim=1)
                if args.root_velocity_reference and args.ramp_steps and phase < 1.:
                    blend_rate = (30. * phase**4 - 60. * phase**3 + 30. * phase**2) / (args.ramp_steps * env.step_dt)
                    root_xy_velocity = ((root_targets[pose_index, :2] - cmd.poses[0, :2])
                                        * blend_rate * args.root_velocity_gain)
                else:
                    root_xy_velocity = torch.zeros(2, device=env.device)
                root_height = float(cmd.poses[0, 2] + blend * (root_targets[pose_index, 2] - cmd.poses[0, 2]))
                act.root_command_override = torch.tensor([[root_height, float(root_xy_velocity[0]),
                                                           float(root_xy_velocity[1]), 1., 0.,
                                                           act.ceer_force_limit]], device=env.device)
                act.reference.q.copy_(reference)
                td["action"] = torch.zeros(1, 29, device=env.device)
                td = env.step(td)["next"]
                root = robot.data.root_pos_w[0]
                gravity = robot.data.projected_gravity_b[0]
                hands = robot.data.body_pos_w[0, [robot.body_names.index("left_hand_mimic"),
                                                robot.body_names.index("right_hand_mimic")]]
                com = (robot.data.body_com_pos_w[0] * body_masses[:, None]).sum(0) / total_mass
                hf = hand_sensor.data.net_forces_w[0]
                table_hf = [s.data.force_matrix_w.reshape(1, -1, 3).sum(1)[0].tolist()
                            for s in table_hand_sensors]
                ff = foot_sensor.data.net_forces_w[0]
                foot_xyz = robot.data.body_pos_w[0, [left_foot_id, right_foot_id]]
                q = robot.data.joint_pos[0, act.joint_ids]
                rows.append(dict(step=step, root_height_m=float(root[2]),
                                 root_xyz_m=root.tolist(),
                                 root_linear_velocity_body_mps=robot.data.root_lin_vel_b[0].tolist(),
                                 root_command=act.root_command_override[0].tolist(),
                                 root_tilt_rad=float(torch.acos((-gravity[2]).clamp(-1, 1))),
                                 hand_drift_m=(hands - hand_start).norm(dim=-1).tolist(),
                                 hand_xy_drift_m=(hands[:, :2] - hand_start[:, :2]).norm(dim=-1).tolist(),
                                 com_xyz_m=com.tolist(),
                                 foot_xyz_m=foot_xyz.tolist(),
                                 left_knee_target_rad=float(reference[0, left_knee_index]),
                                 left_knee_actual_rad=float(q[left_knee_index]),
                                 left_knee_policy_target_rad=float(act.nominal[left_knee_index]
                                     + act.low_action[0, left_knee_index] * act.action_scaling[left_knee_index]),
                                 native_predicted_left_knee_rad=(
                                     float(act.low_policy.last_predicted_joint_raw[0, native_left_knee_index])
                                     if args.controller == "native_ceer" else None),
                                 hand_net_force_N=hf.tolist(), foot_net_force_N=ff.tolist(),
                                 table_hand_force_N=table_hf,
                                 joint_mae_rad=float((q - reference[0]).abs().mean())))
                if writer is not None and step % 2 == 0:
                    image = Image.fromarray(np.asarray(env.render("rgb_array"))[..., :3].astype(np.uint8))
                    draw = ImageDraw.Draw(image, "RGBA")
                    draw.rectangle((0, 0, image.width, 66), fill=(9, 15, 24, 210))
                    actual_lift_cm = max(0., 100. * (float(foot_xyz[0, 2]) - .035))
                    mode_label = ("Native frozen CEER | IK goal" if args.controller == "native_ceer"
                                  else "Frozen actor + direct joint reference | goal")
                    draw.text((12, 8), f"{mode_label} {100 * pose_lifts[pose_index]:.0f} cm"
                              f" | t = {(step + 1) * env.step_dt:.2f} s", font=font, fill="white")
                    draw.text((12, 34), f"actual lift {actual_lift_cm:.1f} cm | left foot {float(ff[0, 2]):.0f} N"
                              f" | table hands {table_hf[0][2]:.0f} / {table_hf[1][2]:.0f} N",
                              font=font, fill=(120, 225, 255))
                    writer.append_data(np.asarray(image))
                if not torch.isfinite(root).all() or root[2] < .45:
                    break
            if writer is not None:
                writer.close()
                print("VIDEO_SAVED", video_path, flush=True)
            tail = rows[min(max(50, args.ramp_steps), len(rows) - 1):]
            left_air_fraction = float(np.mean([
                r["foot_xyz_m"][0][2] > .045 and r["foot_net_force_N"][0][2] < 5.
                for r in tail]))
            right_contact_fraction = float(np.mean([
                r["foot_net_force_N"][1][2] > 30. for r in tail]))
            both_hands_support_fraction = float(np.mean([
                all(r["table_hand_force_N"][i][2] > 5. for i in (0, 1))
                for r in tail]))
            passed = (len(rows) == args.steps and min(x["root_height_m"] for x in rows) > .6
                      and max(x["root_tilt_rad"] for x in rows) < .35
                      and max(max(x["hand_xy_drift_m"]) for x in rows) < .05
                      and right_contact_fraction > .95
                      and both_hands_support_fraction > .9
                      and (pose_index == 0 or left_air_fraction > .9))
            reports.append(dict(pose_index=pose_index, steps_completed=len(rows),
                                ramp_steps=args.ramp_steps,
                                leg_command_gain=args.leg_command_gain,
                                future_reference=args.future_reference,
                                root_velocity_reference=args.root_velocity_reference,
                                root_velocity_gain=args.root_velocity_gain,
                                left_air_fraction=left_air_fraction,
                                right_contact_fraction=right_contact_fraction,
                                both_hands_support_fraction=both_hands_support_fraction,
                                passed=passed,
                                hand_start_xyz_m=hand_start_xyz,
                                hand_end_xyz_m=hands.tolist(),
                                min_root_height_m=min(x["root_height_m"] for x in rows),
                                max_root_tilt_rad=max(x["root_tilt_rad"] for x in rows),
                                mean_hand_vertical_N=[float(np.mean([r["hand_net_force_N"][i][2]
                                                               for r in tail])) for i in range(2)],
                                mean_table_hand_vertical_N=[float(np.mean([r["table_hand_force_N"][i][2]
                                                               for r in tail])) for i in range(2)],
                                mean_right_foot_vertical_N=float(np.mean([r["foot_net_force_N"][1][2]
                                                                          for r in tail])),
                                max_hand_drift_m=max(max(x["hand_drift_m"]) for x in rows),
                                max_hand_xy_drift_m=max(max(x["hand_xy_drift_m"]) for x in rows),
                                com_start_xyz_m=rows[0]["com_xyz_m"],
                                com_end_xyz_m=rows[-1]["com_xyz_m"],
                                mean_joint_mae_rad=float(np.mean([x["joint_mae_rad"] for x in tail])),
                                samples=rows))
            print("POSE_RESULT", pose_index, json.dumps({k: v for k, v in reports[-1].items()
                                                         if k != "samples"}), flush=True)
        checkpoint = ROOT / ".cache/ceer/checkpoint_final.pt"
        result = dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      task=f"{args.controller}, zero residual, static table diagnostic",
                      table_top_m=args.table_center_z + .015,
                      hand_sensor_body_names=hand_sensor.body_names,
                      foot_sensor_body_names=foot_sensor.body_names,
                      poses=reports)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        from isaaclab.sim import SimulationContext
        SimulationContext.instance()._disable_app_control_on_stop_handle = True
        env.close()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        app.close()


if __name__ == "__main__":
    main()

"""Independent table-supported left-foot task; actor receives only target xyz and proprioception."""
import torch

from active_adaptation.envs.mdp import observation, reward
from active_adaptation.utils.math import quat_apply_inverse, quat_apply
from .foot_reach import WallFootReachCommand


class TablePushTaskCommand(WallFootReachCommand):
    supports_symmetry_augmentation = False

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.right_hand_asset = self.asset.body_names.index("right_hand_mimic")
        self.right_hand_motion = self.dataset.body_names.index("right_hand_mimic")
        self._external_targets_w = None

    def set_targets_world(self, targets):
        """Set [root, left hand, right hand, left sole, right sole] world xyz.

        The command is shared by teacher and student at inference.  None
        restores the motion dataset's per-frame targets.
        """
        if targets is None:
            self._external_targets_w = None
            return
        targets = torch.as_tensor(targets, dtype=torch.float32, device=self.device)
        if targets.shape == (5, 3):
            targets = targets.expand(self.num_envs, -1, -1)
        if targets.shape != (self.num_envs, 5, 3) or not torch.isfinite(targets).all():
            raise ValueError("Expected finite [5, 3] or [num_envs, 5, 3] targets")
        self._external_targets_w = targets.clone()

    def target_positions_world(self):
        if self._external_targets_w is not None:
            return self._external_targets_w
        m = self._motion
        left_ankle = m.body_pos_w[:, 0, self.left_motion] + self.env.scene.env_origins
        left_quat = m.body_quat_w[:, 0, self.left_motion]
        left_sole = left_ankle + quat_apply(left_quat, self.sole_offset.expand(self.num_envs, -1))
        right_ankle = m.body_pos_w[:, 0, self.right_motion] + self.env.scene.env_origins
        right_quat = m.body_quat_w[:, 0, self.right_motion]
        right_sole = right_ankle + quat_apply(right_quat, self.sole_offset.expand(self.num_envs, -1))
        return torch.stack((m.root_pos_w[:, 0] + self.env.scene.env_origins,
                            m.body_pos_w[:, 0, self.hand_motion] + self.env.scene.env_origins,
                            m.body_pos_w[:, 0, self.right_hand_motion] + self.env.scene.env_origins,
                            left_sole, right_sole), dim=1)

    @observation
    def target_support_xyz_b(self):
        """Five externally commandable xyz targets expressed in the live root frame."""
        targets = self.target_positions_world()
        delta = targets - self.asset.data.root_pos_w[:, None, :]
        root_quat = self.asset.data.root_quat_w[:, None, :].expand(-1, 5, -1)
        return quat_apply_inverse(root_quat.reshape(-1, 4), delta.reshape(-1, 3)).reshape(self.num_envs, 15)

    def target_support_xyz_b_sym(self):
        from active_adaptation.utils.symmetry import SymmetryTransform
        return SymmetryTransform(torch.arange(15), torch.ones(15).tolist())

    def target_sole_world(self):
        if getattr(self, "_external_targets_w", None) is not None:
            return self._external_targets_w[:, 3]
        return super().target_sole_world()

    @reward
    def wall_hand_tracking(self):
        errors = []
        for asset_id, motion_id in ((self.hand_asset, self.hand_motion),
                                    (self.right_hand_asset, self.right_hand_motion)):
            target = self.target_positions_world()[:, 1 if asset_id == self.hand_asset else 2]
            current = self.asset.data.body_pos_w[:, asset_id]
            errors.append(torch.linalg.vector_norm(target - current, dim=-1, keepdim=True))
        return torch.exp(-torch.stack(errors).mean(0) / .035)

    @reward
    def table_hand_support(self):
        forces = []
        for name in ("eval_left_hand_wall_force", "eval_right_hand_wall_force"):
            sensor = self.env.scene[name]
            forces.append(sensor.data.force_matrix_w.reshape(self.num_envs, -1, 3).sum(1)[:, 2:3])
        # Reward both hands bearing modest weight; saturate to avoid slamming.
        return torch.stack([torch.clamp(f / 15., 0., 1.) for f in forces]).mean(0)

    @reward
    def right_wall_support(self):
        """Reward the right hand bearing a modest load against a negative-y wall."""
        sensor = self.env.scene["eval_right_hand_wall_force"]
        force_y = sensor.data.force_matrix_w.reshape(self.num_envs, -1, 3).sum(1)[:, 1:2]
        return torch.clamp(force_y / 20., 0., 1.)

    @reward
    def box_parallel_progress(self):
        """Track x displacement of the dynamic box as the left-foot x goal advances."""
        target_x = self.target_positions_world()[:, 3, 0:1] - self.env.scene.env_origins[:, 0:1]
        desired_dx = (target_x - .15).clamp(0., .14)
        box = self.env.scene["push_block"]
        actual_dx = box.data.root_pos_w[:, 0:1] - self.env.scene.env_origins[:, 0:1] - .22
        return torch.exp(-torch.abs(actual_dx - desired_dx) / .04)

    @reward
    def box_normal_stability(self):
        """Keep the box near its initial y while it slides parallel to the wall."""
        box = self.env.scene["push_block"]
        dy = box.data.root_pos_w[:, 1:2] - self.env.scene.env_origins[:, 1:2] - .12
        return torch.exp(-torch.abs(dy) / .03)

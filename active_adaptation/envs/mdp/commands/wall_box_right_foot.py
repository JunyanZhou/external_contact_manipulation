"""Right-hand wall support, left-foot stance, right-foot box push."""
import torch

from active_adaptation.envs.mdp import reward
from active_adaptation.utils.math import quat_apply
from .table_push_task import TablePushTaskCommand


class WallBoxRightFootCommand(TablePushTaskCommand):
    def target_sole_world(self):
        return self.target_positions_world()[:, 4]

    @reward
    def swing_foot_tracking(self):
        current = self.asset.data.body_pos_w[:, self.right_asset] + quat_apply(
            self.asset.data.body_quat_w[:, self.right_asset],
            self.sole_offset.expand(self.num_envs, -1))
        error = torch.linalg.vector_norm(self.target_sole_world() - current, dim=-1, keepdim=True)
        self._cum_error[:, 2:3] = error / self._cum_keypoint_scale
        return torch.exp(-error / .06)

    @reward
    def support_foot_tracking(self):
        target = self.target_positions_world()[:, 3]
        current = self.asset.data.body_pos_w[:, self.left_asset] + quat_apply(
            self.asset.data.body_quat_w[:, self.left_asset],
            self.sole_offset.expand(self.num_envs, -1))
        error = torch.linalg.vector_norm(target - current, dim=-1, keepdim=True)
        return torch.exp(-error / .025)

    @reward
    def box_parallel_progress(self):
        target_x = self.target_positions_world()[:, 4, 0:1] - self.env.scene.env_origins[:, 0:1]
        desired_dx = (target_x - .14).clamp(0., .16)
        box = self.env.scene["push_block"]
        actual_dx = box.data.root_pos_w[:, 0:1] - self.env.scene.env_origins[:, 0:1] - .22
        return torch.exp(-torch.abs(actual_dx - desired_dx) / .04)

    @reward
    def box_normal_stability(self):
        box = self.env.scene["push_block"]
        dy = box.data.root_pos_w[:, 1:2] - self.env.scene.env_origins[:, 1:2] + .04
        return torch.exp(-torch.abs(dy) / .03)

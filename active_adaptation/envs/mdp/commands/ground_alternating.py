"""Five-target alternating-foot task without external support objects."""
import torch

from active_adaptation.envs.mdp import reward
from active_adaptation.utils.math import quat_apply
from .table_push_task import TablePushTaskCommand


class GroundAlternatingCommand(TablePushTaskCommand):
    @reward
    def alternating_lift(self):
        targets = self.target_positions_world()
        heights = []
        for index, asset_id in ((3, self.left_asset), (4, self.right_asset)):
            current = self.asset.data.body_pos_w[:, asset_id] + quat_apply(
                self.asset.data.body_quat_w[:, asset_id], self.sole_offset.expand(self.num_envs, -1))
            desired = targets[:, index, 2:3]
            active = (desired > .008).float()
            heights.append(active * (current[:, 2:3] / desired.clamp_min(.008)).clamp(0., 1.))
        return torch.stack(heights).sum(0)

    @reward
    def swing_foot_tracking(self):
        targets = self.target_positions_world()
        errors = []
        for index, asset_id in ((3, self.left_asset), (4, self.right_asset)):
            current = self.asset.data.body_pos_w[:, asset_id] + quat_apply(
                self.asset.data.body_quat_w[:, asset_id], self.sole_offset.expand(self.num_envs, -1))
            errors.append(torch.linalg.vector_norm(targets[:, index] - current, dim=-1, keepdim=True))
        error = torch.stack(errors).mean(0)
        self._cum_error[:, 2:3] = error / self._cum_keypoint_scale
        return torch.exp(-error / .05)

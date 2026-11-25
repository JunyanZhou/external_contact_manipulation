"""Export the five external xyz command streams from a screened motion clip."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--sole-offset-z", type=float, default=-0.03540915)
    args = p.parse_args()
    with np.load(args.source) as clip:
        names = list(clip["body_names"])
        root = clip["root_pos"].astype(np.float64)
        root_r = Rotation.from_quat(clip["root_rot"])
        local_pos = clip["local_body_pos"].astype(np.float64)
        local_rot = clip["local_body_rot"].astype(np.float64)
        def body(name, sole=False):
            idx = names.index(name)
            local = local_pos[:, idx].copy()
            if sole:
                offset = np.tile([0., 0., args.sole_offset_z], (len(root), 1))
                local += Rotation.from_quat(local_rot[:, idx]).apply(offset)
            return root + root_r.apply(local)
        goals = np.stack((root, body("left_hand_mimic"), body("right_hand_mimic"),
                          body("left_ankle_roll_link", True),
                          body("right_ankle_roll_link", True)), axis=1).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, goal_xyz_world=goals,
                        names=np.array(["root", "left_hand", "right_hand", "left_sole", "right_sole"]))
    print(f"{args.output}: {goals.shape[0]} frames, five xyz targets")


if __name__ == "__main__":
    main()

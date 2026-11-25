"""Scale the lateral component of a smooth external left-foot target clip."""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sweep", type=float, required=True,
                        help="Nominal lateral sweep of the input clip, in meters")
    parser.add_argument("--target-sweep", type=float, required=True,
                        help="Desired nominal lateral sweep, in meters")
    args = parser.parse_args()
    if args.source_sweep <= 0 or args.target_sweep <= 0:
        parser.error("Sweep magnitudes must be positive")
    target = np.load(args.input)["target_foot_pos_b"].copy()
    target[:, 1] = target[0, 1] + (target[:, 1] - target[0, 1]) * (
        args.target_sweep / args.source_sweep)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, target_foot_pos_b=target)


if __name__ == "__main__":
    main()

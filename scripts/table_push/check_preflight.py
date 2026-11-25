"""Check whether the Frozen CEER table-push experiments can run here.

This check never starts Isaac Sim or modifies a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / ".cache/ceer/checkpoint_final.pt")
    parser.add_argument("--reference-python", type=Path, default=ROOT / ".venv-reference/bin/python")
    parser.add_argument("--output", type=Path, help="Write the preflight report as JSON")
    args = parser.parse_args()

    packages = {name: importlib.util.find_spec(name) is not None
                for name in ("torch", "isaaclab", "crocoddyl", "pinocchio", "coal", "numpy")}
    checkpoint = args.checkpoint.expanduser().resolve()
    report: dict = {
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.is_file(),
        "packages": packages,
        "robot_usd_exists": (ROOT / "active_adaptation/assets/G1/g1_flat_fullcol.usd").is_file(),
    }
    # Keep the venv path: resolving its python symlink would discard pyvenv.cfg.
    reference_python = args.reference_python.expanduser().absolute()
    report["reference_python"] = str(reference_python)
    if reference_python.is_file():
        probe = subprocess.run(
            [str(reference_python), "-c",
             "import importlib.util,json;print(json.dumps({x:bool(importlib.util.find_spec(x)) for x in ('crocoddyl','pinocchio','coal','pxr','scipy')}))"],
            capture_output=True, text=True, check=False,
        )
        if probe.returncode == 0:
            report["reference_packages"] = json.loads(probe.stdout)
        else:
            report["reference_python_error"] = probe.stderr.strip()
    if checkpoint.is_file():
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        report["checkpoint_sha256"] = digest.hexdigest()
        if packages["torch"]:
            import torch

            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            policy = state.get("policy", {})
            cfg = state.get("cfg")
            report["checkpoint_phase"] = policy.get("last_phase")
            report["has_checkpoint_env_norm"] = "env" in state
            report["has_config"] = cfg is not None
            report["has_student_actor"] = "actor_student" in policy
            if cfg is not None:
                try:
                    report["checkpoint_robot"] = str(cfg.task.robot.name)
                    obs_cfg = cfg.task.observation.policy
                    report["checkpoint_policy_observations"] = list(obs_cfg)
                    report["joint_history_steps"] = list(obs_cfg.joint_pos_history.history_steps)
                except (AttributeError, KeyError, TypeError):
                    report["checkpoint_schema_error"] = "missing expected task observations"
            env_state = state.get("env", {})
            report["normalization_shapes"] = {
                key: list(env_state[key].shape) if key in env_state else None
                for key in ("transforms.2.loc", "transforms.2.scale",
                            "transforms.3.loc", "transforms.3.scale")
            }
            report["checkpoint_compatible"] = (
                report["checkpoint_phase"] == "finetune"
                and report["has_checkpoint_env_norm"] and report["has_student_actor"]
                and report.get("checkpoint_policy_observations") == [
                    "boot_indicator_state", "command", "root_and_wrist_6d",
                    "root_ang_vel_history", "projected_gravity_history",
                    "joint_pos_history", "prev_actions"]
                and report.get("joint_history_steps") == [0, 1, 2, 3, 4]
                and report["normalization_shapes"] == {
                    "transforms.2.loc": [257], "transforms.2.scale": [257],
                    "transforms.3.loc": [145], "transforms.3.scale": [145]}
            )
    report["step1_runtime_prerequisites_present"] = bool(
        report.get("checkpoint_compatible", False)
        and report["robot_usd_exists"]
        and all(packages[name] for name in ("torch", "isaaclab", "numpy"))
    )
    report["step2_runtime_prerequisites_present"] = bool(
        report["step1_runtime_prerequisites_present"]
        and all(report.get("reference_packages", {}).get(name, False)
                for name in ("crocoddyl", "pinocchio", "coal", "pxr", "scipy"))
    )
    result = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n")
    print(result)
    return 0 if report["step2_runtime_prerequisites_present"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

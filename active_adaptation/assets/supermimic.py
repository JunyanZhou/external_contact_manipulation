"""G1 asset initialization for the released SuperMimic controller."""
import json
import os
from pathlib import Path

from .humanoid import G1_COL_FULL

DEPLOY_CONFIG = Path(os.environ.get("SUPERMIMIC_ROOT", str(Path(__file__).resolve().parents[2] / ".cache/supermimic"))) / "pretrained/e11c_0220000/deploy_config.json"
_deploy = json.loads(DEPLOY_CONFIG.read_text())

G1_SUPERMIMIC = G1_COL_FULL.copy()
G1_SUPERMIMIC.init_state.pos = (0.0, 0.0, 0.76)
G1_SUPERMIMIC.init_state.joint_pos = dict(zip(_deploy["joint_names"], _deploy["default_joint_pos"]))

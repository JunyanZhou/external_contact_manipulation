import os
import copy
from isaaclab.assets import Articulation
from .humanoid import *
from .sonic import G1_SONIC


ASSET_PATH = os.path.dirname(__file__)

ROBOTS = {
    "g1": G1_CFG,
    "g1_col_full": G1_COL_FULL,
    "g1_col_full_arm_reach": G1_COL_FULL_ARM_REACH,
    "g1_col_full_arm_contact": G1_COL_FULL_ARM_CONTACT,
    "g1_col_full_arm_contact60": G1_COL_FULL_ARM_CONTACT60,
    "g1_col_full_arm_contact70": G1_COL_FULL_ARM_CONTACT70,
    "g1_col_full_arm_contact70_drawer_palm": G1_COL_FULL_ARM_CONTACT70_DRAWER_PALM,
    "g1_col_full_arm_contact70_drawer_grasp": G1_COL_FULL_ARM_CONTACT70_DRAWER_GRASP,
    "g1_col_full_box_push_palms": G1_COL_FULL_BOX_PUSH_PALMS,
    "g1_sonic": G1_SONIC,
}

# The released SuperMimic asset depends on a separate, optional checkout.
# Keep the standard G1 assets usable when that checkpoint is absent.
try:
    from .supermimic import G1_SUPERMIMIC
except FileNotFoundError:
    pass
else:
    ROBOTS["g1_supermimic"] = G1_SUPERMIMIC

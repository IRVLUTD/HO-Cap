import sys, os
from pathlib import Path
from .mano_info import *
from .device_info import *
from .colors import (
    COLORS,
    OBJ_CLASS_COLORS,
    OBJECT_COLORS,
    HAND_COLORS,
    HAND_BONE_COLORS,
    HAND_JOINT_COLORS,
)

NUM_CPU_CORES = os.cpu_count()

NUM_MANO_VERTS = 778
NUM_MANO_FACES = 1538

PROJ_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = PROJ_ROOT / "external"
XMEM_ROOT = PROJ_ROOT / "external" / "XMem"


def add_path(path):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

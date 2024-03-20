import sys
from pathlib import Path


def add_path(path):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


CURR_DIR = Path(__file__).resolve().parents[0]
PROJ_ROOT = Path(__file__).resolve().parents[1]

add_path(PROJ_ROOT)

import sys
from pathlib import Path


def add_path(path):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


PROJ_ROOT = Path(__file__).resolve().parents[1]

add_path(PROJ_ROOT)

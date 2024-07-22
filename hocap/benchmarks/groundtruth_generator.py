from ..utils import *
from ..utils.common import *
from ..loaders import SequenceLoader


class BenchmarkGTGenerator:
    def __init__(self):
        self._data_root = PROJ_ROOT / "data"

    def generate_hand_pose_gt(self):
        keys_file = (
            self._data_root / "data/benchmarks/hand_pose_benchmark_gt_demo_keys.json"
        )
        keys = read_data_from_json(keys_file)

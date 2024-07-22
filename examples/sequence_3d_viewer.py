from hocap.utils import *
from hocap.loaders import SequenceLoader


class SequenceViewer:
    def __init__(self, sequence_folder, device="cpu") -> None:
        self._loader = SequenceLoader(
            sequence_folder, load_mano=True, load_object=True, device=device
        )

from torchvision import transforms
from ..utils import *

add_path(EXTERNAL_ROOT / "XMem")

from model.network import XMem
from inference.inference_core import InferenceCore
from inference.data.mask_mapper import MaskMapper


XMEM_CONFIG = {
    # "key_dim": 64,
    # "value_dim": 512,
    # "hidden_dim": 64,
    "top_k": 30,
    "mem_every": 3,  # r in paper. Increase to improve running speed
    "deep_update_every": -1,  # Leave -1 normally to synchronize with mem_every
    "single_object": False,
    "enable_long_term": True,
    "enable_long_term_count_usage": True,
    "max_mid_term_frames": 60,  # T_max in paper, decrease to save memory
    "min_mid_term_frames": 3,  # T_min in paper, decrease to save memory
    "num_prototypes": 128,  # P in paper
    "max_long_term_elements": 10000,  # LT_max in paper, increase if objects disappear for a long time
    "device": "cuda",
    "xmem_model_path": "config/xmem/XMem-no-sensory.pth",  # [XMem.pth, XMem-s012.pth, XMem-no-sensory.pth]
}


class XMemWrapper:
    def __init__(self, config=XMEM_CONFIG) -> None:
        model_path = PROJ_ROOT / config.get("xmem_model_path", "config/xmem/XMem.pth")
        if not model_path.exists():
            raise ValueError(f"Cound not find model at {model_path}.")

        self._logger = get_logger("XMemWrapper")
        self._config = config
        self._device = config.get("device", "cuda")

        self._im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load the model checkpoint
        self._network = XMem(self._config, str(model_path)).to(self._device).eval()
        self._network.load_weights(
            torch.load(str(model_path)), init_as_zero_if_needed=True
        )
        self._mapper = MaskMapper()
        self._processor = InferenceCore(self._network, self._config)

        self._first_mask_loaded = False

    def reset(self) -> None:
        self._processor.clear_memory()
        self._processor.all_labels = None
        self._mapper.labels = []
        self._mapper.remappings = {}
        self._first_mask_loaded = False

    @torch.no_grad()
    def get_mask(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray] = None,
        exhaustive: bool = False,
    ) -> np.ndarray:
        """
        Returns a mask for the given frame.

        Args:
            rgb (np.ndarray): RGB image of the frame.
            mask (np.ndarray, optional): Mask of the frame. Defaults to None.
            exhaustive (bool, optional): Flag for exhaustive mask conversion. Defaults to False.

        Returns:
            np.ndarray: The output mask.
        """
        with torch.cuda.amp.autocast(enabled=True):
            rgb = self._im_transform(rgb).to(self._device)

            if not self._first_mask_loaded:
                if mask is not None:
                    self._first_mask_loaded = True
                else:
                    return np.zeros(rgb.shape[1:3], dtype=np.uint8)

            if mask is not None:
                mask = mask.astype(np.uint8)
                mask, labels = self._mapper.convert_mask(mask, exhaustive)
                mask = torch.Tensor(mask).to(self._device)
                self._processor.set_all_labels(list(self._mapper.remappings.values()))
            else:
                labels = None

            prob = self._processor.step(rgb, mask, labels)

            # Probability mask -> index mask
            out_mask = torch.max(prob, dim=0).indices
            out_mask = out_mask.detach().cpu().numpy().astype(np.uint8)
            out_mask = self._mapper.remap_index_mask(out_mask)
            return out_mask

import numpy as np
from mobile_sam import sam_model_registry, SamPredictor
from ..utils import *


class SamWrapper:
    def __init__(self, model_type="vit_t", device="cuda"):
        """Wrapper for Mobile-SAM model.

        Args:
            model_type (str): Model type, one of ["vit_h", "vit_l", "vit_b", "vit_t"]
            device (str): Device, one of ["cuda", "cpu"]
        """

        assert model_type in ["vit_h", "vit_l", "vit_b", "vit_t"]
        self._device = device
        self._model_type = model_type
        self._predictor = self._init_sam_predictor()

    def _init_sam_predictor(self):
        sam = sam_model_registry[self._model_type](
            PROJ_ROOT / f"config/SAM/sam_{self._model_type}.pth"
        )
        sam.to(self._device)
        sam.eval()
        return SamPredictor(sam)

    def set_image(self, rgb_image):
        """Set image for segmentation.

        Args:
            rgb_image (np.ndarray): RGB image, shape (height, width, 3)
        """
        self._img_h = rgb_image.shape[0]
        self._img_w = rgb_image.shape[1]
        self._predictor.set_image(rgb_image)

    def predict(
        self,
        prompt_points=None,
        prompt_labels=None,
        prompt_box=None,
        prompt_mask=None,
    ):
        """Segment image with prompt points and labels.

        Args:

            prompt_points (np.ndarray): Prompt points, shape (num_points, 2)
            prompt_labels (np.ndarray): Prompt labels, shape (num_points,)
            prompt_box (np.ndarray): Prompt box, shape (4,), [x1, y1, x2, y2]
            prompt_mask (np.ndarray): Prompt mask, shape (height, width)

        Returns:
            mask (np.ndarray): Segmentation mask, shape (height, width)
        """

        if prompt_box is not None or prompt_mask is not None:
            masks, _, _ = self._predictor.predict(
                point_coords=prompt_points,
                point_labels=prompt_labels,
                box=prompt_box,
                mask_input=prompt_mask,
                multimask_output=False,
            )
            mask = masks[0]
        elif prompt_points is not None and prompt_labels is not None:  # point prompt
            masks, scores, _ = self._predictor.predict(
                point_coords=prompt_points,
                point_labels=prompt_labels,
                multimask_output=True,
            )
            mask = masks[np.argmax(scores)]
        else:
            mask = np.zeros((self._img_h, self._img_w))

        return mask.astype(bool)

import math
import numpy as np
import mediapipe as mp
from ..utils import *


MP_CONFIG = {
    "max_num_hands": 1,
    "min_hand_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "min_hand_presence_confidence": 0.5,
    "skin_seg_thresh": 0.1,
    "output_confidence_masks": True,
    "output_category_mask": False,
    "running_mode": "image",
    "frame_rate": 30,
    "device": "cpu",
}


class MPHandDetector:
    def __init__(self, config=MP_CONFIG):
        self._config = config
        self._device = config.get("device", "cpu")
        self._mode = config.get("running_mode", "image")
        if self._mode == "video":
            self._delta_time_ms = 1000 // config.get("frame_rate", 30)
            self._timestamp_ms = 0
        self._detector = self._init_mp_hand_detector()

    def _init_mp_hand_detector(self):
        if self._mode == "image":
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(
                        PROJ_ROOT / "config/Mediapipe/hand_landmarker.task"
                    ),
                    delegate=(
                        mp.tasks.BaseOptions.Delegate.CPU
                        if self._device == "cpu"
                        else mp.tasks.BaseOptions.Delegate.GPU
                    ),
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=self._config["max_num_hands"],
                min_hand_detection_confidence=self._config[
                    "min_hand_detection_confidence"
                ],
                min_tracking_confidence=self._config["min_tracking_confidence"],
                min_hand_presence_confidence=self._config[
                    "min_hand_presence_confidence"
                ],
            )
        if self._mode == "video":
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(
                        PROJ_ROOT / "config/Mediapipe/hand_landmarker.task"
                    ),
                    delegate=(
                        mp.tasks.BaseOptions.Delegate.CPU
                        if self._device == "cpu"
                        else mp.tasks.BaseOptions.Delegate.GPU
                    ),
                ),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=self._config["max_num_hands"],
                min_hand_detection_confidence=self._config[
                    "min_hand_detection_confidence"
                ],
                min_tracking_confidence=self._config["min_tracking_confidence"],
                min_hand_presence_confidence=self._config[
                    "min_hand_presence_confidence"
                ],
            )
        return mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb_image):
        """
        Run hand detection on a single image.

        Args:
            rgb_image (np.ndarray): RGB image, shape (height, width, 3)

        Returns:
            hand_marks (list): List of hand marks, shape (num_hands, 21, 2)
            hand_sides (list): List of hand sides, shape (num_hands,)
            hand_scores (list): List of hand scores, shape (num_hands,)
        """

        def normalized_to_pixel_coords(normalized_x, normalized_y):
            pixel_x = min(math.floor(normalized_x * W), W - 1)
            pixel_y = min(math.floor(normalized_y * H), H - 1)
            return pixel_x, pixel_y

        hand_marks = []
        hand_sides = []
        hand_scores = []
        H, W, _ = rgb_image.shape
        mp_image = mp.Image(data=rgb_image.copy(), image_format=mp.ImageFormat.SRGB)

        if self._mode == "image":
            mp_result = self._detector.detect(mp_image)
        if self._mode == "video":
            mp_result = self._detector.detect_for_video(mp_image, self._timestamp_ms)
            self._timestamp_ms += self._delta_time_ms

        if mp_result.hand_landmarks:
            hand_landmarks_list = mp_result.hand_landmarks
            handedness_list = mp_result.handedness
            for idx, hand_landmarks in enumerate(hand_landmarks_list):
                marks = np.array(
                    [
                        normalized_to_pixel_coords(lmk.x, lmk.y)
                        for lmk in hand_landmarks
                    ],
                    dtype=np.int64,
                )
                side = handedness_list[idx][0].category_name.lower()
                score = handedness_list[idx][0].score
                hand_marks.append(marks)
                hand_sides.append(side)
                hand_scores.append(score)

        return hand_marks, hand_sides, hand_scores


class MPSkinSegmenter:
    def __init__(self, config=MP_CONFIG):
        self._config = config
        self._segmenter = self._init_mp_skin_segmenter()

    def _init_mp_image_segmenter(self):
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=(
                    PROJ_ROOT / "config/Mediapipe/selfie_multiclass_256x256.tflite"
                )
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_category_mask=self._config["output_category_mask"],
            output_confidence_masks=self._config["output_confidence_masks"],
        )
        return mp.tasks.vision.ImageSegmenter.create_from_options(options)

    def segment(self, rgb_image):
        mp_image = mp.Image(data=rgb_image, image_format=mp.ImageFormat.SRGB)
        mp_result = self._segmenter.segment(mp_image)
        mask = mp_result.confidence_masks[2].numpy_view()
        mask = mask > self._mp_config["skin_seg_thresh"]
        return mask.astype(np.uint8)

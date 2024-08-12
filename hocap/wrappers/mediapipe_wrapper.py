import mediapipe as mp
from ..utils import *


MP_CONFIG = {
    "max_num_hands": 1,
    "min_hand_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "min_hand_presence_confidence": 0.5,
    "running_mode": "image",
    "frame_rate": 30,
    "device": "cpu",
    "model_asset_path": "config/mediapipe/hand_landmarker.task",
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
        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(PROJ_ROOT / self._config["model_asset_path"]),
            delegate=(
                mp.tasks.BaseOptions.Delegate.CPU
                if self._config.get("device", "cpu") == "cpu"
                else mp.tasks.BaseOptions.Delegate.GPU
            ),
        )
        running_mode = (
            mp.tasks.vision.RunningMode.IMAGE
            if self._mode == "image"
            else mp.tasks.vision.RunningMode.VIDEO
        )
        mp_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_hands=self._config["max_num_hands"],
            min_hand_detection_confidence=self._config["min_hand_detection_confidence"],
            min_tracking_confidence=self._config["min_tracking_confidence"],
            min_hand_presence_confidence=self._config["min_hand_presence_confidence"],
        )

        return mp.tasks.vision.HandLandmarker.create_from_options(mp_options)

    def _normalized_to_pixel_coords(self, normalized_x, normalized_y, width, height):
        pixel_x = min(math.floor(normalized_x * width), width - 1)
        pixel_y = min(math.floor(normalized_y * height), height - 1)
        return pixel_x, pixel_y

    def detect_one(self, rgb_image):
        """
        Run hand detection on a single image.

        Args:
            rgb_image (np.ndarray): RGB image, shape (height, width, 3)

        Returns:
            hand_marks (list): List of hand marks, shape (num_hands, 21, 2)
            hand_sides (list): List of hand sides, shape (num_hands,)
            hand_scores (list): List of hand scores, shape (num_hands,)
        """

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
                        self._normalized_to_pixel_coords(lmk.x, lmk.y, W, H)
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

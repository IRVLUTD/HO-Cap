from hocap.utils import *

MP_CONFIG = {
    "max_num_hands": 2,
    "min_hand_detection_confidence": 0.1,
    "min_tracking_confidence": 0.5,
    "min_hand_presence_confidence": 0.5,
    "running_mode": "video",
    "frame_rate": 30,
    "device": "cpu",
    "model_asset_path": "config/mediapipe/hand_landmarker.task",
}

XMEM_CONFIG = {
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

if __name__ == "__main__":
    print("Creating MediaPipe config...")
    mp_config_file = PROJ_ROOT / "config/mp_config.json"
    write_data_to_json(mp_config_file, MP_CONFIG)

    print("Creating XMem config...")
    xmem_config_file = PROJ_ROOT / "config/xmem_config.json"
    write_data_to_json(xmem_config_file, XMEM_CONFIG)

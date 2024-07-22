import re
from _init_paths import *
from lib.utils import *
from lib.utils.common import *

order_A2J = [20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16]
r_or_l = ["left", "right"]
camera_1 = [378.4029541015625, 377.9852600097656, 322.846435546875, 243.7872772216797]
camera_2 = [381.4822692871094, 381.086669921875, 319.3353271484375, 247.47662353515625]
camera_3 = [384.9375, 384.33966064453125, 308.61553955078125, 243.12005615234375]
camera_4 = [380.6561584472656, 380.3317565917969, 324.2102355957031, 241.03143310546875]
camera_5 = [378.30926513671875, 378.024658203125, 323.66534423828125, 253.562744140625]
camera_6 = [377.67059326171875, 377.2873840332031, 319.260009765625, 242.50064086914062]
camera_7 = [380.1505126953125, 379.7682189941406, 319.6207275390625, 248.99859619140625]
camera_8 = [380.4667663574219, 380.0273132324219, 320.4730529785156, 238.1913604736328]

camera_intrinsics = {
    "037522251142": camera_1,
    "043422252387": camera_2,
    "046122250168": camera_3,
    "105322251225": camera_4,
    "105322251564": camera_5,
    "108222250342": camera_6,
    "115422250549": camera_7,
    "117222250549": camera_8,
}


def merge_a2j_results(results_folder):
    merged_results = {}

    result_files = sorted(results_folder.glob("*.json"))

    for result_file in result_files:
        match = re.search(r"output_(subject_\d+)_(\d{8}_\d{6})\.json", result_file.name)

        if match:
            subject_id = match.group(1)
            sequence_id = match.group(2)
        else:
            raise ValueError(f"Invalid file name: {result_file.name}")

        tqdm.write(f"- Processing {subject_id}/{sequence_id}...")

        data = read_data_from_json(result_file)

        for camera_id in sorted(data.keys()):
            for frame_id in sorted(data[camera_id].keys()):

                print(f"frame_id: {frame_id} {type(frame_id)}")
                if int(frame_id) != 9:
                    continue
                new_key = f"{subject_id}/{sequence_id}/{camera_id}/{int(frame_id):06d}"
                merged_results[new_key] = {
                    "landmarks": {
                        "right": [[-1, -1]] * 21,
                        "left": [[-1, -1]] * 21,
                    },
                    "landmarks_3d": {
                        "right": [[-1, -1, -1]] * 21,
                        "left": [[-1, -1, -1]] * 21,
                    },
                }

                is_right = data[camera_id][frame_id]["is_right"]
                pred_keypoints_2d_full = np.array(
                    data[camera_id][frame_id]["pred_keypoints_2d_full"]
                ).astype(np.int64)

                pred_keypoints_ = np.array(data[camera_id][frame_id]["pred_keypoints_"])

                if len(is_right) != len(np.unique(is_right)):
                    # print("the same")
                    is_right = np.unique(is_right)

                for n in range(len(is_right)):
                    rl = "left" if int(is_right[n]) == 0 else "right"
                    pred_keypoints_2d = pred_keypoints_2d_full[n][order_A2J]
                    if is_right[n] == 1:
                        pred_keypoints_UVD = np.array(pred_keypoints_[n][:21])
                        # pred_keypoints_UVD = pred_keypoints_UVD[order_A2J]

                    elif is_right[n] == 0:
                        pred_keypoints_UVD = np.array(pred_keypoints_[n][21:])
                        # pred_keypoints_UVD = pred_keypoints_UVD[order_A2J]

                    merged_results[new_key]["landmarks"][
                        rl
                    ] = pred_keypoints_2d.tolist()
                    merged_results[new_key]["landmarks_3d"][
                        rl
                    ] = pred_keypoints_UVD.tolist()

                print(merged_results)

                exit()


def merge_hamer_results(results_folder):
    merged_results = {}

    result_files = sorted(results_folder.glob("*.json"))

    for result_file in result_files:
        match = re.search(r"output_(subject_\d+)_(\d{8}_\d{6})\.json", result_file.name)

        if match:
            subject_id = match.group(1)
            sequence_id = match.group(2)
        else:
            raise ValueError(f"Invalid file name: {result_file.name}")

        tqdm.write(f"- Processing {subject_id}/{sequence_id}...")

        data = read_data_from_json(result_file)

        for cam_id in sorted(data.keys()):
            for frame_id in sorted(data[cam_id].keys()):
                new_key = f"{subject_id}/{sequence_id}/{cam_id}/{int(frame_id):06d}"
                merged_results[new_key] = {
                    "landmarks": {
                        "right": [[-1, -1]] * 21,
                        "left": [[-1, -1]] * 21,
                    },
                    "landmarks_3d": {
                        "right": [[-1, -1, -1]] * 21,
                        "left": [[-1, -1, -1]] * 21,
                    },
                }

                out_data = data[cam_id][frame_id]
                is_right = np.array(out_data["is_right"])
                box_size = np.array(out_data["box_size"])
                box_center = np.array(out_data["box_center"])
                pred_keypoints_2d_full = np.array(out_data["pred_keypoints_2d_full"])
                pred_keypoints_3d = np.array(out_data["pred_keypoints_3d"])

                if len(is_right) != len(np.unique(is_right)):
                    # print("the same")
                    is_right = np.unique(is_right)

                for n in range(is_right.shape[0]):
                    rl = r_or_l[int(is_right[n])]


if __name__ == "__main__":

    # merge_a2j_results(PROJ_ROOT / "data/benchmarks/hand_pose/A2J_results")

    merge_hamer_results(PROJ_ROOT / "data/benchmarks/hand_pose/HaMeR_Results")

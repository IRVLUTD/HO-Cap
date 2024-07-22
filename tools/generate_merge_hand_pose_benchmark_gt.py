from _init_paths import *
from lib.utils import *
from lib.utils.common import *


BASE_DIR = Path("/metadisk/HOCap/benchmark/hand_pose")


def merge_hand_pose_benchmarks(bm_folder, save_file):
    benchmark_files = sorted(bm_folder.glob("subject_*/2023*/hand_benchmark.json"))
    merged_data = {}

    for bm_file in benchmark_files:
        subject_id = bm_file.parent.parent.name
        sequence_id = bm_file.parent.name
        print(f"Processing {subject_id}/{sequence_id}")
        data = read_data_from_json(bm_file)

        for cam_id in sorted(data.keys()):
            for frame_id in sorted(data[cam_id].keys()):
                new_key = f"{subject_id}/{sequence_id}/{cam_id}/{int(frame_id):06d}"
                landmarks_2d = data[cam_id][frame_id]["landmarks"]
                landmarks_3d = data[cam_id][frame_id]["landmarks_3d"]
                bbox = data[cam_id][frame_id]["bbox"]
                merged_data[new_key] = {
                    "landmarks_2d": landmarks_2d,
                    "landmarks_3d": landmarks_3d,
                    "bbox": bbox,
                }

    write_data_to_json(save_file, merged_data)


if __name__ == "__main__":

    # benchmark_folder_all = BASE_DIR / "gt_3d_joints"
    # benchmark_folder_small = BASE_DIR / "gt_3d_joints_small"

    # for bm_folder in [
    #     benchmark_folder_all,
    #     benchmark_folder_small,
    # ]:
    #     save_file = bm_folder / "hand_pose_benchmark_merged.json"
    #     merge_hand_pose_benchmarks(bm_folder, save_file)

    benchmark_gt_file = (
        PROJ_ROOT / "data/benchmarks/hand_pose/hand_pose_benchmark_gt_all.json"
    )

    new_pickle_file = benchmark_gt_file.with_suffix(".pkl")

    print(f"Reading {benchmark_gt_file}")
    data = read_data_from_json(benchmark_gt_file)

    # print(f"Writing {new_pickle_file}")
    # write_data_to_pkl(new_pickle_file, data)

    benchmark_keys = sorted(data.keys())
    save_file = benchmark_gt_file.parent / benchmark_gt_file.name.replace(
        ".json", "_keys.json"
    )
    print(f"Writing {save_file}")
    write_data_to_json(save_file, benchmark_keys)

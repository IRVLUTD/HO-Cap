import os

import trimesh

os.environ["PYOPENGL_PLATFORM"] = "egl"  # GPU-based offscreen rendering

import pyrender
from _init_paths import *
from lib.utils import *
from lib.utils.common import *

cvcam_in_glcam = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

release_ready_sequence_ids = (
    "subject_1/20231025_165502",
    "subject_1/20231025_165807",
    "subject_1/20231025_170105",
    "subject_1/20231025_170650",
    "subject_1/20231025_170959",
    "subject_2/20231022_200657",
    "subject_2/20231022_201316",
    "subject_2/20231022_201449",
    "subject_2/20231022_202617",
    "subject_2/20231022_203100",
    "subject_2/20231023_164741",
    "subject_2/20231023_170018",
    "subject_3/20231024_154531",
    "subject_3/20231024_154810",
    "subject_3/20231024_155008",
    "subject_3/20231024_161209",
    "subject_3/20231024_161306",
    "subject_3/20231024_162327",
    "subject_3/20231024_162409",
    "subject_4/20231026_162155",
    "subject_4/20231026_162248",
    "subject_4/20231026_163223",
    "subject_4/20231026_164131",
    "subject_4/20231026_164812",
    "subject_5/20231027_112303",
    "subject_5/20231027_113202",
    "subject_6/20231025_112332",
    "subject_6/20231025_112546",
    "subject_7/20231022_190534",
    "subject_7/20231022_192832",
    "subject_7/20231022_193506",
    "subject_7/20231022_193630",
    "subject_7/20231022_193809",
    "subject_7/20231023_162803",
    "subject_7/20231023_163653",
    "subject_8/20231024_181413",
    "subject_9/20231027_123403",
    "subject_9/20231027_123725",
    "subject_9/20231027_123814",
    "subject_9/20231027_124057",
    "subject_9/20231027_124926",
    "subject_9/20231027_125019",
)


def load_K_from_json(json_file):
    data = read_data_from_json(json_file)
    K = np.array(
        [
            [data["color"]["fx"], 0, data["color"]["ppx"]],
            [0, data["color"]["fy"], data["color"]["ppy"]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    return K


def get_render_image(
    rgb_img, obj_mesh, ob_in_cam, cam_K, znear=0.001, zfar=1000.0, alpha=0.5
):
    height, width = rgb_img.shape[:2]

    scene = pyrender.Scene(bg_color=[0, 0, 0, 1], ambient_light=[1.0, 1.0, 1.0, 1.0])

    # add camera node
    cam_node = scene.add(
        pyrender.IntrinsicsCamera(
            cam_K[0, 0],
            cam_K[1, 1],
            cam_K[0, 2],
            cam_K[1, 2],
            znear,
            zfar,
        ),
        pose=np.eye(4),
    )

    # add object node
    ob_in_glcam = cvcam_in_glcam.dot(ob_in_cam)
    mesh_node = scene.add(
        pyrender.Mesh.from_trimesh(obj_mesh),
        pose=np.eye(4),
        parent_node=cam_node,
    )
    scene.set_pose(mesh_node, ob_in_glcam)

    # render
    r = pyrender.OffscreenRenderer(width, height)
    color, _ = r.render(scene)

    r.delete()

    vis = cv2.addWeighted(rgb_img, 1 - alpha, color, alpha, 0)
    return vis


def generate_new_object_pose_results(json_file, render_folder=None):
    data = read_data_from_json(json_file)
    new_data = {}
    new_data_small = {}

    if render_folder is not None:
        render_folder.mkdir(parents=True, exist_ok=True)

    for obj_id in sorted(data.keys()):

        new_data[obj_id] = {}

        for sub_id in sorted(data[obj_id].keys()):
            for sequence_id in sorted(data[obj_id][sub_id].keys()):

                tqdm.write(f"  - Processing {obj_id} - {sub_id}...")

                for cam_id in tqdm(
                    sorted(data[obj_id][sub_id][sequence_id].keys()), ncols=60
                ):
                    for frame_id in data[obj_id][sub_id][sequence_id][cam_id]:
                        old_pose = np.array(
                            data[obj_id][sub_id][sequence_id][cam_id][frame_id],
                            dtype=np.float32,
                        )
                        img_file = (
                            f"{sub_id}/{sequence_id}/{cam_id}/color_{frame_id}.jpg"
                        )
                        stable_pose = np.loadtxt(
                            fixed_model_folder / obj_id / "stable_pose.txt"
                        ).astype(np.float32)
                        stable_pose_inv = np.linalg.inv(stable_pose)
                        new_pose = np.matmul(old_pose, stable_pose_inv)
                        new_data[obj_id][img_file] = new_pose.tolist()

                        if f"{sub_id}/{sequence_id}" in release_ready_sequence_ids:
                            if obj_id not in new_data_small:
                                new_data_small[obj_id] = {}
                            new_data_small[obj_id][img_file] = new_pose.tolist()

                        if render_folder is not None:
                            save_folder = render_folder / obj_id
                            save_folder.mkdir(parents=True, exist_ok=True)

                            K = load_K_from_json(
                                calib_folder / "intrinsics" / f"{cam_id}_640x480.json"
                            )

                            obj_mesh = trimesh.load_mesh(
                                fixed_model_folder / obj_id / "textured_mesh.obj",
                                process=False,
                            )

                            rgb_image = read_rgb_image(data_folder / img_file)

                            vis = get_render_image(
                                rgb_image,
                                obj_mesh,
                                new_pose,
                                K,
                                alpha=0.618,
                            )
                            write_rgb_image(
                                save_folder
                                / f"vis_{sub_id}-{sequence_id}-{cam_id}-{frame_id}.png",
                                vis,
                            )

    write_data_to_json(save_json_file, new_data)
    write_data_to_json(
        save_json_file.with_name(f"{save_json_file.stem}_small.json"), new_data_small
    )


if __name__ == "__main__":
    # fixed_model_folder = Path("/metadisk/HOCap/release/models").resolve()
    # calib_folder = fixed_model_folder.parent / "calibration"
    # data_folder = fixed_model_folder.parent

    # pose_file = (
    #     PROJ_ROOT
    #     # / "data/benchmarks/object_pose/FoundationPose_results_converted.json"
    #     / "data/benchmarks/object_pose/MegaPose_results-megapose-1.0-RGBD.json"
    # )
    # pose_file_name = pose_file.stem
    # render_folder = pose_file.parent / "rendered_images" / pose_file_name
    # save_json_file = pose_file.parent / f"{pose_file_name}_stable.json"

    # generate_new_object_pose_results(pose_file, render_folder)

    pose_file = (
        PROJ_ROOT
        # / "data/benchmarks/object_pose/FoundationPose_results_converted_stable.json"
        # / "data/benchmarks/object_pose/FoundationPose_results_converted_stable_small.json"
        # / "data/benchmarks/object_pose/MegaPose_results-megapose-1.0-RGBD_stable_all.json"
        # / "data/benchmarks/object_pose/MegaPose_results-megapose-1.0-RGBD_stable_small.json"
        # / "data/benchmarks/object_pose/object_pose_benchmark_gt_all.json"
        / "data/benchmarks/object_pose/object_pose_benchmark_gt_small.json"
    )

    data = read_data_from_json(pose_file)
    new_data = {}

    file_keys = {}

    for object_id in sorted(data.keys()):
        new_data[object_id] = {}
        file_keys[object_id] = []
        for img_file in sorted(data[object_id].keys()):
            # subject_id, sequence_id, cam_id, img_name = img_file.split("/")
            # frame_id = int(img_name.split(".")[0].split("_")[1])
            subject_id, sequence_id, cam_id, frame_id = img_file.split("/")
            new_key = f"{subject_id}/{sequence_id}/{cam_id}/{int(frame_id):06d}"
            new_data[object_id][new_key] = data[object_id][img_file]
            file_keys[object_id].append(new_key)

    write_data_to_json(pose_file, new_data)
    write_data_to_json(pose_file.with_name(f"{pose_file.stem}_keys.json"), file_keys)

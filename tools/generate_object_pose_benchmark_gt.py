import os


os.environ["PYOPENGL_PLATFORM"] = "egl"  # GPU-based offscreen rendering

import trimesh
import pyrender
from _init_paths import *
from lib.utils import *
from lib.utils.common import *

CALIB_ROOT = PROJ_ROOT / "data/calibration"
RELEASE_READY_ROOT = Path("/metadisk/HOCap/for_final_release").resolve()
RELEASE_DATA_ROOT = Path("/metadisk/HOCap/release").resolve()
STABLE_MODELS_ROOT = Path("/metadisk/HOCap/release/models").resolve()
OBJ_POSE_BENCHMARK_ROOT = Path("/metadisk/HOCap/benchmark/object_pose").resolve()

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


def load_intrinsics():
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

    K = {
        s: load_K_from_json(CALIB_ROOT / f"intrinsics/{s}_640x480.json")
        for s in RS_CAMERAS
    }
    return K


def load_extrinsics():
    def load_extrinsics(data):
        pose = np.array(
            [
                [data[0], data[1], data[2], data[3]],
                [data[4], data[5], data[6], data[7]],
                [data[8], data[9], data[10], data[11]],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        return pose

    data = read_data_from_json(
        CALIB_ROOT / "extrinsics/extrinsics_20231014/extrinsics.json"
    )["extrinsics"]

    tag_0 = load_extrinsics(data["tag_0"])
    tag_0_inv = np.linalg.inv(tag_0)

    tag_1 = load_extrinsics(data["tag_1"])
    tag_1_inv = np.linalg.inv(tag_1)

    extr2master = {s: load_extrinsics(data[s]) for s in RS_CAMERAS}
    extr2master_inv = {s: np.linalg.inv(extr2master[s]) for s in RS_CAMERAS}

    extr2world = {s: tag_1_inv @ extr2master[s] for s in RS_CAMERAS}
    extr2world_inv = {s: np.linalg.inv(extr2world[s]) for s in RS_CAMERAS}

    return (
        tag_0,
        tag_0_inv,
        tag_1,
        tag_1_inv,
        extr2master,
        extr2master_inv,
        extr2world,
        extr2world_inv,
    )


def project_points_to_image(points, K):
    points = points.copy()
    points = points / points[:, 2, None]
    points = points[:, :2]
    points = points @ K[:2, :2].T + K[:2, 2]
    return points


if __name__ == "__main__":
    gt_data_all = {}
    gt_data_small = {}

    K_dict = load_intrinsics()

    (
        tag_0,
        tag_0_inv,
        tag_1,
        tag_1_inv,
        extr2master,
        extr2master_inv,
        extr2world,
        extr2world_inv,
    ) = load_extrinsics()

    sequence_folders = sorted(
        [d for d in OBJ_POSE_BENCHMARK_ROOT.glob("subject_*/2023*") if d.is_dir()]
    )
    for sequence_folder in sequence_folders:
        sequence_id = sequence_folder.stem
        subject_id = sequence_folder.parent.stem
        tqdm.write(f">>>>>>>>>> Processing {subject_id}/{sequence_id} <<<<<<<<<<<<")
        meta_data = read_data_from_json(
            RELEASE_DATA_ROOT / f"{subject_id}/{sequence_id}/meta.json"
        )
        object_ids = meta_data["object_ids"]
        # tqdm.write(f"  - Object IDs: {object_ids}")
        cam_ids = sorted([d.name for d in sequence_folder.iterdir() if d.is_dir()])
        # tqdm.write(f"  - Camera IDs: {cam_ids}")

        pose_file = (
            RELEASE_DATA_ROOT
            / f"{subject_id}/{sequence_id}/processed/pose_solver/joint_pose/poses_o.npy"
        )
        if not pose_file.exists():
            pose_file = RELEASE_DATA_ROOT / f"{subject_id}/{sequence_id}/pose_o.npy"

        if not pose_file.exists():
            tqdm.write(f"  - Pose file not found: {pose_file}")
            continue

        poses_o = np.load(pose_file)
        tqdm.write(f"  - Object Poses Loaded: {poses_o.shape}")

        tqdm.write("  - Updating Object Poses...")
        for idx, object_id in enumerate(object_ids):
            poses = quat_to_mat(poses_o[idx])
            stable_pose = np.loadtxt(
                STABLE_MODELS_ROOT / object_id / "stable_pose.txt"
            ).astype(np.float32)
            stable_pose_inv = np.linalg.inv(stable_pose)
            poses = np.einsum("nij,jk->nik", poses, stable_pose_inv)
            poses_o[idx] = mat_to_quat(poses)

        tqdm.write("  - Saving Updated Object Poses...")
        # np.save(
        #     OBJ_POSE_BENCHMARK_ROOT / subject_id / sequence_id / "poses_o_stable.npy",
        #     poses_o,
        # )

        tqdm.write("  - Projecting Object Poses to Camera...")
        for cam_id in cam_ids:
            K = K_dict[cam_id]
            extr = extr2world_inv[cam_id]
            frame_ids = sorted(
                [
                    int(d.stem.split("_")[-1])
                    for d in (
                        OBJ_POSE_BENCHMARK_ROOT / subject_id / sequence_id / cam_id
                    ).glob("color_*.jpg")
                ]
            )
            # tqdm.write(f"    - Cam_ID - Frame IDs: {cam_id}\n{frame_ids}")
            for frame_id in frame_ids:
                for idx, object_id in enumerate(object_ids):
                    p_world_qt = poses_o[idx, frame_id]
                    p_world_mat = quat_to_mat(p_world_qt)
                    ob_in_cam = extr @ p_world_mat

                    data_key = f"{subject_id}/{sequence_id}/{cam_id}/{frame_id:06d}"
                    if object_id not in gt_data_all:
                        gt_data_all[object_id] = {}
                    gt_data_all[object_id][data_key] = ob_in_cam.tolist()

                    if f"{subject_id}/{sequence_id}" in release_ready_sequence_ids:
                        if object_id not in gt_data_small:
                            gt_data_small[object_id] = {}
                        gt_data_small[object_id][data_key] = ob_in_cam.tolist()

    write_data_to_json(
        PROJ_ROOT / "data/benchmarks/object_pose" / "object_pose_benchmark_gt_all.json",
        gt_data_all,
    )
    write_data_to_json(
        PROJ_ROOT
        / "data/benchmarks/object_pose"
        / "object_pose_benchmark_gt_small.json",
        gt_data_small,
    )

    exit()

    data_root = DATA_ROOT / "benchmarks/object_pose_benchmark"
    release_root = PROJ_ROOT / "release_dataset/sequences"
    subject_ids = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    for subject_id in subject_ids:
        subject_folder = data_root / subject_id
        sequence_names = sorted(
            [d.name for d in subject_folder.iterdir() if d.is_dir()]
        )
        for sequence_name in sequence_names:
            object_ids = read_data_from_json(
                subject_folder / sequence_name / "meta.json"
            )["object_ids"]
            camera_serials = sorted(
                [
                    d.name
                    for d in (subject_folder / sequence_name).iterdir()
                    if d.is_dir()
                ]
            )

            pose_file = release_root / subject_id / sequence_name / "pose_o.npy"
            pose_o = np.load(pose_file)

            for serial in camera_serials:
                K = K_dict[serial]
                extr = extr2world_inv[serial]
                frame_ids = sorted(
                    [
                        d.stem.split("_")[-1]
                        for d in (subject_folder / sequence_name / serial).glob(
                            "color_*.jpg"
                        )
                    ]
                )
                for frame_id in frame_ids:
                    for idx, obj_id in enumerate(object_ids):
                        p_world_qt = pose_o[idx, int(frame_id)]
                        p_world_mat = qt_to_mat(p_world_qt)
                        p_cam = extr @ p_world_mat
                        pose_results[obj_id][subject_id][sequence_name][serial][
                            frame_id
                        ] = p_cam.tolist()

    write_data_to_json(
        PROJ_ROOT
        / "release_dataset/benchmarks/results/object_pose"
        / "gt_object_pose_results.json",
        pose_results,
    )

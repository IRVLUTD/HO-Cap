import argparse
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from _init_paths import *
from lib.utils import *
from lib.utils.common import *
from lib.loaders import SequenceLoader

HAND_POSE_DATASET_ROOT = Path("/metadisk/HOCap/benchmark/hand_pose").resolve()
RELEASE_ROOT = Path("/metadisk/HOCap/release").resolve()
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


class HandPoseDatasetGenerator:
    def __init__(self, sequence_folder, device="cuda") -> None:
        self._device = device
        self._handpose_data_folder = Path(sequence_folder).resolve()
        subject_id = self._handpose_data_folder.parent.name
        sequence_id = self._handpose_data_folder.name

        self._data_folder = RELEASE_ROOT / subject_id / sequence_id

        self._loader = SequenceLoader(
            sequence_folder=self._data_folder,
            load_mano=True,
            device=device,
        )
        # load sequence info from loader
        self._rs_serials = self._loader.rs_serials
        self._rs_width = self._loader.rs_width
        self._rs_height = self._loader.rs_height
        self._intrinsics = self._loader.intrinsics.cpu().numpy()
        self._extrinsics = self._loader.extrinsics2world.cpu().numpy()
        self._extrinsics_inv = self._loader.extrinsics2world_inv.cpu().numpy()
        self._M = self._loader.M2world.cpu().numpy()
        self._num_frames = self._loader.num_frames
        self._mano_sides = self._loader.mano_sides
        self._num_manos = len(self._mano_sides)
        self._num_cameras = len(self._rs_serials)
        # self._poses_m = np.load(self._data_folder / "pose_m.npy")
        poses_m_file = (
            self._data_folder
            / "processed"
            / "pose_solver"
            / "joint_pose"
            / "poses_m.npy"
        )
        if not poses_m_file.exists():
            poses_m_file = (
                self._data_folder
                / "processed"
                / "pose_solver"
                / "mano_pose_all"
                / "poses_m.npy"
            )

        if not poses_m_file.exists():
            raise FileNotFoundError(f"Cannot find poses_m.npy in {self._data_folder}")

        self._poses_m = np.load(poses_m_file)

    def down_sample_frames(self, frame_ids, sample_fps=5, original_fps=30):
        interval = original_fps // sample_fps

        keys = sorted(set(f_id // interval for f_id in frame_ids))
        data_dict = {key: [] for key in keys}

        for frame_id in frame_ids:
            key = frame_id // interval
            data_dict[key].append(frame_id)

        sampled_ids = sorted(random.choice(data_dict[key]) for key in data_dict)

        return sampled_ids

    def _point_3d_to_2d(self, point_3d, M):
        point_2d = M @ np.append(point_3d, 1)
        point_2d /= point_2d[2]
        point_2d = point_2d[:2].astype(np.int64)
        # check if the point is within the image
        if (
            point_2d[0] < 0
            or point_2d[0] >= self._rs_width
            or point_2d[1] < 0
            or point_2d[1] >= self._rs_height
        ):
            return None
        return point_2d

    def _points_3d_to_2d(self, points_3d, Ms):
        points_2d = np.full((len(Ms), len(points_3d), 2), -1)
        for i, p_3d in enumerate(points_3d):
            if np.any(p_3d == -1):
                continue
            tmp_pts_2d = [self._point_3d_to_2d(p_3d, M) for M in Ms]
            # replace invalid points with -1
            points_2d[:, i] = [
                p_2d if p_2d is not None else np.array([-1, -1]) for p_2d in tmp_pts_2d
            ]
        return points_2d.astype(np.int64)

    def get_2d_landmarks(self, frame_id, serial):

        _, joints_m = self._get_mano_verts_and_joints(frame_id)
        joints_m = joints_m.cpu().numpy()

        pts_2d = [
            self._point_3d_to_2d(p_3d, self._M[self._rs_serials.index(serial)])
            for p_3d in joints_m
        ]

        pts_2d = [
            pt_2d if pt_2d is not None else np.array([-1, -1]) for pt_2d in pts_2d
        ]

        landmarks = np.full((2, 21, 2), -1)
        pts_2d = [
            self._point_3d_to_2d(p_3d, self._M[self._rs_serials.index(serial)])
            for p_3d in joints_m
        ]
        pts_2d = [
            pt_2d if pt_2d is not None else np.array([-1, -1]) for pt_2d in pts_2d
        ]
        for i, side in enumerate(self._mano_sides):
            h_id = 0 if side == "right" else 1
            landmarks[h_id] = pts_2d[i * 21 : (i + 1) * 21]

        return landmarks

    def get_rendered_mano_landmarks(self, frame_id, serial):
        _, joints_m = self._get_mano_verts_and_joints(frame_id)
        joints_m = joints_m.cpu().numpy()

        landmarks = np.full((2, 21, 2), -1)
        pts_2d = [
            self._point_3d_to_2d(p_3d, self._M[self._rs_serials.index(serial)])
            for p_3d in joints_m
        ]
        pts_2d = [
            pt_2d if pt_2d is not None else np.array([-1, -1]) for pt_2d in pts_2d
        ]
        for i, side in enumerate(self._mano_sides):
            h_id = 0 if side == "right" else 1
            landmarks[h_id] = pts_2d[i * 21 : (i + 1) * 21]

        rgb = self._loader.get_rgb_image(frame_id, serial)
        vis = draw_debug_image(rgb, hand_marks=landmarks)
        return vis

    def get_bbox_from_2d_landmarks(self, landmarks):
        if np.all(landmarks == -1):
            return np.array([-1, -1, -1, -1])
        box = get_bbox_from_landmarks(
            landmarks, self._rs_width, self._rs_height, margin=5
        )
        return box

    def _get_mano_verts_and_joints(self, frame_id):
        p = [
            torch.from_numpy(self._poses_m[0 if side == "right" else 1, frame_id])
            .unsqueeze(0)
            .to(self._device)
            for side in self._mano_sides
        ]
        verts, joints = self._loader.mano_group_layer_forward(p)
        return verts, joints

    def _mano_3d_joints_world_to_camera(self, landmarks_3d, T_matrix):
        pts_3d = landmarks_3d.copy().reshape(-1, 3)
        # convert to camera coordinate
        pts_3d = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
        pts_3d = np.dot(T_matrix, pts_3d.T).T
        pts_3d = pts_3d[:, :3] / pts_3d[:, 3:]
        return pts_3d

    def get_3d_landmarks_in_camera(self, frame_id, serial):
        _, joints_m = self._get_mano_verts_and_joints(frame_id)
        joints_m = joints_m.cpu().numpy()
        T_matrix = self._extrinsics_inv[self._rs_serials.index(serial)]
        joints_3d = self._mano_3d_joints_world_to_camera(joints_m, T_matrix)
        landmarks_3d = np.full((2, 21, 3), -1.0)
        for i, side in enumerate(self._mano_sides):
            h_id = 0 if side == "right" else 1
            landmarks_3d[h_id] = joints_3d[i * 21 : (i + 1) * 21]
        return landmarks_3d

    def draw_project_3d_points_to_image(self, rgb_image, points_3d, K):
        height, width, _ = rgb_image.shape
        points_2d = np.dot(K, points_3d.T).T
        points_2d = points_2d[:, :2] / points_2d[:, 2:]
        points_2d = points_2d.astype(np.int32)
        points_2d = points_2d[
            (points_2d[:, 0] >= 0)
            & (points_2d[:, 0] < width)
            & (points_2d[:, 1] >= 0)
            & (points_2d[:, 1] < height)
        ]
        for point in points_2d:
            cv2.circle(rgb_image, tuple(point), 3, (0, 255, 0), -1)
        return rgb_image

    def run(self):
        subject_id = self._data_folder.parent.name
        sequence_id = self._data_folder.name

        target_folder = (
            HAND_POSE_DATASET_ROOT
            / "hand_pose_benchmark_gt"
            / subject_id
            / sequence_id
            # HAND_POSE_DATASET_ROOT
            # / "gt_3d_joints_small"
            # / subject_id
            # / sequence_id
        )
        make_clean_folder(target_folder)

        # ========== generate hand benchmark data
        tqdm.write("- generating Hand benchmark info...")
        benchmark_info = {}
        for serial in self._rs_serials:
            tqdm.write(f"  ** {serial}...")

            camera_folder = self._handpose_data_folder / serial
            sampled_ids = sorted(
                int(x.stem.split("_")[-1]) for x in (camera_folder).glob("color_*.jpg")
            )

            benchmark_info[serial] = {}

            for frame_id in tqdm(sampled_ids, ncols=60):
                data_elem = {
                    "landmarks": {"right": None, "left": None},
                    "bbox": {"right": None, "left": None},
                }
                landmarks = self.get_2d_landmarks(frame_id, serial)
                landmarks_3d = self.get_3d_landmarks_in_camera(frame_id, serial)
                data_elem["landmarks_3d"] = {
                    "right": landmarks_3d[0].tolist(),
                    "left": landmarks_3d[1].tolist(),
                }
                for i, marks in enumerate(landmarks):
                    side = "right" if i == 0 else "left"
                    bbox = self.get_bbox_from_2d_landmarks(marks)
                    data_elem["landmarks"][side] = marks.tolist()
                    data_elem["bbox"][side] = bbox.tolist()
                benchmark_info[serial][frame_id] = data_elem
        with open(target_folder / "hand_benchmark.json", "w") as f:
            json.dump(benchmark_info, f)


def args_parser():
    parser = argparse.ArgumentParser(description="Hand pose dataset generator")
    parser.add_argument(
        "--sequence_folder",
        type=str,
        required=True,
        help="The folder of the sequence to process",
    )
    parser.add_argument(
        "--sample_fps",
        type=int,
        default=10,
        help="The target frame rate of sampled sequence",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # args = args_parser()
    # generator = HandPoseDatasetGenerator(
    # sequence_folder = args.sequence_folder,
    #     sample_fps=args.sample_fps,
    # )
    # generator.run()
    # tqdm.write(f">>>>>>>>>> Processing {subject_id}/{sequence_id} <<<<<<<<<<<<")
    # generator = HandPoseDatasetGenerator(sequence_folder)
    # generator.run()

    all_sequence_folders = sorted(
        [f for f in (HAND_POSE_DATASET_ROOT.glob("subject_*/2023*")) if f.is_dir()]
    )
    for sequence_folder in all_sequence_folders:
        subject_id = sequence_folder.parent.name
        sequence_id = sequence_folder.name

        tqdm.write(f">>>>>>>>>> Processing {subject_id}/{sequence_id} <<<<<<<<<<<<")

        if f"{subject_id}/{sequence_id}" not in release_ready_sequence_ids:
            tqdm.write(f"- Skipping {subject_id}/{sequence_id}...")
            continue

        generator = HandPoseDatasetGenerator(sequence_folder)
        generator.run()

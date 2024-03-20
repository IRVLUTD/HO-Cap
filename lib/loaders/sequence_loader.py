import numpy as np
import torch
import trimesh
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from ..utils import *
from ..utils.common import (
    init_logger,
    read_data_from_json,
    read_rgb_image,
    read_depth_image,
    read_mask_image,
    erode_mask,
)


class SequenceLoader:
    def __init__(
        self,
        sequence_folder,
        load_mano=False,
        load_object=False,
        in_world=True,
        preload=False,
        device="cuda",
        debug=False,
    ):
        self._data_folder = Path(sequence_folder).resolve()
        self._calib_folder = self._data_folder.parents[1] / "calibration"
        self._models_folder = self._data_folder.parents[1] / "models"
        self._load_mano = load_mano
        self._load_object = load_object
        self._in_world = in_world
        self._preload = preload
        self._device = device
        self._debug = debug

        self._logger = init_logger(
            log_level="DEBUG" if debug else "INFO", name="SequenceLoader"
        )
        # crop limits in world frame, [x_min, x_max, y_min, y_max, z_min, z_max]
        self._crop_lim = [-0.60, +0.60, -0.35, +0.35, -0.01, +0.80]

        # load metadata
        self._load_metadata()

        # load intrinsics
        self._load_intrinsics()

        # load extrinsics
        self._load_extrinsics()

        # load mano beta
        self._mano_beta = torch.from_numpy(self._load_mano_beta()).to(self._device)

        # load mano group layer
        self._mano_group_layer = self._init_mano_group_layer()

        # load object group layer
        self._object_group_layer = self._init_object_group_layer()

        # create mapping from 2D coordinates to 3D rays
        self._Rays = self._create_3d_rays()  # [N, 3, H*W]

        # create projection matrices from camera to master/world
        self._M_master = torch.bmm(self._K, self._extr2master_inv[:, :3, :])
        self._M_world = torch.bmm(self._K, self._extr2world_inv[:, :3, :])

        # load images to memory
        self._color_images = self._load_color_images() if self._preload else None
        self._depth_images = self._load_depth_images() if self._preload else None
        self._mask_images = self._load_mask_images() if self._preload else None

        # initialize
        self._frame_id = -1
        self._points = torch.zeros(
            (self.num_cameras, self._rs_height * self._rs_width, 3),
            dtype=torch.float32,
            device=self._device,
        )
        self._colors = torch.zeros(
            (self.num_cameras, self._rs_height * self._rs_width, 3),
            dtype=torch.float32,
            device=self._device,
        )
        self._masks = torch.zeros(
            (self.num_cameras, self._rs_height * self._rs_width),
            dtype=torch.bool,
            device=self._device,
        )

    def _load_color_images(self):
        self._logger.info("Loading color images...")
        images = [[None] * self._num_frames] * self._num_cameras
        workers = []
        tqbar = tqdm(total=self._num_frames * self._num_cameras, ncols=60)
        with ThreadPoolExecutor() as executor:
            for c_idx, serial in enumerate(self._rs_serials):
                for f_idx in range(self._num_frames):
                    workers.append(
                        executor.submit(
                            read_rgb_image,
                            self._data_folder / f"{serial}/color_{f_idx:06d}.jpg",
                            idx=(c_idx, f_idx),
                        )
                    )
        for worker in workers:
            img, (c_idx, f_idx) = worker.result()
            images[c_idx][f_idx] = img
            tqbar.update(1)
            tqbar.refresh()
        tqbar.close()

        return images

    def _load_depth_images(self):
        self._logger.info("Loading depth images...")
        images = [[None] * self._num_frames] * self._num_cameras
        workers = []
        tqbar = tqdm(total=self._num_frames * self._num_cameras, ncols=60)
        with ThreadPoolExecutor() as executor:
            for c_idx, serial in enumerate(self._rs_serials):
                for f_idx in range(self._num_frames):
                    workers.append(
                        executor.submit(
                            read_depth_image,
                            self._data_folder / f"{serial}/depth_{f_idx:06d}.png",
                            idx=(c_idx, f_idx),
                        )
                    )
        for worker in workers:
            img, (c_idx, f_idx) = worker.result()
            images[c_idx][f_idx] = img
            tqbar.update(1)
            tqbar.refresh()
        tqbar.close()

        return images

    def _load_mask_images(self):
        self._logger.info("Loading mask images...")
        images = [[None] * self._num_frames] * self._num_cameras
        workers = []
        tqbar = tqdm(total=self._num_frames * self._num_cameras, ncols=60)
        with ThreadPoolExecutor() as executor:
            for c_idx, serial in enumerate(self._rs_serials):
                for f_idx in range(self._num_frames):
                    workers.append(
                        executor.submit(
                            read_mask_image,
                            self._data_folder / f"{serial}/mask_{f_idx:06d}.png",
                            idx=(c_idx, f_idx),
                        )
                    )
        for worker in workers:
            img, (c_idx, f_idx) = worker.result()
            images[c_idx][f_idx] = img
            tqbar.update(1)
            tqbar.refresh()
        tqbar.close()

        return images

    def _load_metadata(self):
        data = read_data_from_json(self._data_folder / "meta.json")
        # sequnece
        self._num_frames = data["num_frames"]
        self._object_ids = data["object_ids"]
        self._mano_sides = data["mano_sides"]
        self._task_id = data["task_id"]
        # realsense
        self._rs_serials = data["realsense"]["serials"]
        self._rs_width = data["realsense"]["width"]
        self._rs_height = data["realsense"]["height"]
        self._num_cameras = len(self._rs_serials)
        # hololens
        self._hl_serials = data["hololens"]["serials"]
        self._hl_depth_mode = data["hololens"]["depth_mode"]
        self._hl_pv_width = data["hololens"]["pv_width"]
        self._hl_pv_height = data["hololens"]["pv_height"]
        self._hl_depth_width = data["hololens"]["depth_width"]
        self._hl_depth_height = data["hololens"]["depth_height"]
        self._num_hololens = len(self._hl_serials)
        # calibration
        self._extr_file = (
            self._calib_folder
            / "extrinsics"
            / data["calibration"]["extrinsics"]
            / "extrinsics.json"
        )
        self._subject_id = data["calibration"]["mano"]
        self._mano_file = self._calib_folder / "mano" / self._subject_id / "mano.json"

    def _load_intrinsics(self):
        def read_K_from_json(serial):
            json_path = (
                self._calib_folder
                / "intrinsics"
                / f"{serial}_{self._rs_width}x{self._rs_height}.json"
            )
            data = read_data_from_json(json_path)
            K = np.array(
                [
                    [data["color"]["fx"], 0.0, data["color"]["ppx"]],
                    [0.0, data["color"]["fy"], data["color"]["ppy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            return K

        self._K = np.stack(
            [read_K_from_json(serial) for serial in self._rs_serials], axis=0
        )
        self._K_inv = np.stack([np.linalg.inv(K) for K in self._K], axis=0)

        self._K = torch.from_numpy(self._K).to(self._device)
        self._K_inv = torch.from_numpy(self._K_inv).to(self._device)

    def _load_extrinsics(self):
        data = read_data_from_json(self._extr_file)
        self._rs_master = data["rs_master"]
        tag_0 = np.array(
            [
                [
                    data["extrinsics"]["tag_0"][0],
                    data["extrinsics"]["tag_0"][1],
                    data["extrinsics"]["tag_0"][2],
                    data["extrinsics"]["tag_0"][3],
                ],
                [
                    data["extrinsics"]["tag_0"][4],
                    data["extrinsics"]["tag_0"][5],
                    data["extrinsics"]["tag_0"][6],
                    data["extrinsics"]["tag_0"][7],
                ],
                [
                    data["extrinsics"]["tag_0"][8],
                    data["extrinsics"]["tag_0"][9],
                    data["extrinsics"]["tag_0"][10],
                    data["extrinsics"]["tag_0"][11],
                ],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        tag_0_inv = np.linalg.inv(tag_0)

        tag_1 = np.array(
            [
                [
                    data["extrinsics"]["tag_1"][0],
                    data["extrinsics"]["tag_1"][1],
                    data["extrinsics"]["tag_1"][2],
                    data["extrinsics"]["tag_1"][3],
                ],
                [
                    data["extrinsics"]["tag_1"][4],
                    data["extrinsics"]["tag_1"][5],
                    data["extrinsics"]["tag_1"][6],
                    data["extrinsics"]["tag_1"][7],
                ],
                [
                    data["extrinsics"]["tag_1"][8],
                    data["extrinsics"]["tag_1"][9],
                    data["extrinsics"]["tag_1"][10],
                    data["extrinsics"]["tag_1"][11],
                ],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        tag_1_inv = np.linalg.inv(tag_1)

        extr2master = np.stack(
            [
                np.array(
                    [
                        [
                            data["extrinsics"][s][0],
                            data["extrinsics"][s][1],
                            data["extrinsics"][s][2],
                            data["extrinsics"][s][3],
                        ],
                        [
                            data["extrinsics"][s][4],
                            data["extrinsics"][s][5],
                            data["extrinsics"][s][6],
                            data["extrinsics"][s][7],
                        ],
                        [
                            data["extrinsics"][s][8],
                            data["extrinsics"][s][9],
                            data["extrinsics"][s][10],
                            data["extrinsics"][s][11],
                        ],
                        [0, 0, 0, 1],
                    ],
                    dtype=np.float32,
                )
                for s in self._rs_serials
            ],
            axis=0,
        )
        extr2master_inv = np.stack([np.linalg.inv(t) for t in extr2master], axis=0)
        extr2world = np.stack([tag_1_inv @ t for t in extr2master], axis=0)
        extr2world_inv = np.stack([np.linalg.inv(t) for t in extr2world], axis=0)

        self._tag_0 = torch.from_numpy(tag_0).to(self._device)
        self._tag_0_inv = torch.from_numpy(tag_0_inv).to(self._device)
        self._tag_1 = torch.from_numpy(tag_1).to(self._device)
        self._tag_1_inv = torch.from_numpy(tag_1_inv).to(self._device)
        self._extr2master = torch.from_numpy(extr2master).to(self._device)
        self._extr2master_inv = torch.from_numpy(extr2master_inv).to(self._device)
        self._extr2world = torch.from_numpy(extr2world).to(self._device)
        self._extr2world_inv = torch.from_numpy(extr2world_inv).to(self._device)

    def _load_mano_beta(self):
        data = read_data_from_json(self._mano_file)
        return np.array(data["betas"], dtype=np.float32)

    def _create_3d_rays(self):
        def create_2d_coords():
            xv, yv = torch.meshgrid(
                torch.arange(self._rs_width),
                torch.arange(self._rs_height),
                indexing="xy",
            )
            coord_2d = torch.stack(
                (xv, yv, torch.ones_like(xv)), dim=0
            ).float()  # (3, H, W)
            coords_2d = (
                coord_2d.unsqueeze(0)
                .repeat(self._num_cameras, 1, 1, 1)
                .view(self._num_cameras, 3, -1)
            )  # (N, 3, H*W)
            coords_2d = coords_2d.to(self._device)
            return coords_2d

        coords_2d = create_2d_coords()
        return torch.bmm(self._K_inv, coords_2d)

    def _init_mano_group_layer(self):
        if not self._load_mano:
            return None
        from ..layers import MANOGroupLayer

        mano_group_layer = MANOGroupLayer(
            self._mano_sides, [self._load_mano_beta() for _ in self._mano_sides]
        ).to(self._device)
        return mano_group_layer

    def _init_object_group_layer(self):
        if not self._load_object:
            return None
        from ..layers import ObjectGroupLayer

        verts, faces, norms = [], [], []
        for obj_file in self.object_cleaned_mesh_files:
            m = trimesh.load(obj_file, process=False)
            verts.append(m.vertices)
            faces.append(m.faces)
            norms.append(m.vertex_normals)
        object_group_layer = ObjectGroupLayer(verts, faces, norms).to(self._device)
        return object_group_layer

    def _deproject(self, colors, depths, depth_scale=1000.0):
        """Deproject depth images to 3D points.

        Args:
            colors (list): color images, [N, H, W, 3], dtype=uint8
            depths (list): depth images, [N, H, W], dtype=uint16
            min_depth (float, optional): minimum depth. Defaults to 0.1.
            max_depth (float, optional): maximum depth. Defaults to 1.4.

        Returns:
            torch.Tensor: 3D points, [N, H, W, 3]
        """
        # process color images
        colors = (
            np.stack(colors, axis=0, dtype=np.float32).reshape(self._num_cameras, -1, 3)
            / 255.0
        )  # [N, H*W, 3]
        colors = torch.from_numpy(colors).to(self._device)

        # process depth images
        depths = (
            np.stack(depths, axis=0, dtype=np.float32).reshape(self._num_cameras, 1, -1)
            / depth_scale
        )  # [N, 1, H*W]
        depths = torch.from_numpy(depths).to(self._device)

        # deproject depth images to 3D points in camera frame
        pts_c = self._Rays * depths  #  [N, 3, H*W]
        # transform 3D points from camera frame to world frame
        pts = torch.baddbmm(
            self._extr2world[:, :3, 3].unsqueeze(2),
            self._extr2world[:, :3, :3],
            pts_c,
        ).permute(
            0, 2, 1
        )  # (N, H*W, 3)

        # crop 3D points
        mx1 = pts[..., 0] > self._crop_lim[0]
        mx2 = pts[..., 0] < self._crop_lim[1]
        my1 = pts[..., 1] > self._crop_lim[2]
        my2 = pts[..., 1] < self._crop_lim[3]
        mz1 = pts[..., 2] > self._crop_lim[4]
        mz2 = pts[..., 2] < self._crop_lim[5]
        masks = mx1 & mx2 & my1 & my2 & mz1 & mz2

        # transform 3D points from world frame to master frame if necessary
        if not self._in_world:
            pts = torch.baddbmm(
                self._extr2master[:, :3, 3].unsqueeze(2),
                self._extr2master[:, :3, :3],
                pts_c,
            ).permute(
                0, 2, 1
            )  # [N, H*W, 3]

        return colors, pts, masks

    def _update_pcd(self, frame_id):
        colors, points, masks = self._deproject(
            self.get_rgb_image(frame_id), self.get_depth_image(frame_id)
        )
        self._points.copy_(points)
        self._colors.copy_(colors)
        self._masks.copy_(masks)

    def get_rgb_image(self, frame_id, serial=None):
        """Get RGB image in numpy format, dtype=uint8, [H, W, 3]"""
        if serial is None:
            if self._preload:
                data = [
                    self._color_images[i][frame_id] for i in range(self._num_cameras)
                ]
            else:
                data = [None] * self._num_cameras
                with ThreadPoolExecutor() as executor:
                    workers = [
                        executor.submit(
                            read_rgb_image,
                            self._data_folder / f"{s}/color_{frame_id:06d}.jpg",
                            idx=i,
                        )
                        for i, s in enumerate(self._rs_serials)
                    ]
                    for worker in workers:
                        img, idx = worker.result()
                        data[idx] = img
        else:
            if self._preload:
                data = self._color_images[self._rs_serials.index(serial)][frame_id]
            else:
                data = read_rgb_image(
                    self._data_folder / f"{serial}/color_{frame_id:06d}.jpg"
                )
        return data

    def get_depth_image(self, frame_id, serial=None):
        """Get depth image in numpy format, dtype=uint16, [H, W]"""
        if serial is None:
            if self._preload:
                data = [
                    self._depth_images[i][frame_id] for i in range(self._num_cameras)
                ]
            else:
                data = [None] * self._num_cameras
                with ThreadPoolExecutor() as executor:
                    workers = [
                        executor.submit(
                            read_depth_image,
                            self._data_folder / f"{s}/depth_{frame_id:06d}.png",
                            idx=i,
                        )
                        for i, s in enumerate(self._rs_serials)
                    ]
                    for worker in workers:
                        img, idx = worker.result()
                        data[idx] = img
        else:
            if self._preload:
                data = self._depth_images[self._rs_serials.index(serial)][frame_id]
            else:
                data = read_depth_image(
                    self._data_folder / f"{serial}/depth_{frame_id:06d}.png"
                )
        return data

    def get_mask_image(self, frame_id, serial=None, erode_kernel=0):
        """Get mask image in numpy format, dtype=uint8, [H, W]"""
        if serial is None:
            if self._preload:
                data = [self._masks[i][frame_id] for i in range(self._num_cameras)]
            else:
                data = [None for _ in self._rs_serials]
                workers = []
                with ThreadPoolExecutor() as executor:
                    for i, s in enumerate(self._rs_serials):
                        mask_file = self._data_folder / f"{s}/mask_{frame_id:06d}.png"
                        if mask_file.exists():
                            workers.append(
                                executor.submit(
                                    read_mask_image,
                                    mask_file,
                                    idx=i,
                                )
                            )

                        else:
                            data[i] = np.zeros(
                                (self.rs_height, self.rs_width), dtype=np.uint8
                            )
                    for worker in workers:
                        img, idx = worker.result()
                        data[idx] = erode_mask(img, erode_kernel)
        else:
            if self._preload:
                data = self._masks[self._rs_serials.index(serial)][frame_id]
            else:
                mask_file = self._data_folder / f"{serial}/mask_{frame_id:06d}.png"
                if mask_file.exists():
                    data = read_mask_image(mask_file)
                else:
                    data = np.zeros((self.rs_height, self.rs_width), dtype=np.uint8)
        return data

    def object_group_layer_forward(self, poses, subset=None):
        p = torch.cat(poses, dim=1)
        v, n = self._object_group_layer(p, inds=subset)
        v = v[0]
        n = n[0]
        return v, n

    def mano_group_layer_forward(self, poses, subset=None):
        p = torch.cat(poses, dim=1)
        v, j = self._mano_group_layer(p, inds=subset)
        v = v[0]
        j = j[0]
        return v, j

    def load_object_seg_masks(self):
        """Load object segmentation masks.

        Returns:
            List of segmentation masks, NUM_FRAMES * [NUM_CAMS, H, W], dtype=uint8
        """
        seg_masks = [None for _ in range(self.num_cameras)] * self.num_frames
        workers = []
        with ThreadPoolExecutor() as executor:
            for f_idx in range(self.num_frames):
                for c_idx, serial in enumerate(self._rs_serials):
                    mask_file = self._data_folder / f"{serial}/mask_{f_idx:06d}.png"
                    if mask_file.exists():
                        workers.append(
                            executor.submit(
                                read_mask_image, mask_file, idx=(f_idx, c_idx)
                            )
                        )
                    else:
                        seg_masks[f_idx][c_idx] = np.zeros(
                            (self.rs_height, self.rs_width), dtype=np.uint8
                        )
        for worker in workers:
            img, (f_idx, c_idx) = worker.result()
            seg_masks[f_idx][c_idx] = img
        seg_masks = np.stack(seg_masks, axis=0)
        return seg_masks

    def step(self):
        """Step to the next frame."""
        self._frame_id = (self._frame_id + 1) % self._num_frames
        self._update_pcd(self._frame_id)

    def step_by_frame_id(self, frame_id):
        frame_id = frame_id % self._num_frames
        self._update_pcd(frame_id)

    @property
    def sequence_folder(self):
        return str(self._data_folder)

    @property
    def load_mano(self):
        return self._load_mano

    @property
    def load_object(self):
        return self._load_object

    @property
    def in_world(self):
        return self._in_world

    @property
    def device(self):
        return self._device

    @property
    def object_ids(self):
        return self._object_ids

    @property
    def group_id(self):
        return self._object_ids[0].split("_")[0]

    @property
    def subject_id(self):
        return self._subject_id

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def rs_width(self):
        return self._rs_width

    @property
    def rs_height(self):
        return self._rs_height

    @property
    def rs_serials(self):
        return self._rs_serials

    @property
    def rs_master(self):
        return self._rs_master

    @property
    def num_cameras(self):
        return self._num_cameras

    @property
    def holo_serials(self):
        return self._hl_serials

    @property
    def holo_pv_width(self):
        return self._hl_pv_width

    @property
    def holo_pv_height(self):
        return self._hl_pv_height

    @property
    def holo_depth_width(self):
        return self._hl_depth_width

    @property
    def holo_depth_height(self):
        return self._hl_depth_height

    @property
    def holo_depth_mode(self):
        return self._hl_depth_mode

    @property
    def mano_beta(self):
        return self._mano_beta

    @property
    def mano_sides(self):
        return self._mano_sides

    @property
    def intrinsics(self):
        return self._K

    @property
    def intrinsics_inv(self):
        return self._K_inv

    @property
    def extrinsics2master(self):
        return self._extr2master

    @property
    def extrinsics2master_inv(self):
        return self._extr2master_inv

    @property
    def extrinsics2world(self):
        return self._extr2world

    @property
    def extrinsics2world_inv(self):
        return self._extr2world_inv

    @property
    def tag_0(self):
        """tag_0 to rs_master transformation matrix"""
        return self._tag_0

    @property
    def tag_0_inv(self):
        """rs_master to tag_0 transformation matrix"""
        return self._tag_0_inv

    @property
    def tag_1(self):
        """tag_1 to rs_master transformation matrix"""
        return self._tag_1

    @property
    def tag_1_inv(self):
        """rs_master to tag_1 transformation matrix"""
        return self._tag_1_inv

    @property
    def M2master(self):
        """camera to master transformation matrix"""
        return self._M_master

    @property
    def M2world(self):
        """camera to world transformation matrix"""
        return self._M_world

    @property
    def frame_id(self):
        return self._frame_id

    @property
    def mano_group_layer(self):
        return self._mano_group_layer

    @property
    def object_group_layer(self):
        return self._object_group_layer

    @property
    def object_textured_mesh_files(self):
        return [
            str(self._models_folder / f"{object_id}/textured_mesh.obj")
            for object_id in self._object_ids
        ]

    @property
    def object_cleaned_mesh_files(self):
        return [
            str(self._models_folder / f"{object_id}/cleaned_mesh_10000.obj")
            for object_id in self._object_ids
        ]

    @property
    def points(self):
        return self._points

    @property
    def colors(self):
        return self._colors

    @property
    def masks(self):
        return self._masks

    @property
    def points_map(self):
        return self._points.view(self._num_cameras, self._rs_height, self._rs_width, 3)

    @property
    def colors_map(self):
        return self._colors.view(self._num_cameras, self._rs_height, self._rs_width, 3)

    @property
    def masks_map(self):
        return self._masks.view(self._num_cameras, self._rs_height, self._rs_width)

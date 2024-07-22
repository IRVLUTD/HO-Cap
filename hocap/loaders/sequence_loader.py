from ..utils import *


class SequenceLoader:
    """Class for loading and processing sequence data."""

    def __init__(
        self,
        sequence_folder,
        load_mano=False,
        load_object=False,
        in_world=True,
        device="cpu",
    ):
        self._data_folder = Path(sequence_folder).resolve()
        self._calib_folder = self._data_folder.parent.parent / "calibration"
        self._models_folder = self._data_folder.parent.parent / "models"
        self._device = device
        self._load_mano = load_mano
        self._load_object = load_object
        self._in_world = in_world

        # Crop limits in world frame, [x_min, x_max, y_min, y_max, z_min, z_max]
        self._crop_lim = [-0.60, +0.60, -0.35, +0.35, -0.01, +0.80]

        # Load metadata
        self._load_metadata()

        # Load MANO group layer
        self._mano_group_layer = self._init_mano_group_layer()

        # Load object group layer
        self._object_group_layer = self._init_object_group_layer()

        # Create mapping from 2D coordinates to 3D rays
        self._rays = self._create_3d_rays()

        # Create projection matrices from camera to master/world
        self._M2master = torch.bmm(self._K, self._extr2master_inv[:, :3, :])
        self._M2world = torch.bmm(self._K, self._extr2world_inv[:, :3, :])

        # Initialize points, colors, and masks
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

    def _load_metadata(self):
        data = read_data_from_json(self._data_folder / "meta.json")

        self._num_frames = data["num_frames"]
        self._object_ids = data["object_ids"]
        self._mano_sides = data["mano_sides"]
        self._task_id = data["task_id"]

        # RealSense camera metadata
        self._rs_serials = data["realsense"]["serials"]
        self._rs_width = data["realsense"]["width"]
        self._rs_height = data["realsense"]["height"]
        self._num_cams = len(self._rs_serials)

        # HoloLens metadata
        self._hl_serials = data["hololens"]["serials"]
        self._hl_depth_mode = data["hololens"]["depth_mode"]
        self._hl_pv_width = data["hololens"]["pv_width"]
        self._hl_pv_height = data["hololens"]["pv_height"]
        self._hl_depth_width = data["hololens"]["depth_width"]
        self._hl_depth_height = data["hololens"]["depth_height"]
        self._num_hololens = len(self._hl_serials)

        # Calibration metadata
        self._subject_id = data["calibration"]["mano"]

        # Load intrinsics
        self._load_intrinsics(self._rs_serials)

        # Load extrinsics
        self._load_extrinsics(
            self._calib_folder
            / "extrinsics"
            / data["calibration"]["extrinsics"]
            / "extrinsics.json"
        )

        # Load MANO beta
        self._mano_beta = self._load_mano_beta(
            self._calib_folder / "mano" / self._subject_id / "mano.json"
        )

    def _load_intrinsics(self, serials: list):
        """Load camera intrinsics from JSON files."""

        def read_K_from_json(json_file):
            data = read_data_from_json(json_file)
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
            [
                read_K_from_json(
                    self._calib_folder
                    / "intrinsics"
                    / f"{serial}_{self._rs_width}x{self._rs_height}.json"
                )
                for serial in serials
            ],
            axis=0,
        )
        self._K_inv = np.stack([np.linalg.inv(K) for K in self._K], axis=0)

        self._K = torch.from_numpy(self._K).to(self._device)
        self._K_inv = torch.from_numpy(self._K_inv).to(self._device)

    def _load_extrinsics(self, extr_file: str):
        """Load camera extrinsics from a JSON file."""
        data = read_data_from_json(extr_file)
        self._rs_master = data["rs_master"]
        tag_0 = np.array(
            [
                data["extrinsics"]["tag_0"][:4],
                data["extrinsics"]["tag_0"][4:8],
                data["extrinsics"]["tag_0"][8:12],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        tag_0_inv = np.linalg.inv(tag_0)

        tag_1 = np.array(
            [
                data["extrinsics"]["tag_1"][:4],
                data["extrinsics"]["tag_1"][4:8],
                data["extrinsics"]["tag_1"][8:12],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        tag_1_inv = np.linalg.inv(tag_1)

        extr2master = np.stack(
            [
                np.array(
                    [
                        data["extrinsics"][s][:4],
                        data["extrinsics"][s][4:8],
                        data["extrinsics"][s][8:12],
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

    def _load_mano_beta(self, mano_file) -> torch.Tensor:
        """Load MANO beta values from a JSON file."""
        data = read_data_from_json(mano_file)
        return torch.tensor(data["betas"], dtype=torch.float32, device=self._device)

    def _create_3d_rays(self) -> torch.Tensor:
        """Create 3D rays for deprojecting depth images."""

        def create_2d_coords() -> torch.Tensor:
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
                .repeat(self._num_cams, 1, 1, 1)
                .view(self._num_cams, 3, -1)
            )  # (N, 3, H*W)
            coords_2d = coords_2d.to(self._device)
            return coords_2d

        coords_2d = create_2d_coords()
        return torch.bmm(self._K_inv, coords_2d)  # (N, 3, H*W)

    def _init_mano_group_layer(self):
        """Initialize the MANO group layer."""
        if not self._load_mano:
            return None
        from ..layers import MANOGroupLayer

        mano_group_layer = MANOGroupLayer(
            self._mano_sides, [self._mano_beta.cpu().numpy()] * len(self._mano_sides)
        ).to(self._device)
        return mano_group_layer

    def _init_object_group_layer(self):
        """Initialize the object group layer."""
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

    def _deproject(self, colors, depths, depth_scale=1000.0) -> tuple:
        """Deproject depth images to 3D points.

        Args:
            colors (list): List of color images, [N, H, W, 3], dtype=uint8.
            depths (list): List of depth images, [N, H, W], dtype=uint16.
            depth_scale (float, optional): Depth scale factor. Defaults to 1000.0.

        Returns:
            tuple: Colors, 3D points, and masks.
        """
        # Process color images
        colors = (
            torch.from_numpy(
                np.stack(colors, axis=0, dtype=np.float32).reshape(
                    self._num_cams, -1, 3
                )
            ).to(self._device)
            / 255.0
        )  # [N, H*W, 3]

        # Process depth images
        depths = (
            torch.from_numpy(
                np.stack(depths, axis=0, dtype=np.float32).reshape(
                    self._num_cams, 1, -1
                )
            ).to(self._device)
            / depth_scale
        )  # [N, 1, H*W]

        # Deproject depth images to 3D points in camera frame
        pts_c = self._rays * depths  #  [N, 3, H*W]
        # Transform 3D points from camera frame to world frame
        pts = torch.baddbmm(
            self._extr2world[:, :3, 3].unsqueeze(2),
            self._extr2world[:, :3, :3],
            pts_c,
        ).permute(
            0, 2, 1
        )  # (N, H*W, 3)

        # Crop 3D points
        mx1 = pts[..., 0] > self._crop_lim[0]
        mx2 = pts[..., 0] < self._crop_lim[1]
        my1 = pts[..., 1] > self._crop_lim[2]
        my2 = pts[..., 1] < self._crop_lim[3]
        mz1 = pts[..., 2] > self._crop_lim[4]
        mz2 = pts[..., 2] < self._crop_lim[5]
        masks = mx1 & mx2 & my1 & my2 & mz1 & mz2

        # Transform 3D points from world frame to master frame if necessary
        if not self._in_world:
            pts = torch.baddbmm(
                self._extr2master[:, :3, 3].unsqueeze(2),
                self._extr2master[:, :3, :3],
                pts_c,
            ).permute(
                0, 2, 1
            )  # [N, H*W, 3]

        return colors, pts, masks

    def _update_pcd(self, frame_id: int):
        """Update point cloud data."""
        colors, points, masks = self._deproject(
            self.get_rgb_image(frame_id), self.get_depth_image(frame_id)
        )
        self._points.copy_(points)
        self._colors.copy_(colors)
        self._masks.copy_(masks)

    def _read_mask_image(self, mask_file, erode_kernel=0, idx=None) -> np.ndarray:
        """Read mask image from file."""
        if mask_file.exists():
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if erode_kernel > 0:
                mask = erode_mask(mask, erode_kernel)
        else:
            mask = np.zeros((self._rs_height, self._rs_width), dtype=np.uint8)
        return mask if idx is None else (mask, idx)

    def get_rgb_image(self, frame_id: int, serial: str = None) -> np.ndarray:
        """Get RGB image in numpy format, dtype=uint8, [H, W, 3]."""
        if serial is None:
            data = [None] * self._num_cams
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        read_rgb_image,
                        self._data_folder / s / f"color_{frame_id:06d}.jpg",
                    ): i
                    for i, s in enumerate(self._rs_serials)
                }
                for future in concurrent.futures.as_completed(futures):
                    data[futures[future]] = future.result()
        else:
            data = read_rgb_image(
                self._data_folder / serial / f"color_{frame_id:06d}.jpg"
            )
        return data

    def get_depth_image(self, frame_id: int, serial: str = None) -> np.ndarray:
        """Get depth image in numpy format, dtype=uint16, [H, W]."""
        if serial is None:
            data = [None] * self._num_cams
            with ThreadPoolExecutor() as executor:
                workers = [
                    executor.submit(
                        read_depth_image,
                        self._data_folder / s / f"depth_{frame_id:06d}.png",
                        idx=i,
                    )
                    for i, s in enumerate(self._rs_serials)
                ]
                for worker in workers:
                    img, idx = worker.result()
                    data[idx] = img
        else:
            data = read_depth_image(
                self._data_folder / serial / f"depth_{frame_id:06d}.png"
            )
        return data

    def get_mask_image(
        self, frame_id: int, serial: str = None, erode_kernel: int = 0
    ) -> np.ndarray:
        """Get mask image in numpy format, dtype=uint8, [H, W]."""
        if serial is None:
            data = [None for _ in self._rs_serials]
            workers = []
            with ThreadPoolExecutor() as executor:
                for i, s in enumerate(self._rs_serials):
                    workers.append(
                        executor.submit(
                            self._read_mask_image,
                            self._data_folder / s / f"mask_{frame_id:06d}.png",
                            erode_kernel,
                            idx=i,
                        )
                    )
                for worker in workers:
                    img, idx = worker.result()
                    data[idx] = erode_mask(img, erode_kernel)
        else:
            data = self._read_mask_image(
                self._data_folder / serial / f"mask_{frame_id:06d}.png", erode_kernel
            )

        return data

    def object_group_layer_forward(
        self, poses: list[torch.Tensor], subset: list[int] = None
    ) -> tuple:
        """Forward pass for the object group layer."""
        p = torch.cat(poses, dim=1)
        v, n = self._object_group_layer(p, inds=subset)
        if p.size(0) == 1:
            v = v.squeeze(0)
            n = n.squeeze(0)
        return v, n

    def mano_group_layer_forward(
        self, poses: list[torch.Tensor], subset: list[int] = None
    ) -> tuple:
        """Forward pass for the MANO group layer."""
        p = torch.cat(poses, dim=1)
        v, j = self._mano_group_layer(p, inds=subset)
        if p.size(0) == 1:
            v = v.squeeze(0)
            j = j.squeeze(0)
        return v, j

    def load_object_seg_masks(self) -> np.ndarray:
        """Load object segmentation masks.

        Returns:
            np.ndarray: List of segmentation masks, NUM_FRAMES * [NUM_CAMS, H, W], dtype=uint8.
        """
        seg_masks = [None for _ in range(self.num_cameras)] * self.num_frames
        workers = []
        with ThreadPoolExecutor() as executor:
            for f_idx in range(self.num_frames):
                for c_idx, serial in enumerate(self._rs_serials):
                    workers.append(
                        executor.submit(
                            self._read_mask_image,
                            self._data_folder / serial / f"mask_{f_idx:06d}.png",
                            idx=(f_idx, c_idx),
                        )
                    )
            for worker in workers:
                msk, (f_idx, c_idx) = worker.result()
                seg_masks[f_idx][c_idx] = msk
        seg_masks = np.stack(seg_masks, axis=0)
        return seg_masks

    def step(self):
        """Step to the next frame."""
        self._frame_id = (self._frame_id + 1) % self._num_frames
        self._update_pcd(self._frame_id)

    def step_by_frame_id(self, frame_id: int):
        """Step to a specific frame."""
        self._frame_id = frame_id % self._num_frames
        self._update_pcd(self._frame_id)

    @property
    def sequence_folder(self) -> str:
        return str(self._data_folder)

    @property
    def load_mano(self) -> bool:
        return self._load_mano

    @property
    def load_object(self) -> bool:
        return self._load_object

    @property
    def in_world(self) -> bool:
        return self._in_world

    @property
    def device(self) -> str:
        return self._device

    @property
    def object_ids(self) -> list:
        return self._object_ids

    @property
    def group_id(self) -> str:
        return self._object_ids[0].split("_")[0]

    @property
    def subject_id(self) -> str:
        return self._subject_id

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def rs_width(self) -> int:
        return self._rs_width

    @property
    def rs_height(self) -> int:
        return self._rs_height

    @property
    def rs_serials(self) -> list:
        return self._rs_serials

    @property
    def rs_master(self) -> str:
        return self._rs_master

    @property
    def num_cameras(self) -> int:
        return self._num_cams

    @property
    def holo_serials(self) -> list:
        return self._hl_serials

    @property
    def holo_pv_width(self) -> int:
        return self._hl_pv_width

    @property
    def holo_pv_height(self) -> int:
        return self._hl_pv_height

    @property
    def holo_depth_width(self) -> int:
        return self._hl_depth_width

    @property
    def holo_depth_height(self) -> int:
        return self._hl_depth_height

    @property
    def holo_depth_mode(self) -> str:
        return self._hl_depth_mode

    @property
    def mano_beta(self) -> torch.Tensor:
        return self._mano_beta

    @property
    def mano_sides(self) -> list:
        return self._mano_sides

    @property
    def intrinsics(self) -> torch.Tensor:
        return self._K

    @property
    def intrinsics_inv(self) -> torch.Tensor:
        return self._K_inv

    @property
    def extrinsics2master(self) -> torch.Tensor:
        return self._extr2master

    @property
    def extrinsics2master_inv(self) -> torch.Tensor:
        return self._extr2master_inv

    @property
    def extrinsics2world(self) -> torch.Tensor:
        return self._extr2world

    @property
    def extrinsics2world_inv(self) -> torch.Tensor:
        return self._extr2world_inv

    @property
    def tag_0(self) -> torch.Tensor:
        """tag_0 to rs_master transformation matrix"""
        return self._tag_0

    @property
    def tag_0_inv(self) -> torch.Tensor:
        """rs_master to tag_0 transformation matrix"""
        return self._tag_0_inv

    @property
    def tag_1(self) -> torch.Tensor:
        """tag_1 to rs_master transformation matrix"""
        return self._tag_1

    @property
    def tag_1_inv(self) -> torch.Tensor:
        """rs_master to tag_1 transformation matrix"""
        return self._tag_1_inv

    @property
    def M2master(self) -> torch.Tensor:
        """camera to master transformation matrix"""
        return self._M2master

    @property
    def M2world(self) -> torch.Tensor:
        """camera to world transformation matrix"""
        return self._M2world

    @property
    def frame_id(self) -> int:
        return self._frame_id

    @property
    def mano_group_layer(self):
        return self._mano_group_layer

    @property
    def object_group_layer(self):
        return self._object_group_layer

    @property
    def object_textured_mesh_files(self) -> list:
        return [
            str(self._models_folder / f"{object_id}/textured_mesh.obj")
            for object_id in self._object_ids
        ]

    @property
    def object_cleaned_mesh_files(self) -> list:
        return [
            str(self._models_folder / f"{object_id}/cleaned_mesh_10000.obj")
            for object_id in self._object_ids
        ]

    @property
    def points(self) -> torch.Tensor:
        return self._points

    @property
    def colors(self) -> torch.Tensor:
        return self._colors

    @property
    def masks(self) -> torch.Tensor:
        return self._masks

    @property
    def points_map(self) -> torch.Tensor:
        return self._points.view(self._num_cams, self._rs_height, self._rs_width, 3)

    @property
    def colors_map(self) -> torch.Tensor:
        return self._colors.view(self._num_cams, self._rs_height, self._rs_width, 3)

    @property
    def masks_map(self) -> torch.Tensor:
        return self._masks.view(self._num_cams, self._rs_height, self._rs_width)

    def load_object_poses(self):
        poses = np.load(self._data_folder / "poses_o.npy")
        return poses

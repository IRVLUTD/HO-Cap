import os

os.environ["PYOPENGL_PLATFORM"] = "egl"  # GPU-based offscreen rendering

from hocap.utils import *
from hocap.loaders import SequenceLoader


class SequenceRenderer:
    def __init__(self, sequence_folder) -> None:
        self._seq_folder = Path(sequence_folder).resolve()
        self._loader = SequenceLoader(sequence_folder, load_mano=True)
        self._num_frames = self._loader.num_frames
        self._object_ids = self._loader.object_ids
        # Realsense cameras
        self._rs_serials = self._loader.rs_serials
        self._rs_width = self._loader.rs_width
        self._rs_height = self._loader.rs_height
        self._rs_intrinsics = self._loader.intrinsics.cpu().numpy()
        self._rs_extrinsics = self._loader.extrinsics2world.cpu().numpy()
        # Hololens cameras
        self._hl_serial = self._loader.holo_serials[0]
        self._hl_pv_width = self._loader.holo_pv_width
        self._hl_pv_height = self._loader.holo_pv_height
        self._hl_pv_intrinsics = self._load_holo_pv_intrinsics(self._hl_serial)

        # Load poses
        self._poses_o = self._load_object_poses()
        self._poses_m = self._load_mano_poses()
        self._poses_pv = self._load_holo_poses()

        # Rendering flags
        self._rgb_flags = (
            pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.SHADOWS_ALL
        )
        self._depth_flags = (
            pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY
        )
        self._mask_flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.SEG

        # Load object meshes
        self._obj_meshes = [
            pyrender.Mesh.from_trimesh(trimesh.load_mesh(f, process=False))
            for f in self._loader.object_textured_mesh_files
        ]

        # Set up scene
        self._create_scene()

    def _create_scene(self):
        self._scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0], ambient_light=[1.0, 1.0, 1.0]
        )

        # Add world node
        self._world_node = self._scene.add_node(pyrender.Node(name="world"))

        # Add realsense camera nodes
        self._camera_nodes = {
            serial: self._scene.add(
                pyrender.IntrinsicsCamera(
                    fx=cam_K[0, 0],
                    fy=cam_K[1, 1],
                    cx=cam_K[0, 2],
                    cy=cam_K[1, 2],
                    znear=0.01,
                    zfar=10.0,
                ),
                parent_node=self._world_node,
                name=serial,
                pose=cam_RT @ cvcam_in_glcam,
            )
            for serial, cam_K, cam_RT in zip(
                self._rs_serials, self._rs_intrinsics, self._rs_extrinsics
            )
        }

        # Add hololens camera node
        self._camera_nodes[self._hl_serial] = self._scene.add(
            pyrender.IntrinsicsCamera(
                fx=self._hl_pv_intrinsics[0, 0],
                fy=self._hl_pv_intrinsics[1, 1],
                cx=self._hl_pv_intrinsics[0, 2],
                cy=self._hl_pv_intrinsics[1, 2],
                znear=0.01,
                zfar=10.0,
            ),
            parent_node=self._world_node,
            name=self._hl_serial,
            pose=np.eye(4),
        )

        # Add object nodes
        self._object_nodes = [
            self._scene.add(
                pyrender.Mesh.from_trimesh(trimesh.load_mesh(mesh_file, process=False)),
                parent_node=self._world_node,
                name=object_id,
                pose=np.eye(4),
            )
            for object_id, mesh_file in zip(
                self._object_ids, self._loader.object_textured_mesh_files
            )
        ]

    def _load_holo_pv_intrinsics(self, serial):
        K = np.fromfile(
            self._loader._calib_folder
            / f"hololens/{serial}/personal_video"
            / f"1000_{self._hl_pv_width}_{self._hl_pv_height}/intrinsics.bin",
            dtype=np.float32,
        ).reshape(4, 4)[:3, :3]
        K[0, 0] = -K[0, 0]
        return K.T

    def _load_object_poses(self):
        pose_file = self._seq_folder / "poses_o.npy"
        poses = np.load(pose_file)
        poses = np.stack([quat_to_mat(p) for p in poses], axis=0)
        return poses

    def _load_mano_poses(self):
        pose_file = self._seq_folder / "poses_m.npy"
        poses = np.load(pose_file)
        return poses

    def _load_holo_poses(self):
        pose_file = self._seq_folder / "poses_pv_fd.npy"
        poses = np.linalg.inv(np.load(pose_file))
        return poses

    def get_rgb_image(self, frame_id, serial):
        return self._loader.get_rgb_image(frame_id, serial)

    def get_rendered_image(self, frame_id, serial):

    def get_rendered_mesh(self, frame_id, serial):
        obj_nodes = [
            self._scene.add(
                mesh,
                name=f"mesh_{i}",
                parent_node=self._world_node,
            )
            for i, mesh in enumerate(self._obj_meshes)
        ]
        # set object pose
        seg_node_map = {}
        for i, obj_node in enumerate(obj_nodes):
            self._scene.set_pose(obj_node, self._poses_o[i, frame_id])
            seg_node_map[obj_node] = OBJ_CLASS_COLORS[i + 1].rgb

        if serial == self._hl_serial:
            # set hololens camera pose
            self._scene.set_pose(
                node=self._cam_nodes[serial],
                pose=self._poses_pv[frame_id] @ cvcam_in_glcam,
            )
            r = pyrender.OffscreenRenderer(
                viewport_width=self._hl_pv_width,
                viewport_height=self._hl_pv_height,
            )

        else:
            r = pyrender.OffscreenRenderer(
                viewport_width=self._rs_width,
                viewport_height=self._rs_height,
            )

        # set camera node
        self._scene.main_camera_node = self._cam_nodes[serial]

        # render rgb, depth, mask
        color, depth = r.render(self._scene, flags=self._rgb_flags)
        mask, _ = r.render(
            self._scene, flags=self._mask_flags, seg_node_map=seg_node_map
        )
        # mask = mask[:, :, 0]

        # release source
        for obj_node in obj_nodes:
            self._scene.remove_node(obj_node)
        r.delete()

        return color, depth, mask

    def get_rendered_point_cloud(self, frame_id, serial):
        self._loader.step_by_frame_id(frame_id)
        # pcd_masks = self._loader.masks
        # pcd_points = self._loader.points[pcd_masks].cpu().numpy()
        # pcd_colors = self._loader.colors[pcd_masks].cpu().numpy()
        pcd_points = self._loader.points.cpu().numpy().reshape(-1, 3)
        pcd_colors = self._loader.colors.cpu().numpy().reshape(-1, 3)

        pcd_node = self._scene.add(
            pyrender.Mesh.from_points(points=pcd_points, colors=pcd_colors),
            name="pcd",
            parent_node=self._world_node,
        )

        if serial == self._hl_serial:
            # set hololens camera pose
            self._scene.set_pose(
                node=self._cam_nodes[serial],
                pose=self._poses_pv[frame_id] @ cvcam_in_glcam,
            )
            r = pyrender.OffscreenRenderer(
                viewport_width=self._hl_pv_width,
                viewport_height=self._hl_pv_height,
                point_size=1.0,
            )

        else:
            r = pyrender.OffscreenRenderer(
                viewport_width=self._rs_width,
                viewport_height=self._rs_height,
                point_size=1.0,
            )

        # set camera node
        self._scene.main_camera_node = self._cam_nodes[serial]

        color, depth = r.render(self._scene, flags=self._rgb_flags)
        depth = (depth * 1000.0).astype(np.int16)  # convert to mm
        mask = np.zeros_like(depth).astype(np.uint8)

        # release source
        self._scene.remove_node(pcd_node)
        r.delete()

        return color, depth, mask

    def get_rendered_scene(self, frame_id, serial):
        obj_nodes = [
            self._scene.add(
                mesh,
                name=f"mesh_{i}",
                parent_node=self._world_node,
            )
            for i, mesh in enumerate(self._obj_meshes)
        ]

        # set object pose
        for i, obj_node in enumerate(obj_nodes):
            self._scene.set_pose(obj_node, self._poses_o[i, frame_id])

        if serial == self._hl_serial:
            # set hololens camera pose
            self._scene.set_pose(
                node=self._cam_nodes[serial],
                pose=self._poses_pv[frame_id] @ cvcam_in_glcam,
            )
            r = pyrender.OffscreenRenderer(
                viewport_width=self._hl_pv_width,
                viewport_height=self._hl_pv_height,
            )

        else:
            r = pyrender.OffscreenRenderer(
                viewport_width=self._rs_width,
                viewport_height=self._rs_height,
            )

        # set camera node
        self._scene.main_camera_node = self._cam_nodes[serial]

        # render rgb, depth, mask
        mask, _ = r.render(
            self._scene,
            flags=self._mask_flags,
            seg_node_map={
                # obj_node: OBJ_CLASS_COLORS[i + 1].rgb
                obj_node: [i + 1] * 3
                for i, obj_node in enumerate(obj_nodes)
            },
        )
        mask = mask[:, :, 0]
        print(np.unique(mask), mask.shape, mask.dtype, mask.min(), mask.max())

        self._loader.step_by_frame_id(frame_id)
        # pcd_masks = self._loader.masks
        # pcd_points = self._loader.points[pcd_masks].cpu().numpy()
        # pcd_colors = self._loader.colors[pcd_masks].cpu().numpy()
        pcd_points = self._loader.points.cpu().numpy().reshape(-1, 3)
        pcd_colors = self._loader.colors.cpu().numpy().reshape(-1, 3)
        pcd_colors = pcd_colors[pcd_points[:, 2] > 0.0]
        pcd_points = pcd_points[pcd_points[:, 2] > 0.0]

        pcd_node = self._scene.add(
            pyrender.Mesh.from_points(points=pcd_points, colors=pcd_colors),
            name="pcd",
            parent_node=self._world_node,
        )

        color, depth = r.render(self._scene, flags=self._rgb_flags)

        # release source
        # for obj_node in obj_nodes:
        #     self._scene.remove_node(obj_node)
        self._scene.remove_node(pcd_node)
        r.delete()

        return color, depth, mask

    def _create_frame_scene(self, frame_id):
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0], ambient_light=[1.0, 1.0, 1.0])
        # Add world node
        scene.add_node(pyrender.Node(name="world_node"))
        # Add camera nodes
        for i, serial in enumerate(self._rs_serials):
            scene.add(
                self._cameras[serial],
                name=serial,
                parent_name="world_node",
                pose=self._rs_extrinsics[i] @ cvcam_in_glcam,
            )
        scene.add(
            self._cameras[self._hl_serial],
            name=self._hl_serial,
            parent_name="world_node",
            pose=self._poses_pv[frame_id] @ cvcam_in_glcam,
        )
        # Add object nodes
        for i, obj_id in enumerate(self._object_ids):
            scene.add(
                self._obj_meshes[i],
                name=obj_id,
                parent_name="world_node",
                pose=self._poses_o[i, frame_id],
            )

        return scene

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def rs_serials(self):
        return self._rs_serials

    @property
    def holo_serial(self):
        return self._hl_serial


def plot_and_save_images(images):
    """
    Plot the images in the specified layout and save as 1080P PNG.

    Parameters:
    images (list of numpy arrays): List of 10 images to be displayed.
    frame_id (int): The frame ID to be used in the filename.
    output_folder (str): The folder where the output images will be saved.
    """
    if len(images) != 10:
        raise ValueError("The function expects exactly 10 images.")

    # Create a figure with 1920x1080 resolution
    fig = plt.figure(
        figsize=(19.2, 10.8), dpi=100
    )  # figsize in inches, dpi=100 for 1920x1080 pixels

    # Create a GridSpec with 3 rows and 4 columns
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.5])

    # Plot the first 8 images in a 2x4 grid
    for i in range(8):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        ax.imshow(images[i])
        ax.axis("off")  # Hide the axes

    # Plot the 9th image on the bottom left
    ax = fig.add_subplot(gs[2, :2])
    ax.imshow(images[8])
    ax.axis("off")  # Hide the axes

    # Plot the 10th image on the bottom right
    ax = fig.add_subplot(gs[2, 2:])
    ax.imshow(images[9])
    ax.axis("off")  # Hide the axes

    # Display the plot
    plt.tight_layout()
    plt.show()


def args_parser():
    parser = argparse.ArgumentParser(description="Sequence Renderer")
    parser.add_argument(
        "--sequence_folder",
        type=str,
        default="data/subject_1/20231025_165502",
        help="Sequence path relative to the data folder",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # args = args_parser()
    # sequence_folder = Path(args.sequence_folder).resolve()

    sequence_folder = PROJ_ROOT / "data/subject_1/20231025_165502"
    renderer = SequenceRenderer(sequence_folder)

    frame_id = 0
    scene = renderer._create_frame_scene(frame_id)

    r = pyrender.OffscreenRenderer(640, 480)
    hl_renderer = pyrender.OffscreenRenderer(1280, 720)

    rs_serials = renderer.rs_serials
    hl_serial = renderer.holo_serial

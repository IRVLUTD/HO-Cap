import os

os.environ["PYOPENGL_PLATFORM"] = "egl"  # GPU-based offscreen rendering

from hocap.utils import *
from hocap.loaders import SequenceLoader


class SequenceRenderer:
    def __init__(self, sequence_folder, device="cpu") -> None:
        self._seq_folder = Path(sequence_folder).resolve()
        self._device = device
        self._loader = SequenceLoader(sequence_folder, load_mano=True, device=device)
        self._num_frames = self._loader.num_frames
        self._object_ids = self._loader.object_ids
        self._mano_sides = self._loader.mano_sides
        self._mano_group_layer = self._loader.mano_group_layer
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

        # Load object meshes
        self._obj_meshes = [
            pyrender.Mesh.from_trimesh(trimesh.load_mesh(f, process=False))
            for f in self._loader.object_textured_mesh_files
        ]

        # Get verts, faces, colors for MANO
        self._mano_verts = self._get_mano_verts()
        self._mano_faces = self._get_mano_faces()
        self._mano_colors = self._get_mano_colors()

        # Rendering flags
        self._rgb_flags = (
            pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.SHADOWS_ALL
        )
        self._depth_flags = (
            pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY
        )
        self._mask_flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.SEG

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
        poses = [
            torch.from_numpy(poses[0 if side == "right" else 1]).to(self._device)
            for side in self._mano_sides
        ]
        return poses

    def _load_holo_poses(self):
        pose_file = self._seq_folder / "poses_pv_fd.npy"
        poses = np.linalg.inv(np.load(pose_file))
        return poses

    def _get_mano_verts(self):
        p = torch.cat(self._poses_m, dim=1)
        v, _ = self._mano_group_layer(p)
        if p.size(0) == 1:
            v = v[0]
        return v.cpu().numpy()

    def _get_mano_faces(self):
        mano_faces = [
            np.concatenate(
                [
                    self._mano_group_layer.f.cpu().numpy(),
                    (
                        np.array(NEW_MANO_FACES)
                        if side == "right"
                        else np.array(NEW_MANO_FACES)[:, ::-1]
                    ),
                ]
            )
            for side in self._mano_sides
        ]
        return mano_faces

    def _get_mano_colors(self):
        mano_colors = np.stack(
            [
                [HAND_COLORS[1].rgb if side == "right" else HAND_COLORS[2].rgb]
                * NUM_MANO_VERTS
                for side in self._mano_sides
            ]
        )
        return mano_colors

    def _get_mano_meshes(self, frame_id):
        meshes = [
            trimesh.Trimesh(
                vertices=self._mano_verts[frame_id][
                    i * NUM_MANO_VERTS : (i + 1) * NUM_MANO_VERTS
                ],
                faces=self._mano_faces[i],
                vertex_colors=self._mano_colors[i],
                process=False,
            )
            for i in range(len(self._mano_sides))
        ]
        meshes = [pyrender.Mesh.from_trimesh(mesh) for mesh in meshes]
        return meshes

    def create_scene(self, frame_id):
        self._scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0], ambient_light=[1.0, 1.0, 1.0]
        )

        # Add world node
        world_node = self._scene.add_node(pyrender.Node(name="world"))

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
                parent_node=world_node,
                name=f"cam_{serial}",
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
            parent_node=world_node,
            name=f"cam_{self._hl_serial}",
            pose=self._poses_pv[frame_id] @ cvcam_in_glcam,
        )

        # Add object nodes
        self._object_nodes = [
            self._scene.add(
                obj_mesh,
                parent_node=world_node,
                name=f"obj_{self._object_ids[i]}",
                pose=self._poses_o[i, frame_id],
            )
            for i, obj_mesh in enumerate(self._obj_meshes)
        ]

        # Add MANO nodes
        self._mano_nodes = [
            self._scene.add(
                mano_mesh,
                parent_node=world_node,
                name=f"mano_{self._mano_sides[i]}",
                pose=np.eye(4),
            )
            for i, mano_mesh in enumerate(self._get_mano_meshes(frame_id))
        ]

        self._seg_node_map = {}
        for i, obj_node in enumerate(self._object_nodes):
            self._seg_node_map[obj_node] = OBJ_CLASS_COLORS[i + 1].rgb

        for i, side in enumerate(self._mano_sides):
            hand_color_idx = 1 if side == "right" else 2
            self._seg_node_map[self._mano_nodes[i]] = HAND_COLORS[hand_color_idx].rgb

    def get_rgb_image(self, frame_id, serial):
        return self._loader.get_rgb_image(frame_id, serial)

    def get_render_colors(self):
        color_images = {}
        # Render color images for realsense cameras
        r = pyrender.OffscreenRenderer(self._rs_width, self._rs_height)
        for serial in self._rs_serials:
            self._scene.main_camera_node = self._camera_nodes[serial]
            color, _ = r.render(self._scene, flags=self._rgb_flags)
            color_images[serial] = color
        r.delete()
        # Render color image for hololens camera
        r = pyrender.OffscreenRenderer(self._hl_pv_width, self._hl_pv_height)
        self._scene.main_camera_node = self._camera_nodes[self._hl_serial]
        color, _ = r.render(self._scene, flags=self._rgb_flags)
        color_images[self._hl_serial] = color
        r.delete()
        return color_images

    def get_render_depths(self):
        depth_images = {}
        # Render depth images for realsense cameras
        r = pyrender.OffscreenRenderer(self._rs_width, self._rs_height)
        for serial in self._rs_serials:
            self._scene.main_camera_node = self._camera_nodes[serial]
            depth = r.render(self._scene, flags=self._depth_flags)
            depth_images[serial] = depth
        r.delete()
        # Render depth image for hololens camera
        r = pyrender.OffscreenRenderer(self._hl_pv_width, self._hl_pv_height)
        self._scene.main_camera_node = self._camera_nodes[self._hl_serial]
        depth = r.render(self._scene, flags=self._depth_flags)
        depth_images[self._hl_serial] = depth
        r.delete()
        return depth_images

    def get_render_masks(self):
        mask_images = {}
        # Render mask images for realsense cameras
        r = pyrender.OffscreenRenderer(self._rs_width, self._rs_height)
        for serial in self._rs_serials:
            self._scene.main_camera_node = self._camera_nodes[serial]
            mask, _ = r.render(
                self._scene, flags=self._mask_flags, seg_node_map=self._seg_node_map
            )
            mask_images[serial] = mask
        r.delete()
        # Render mask image for hololens camera
        r = pyrender.OffscreenRenderer(self._hl_pv_width, self._hl_pv_height)
        self._scene.main_camera_node = self._camera_nodes[self._hl_serial]
        mask, _ = r.render(
            self._scene, flags=self._mask_flags, seg_node_map=self._seg_node_map
        )
        mask_images[self._hl_serial] = mask
        r.delete()
        return mask_images

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


if __name__ == "__main__":
    sequence_folder = PROJ_ROOT / "data/subject_1/20231025_165502"
    renderer = SequenceRenderer(sequence_folder)

    frame_id = 70

    # rs_serials = renderer.rs_serials
    # hl_serial = renderer.holo_serial

    # for serial in rs_serials:
    #     rgb = renderer.get_rgb_image(frame_id, serial)
    #     write_rgb_image(f"color_{serial}.jpg", rgb)

    # rgb = renderer.get_rgb_image(frame_id, hl_serial)
    # write_rgb_image(f"color_{hl_serial}.jpg", rgb)

    renderer.create_scene(frame_id)
    # render_colors = renderer.get_render_colors()
    # render_colors = renderer.get_render_depths()
    render_colors = renderer.get_render_masks()

    for serial, render_color in render_colors.items():
        color = renderer.get_rgb_image(frame_id, serial)
        color = cv2.addWeighted(color, 0.5, render_color, 0.5, 0)
        # render_color = get_depth_colormap(render_color)
        write_rgb_image(f"color_{serial}.png", color)

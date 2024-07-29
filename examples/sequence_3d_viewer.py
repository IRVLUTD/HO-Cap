import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from time import sleep
from torch.utils import dlpack
from hocap.utils import *
from hocap.loaders import SequenceLoader


HELP_INFO = """
=============================
Keyboard commands:
=============================
H: display control panel
SPACE: pause
Q: quit
R: reset camera
=============================
"""


class SequenceViewer:
    def __init__(self, sequence_folder, device="cuda") -> None:
        self._data_folder = Path(sequence_folder).resolve()
        self._device = device

        self._loader = SequenceLoader(sequence_folder, load_mano=True, device=device)
        self._num_frames = self._loader.num_frames
        self._rs_serials = self._loader.rs_serials
        self._rs_master = self._loader.rs_master
        self._master_id = self._rs_serials.index(self._rs_master)
        self._num_cameras = self._loader.num_cameras
        self._rs_height = self._loader.rs_height
        self._rs_width = self._loader.rs_width
        self._load_mano = self._loader.load_mano
        self._intrinsics = self._loader.intrinsics.cpu().numpy()
        self._extrinsics = self._loader.extrinsics2world.cpu().numpy()
        self._mano_sides = self._loader.mano_sides

        if self._load_mano:
            self._mano_group_layer = self._loader.mano_group_layer
            self._mano_verts = self._get_mano_verts()
            self._mano_faces = self._get_mano_faces()
            self._mano_colors = self._get_mano_colors()

    def run(self):
        self._is_done = False
        self._frame_id = -1

        # rendering settings
        self._bg_color = (0.0, 0.0, 0.0, 1.0)  # black
        self._point_size = 1
        self._update_flag = (
            rendering.Scene.UPDATE_POINTS_FLAG | rendering.Scene.UPDATE_COLORS_FLAG
        )  # update points and colors

        # control flags
        self._cropped = False  # crop points
        self._is_paused = False  # pause
        self._show_skybox = False  # show skybox background
        self._show_axes = False  # show axes frame
        self._show_pcds = True  # show point clouds
        self._show_mano = False  # show mano mesh
        self._show_object = False  # show object mesh
        # self._cam_id = self._master_id  # camera view
        self._cam_id = self._rs_serials.index(self._rs_master)  # camera view

        # materials
        self._mat_pcd = rendering.MaterialRecord()
        self._mat_pcd.shader = "defaultUnlit"
        self._mat_pcd.point_size = self._point_size
        self._mat_mesh = rendering.MaterialRecord()
        self._mat_mesh.shader = "defaultUnlit"
        self._mat_line = rendering.MaterialRecord()
        self._mat_line.shader = "unlitLine"

        # dummy geometry
        zeros = o3c.Tensor.zeros(
            (self._rs_width * self._rs_height * self._num_cameras, 3), dtype=o3c.float32
        )
        self._pcd = o3d.t.geometry.PointCloud()
        self._pcd.point.positions = zeros
        self._pcd.point.colors = zeros
        self._pcd.point.normals = zeros

        if self._load_mano:
            mano_mesh = o3d.geometry.TriangleMesh()
            mano_mesh.vertices = o3d.utility.Vector3dVector(self._mano_verts[0].numpy())
            mano_mesh.triangles = o3d.utility.Vector3iVector(self._mano_faces)
            mano_mesh.vertex_colors = o3d.utility.Vector3dVector(self._mano_colors)
            mano_mesh.compute_vertex_normals()
            mano_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mano_mesh)
            mano_ls.paint_uniform_color((0.0, 0.0, 0.0))  # black
            self._mano_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mano_mesh)
            self._mano_ls = o3d.t.geometry.LineSet.from_legacy(mano_ls)

        # init gui
        self._app = gui.Application.instance
        self._app.initialize()

        # create window
        self._window = self._create_window()

        # set callbacks
        self._window.set_on_layout(self._on_layout)
        self._window.set_on_key(self._on_key)
        self._window.set_on_close(self._on_close)

        # add initial dummy geometry
        self._widget3d.scene.add_geometry("pcd", self._pcd, self._mat_pcd)

        # update camera
        self._reset_camera()

        # run
        self._app.run_in_thread(self.update)
        self._app.run()

    def _create_window(self, title="Sequence Viewer", width=640, height=480):
        # create window
        window = self._app.create_window(title, width, height)

        ## add widget3d
        self._widget3d = gui.SceneWidget()
        self._widget3d.scene = rendering.Open3DScene(window.renderer)
        self._widget3d.scene.set_background(self._bg_color)
        view = self._widget3d.scene.view
        view.set_post_processing(False)
        window.add_child(self._widget3d)

        ## add settings panel
        em = window.theme.font_size
        margin = 0.25 * em
        self._panel = gui.Vert(margin, gui.Margins(margin, margin, margin, margin))

        ### render settings
        settings = gui.CollapsableVert(
            "Render Settings", margin, gui.Margins(margin, margin, margin, margin)
        )
        settings.set_is_open(True)
        render_blk = gui.VGrid(2, margin)
        self._bg_color_edit = gui.ColorEdit()
        self._bg_color_edit.color_value = gui.Color(*self._bg_color)
        self._bg_color_edit.set_on_value_changed(self._on_bg_color)
        render_blk.add_child(gui.Label("Background Color"))
        render_blk.add_child(self._bg_color_edit)
        point_size = gui.Slider(gui.Slider.INT)
        point_size.double_value = self._point_size
        point_size.set_limits(1, 10)
        point_size.set_on_value_changed(self._on_point_size)
        render_blk.add_child(gui.Label("Point Size"))
        render_blk.add_child(point_size)
        chk_box = gui.Checkbox("Show Skybox")
        chk_box.checked = self._show_skybox
        chk_box.set_on_checked(self._on_skybox)
        render_blk.add_child(chk_box)
        chk_box = gui.Checkbox("Show Axes")
        chk_box.checked = self._show_axes
        chk_box.set_on_checked(self._on_axes)
        render_blk.add_child(chk_box)
        crop_box = gui.Checkbox("Crop Points")
        crop_box.checked = self._cropped
        crop_box.set_on_checked(self._on_crop)
        render_blk.add_child(crop_box)
        settings.add_child(render_blk)
        self._panel.add_child(settings)
        ### geometry settings
        settings = gui.CollapsableVert(
            "Geometry Settings", margin, gui.Margins(margin, margin, margin, margin)
        )
        settings.set_is_open(True)
        geo_blk = gui.Vert(margin, gui.Margins(margin, margin, margin, margin))
        chk_box = gui.Checkbox("Point Clouds")
        chk_box.checked = self._show_pcds
        chk_box.set_on_checked(self._on_pcds)
        geo_blk.add_child(chk_box)
        chk_box = gui.Checkbox("Hand Mesh")
        chk_box.enabled = self._load_mano
        chk_box.checked = self._show_mano
        chk_box.set_on_checked(self._on_mano)
        geo_blk.add_child(chk_box)
        chk_box = gui.Checkbox("Object Mesh")
        chk_box.enabled = False
        chk_box.checked = self._show_object
        chk_box.set_on_checked(self._on_object)
        geo_blk.add_child(chk_box)
        settings.add_child(geo_blk)
        self._panel.add_child(settings)
        ### progress bar
        bar = gui.VGrid(3, margin)
        self._slider = gui.Slider(gui.Slider.INT)
        self._slider.set_limits(0, self._num_frames - 1)
        self._slider.set_on_value_changed(self._on_progress_slider)
        self._num_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self._num_edit.set_limits(0, self._num_frames - 1)
        self._num_edit.set_on_value_changed(self._on_progress_slider)
        bar.add_child(gui.Label("Frame Slider"))
        bar.add_child(self._slider)
        bar.add_child(self._num_edit)
        self._panel.add_child(bar)
        ### reset button
        btns = gui.Horiz(margin, gui.Margins(margin, margin, margin, margin))
        botton1 = gui.Button("Reset")
        botton1.set_on_clicked(self._on_reset)
        botton2 = gui.Button("Pause/Play")
        botton2.set_on_clicked(self._on_pause)
        botton3 = gui.Button("Exit")
        botton3.set_on_clicked(self._on_exit)
        btns.add_stretch()
        btns.add_child(botton1)
        btns.add_child(botton2)
        btns.add_child(botton3)
        btns.add_stretch()
        self._panel.add_child(btns)

        self._panel.add_stretch()
        ####################
        # add tab control
        self._tabs = gui.TabControl()
        help_tab = gui.Vert(margin, gui.Margins(margin, margin, margin, margin))
        help_info = gui.VGrid(2, margin)
        help_info.add_child(gui.Label(HELP_INFO))
        help_tab.add_child(help_info)
        self._tabs.add_tab("Settings", self._panel)
        self._tabs.add_tab("Help", help_tab)

        # add tabs
        window.add_child(self._tabs)

        return window

    def _on_layout(self, ctx):
        r = self._window.content_rect
        panel_size = self._tabs.calc_preferred_size(ctx, gui.Widget.Constraints())
        if (r.width < self._rs_width + panel_size.width) or r.height < self._rs_height:
            self._window.size = gui.Size(
                self._rs_width + panel_size.width, self._rs_height
            )
        self._width = r.width - panel_size.width
        self._height = r.height
        self._widget3d.frame = gui.Rect(0, 0, self._width, self._height)
        self._tabs.frame = gui.Rect(
            self._widget3d.frame.get_right(), 0, panel_size.width, self._height
        )
        self._update_camera_K()

    def _on_close(self):
        self._is_done = True
        sleep(0.10)
        return True

    def _on_key(self, event):
        if event.key == gui.KeyName.Q:  # quit
            if event.type == gui.KeyEvent.DOWN:
                self._window.close()
                return True

        if event.key == gui.KeyName.SPACE:  # pause
            if event.type == gui.KeyEvent.DOWN:
                self._on_pause()
                return True

        if event.key == gui.KeyName.R:  # reset camera
            if event.type == gui.KeyEvent.DOWN:
                self._reset_camera()
                return True

        return False

    def _on_exit(self):
        self._window.close()
        self._app.quit()

    def _on_pause(self):
        self._is_paused = not self._is_paused

    def _on_reset(self):
        self._cam_id = self._rs_serials.index(self._rs_master)
        self._reset_camera()
        self._frame_id = -1
        self._slider.int_value = 0
        self._num_edit.int_value = 0

    def _on_progress_slider(self, value):
        value = int(value) % self._num_frames
        self._frame_id = value
        self._num_edit.int_value = value

    def _on_bg_color(self, color):
        self._bg_color_edit.color_value = color
        self._widget3d.scene.set_background(
            [color.red, color.green, color.blue, color.alpha]
        )

    def _on_skybox(self, checked):
        self._widget3d.scene.show_skybox(checked)

    def _on_axes(self, checked):
        self._widget3d.scene.show_axes(checked)

    def _on_crop(self, checked):
        self._cropped = checked

    def _on_pcds(self, checked):
        self._show_pcds = checked
        self._widget3d.scene.show_geometry("pcd", checked)

    def _on_mano(self, checked):
        self._show_mano = checked
        self._widget3d.scene.show_geometry("mano", checked)
        self._widget3d.scene.show_geometry("mano_ls", checked)

    def _on_object(self, checked):
        self._show_object = checked
        self._widget3d.scene.show_geometry("object", checked)

    def _on_point_size(self, value):
        self._mat_pcd.point_size = int(value)
        self._widget3d.scene.modify_geometry_material("pcd", self._mat_pcd)

    def _reset_camera(self):
        self._widget3d.scene.camera.look_at([0, 0, 0], [0, 0, 0.8], [0, -1, 0])

    def _update_camera_K(self):
        def create_K_matrix(image_width, image_height, fov_degrees):
            # The principal point is at the center of the image.
            cx = image_width / 2.0
            cy = image_height / 2.0
            # Compute the focal length from the field of view.
            fov_rad = np.deg2rad(fov_degrees)
            fx = fy = cx / np.tan(fov_rad / 2)
            # Create the intrinsic matrix.
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            return K

        K = create_K_matrix(self._width, self._height, 90)
        self._widget3d.scene.camera.set_projection(
            K, 0.001, 1000.0, self._width, self._height
        )

    def _update_camera_pose(self):
        def extrinsics_to_look_at(pose):
            R = pose[:3, :3]
            T = pose[:3, 3]
            # The camera's position (eye) is the negative rotation by R of T.
            eye = -np.matmul(R.T, T)
            # The center point is one unit down the z-axis in the camera's space,
            # then transformed to world space by the pose matrix.
            center = np.matmul(R.T, np.array([0, 0, 1])) + eye
            # The up vector is the y-axis in the camera's space, then transformed to world space by the rotation matrix.
            # This assumes that the y-axis is the down-direction in the camera's local space.
            up = np.matmul(R.T, np.array([0, -1, 0]))
            return center, eye, up

        extrinsics = self._extrinsics[self._cam_id]
        center, eye, up = extrinsics_to_look_at(extrinsics)
        # self._widget3d.scene.camera.look_at(center, eye, up)
        self._widget3d.look_at(center, eye, up)

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

    def _get_mano_verts(self):
        pose_file = self._data_folder / "poses_m.npy"
        poses = np.load(pose_file)
        poses = [
            torch.from_numpy(poses[0 if side == "right" else 1]).to(self._device)
            for side in self._mano_sides
        ]
        p = torch.cat(poses, dim=1)
        v, _ = self._mano_group_layer(p)
        if p.size(0) == 1:
            v = v[0]
        return v.cpu()

    def _get_mano_faces(self):
        mano_faces = np.stack(
            [
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
        ).reshape(-1, 3)
        return mano_faces

    def _get_mano_colors(self):
        mano_colors = np.stack(
            [
                [HAND_COLORS[1].rgb if side == "right" else HAND_COLORS[2].rgb]
                * NUM_MANO_VERTS
                for side in self._mano_sides
            ]
        ).reshape(-1, 3)
        return mano_colors

    def step(self):
        if not self._is_paused:
            self._frame_id = (self._frame_id + 1) % self._num_frames
            self._slider.int_value = self._frame_id
            self._num_edit.int_value = self._frame_id
        self._loader.step_by_frame_id(self._frame_id)

    def update(self):
        def update():
            if self._show_pcds:
                points = self._loader.points
                colors = self._loader.colors
                masks = self._loader.masks
                if self._cropped:
                    points[~masks] = 0.0
                    colors[~masks] = 0.0
                self._pcd.point.positions = o3c.Tensor.from_dlpack(
                    dlpack.to_dlpack(points.cpu().view((-1, 3)))
                )
                self._pcd.point.colors = o3c.Tensor.from_dlpack(
                    dlpack.to_dlpack(colors.cpu().view((-1, 3)))
                )
                self._widget3d.scene.scene.update_geometry(
                    "pcd", self._pcd, self._update_flag
                )

            if self._show_mano:
                verts_m = self._mano_verts[self._frame_id]
                verts_m = o3c.Tensor.from_dlpack(dlpack.to_dlpack(verts_m))
                self._mano_mesh.vertex.positions = verts_m
                self._mano_ls.point.positions = verts_m
                # self._mano_mesh = self._mano_mesh.transform(np.eye(4))
                # self._mano_ls = self._mano_ls.transform(np.eye(4))
                self._widget3d.scene.remove_geometry("mano")
                self._widget3d.scene.add_geometry(
                    "mano", self._mano_mesh, self._mat_mesh
                )
                self._widget3d.scene.remove_geometry("mano_ls")
                self._widget3d.scene.add_geometry(
                    "mano_ls", self._mano_ls, self._mat_line
                )

        while not self._is_done:
            sleep(0.067)
            if not self._is_done:
                self.step()
                self._app.post_to_main_thread(self._window, update)


if __name__ == "__main__":
    sequence_name = "subject_1/20231025_165502"

    viewer = SequenceViewer(
        sequence_folder=PROJ_ROOT / "data" / sequence_name, device="cuda"
    )
    viewer.run()

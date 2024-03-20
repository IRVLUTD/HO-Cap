import open3d as o3d
from open3d.visualization import rendering
import numpy as np
import torch
from ..utils import *


# OpenGL RH y UP (Pyrender)
#    y
#    |
#    +---x
#   /
#  z

# Blender RH z UP
#    z
#    | y
#    |/
#    +---x

# Direct X, LH y UP
#    y
#    | z
#    |/
#    +---x

# Unreal Engine, RH, x RIGHT y UP
#    y
#    |
#    +---x
#   /
#  z




class OffscreenRenderer:
    def __init__(self, width=640, height=480):
        self._width = width
        self._height = height

        # materials
        self._mat_mesh = rendering.MaterialRecord()
        self._mat_mesh.shader = "defaultUnlit"
        self._mat_mesh.base_color = (0.8, 0.8, 0.8, 1.0)
        self._mat_pcd = rendering.MaterialRecord()
        self._mat_pcd.shader = "defaultUnlit"
        self._mat_pcd.base_color = (0.8, 0.8, 0.8, 1.0)
        self._mat_pcd.point_size = 2.0

        # rendering settings
        self._bg_color = (0.0, 0.0, 0.0, 1.0)
        self._show_axes = False
        self._enable_shadow = False
        self._enable_post_processing = False
        self._lighting_profile = rendering.Open3DScene.LightingProfile.NO_SHADOWS

        # init render
        self._render = rendering.OffscreenRenderer(width, height)
        self._scene = self._render.scene
        self._scene.set_background(self._bg_color)
        self._scene.show_axes(self._show_axes)
        self._scene.view.set_shadowing(self._enable_shadow)
        self._scene.view.set_post_processing(self._enable_post_processing)
        self._scene.scene.enable_sun_light(False)
        self._scene.scene.set_indirect_light_intensity(60000.0)

    def enable_post_processing(self, enable):
        self._enable_post_processing = enable
        if enable:
            self._mat_mesh.shader = "defaultLit"
        else:
            self._mat_mesh.shader = "defaultUnlit"

        self._scene.view.set_post_processing(self._enable_post_processing)
        self._scene.view.set_shadowing(self._enable_shadow)

    def add_mesh(self, verts, colors, faces, normals=None, name="mesh"):
        if self._scene.has_geometry(name):
            self._scene.remove_geometry(name)
        if isinstance(verts, torch.Tensor):
            verts = verts.cpu().numpy()
        if isinstance(colors, torch.Tensor):
            colors = colors.cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
        if isinstance(normals, torch.Tensor):
            normals = normals.cpu().numpy()

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        if normals is not None:
            mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        else:
            mesh.compute_vertex_normals()
        self._scene.add_geometry(name, mesh, self._mat_mesh, False)

    def add_model(self, mesh_model, name="model"):
        if self._scene.has_geometry(name):
            self._scene.remove_geometry(name)
        self._scene.add_model(name, mesh_model)

    def add_o3d_mesh(self, mesh, name="mesh"):
        if self._scene.has_geometry(name):
            self._scene.remove_geometry(name)
        self._scene.add_geometry(name, mesh, self._mat_mesh, False)

    def add_pcd(self, points, colors, name="pcd"):
        if self._scene.has_geometry(name):
            self._scene.remove_geometry(name)
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        if isinstance(colors, torch.Tensor):
            colors = colors.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        self._scene.add_geometry(name, pcd, self._mat_pcd, False)

    def set_camera(self, K, cam_pose):
        self._K = K.astype(np.float64)
        self._cam_pose = cam_pose.astype(np.float64)
        self._render.setup_camera(self._K, self._cam_pose, self._width, self._height)
        # compute sun light direction from camera pose
        # sun_light_direction = self._cam_pose[:3, 2]
        # sun_light_direction = self._cam_pose[:3, :3] @ np.array([0, 0, -1])
        # self._scene.set_lighting(self._lighting_profile, sun_light_direction)
        # self._scene.scene.set_sun_light(sun_light_direction, [1.0, 1.0, 1.0], 300000)

    def clear_geometry(self):
        self._scene.clear_geometry()

    def remove_geometry(self, name):
        if self._scene.has_geometry(name):
            self._scene.remove_geometry(name)

    def render_to_rgb_image(self):
        img = self._render.render_to_image()
        return np.asarray(img)

    def render_to_depth_image(self, z_in_view_space=False):
        depth = self._render.render_to_depth_image(z_in_view_space)
        return np.asarray(depth)

    def set_background(self, color, image=None):
        self._bg_color = color
        if image is not None:
            image = o3d.geometry.Image(image)
        self._scene.set_background(self._bg_color, image)

    def set_geometry_transform(self, name, transform):
        self._scene.set_geometry_transform(name, transform)

    def set_view_size(self, width, height):
        self._width = width
        self._height = height
        self._scene.set_view_size(width, height)

    def show_geometry(self, name, show):
        self._scene.show_geometry(name, show)

    def show_axes(self, show):
        self._show_axes = show
        self._scene.show_axes(show)

    def _extrinsics_to_look_at(self, cam_pose):
        R = cam_pose[:3, :3]
        T = cam_pose[:3, 3]
        # The camera's position (eye) is the negative rotation by R of T.
        eye = -np.matmul(R.T, T)
        # The center point is one unit down the z-axis in the camera's space,
        # then transformed to world space by the pose matrix.
        center = np.matmul(R.T, np.array([0, 0, 1])) + eye
        # The up vector is the y-axis in the camera's space, then transformed to world space by the rotation matrix.
        # This assumes that the y-axis is the down-direction in the camera's local space.
        up = np.matmul(R.T, np.array([0, -1, 0]))
        return center, eye, up

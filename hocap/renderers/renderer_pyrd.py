import os

os.environ["PYOPENGL_PLATFORM"] = "egl"  # GPU-based offscreen rendering

import numpy as np
import pyrender
from pyrender import RenderFlags

# OpenGL RH y UP (Pyrender)
#    y
#    |
#    +---x
#   /
#  z

# CV Camera RH y DOWN, x RIGHT, z FRONT
#      z
#     /
#    +---x
#    |
#    y


class OffscreenRenderer:
    def __init__(self, width, height, znear=0.001, zfar=1000.0, pose_type="cv") -> None:
        assert pose_type in ["cv", "gl"], "Invalid pose type. Must be 'cv' or 'gl'"
        self._pose_type = pose_type
        self._width = width
        self._height = height
        self._znear = znear
        self._zfar = zfar
        self._bg_color = np.array([0.0, 0.0, 0.0, 1.0])
        self._ambient_light = np.array([1.0, 1.0, 1.0, 1.0])

        self._cam = pyrender.PerspectiveCamera(
            yfov=np.pi / 3.0, znear=self._znear, zfar=self._zfar
        )

        self._glcam2cvcam = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        self._cvcam2glcam = np.linalg.inv(self._glcam2cvcam)

    def get_render_image(
        self, meshes, mesh_poses=None, cam_K=None, cam_pose=None, idx=None
    ):
        poses_m = (
            mesh_poses
            if mesh_poses is not None
            else [np.eye(4) for _ in range(len(meshes))]
        )
        pose_c = cam_pose if cam_pose is not None else np.eye(4)
        if self._pose_type == "cv":
            pose_c = pose_c @ self._cvcam2glcam

        scene = pyrender.Scene(
            bg_color=self._bg_color, ambient_light=self._ambient_light
        )

        # add dummy world node
        world_node = scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0))

        # add camera
        scene.main_camera_node = scene.add(
            pyrender.IntrinsicsCamera(
                fx=cam_K[0, 0],
                fy=cam_K[1, 1],
                cx=cam_K[0, 2],
                cy=cam_K[1, 2],
                znear=self._znear,
                zfar=self._zfar,
            ),
            name="camera",
            pose=pose_c,
            parent_node=world_node,
        )

        # add meshes
        for i, mesh in enumerate(meshes):
            scene.add(
                pyrender.Mesh.from_trimesh(mesh),
                name=f"mesh_{i}",
                pose=poses_m[i],
                parent_node=world_node,
            )
        # render
        r = pyrender.OffscreenRenderer(self._width, self._height)
        color, depth = r.render(
            scene,
            flags=RenderFlags.ALL_SOLID,
        )
        r.delete()

        if idx is not None:
            return color, depth, idx
        return color, depth


class SequenceRenderer(OffscreenRenderer):
    def __init__(self, width, height, znear=0.001, zfar=1000, pose_type="cv") -> None:
        super().__init__(width, height, znear, zfar, pose_type)

    def render_sequence(
        self,
        mano_meshes=None,
        object_meshes=None,
        object_poses=None,
        cam_K=None,
        cam_pose=None,
        width=None,
        height=None,
        idx=None,
    ):
        scene = pyrender.Scene(
            bg_color=self._bg_color, ambient_light=self._ambient_light
        )

        # add dummy world node
        world_node = scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0))

        # add camera
        scene.main_camera_node = scene.add(
            pyrender.IntrinsicsCamera(
                fx=cam_K[0, 0],
                fy=cam_K[1, 1],
                cx=cam_K[0, 2],
                cy=cam_K[1, 2],
                znear=self._znear,
                zfar=self._zfar,
            ),
            name="camera",
            pose=cam_pose if self._pose_type == "gl" else cam_pose @ self._cvcam2glcam,
            parent_node=world_node,
        )

        # add mano meshes
        if mano_meshes is not None:
            for i, mesh in enumerate(mano_meshes):
                scene.add(
                    pyrender.Mesh.from_trimesh(mesh),
                    name=f"mano_{i}",
                    pose=np.eye(4),
                    parent_node=world_node,
                )

        # add object meshes
        if object_meshes is not None:
            for i, mesh in enumerate(object_meshes):
                scene.add(
                    pyrender.Mesh.from_trimesh(mesh),
                    name=f"object_{i}",
                    pose=object_poses[i],
                    parent_node=world_node,
                )

        # render
        if width is not None and height is not None:
            r = pyrender.OffscreenRenderer(width, height)
        else:
            r = pyrender.OffscreenRenderer(self._width, self._height)
        color, _ = r.render(
            scene,
            flags=RenderFlags.ALL_SOLID,
        )
        r.delete()

        if idx is not None:
            return color, idx
        return color

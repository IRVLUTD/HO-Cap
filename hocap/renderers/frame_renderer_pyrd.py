import os

os.environ["PYOPENGL_PLATFORM"] = "egl"  # GPU-based offscreen rendering

import numpy as np
import pyrender


class FrameRenderer:
    def __init__(self, cam_K, cam_RT, image_size):
        self.cam_K = cam_K
        self.cam_RT = cam_RT
        self.image_size = image_size
        self.meshes = []

    def add_mesh(self, tri_mesh):
        pyrender_mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        self.meshes.append(pyrender_mesh)

    def render_scene(self):
        scene = pyrender.Scene(
            bg_color=np.array([0.0, 0.0, 0.0, 1.0]),
            ambient_light=np.array([1.0, 1.0, 1.0, 1.0]),
        )
        for mesh in self.meshes:
            scene.add(mesh)

        camera = pyrender.IntrinsicsCamera(
            fx=self.cam_K[0, 0],
            fy=self.cam_K[1, 1],
            cx=self.cam_K[0, 2],
            cy=self.cam_K[1, 2],
        )
        camera_node = pyrender.Node(camera=camera, matrix=self.cam_RT)
        scene.add_node(camera_node)

        renderer = pyrender.OffscreenRenderer(
            viewport_width=self.image_size[0], viewport_height=self.image_size[1]
        )
        _, depth = renderer.render(scene)
        renderer.delete()

        return depth

    def render_mesh_with_depth(self, mesh, depth_ref):
        scene = pyrender.Scene(
            bg_color=np.array([0.0, 0.0, 0.0, 1.0]),
            ambient_light=np.array([1.0, 1.0, 1.0, 1.0]),
        )
        scene.add(mesh)

        camera = pyrender.IntrinsicsCamera(
            fx=self.cam_K[0, 0],
            fy=self.cam_K[1, 1],
            cx=self.cam_K[0, 2],
            cy=self.cam_K[1, 2],
        )
        camera_node = pyrender.Node(camera=camera, matrix=self.cam_RT)
        scene.add_node(camera_node)

        renderer = pyrender.OffscreenRenderer(
            viewport_width=self.image_size[0], viewport_height=self.image_size[1]
        )
        _, depth = renderer.render(scene)
        renderer.delete()

        mask = (depth > 0) & (depth <= depth_ref)
        return mask

    def get_label_mask(self):
        depth_ref = self.render_scene()
        label_mask = np.zeros(self.image_size, dtype=np.uint8)

        for i, mesh in enumerate(self.meshes):
            mask = self.render_mesh_with_depth(mesh, depth_ref)
            label_mask[mask] = i + 1  # Label IDs start from 1

        return label_mask

    def get_rendered_scene(self):
        scene = pyrender.Scene(
            bg_color=np.array([0.0, 0.0, 0.0, 1.0]),
            ambient_light=np.array([1.0, 1.0, 1.0, 1.0]),
        )
        for mesh in self.meshes:
            scene.add(mesh)

        camera = pyrender.IntrinsicsCamera(
            fx=self.cam_K[0, 0],
            fy=self.cam_K[1, 1],
            cx=self.cam_K[0, 2],
            cy=self.cam_K[1, 2],
        )
        camera_node = pyrender.Node(camera=camera, matrix=self.cam_RT)
        scene.add_node(camera_node)

        renderer = pyrender.OffscreenRenderer(
            viewport_width=self.image_size[0], viewport_height=self.image_size[1]
        )
        color, depth = renderer.render(scene)
        renderer.delete()

        return color, depth

import numpy as np
from tqdm import tqdm
import trimesh
import torch

from _init_paths import *
from lib.utils import *
from lib.utils.common import mat_to_rvt
from lib.mesh_to_sdf import get_surface_point_cloud
from lib.optas.visualize import Visualizer
from lib.optas.spatialmath import angvec2r
from lib.layers.object_layer import ObjectLayer
from pose_solver import SDFTaskModel, PoseSolver


"""
The MeshSDFLoss will return three values:
1. loss: the loss value
2. dist: the distance of each point to the mesh
3. asso: the index of the nearest face of each point

** When the dist is minimized, the sdf is optimized.
** I also use 'dist < dist_thresh' to mask the points that are too far from the mesh.
** The asso is helpful to distinguish which (point, face) pair is used to calculate the dist.
"""


class MyLoader:
    def __init__(self, sequence_folder, device="cuda") -> None:
        self.data_folder = Path(sequence_folder).resolve()
        self.device = device

        self.poses = self.load_poses()
        self.points = self.load_points()
        self.cloud, self.object_mesh, self.mesh_file = self.init_object_point_cloud()
        self.object_layer = self.init_object_layer()       

        self.num_frames = len(self.points)

    def load_poses(self):
        pose_files = sorted(self.data_folder.glob("pose/pose_*.txt"))
        poses = [mat_to_rvt(np.loadtxt(str(f))) for f in pose_files]
        return [torch.from_numpy(p).to(self.device) for p in poses]

    def load_points(self):
        point_files = sorted(self.data_folder.glob("point/points_*.npy"))
        points = [np.load(str(f)) for f in point_files]
        return [torch.from_numpy(pts).to(self.device) for pts in points]

    def init_object_point_cloud(self):
        mesh_file = self.data_folder / "mesh" / "cleaned_mesh_10000.obj"
        m = trimesh.load(str(mesh_file), process=False)
        print("Scanning...", mesh_file)
        cloud = get_surface_point_cloud(m, surface_point_method='scan', scan_count=20, scan_resolution=400)
        return cloud, m, mesh_file
    
    def init_object_layer(self):
        mesh_file = self.data_folder / "mesh" / "cleaned_mesh_10000.obj"
        m = trimesh.load(str(mesh_file), process=False)
        verts = m.vertices
        faces = m.faces
        norms = m.vertex_normals
        object_layer = ObjectLayer(verts, faces, norms).to(self.device)
        return object_layer


if __name__ == "__main__":
    sequence_folder = PROJ_ROOT / "data/for_optas/sequences/20231024_180111"
    loader = MyLoader(sequence_folder, device="cuda")

    # task model
    name = 'object_pose_estimator'
    dim = 6
    task_model = SDFTaskModel(name, dim)
    points = loader.cloud.points
    normals = loader.cloud.normals
    task_model.setup_points_field(points)

    # sdf field
    world_points = task_model.workspace_points
    sdf_cost = loader.cloud.get_sdf(world_points)
    sdf_cost = sdf_cost.reshape(task_model.field_shape)

    # pose solver
    pose_solver = PoseSolver(task_model, sdf_cost)
    # sample 5000 points for optimization, we need to setup optimization with fixed number of points
    num_points = 5000
    pose_solver.setup_optimization(num_points)
    print('finish setup optimization')

    # initial pose
    start = 0
    pose_0 = loader.poses[start].unsqueeze(0)
    RT = np.eye(4, dtype=np.float32)
    rv = pose_0[0, :3].cpu().numpy()
    angle = np.linalg.norm(rv)
    axis = rv / angle
    RT[:3, :3] = angvec2r(angle, axis)
    RT[:3, 3] = pose_0[:, 3:].cpu().numpy()
    RT_before = RT.copy()

    # optimize the pose
    for frame_id in range(start, loader.num_frames, 50):
        # object points
        dpts = loader.points[frame_id]

        # solve pose
        RT_inv = np.linalg.inv(RT_before)
        points = dpts.cpu().numpy()
        print('frame %d, num points %d' % (frame_id, points.shape[0]))

        # sample points
        index = np.random.choice(points.shape[0], size=num_points, replace=True)
        y = pose_solver.solve_pose(RT_inv, points[index])

        # construct RT
        angle = np.linalg.norm(y[:3])
        axis = y[:3] / angle
        RT[:3, :3] = angvec2r(angle, axis)        
        RT[:3, 3] = y[3:]
        RT = np.linalg.inv(RT)
        print('before:')
        print(RT_before)
        print('after:')
        print(RT)
        RT_before = RT.copy()
        
        # visualize
        RT_inv = np.linalg.inv(RT)
        points_tf = RT_inv[:3, :3] @ points.T + RT_inv[:3, 3].reshape((3, 1))
        vis = Visualizer(camera_position=[3, 2, 4])
        vis.grid_floor()
        vis.obj(
            loader.mesh_file,
        )
        # vis.points(
        #     world_points,
        #     rgb = [0, 1, 0],
        #     size=1,
        # )
        vis.points(
            points_tf.T,
            rgb = [1, 0, 0],
            size=5,
        )              
        vis.start()
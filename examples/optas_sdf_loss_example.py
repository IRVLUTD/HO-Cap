import numpy as np
from tqdm import tqdm
import trimesh
import torch
from transforms3d.quaternions import mat2quat, quat2mat

from _init_paths import *
from lib.utils import *
from lib.utils.common import mat_to_rvt
from lib.mesh_to_sdf import get_surface_point_cloud
from lib.optas.visualize import Visualizer
from lib.layers.object_layer import ObjectLayer
from lib.layers.meshsdf_loss import MeshSDFLoss
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
        self.field_margin = 0.4
        self.grid_resolution = 0.05        

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
    dim = 7
    task_model = SDFTaskModel(name, dim)
    points = loader.cloud.points
    task_model.setup_points_field(points)

    # sdf field
    world_points = task_model.workspace_points
    sdf_cost = loader.cloud.get_sdf(world_points)
    print(sdf_cost, sdf_cost.shape)

    # pose solver
    pose_solver = PoseSolver(task_model)

    # optimize the pose
    for frame_id in range(0, loader.num_frames, 50):
        pose_0 = loader.poses[frame_id].unsqueeze(0)
        dpts = loader.points[frame_id]

        # construct pose
        RT = np.eye(4, dtype=np.float32)
        RT[:3, :3] = loader.object_layer.rv2dcm(pose_0[:, :3])[0].cpu().numpy()
        RT[:3, 3] = pose_0[:, 3:].cpu().numpy()
        print('before', RT, RT.shape)

        # solve pose
        RT_inv = np.linalg.inv(RT)
        points = dpts.cpu().numpy()
        pose_solver.setup_optimization(num_points=points.shape[0])   
        y = pose_solver.solve_pose(RT_inv, points, sdf_cost)

        # construct RT
        RT[:3, :3] = quat2mat([y[3], y[0], y[1], y[2]])
        RT[:3, 3] = y[4:]
        RT = np.linalg.inv(RT)
        print('after', RT, RT.shape)
        
        # visualization
        quat = mat2quat(RT[:3, :3])
        # scalar-last (x, y, z, w) format in optas
        orientation = [quat[1], quat[2], quat[3], quat[0]]

        vis = Visualizer(camera_position=[3, 2, 4])
        vis.grid_floor()
        vis.obj(
            loader.mesh_file,
            position=RT[:3, 3],
            orientation=orientation,            
        )
        vis.points(
            dpts.cpu().numpy(),
            rgb = [1, 0, 0],
            size=5,
        )        
        vis.start()
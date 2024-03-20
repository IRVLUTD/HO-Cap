import numpy as np
from tqdm import tqdm
import trimesh
import torch

from _init_paths import *
from lib.utils import *
from lib.utils.common import mat_to_rvt
from lib.layers.object_layer import ObjectLayer
from lib.layers.meshsdf_loss import MeshSDFLoss

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

    def init_object_layer(self):
        mesh_file = self.data_folder / "mesh" / "cleaned_mesh_10000.obj"
        m = trimesh.load(str(mesh_file), process=False)
        verts = m.vertices
        faces = m.faces
        norms = m.vertex_normals
        object_layer = ObjectLayer(verts, faces, norms).to(self.device)
        return object_layer


class PoseOptimizer:
    def __init__(self, object_layer, device="cuda") -> None:
        self.device = device
        self.dist_thresh = 0.03
        self.object_layer = object_layer
        self.faces = object_layer.f

        self.optim_pose = torch.tensor(
            [[0] * 6], dtype=torch.float32, device=device, requires_grad=True
        )
        self.meshsdf_loss = MeshSDFLoss().to(self.device)
        self.optimizer = torch.optim.Adam([self.optim_pose], lr=1e-2)

    def solve(self, pose_0, dpts, steps=100):
        loss_log = []
        self.optim_pose.data = pose_0.clone()

        for step in tqdm(range(steps), ncols=60):
            self.optimizer.zero_grad()
            verts, _ = self.object_layer_forward(self.optim_pose)

            loss = self.loss_sdf(verts, self.faces, dpts)
            loss.backward()
            self.optimizer.step()

            loss_log.append(loss.item())

            if (step + 1) % 10 == 0:
                tqdm.write(f"{step+1:04d}/{steps:04d} | sdf_loss={loss.item():11.8f}")

        loss_log = np.array(loss_log)
        optim_pose = self.optim_pose.detach().cpu().numpy()
        return optim_pose, loss_log

    def loss_sdf(self, verts, faces, dpts):
        loss, _, _ = self.meshsdf_loss(verts, faces, dpts)
        loss *= 1e6 / dpts.size(0)
        return loss

    def get_sdf_mask(self, verts, faces, dpts):
        _, dist, _ = self.meshsdf_loss(verts, faces, dpts)
        mask = dist < self.dist_thresh
        return mask

    def object_layer_forward(self, pose):
        verts, norms = self.object_layer(pose[:, :3], pose[:, 3:])
        verts = verts.contiguous()
        norms = norms.contiguous()
        if pose.size(0) == 1:
            verts = verts.squeeze(0)
            norms = norms.squeeze(0)
        return verts, norms


if __name__ == "__main__":
    sequence_folder = PROJ_ROOT / "data/for_optas/sequences/20231024_180111"
    loader = MyLoader(sequence_folder, device="cuda")
    optimizer = PoseOptimizer(loader.object_layer, device="cuda")

    # optimize the pose
    for frame_id in range(loader.num_frames):
        pose_0 = loader.poses[frame_id].unsqueeze(0)
        dpts = loader.points[frame_id]
        optim_pose, loss_log = optimizer.solve(pose_0, dpts, steps=100)
        print(optim_pose)
        break

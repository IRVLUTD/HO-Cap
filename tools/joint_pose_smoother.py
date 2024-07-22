import open3d as o3d
from open3d import core as o3c

import numpy as np
import torch
from torch.utils.dlpack import to_dlpack, from_dlpack
import trimesh
import time
from tqdm import tqdm
import concurrent.futures
import argparse

from _init_paths import *
from lib.utils import *
from lib.utils.common import *
from lib.layers import MANOGroupLayer, ObjectGroupLayer, MeshSDFLoss
from lib.loaders import SequenceLoader
from lib.renderers.renderer_pyrd import SequenceRenderer


# force the multiprocessing start method to spawn
import multiprocessing

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass


OPTIM_CONFIG = {
    "learning_rate": 0.001,
    "lambda_sdf": 1.0,
    "lambda_reg": 1.0,
    "lambda_smooth": 0.0,
    "lambda_robust": 0.0,
    "window_size": 3,
    "steps": 10,
    "dist_thresh": 0.015,  # 1.5 cm
}


def runner_draw_rendered_image_by_frame_id(
    image_width,
    image_height,
    rgb_images,
    object_meshes,
    object_poses,
    cam_Ks,
    cam_Poses,
    idx=None,
):
    renderer = SequenceRenderer(image_width, image_height)

    i = 3
    vis_image = renderer.render_sequence(
        object_meshes=object_meshes,
        object_poses=object_poses,
        cam_K=cam_Ks[i],
        cam_pose=cam_Poses[i],
    )
    vis_image = cv2.addWeighted(rgb_images[i], 0.3, vis_image, 0.7, 0)
    return vis_image, idx

    # vis_images = []
    # for i in range(len(cam_Ks)):
    #     vis_image = renderer.render_sequence(
    #         object_meshes=object_meshes,
    #         object_poses=object_poses,
    #         cam_K=cam_Ks[i],
    #         cam_pose=cam_Poses[i],
    #     )
    #     vis_image = cv2.addWeighted(rgb_images[i], 0.3, vis_image, 0.7, 0)
    #     vis_images.append(vis_image)
    # if idx is None:
    #     return display_images(vis_images, facecolor="black")
    # return display_images(vis_images, facecolor="black", return_array=True), idx


def runner_loss_sdf(verts, faces, dpts, idx=None):
    meshsdf_loss = MeshSDFLoss()
    loss, _, _ = meshsdf_loss(verts, faces, dpts)
    loss /= dpts.size(0)
    if idx is None:
        return loss
    return loss, idx


class ObjectPoseSolver:
    def __init__(
        self,
        mesh_files,
        mano_sides,
        mano_beta,
        optim_config=OPTIM_CONFIG,
        device="cuda",
    ) -> None:
        self._device = device

        # load optimization config
        self._lr = optim_config["learning_rate"]
        self._w_sdf = optim_config["lambda_sdf"]
        self._w_reg = optim_config["lambda_reg"]
        self._w_smooth = optim_config["lambda_smooth"]
        self._w_robust = optim_config["lambda_robust"]
        self._window_size = optim_config["window_size"]
        self._steps = optim_config["steps"]
        self._dist_thresh = optim_config["dist_thresh"]

        # initialize object group layer
        self._object_group_layer = self._init_object_group_layer(mesh_files)

        # initialize mano group layer
        self._mano_group_layer = self._init_mano_group_layer(mano_sides, mano_beta)

        self._meshsdf_loss = MeshSDFLoss()
        self._mse_loss = torch.nn.MSELoss(reduction="sum").to(self._device)
        self._zero = torch.zeros(
            (), dtype=torch.float32, device=self._device, requires_grad=True
        )

    def _init_object_group_layer(self, mesh_files):
        verts = []
        faces = []
        norms = []
        for mesh_file in mesh_files:
            mesh = trimesh.load_mesh(str(mesh_file), process=False)
            verts.append(mesh.vertices)
            faces.append(mesh.faces)
            norms.append(mesh.vertex_normals)
        object_group_layer = ObjectGroupLayer(verts, faces, norms).to(self._device)
        return object_group_layer

    def _init_mano_group_layer(self, mano_sides, mano_beta):
        mano_betas = [mano_beta for _ in mano_sides]
        mano_group_layer = MANOGroupLayer(mano_sides, mano_betas).to(device)
        return mano_group_layer

    def _object_group_layer_forward(self, subset=None):
        p = torch.cat(self._pose_o, dim=1)
        v, n = self._object_group_layer(p, subset)
        if p.size(0) == 1:
            v = v.squeeze(0)
            n = n.squeeze(0)
        return v, n

    def _mano_group_layer_forward(self, subset=None):
        p = torch.cat(self._pose_m, dim=1)
        v, j = self._mano_group_layer(p, inds=subset)
        if p.size(0) == 1:
            v = v.squeeze(0)
            j = j.squeeze(0)
        return v, j

    def _get_dpts_for_loss_sdf(self, verts, faces, dpts):
        _, dist, _ = self._meshsdf_loss(verts, faces, dpts)
        mask = dist < self._dist_thresh
        return dpts[mask]

    def _loss_sdf(self, verts, faces, sdf_dpts):
        num_frames = len(sdf_dpts)
        loss = 0.0
        workers = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for i in range(num_frames):
                workers.append(
                    executor.submit(
                        runner_loss_sdf,
                        verts[i],
                        faces,
                        sdf_dpts[i],
                        idx=i,
                    )
                )
            for worker in concurrent.futures.as_completed(workers):
                l, i = worker.result()
                loss += l
        del workers

        loss /= num_frames
        return loss

    def _loss_reg(self):
        p = torch.stack(self._pose_m, dim=0)[..., 3:48]
        p_zeros = torch.zeros_like(p)
        loss = self._mse_loss(p, p_zeros)
        loss /= p.size(0) * p.size(1) * p.size(2)
        return loss

    def _loss_temporal_robust_o(self, delta=1.0):
        """Loss for robust temporal smoothing of poses.

        Args:
            delta (float): Threshold for the linear part of the loss function.
        """
        poses = torch.stack(self._pose_o, dim=0)
        num_hands, num_frames, pose_dim = poses.shape
        loss = 0.0
        for k in range(num_hands):
            for n in range(1, num_frames):
                diff = poses[k, n] - poses[k, n - 1]
                abs_diff = torch.abs(diff)
                quadratic_part = torch.minimum(
                    abs_diff, torch.full_like(abs_diff, delta)
                )
                linear_part = abs_diff - quadratic_part
                loss += torch.sum(0.5 * quadratic_part**2 + delta * linear_part)
        loss /= num_hands * num_frames * pose_dim
        return loss

    def _loss_temporal_robust_m(self, delta=1.0):
        """Loss for robust temporal smoothing of poses.

        Args:
            delta (float): Threshold for the linear part of the loss function.
        """
        p = torch.stack(self._pose_m, dim=0)  # Assuming _pose_m is a list of tensors
        num_hands, num_frames, pose_dim = p.shape

        # Calculate differences between consecutive frames, along the frame dimension
        diffs = p[:, 1:] - p[:, :-1]

        # Calculate the robust loss components
        abs_diffs = torch.abs(diffs)
        quadratic_part = torch.minimum(abs_diffs, delta * torch.ones_like(abs_diffs))
        linear_part = abs_diffs - quadratic_part

        # Compute the robust loss function
        loss = 0.5 * quadratic_part**2 + delta * linear_part
        total_loss = torch.sum(loss)  # Sum over all elements

        # Normalize the loss by the number of elements contributing to the loss
        # Use num_frames - 1 because diff is between consecutive frames
        total_loss /= num_hands * (num_frames - 1)

        return total_loss

    def _loss_temporal_smooth_o(self, window_size=1, w_rot=1.0, w_trans=1.0):
        """Loss for temporal smoothing of poses using L2 norm.

        Args:
            poses (torch.Tensor): Poses of shape (num_objects, num_frames, 7).
            window_size (int): Number of frames on either side to include in the smoothing.
            weight_rotation (float): Weight for the rotation loss.
            weight_translation (float): Weight for the translation loss.
        """
        poses = torch.stack(self._pose_o, dim=0)
        num_objects, num_frames, pose_dim = poses.shape
        rotation_loss = 0.0
        translation_loss = 0.0

        for k in range(num_objects):
            for n in range(window_size, num_frames - window_size):
                for j in range(1, window_size + 1):
                    rotation_loss += torch.norm(
                        poses[k, n, :3] - poses[k, n - j, :3], p=2
                    )
                    rotation_loss += torch.norm(
                        poses[k, n, :3] - poses[k, n + j, :3], p=2
                    )
                    translation_loss += torch.norm(
                        poses[k, n, -3:] - poses[k, n - j, -3:], p=2
                    )
                    translation_loss += torch.norm(
                        poses[k, n, -3:] - poses[k, n + j, -3:], p=2
                    )

        total_loss = w_rot * rotation_loss + w_trans * translation_loss
        total_loss /= num_objects * (num_frames - 2 * window_size) * pose_dim
        return total_loss

    def _loss_temporal_smooth_m(self, window_size=1, w_rot=1.0, w_trans=1.0):
        """Loss for temporal smoothing of poses using L2 norm.

        Args:
            poses (torch.Tensor): Poses of shape (num_objects, num_frames, 7).
            window_size (int): Number of frames on either side to include in the smoothing.
            weight_rotation (float): Weight for the rotation loss.
            weight_translation (float): Weight for the translation loss.
        """
        poses = torch.stack(self._pose_m, dim=0)
        num_objects, num_frames, pose_dim = poses.shape
        rotation_loss = 0.0
        translation_loss = 0.0

        for k in range(num_objects):
            for n in range(window_size, num_frames - window_size):
                for j in range(1, window_size + 1):
                    rotation_loss += torch.norm(
                        poses[k, n, :3] - poses[k, n - j, :3], p=2
                    )
                    rotation_loss += torch.norm(
                        poses[k, n, :3] - poses[k, n + j, :3], p=2
                    )
                    translation_loss += torch.norm(
                        poses[k, n, -3:] - poses[k, n - j, -3:], p=2
                    )
                    translation_loss += torch.norm(
                        poses[k, n, -3:] - poses[k, n + j, -3:], p=2
                    )

        total_loss = w_rot * rotation_loss + w_trans * translation_loss
        total_loss /= num_objects * (num_frames - 2 * window_size) * pose_dim
        return total_loss

    def solve(
        self,
        batch_pose_o_0,
        batch_pose_m_0,
        batch_dpts,
        subset_o=None,
        subset_m=None,
    ):
        # initialize optimizer
        self._pose_o = [
            torch.nn.Parameter(p.clone(), requires_grad=True) for p in batch_pose_o_0
        ]
        self._pose_m = [
            torch.nn.Parameter(p.clone(), requires_grad=True) for p in batch_pose_m_0
        ]
        self._optimizer = torch.optim.Adam(self._pose_o + self._pose_m, lr=self._lr)

        # get faces
        faces_o, _ = self._object_group_layer.get_f_from_inds(subset_o)
        faces_m, _ = self._mano_group_layer.get_f_from_inds(subset_m)
        faces = torch.cat(
            [
                faces_o,
                faces_m + self._object_group_layer.get_num_verts_from_inds(subset_o),
            ]
        )
        # tqdm.write(f"faces: {faces.size()}")

        num_frames = len(batch_dpts)
        losses = []

        for step in tqdm(range(self._steps), ncols=60, colour="green"):
            t_s = time.time()

            self._optimizer.zero_grad()

            verts_o, _ = self._object_group_layer_forward(subset_o)
            # tqdm.write(f"verts_o: {verts_o.size()}")
            verts_m, _ = self._mano_group_layer_forward(subset_m)
            # tqdm.write(f"verts_m: {verts_m.size()}")
            verts = torch.cat([verts_o, verts_m], dim=1)
            # tqdm.write(f"verts: {verts.size()}")

            # get dpts for sdf loss
            if step == 0:
                sdf_dpts = [
                    self._get_dpts_for_loss_sdf(verts, faces, dpts)
                    for dpts in batch_dpts
                ]
                tqdm.write(f"batch_dpts: {len(sdf_dpts)}")

            # compute sdf loss
            if self._w_sdf == 0:
                loss_sdf = self._zero
            else:
                loss_sdf = self._loss_sdf(verts, faces, sdf_dpts)
                loss_sdf *= self._w_sdf
            # tqdm.write(f"loss_sdf: {loss_sdf.item()}")

            if self._w_reg == 0:
                loss_reg = self._zero
            else:
                loss_reg = self._loss_reg()
                loss_reg *= self._w_reg
            # tqdm.write(f"loss_reg: {loss_reg.item()}")

            if self._w_smooth == 0:
                loss_smooth = self._zero
            else:
                loss_smooth = self._loss_temporal_smooth_o(
                    self._window_size
                ) + self._loss_temporal_smooth_m(self._window_size)
                loss_smooth *= self._w_smooth
            # tqdm.write(f"loss_smooth: {loss_smooth.item()}")

            if self._w_robust == 0:
                loss_robust = self._zero
            else:
                loss_robust = self._loss_temporal_robust_o(
                    delta=1.0
                ) + self._loss_temporal_robust_m(delta=1.0)
                loss_robust *= self._w_robust
            # tqdm.write(f"loss_robust: {loss_robust.item()}")

            loss = loss_sdf + loss_reg + loss_smooth + loss_robust

            # Set grad to None to prevent Adam from updating the parameters even when
            # the grad are all zeros. An alternative is to place this before the
            # forward pass, but the currently implementation of group layers does not
            # allow this since the non-active poses will still always be used due to
            # concatenation and hence their grad will be 0.0 rather than None.
            for i, p in enumerate(self._pose_o):
                if i not in subset_o and p.grad is not None:
                    assert torch.all(p.grad == 0.0)
                    p.grad = None
            for i, p in enumerate(self._pose_m):
                if i not in subset_m and p.grad is not None:
                    assert torch.all(p.grad == 0.0)
                    p.grad = None

            loss.backward()
            self._optimizer.step()

            t_e = time.time()

            losses.append(
                [
                    loss.item(),
                    loss_sdf.item(),
                    loss_reg.item(),
                    loss_smooth.item(),
                    loss_robust.item(),
                ]
            )

            if (step + 1) % 1 == 0:
                tqdm.write(
                    f"loss: {loss.item():11.8f} "
                    + f"| loss_sdf: {loss_sdf.item():11.8f} "
                    + f"| loss_reg: {loss_reg.item():11.8f} "
                    + f"| loss_smooth: {loss_smooth.item():11.8f} "
                    + f"| loss_robust: {loss_robust.item():11.8f} "
                    + f"| time: {t_e - t_s:.2f}s"
                )

        losses = np.array(losses, dtype=np.float32)
        pose_o = (
            torch.stack([p.data for p in self._pose_o], dim=1).squeeze(0).cpu().numpy()
        )
        _pose_m = (
            torch.stack([p.data for p in self._pose_m], dim=1).squeeze(0).cpu().numpy()
        )
        pose_m = np.full((num_frames, 2, 51), -1, dtype=np.float32)
        for idx, mano_side in enumerate(mano_sides):
            if mano_side == "right":
                pose_m[:, 0] = _pose_m[:, idx]
            elif mano_side == "left":
                pose_m[:, 1] = _pose_m[:, idx]

        return pose_o, pose_m, losses


def process_points(
    points, colors=None, voxel_size=0.0, nb_neighbors=100, std_ratio=1.0
):
    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3c.Tensor.from_dlpack(to_dlpack(points.cpu()))
    if colors is not None:
        pcd.point.colors = o3c.Tensor.from_dlpack(to_dlpack(colors.cpu()))
    if voxel_size > 0.0:
        pcd = pcd.voxel_down_sample(voxel_size)
    pcd, mask = pcd.remove_statistical_outliers(nb_neighbors, std_ratio)
    # mask = torch.from_numpy(mask.cpu().numpy()).to(points.device)
    pts = from_dlpack(pcd.point.positions.to_dlpack()).to(points)
    if colors is not None:
        clrs = from_dlpack(pcd.point.colors.to_dlpack())
        return pts, clrs
    return pts


def args_parser():
    parser = argparse.ArgumentParser(description="Object Pose Smoother")
    parser.add_argument(
        "--sequence_folder",
        type=str,
        required=True,
        help="Path to the sequence folder.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the optimization.",
    )
    # parser.add_argument(
    #     "--dpts_thresh",
    #     type=float,
    #     default=100,
    #     help="Distance threshold for the depth points.",
    # )
    # parser.add_argument(
    #     "--render_results",
    #     action="store_true",
    #     help="Render the results.",
    # )
    # parser.add_argument(
    #     "--debug",
    #     action="store_true",
    #     help="Debug mode.",
    # )
    return parser.parse_args()


if __name__ == "__main__":
    dpts_thresh = 100
    render_results = True
    debug = True

    sequence_folder = (
        PROJ_ROOT / "data/HOT_Dataset/release_dataset/subject_1/20231025_165502"
    )
    device = "cuda"

    # args = args_parser()
    # sequence_folder = Path(args.sequence_folder).resolve()
    # device = args.device

    # load sequence
    loader = SequenceLoader(sequence_folder, device=device)
    num_frames = loader.num_frames
    mano_sides = loader.mano_sides
    object_ids = loader.object_ids
    mesh_files = loader.object_cleaned_mesh_files

    # load object initial poses
    poses_o_file = sequence_folder / "fd_poses_new.npy"
    poses_m_file = sequence_folder / "hamer_poses.npy"
    mano_beta_files = sequence_folder.parent.glob("2023*/hamer_betas.npy")
    poses_o = np.load(poses_o_file)
    poses_m = np.load(poses_m_file)
    mano_beta = np.mean([np.load(f) for f in mano_beta_files], axis=0)

    # initialize solver
    solver = ObjectPoseSolver(mesh_files, mano_sides, mano_beta, device=device)

    # maske save folder
    save_folder = sequence_folder / "processed" / "pose_solver" / "joint_pose_smooth"
    save_folder.mkdir(parents=True, exist_ok=True)

    save_dpts_folder = save_folder / "dpts"
    save_dpts_folder.mkdir(parents=True, exist_ok=True)

    # >>>>>>>>>> prepare data for pose optimization >>>>>>>>>>
    batch_pose_o_0 = []
    batch_pose_m_0 = []
    batch_dpts = []
    subset_o = list(range(len(object_ids)))
    subset_m = list(range(len(mano_sides)))

    tqdm.write("Preparing data for optimization...")
    for frame_id in tqdm(range(num_frames), ncols=60, colour="green"):
        dpts_file = save_dpts_folder / f"dpts_{frame_id:06d}.npy"
        if dpts_file.exists():
            dpts = np.load(dpts_file)
            dpts = torch.from_numpy(dpts).to(device)
        else:
            # prepare dpts
            loader.step_by_frame_id(frame_id)
            pcd_points = loader.points[loader.masks].view(-1, 3)
            # process points
            dpts = process_points(pcd_points, voxel_size=0.005)
            # save dpts
            np.save(dpts_file, dpts.cpu().numpy())
        batch_dpts.append(dpts)

        # prepare initial object poses
        pose_o_0 = np.stack([quat_to_rvt(ps) for ps in poses_o[:, frame_id]])
        pose_o_0 = torch.from_numpy(pose_o_0).unsqueeze(0).to(device)
        batch_pose_o_0.append(pose_o_0)

        # prepare initial mano poses
        pose_m_0 = np.stack(
            [poses_m[0 if side == "right" else 1, frame_id] for side in mano_sides]
        )
        pose_m_0 = torch.from_numpy(pose_m_0).unsqueeze(0).to(device)
        batch_pose_m_0.append(pose_m_0)

    batch_pose_o_0 = torch.cat(batch_pose_o_0, dim=0)
    batch_pose_o_0 = torch.permute(batch_pose_o_0, (1, 0, 2))
    batch_pose_m_0 = torch.cat(batch_pose_m_0, dim=0)
    batch_pose_m_0 = torch.permute(batch_pose_m_0, (1, 0, 2))

    # >>>>>>>>>> solve object poses >>>>>>>>>>

    # solve object pose
    optim_poses_o, optim_poses_m, losses_o = solver.solve(
        batch_pose_o_0=batch_pose_o_0,
        batch_pose_m_0=batch_pose_m_0,
        batch_dpts=batch_dpts,
        subset_o=subset_o,
        subset_m=subset_m,
    )

    tqdm.write("Saving results...")
    # draw loss curve
    draw_losses_curve(
        losses=losses_o,
        loss_names=[
            "total",
            "sdf",
            "reg",
            "smooth",
            "robust",
        ],
        save_path=save_folder / "loss_curve.png",
    )
    np.save(save_folder / "losses.npy", losses_o)

    # save solved poses
    optim_poses_o = np.swapaxes(optim_poses_o, 0, 1)  # (num_objects, num_frames, 7)
    optim_poses_o = np.stack([rvt_to_quat(ps) for ps in optim_poses_o])
    np.save(save_folder / "poses_o.npy", optim_poses_o)

    optim_poses_m = np.swapaxes(optim_poses_m, 0, 1)  # (num_hands, num_frames, 51)
    np.save(save_folder / "poses_m.npy", optim_poses_m)

    tqdm.write(">>>>>>>>>> Done!!! <<<<<<<<<<")

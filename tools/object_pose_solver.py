import open3d as o3d
from open3d import core as o3c

import numpy as np
import torch
from torch.utils.dlpack import to_dlpack, from_dlpack
import trimesh
import time
from tqdm import tqdm

from _init_paths import *
from lib.utils import *
from lib.utils.common import *
from lib.layers import ObjectGroupLayer, MeshSDFLoss
from lib.loaders import SequenceLoader
from lib.renderers.renderer_pyrd import SequenceRenderer

OPTIM_CONFIG = {
    "learning_rate": 0.005,
    "lambda_sdf": 1.0,
    "lambda_reg": 1.0,
}


class ObjectPoseSolver:
    def __init__(self, mesh_files, optim_config=OPTIM_CONFIG, device="cuda") -> None:
        self._device = device

        # load optimization config
        self._lr = optim_config["learning_rate"]
        self._w_sdf = optim_config["lambda_sdf"]
        self._w_reg = optim_config["lambda_reg"]

        # initialize object group layer
        self._object_group_layer = self._init_object_group_layer(mesh_files)

        # initialize optimizer
        self._meshsdf_loss = MeshSDFLoss()
        self._mse_loss = torch.nn.MSELoss(reduction="sum").to(self._device)
        self._zero = torch.zeros(
            (), dtype=torch.float32, device=self._device, requires_grad=True
        )
        self._pose_o = [
            torch.tensor(
                [[0.0] * 6],
                dtype=torch.float32,
                device=self._device,
                requires_grad=True,
            )
            for _ in range(self._object_group_layer.num_obj)
        ]
        self._optimizer = torch.optim.Adam(self._pose_o, lr=self._lr)

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

    def _object_group_layer_forward(self, subset=None):
        p = torch.cat(self._pose_o, dim=1)
        v, n = self._object_group_layer(p, subset)
        if p.size(0) == 1:
            v = v.squeeze(0)
            n = n.squeeze(0)
        return v, n

    def _get_sdf_dpts_by_dist_thresh(self, verts, faces, dpts, dist_thresh):
        _, dist, _ = self._meshsdf_loss(verts, faces, dpts)
        mask = dist < dist_thresh
        return dpts[mask]

    def _loss_sdf(self, verts, faces, dpts):
        loss, _, _ = self._meshsdf_loss(verts, faces, dpts)
        loss *= 1e6 / dpts.size(0)
        return loss

    def _loss_reg(self):
        p = torch.cat(self._pose_o, dim=1)
        zero = torch.zeros_like(p)
        loss = self._mse_loss(p, zero)
        loss /= p.size(0)
        return loss

    def solve(self, pose_o_0, dpts_o, subset_o=None, dist_thresh_o=None, steps=100):
        # copy initial pose
        for i in range(self._object_group_layer.num_obj):
            self._pose_o[i].data = pose_o_0[:, i].clone()

        faces_o, _ = self._object_group_layer.get_f_from_inds(subset_o)

        for step in tqdm(range(steps), ncols=60, colour="green"):
            self._optimizer.zero_grad()

            verts_o, _ = self._object_group_layer_forward(subset_o)
            if step == 0 and dist_thresh_o is not None:
                dpts_o = self._get_sdf_dpts_by_dist_thresh(
                    verts_o, faces_o, dpts_o, dist_thresh_o
                )

            loss_sdf_o = self._loss_sdf(verts_o, faces_o, dpts_o)
            loss_sdf_o *= self._w_sdf

            loss_reg_o = self._loss_reg()
            loss_reg_o *= self._w_reg

            loss = loss_sdf_o + loss_reg_o

            loss.backward()

            # Set grad to None to prevent Adam from updating the parameters even when
            # the grad are all zeros. An alternative is to place this before the
            # forward pass, but the currently implementation of group layers does not
            # allow this since the non-active poses will still always be used due to
            # concatenation and hence their grad will be 0.0 rather than None.
            for i, p in enumerate(self._pose_o):
                if i not in subset_o and p.grad is not None:
                    assert p.grad.sum() == 0.0
                    p.grad = None

            self._optimizer.step()

        losses = torch.stack([loss, loss_sdf_o, loss_reg_o]).data.clone().cpu().numpy()
        pose_o = (
            torch.stack([p.data for p in self._pose_o], dim=1).squeeze(0).cpu().numpy()
        )

        return pose_o, losses


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


if __name__ == "__main__":
    sequence_folder = (
        PROJ_ROOT / "data/HOT_Dataset/release_dataset/subject_8/20231024_180111"
    )
    device = "cuda"
    dpts_thresh = 1000
    render_results = True
    debug = True

    # load sequence
    loader = SequenceLoader(sequence_folder, device=device)
    num_frames = loader.num_frames
    object_ids = loader.object_ids
    mesh_files = loader.object_cleaned_mesh_files

    # load object initial poses
    # poses_o = np.load(sequence_folder / "fd_poses.npy")
    poses_o = np.load(sequence_folder / "fd_poses_new.npy")

    # initialize solver
    solver = ObjectPoseSolver(mesh_files, device=device)

    # maske save folder
    save_folder = sequence_folder / "processed" / "pose_solver" / "object_pose"
    make_clean_folder(save_folder)

    log_poses = []
    log_losses = []

    # initialize renderer
    if render_results:
        rs_serials = loader.rs_serials
        rs_width = loader.rs_width
        rs_height = loader.rs_height
        rs_intrinsics = loader.intrinsics.cpu().numpy()
        rs_extrinsics = loader.extrinsics2world.cpu().numpy()
        textured_meshes = [
            trimesh.load_mesh(str(f), process=False)
            for f in loader.object_textured_mesh_files
        ]

        save_vis_folder = save_folder / "vis"
        make_clean_folder(save_vis_folder)

        renderer = SequenceRenderer(rs_width, rs_height)

    for frame_id in range(num_frames):
        tqdm.write(f"- Optimizing frame {frame_id}...")
        t_s = time.time()

        # prepare dpts
        seg_masks = np.stack(loader.get_mask_image(frame_id, erode_kernel=5), axis=0)
        # get subset_o
        subset_o = []
        masks_o = np.zeros_like(seg_masks).astype(bool)
        for i in range(len(object_ids)):
            m = seg_masks == (i + 1)
            if m.sum() < dpts_thresh:
                continue
            subset_o.append(i)
            masks_o = np.logical_or(masks_o, m)
        if len(subset_o) == 0:
            tqdm.write("No object has enough points.")
            log_poses.append(poses_o[:, frame_id])
            continue

        masks_o = torch.from_numpy(masks_o.reshape(-1)).to(device)
        loader.step_by_frame_id(frame_id)
        masks_o = torch.logical_and(masks_o, loader.masks.view(-1))
        pcd_points = loader.points.view(-1, 3)[masks_o]
        # process points
        dpts_o = process_points(
            pcd_points,
            voxel_size=0.005,
            nb_neighbors=100,
            std_ratio=1.0,
        )
        # print(f"dpts_o: {dpts_o.shape}")
        pose_o_0 = np.stack([quat_to_rvt(ps) for ps in poses_o[:, frame_id]])
        pose_o_0 = torch.from_numpy(pose_o_0).unsqueeze(0).to(device)
        # print(f"pose_o_0: {pose_o_0.shape}")

        # solve object pose
        optim_pose_o, losses = solver.solve(
            pose_o_0=pose_o_0,
            dpts_o=dpts_o,
            subset_o=subset_o,
            steps=20,
        )

        t_e = time.time()

        tqdm.write(
            f"loss: {losses[0]:11.8f} | loss_sdf: {losses[1]:11.8f} | loss_reg: {losses[2]:11.8f} | time: {t_e - t_s:.2f}s"
        )

        log_poses.append([rvt_to_quat(rvt) for rvt in optim_pose_o])
        log_losses.append(losses)

        if render_results:
            vis_images = []
            for cam_idx, serial in enumerate(rs_serials):
                rgb_image = loader.get_rgb_image(frame_id, serial)
                vis_image = renderer.render_sequence(
                    object_meshes=textured_meshes,
                    object_poses=[rvt_to_mat(rvt) for rvt in optim_pose_o],
                    cam_K=rs_intrinsics[cam_idx],
                    cam_pose=rs_extrinsics[cam_idx],
                )
                vis_image = cv2.addWeighted(rgb_image, 0.3, vis_image, 0.7, 0)
                vis_images.append(vis_image)
                display_images(
                    vis_images,
                    rs_serials,
                    facecolor="black",
                    save_path=save_vis_folder / f"vis_{frame_id:06d}.png",
                )

    # save results
    tqdm.write("Saving results...")
    log_poses = np.stack(log_poses)  # (num_frames, num_objects, 7)
    log_poses = np.swapaxes(log_poses, 0, 1)  # (num_objects, num_frames, 7)
    np.save(save_folder / "poses.npy", log_poses)

    log_losses = np.stack(log_losses)  # (num_frames, 3)
    log_losses = np.swapaxes(log_losses, 0, 1)  # (3, num_frames)
    np.save(save_folder / "losses.npy", log_losses)

    if render_results:
        tqdm.write("Creating vis video...")
        vis_images = [None] * num_frames
        workers = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for frame_id in range(num_frames):
                workers.append(
                    executor.submit(
                        read_rgb_image,
                        save_vis_folder / f"vis_{frame_id:06d}.png",
                        idx=frame_id,
                    )
                )
            for worker in concurrent.futures.as_completed(workers):
                vis_image, idx = worker.result()
                vis_images[idx] = vis_image

        create_video_from_rgb_images(
            save_folder / "vis_object_pose.mp4", vis_images, fps=30
        )

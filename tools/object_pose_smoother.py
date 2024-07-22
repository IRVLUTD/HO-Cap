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
from lib.layers import ObjectGroupLayer, MeshSDFLoss
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
    "lambda_smooth": 1.0,
    "lambda_robust": 2.0,
    "window_size": 3,
    "steps": 15,
    "subset_o": [
        0,
        1,
        2,
        3,
    ],
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
    if idx is None:
        return loss
    return loss, idx


class ObjectPoseSolver:
    def __init__(self, mesh_files, optim_config=OPTIM_CONFIG, device="cuda") -> None:
        self._device = device

        # load optimization config
        self._lr = optim_config["learning_rate"]
        self._w_sdf = optim_config["lambda_sdf"]
        self._w_reg = optim_config["lambda_reg"]
        self._w_smooth = optim_config["lambda_smooth"]
        self._w_robust = optim_config["lambda_robust"]
        self._window_size = optim_config["window_size"]
        self._steps = optim_config["steps"]

        # initialize object group layer
        self._object_group_layer = self._init_object_group_layer(mesh_files)

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

    def _object_group_layer_forward(self, subset=None):
        p = torch.cat(self._pose_o, dim=1)
        v, n = self._object_group_layer(p, subset)
        if p.size(0) == 1:
            v = v.sequeeze(0)
            n = n.seqeeze(0)
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

    def _loss_temporal_robust(self, delta=1.0):
        """Loss for robust temporal smoothing of poses.

        Args:
            delta (float): Threshold for the linear part of the loss function.
        """
        poses = torch.stack(self._pose_o, dim=0)
        loss = 0.0
        for k in range(poses.size(0)):
            for n in range(1, poses.size(1)):
                diff = poses[k, n] - poses[k, n - 1]
                abs_diff = torch.abs(diff)
                quadratic_part = torch.minimum(
                    abs_diff, torch.full_like(abs_diff, delta)
                )
                linear_part = abs_diff - quadratic_part
                loss += torch.sum(0.5 * quadratic_part**2 + delta * linear_part)
        loss /= poses.size(0) * poses.size(1)
        return loss

    def _loss_temporal_smooth(self, window_size=1, w_rot=1.0, w_trans=1.0):
        """Loss for temporal smoothing of poses using L2 norm.

        Args:
            poses (torch.Tensor): Poses of shape (num_objects, num_frames, 7).
            window_size (int): Number of frames on either side to include in the smoothing.
            weight_rotation (float): Weight for the rotation loss.
            weight_translation (float): Weight for the translation loss.
        """
        poses = torch.stack(self._pose_o, dim=0)
        num_objects, num_frames, _ = poses.shape
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
                        poses[k, n, 3:] - poses[k, n - j, 3:], p=2
                    )
                    translation_loss += torch.norm(
                        poses[k, n, 3:] - poses[k, n + j, 3:], p=2
                    )

        total_loss = w_rot * rotation_loss + w_trans * translation_loss
        total_loss /= num_objects * (num_frames - 2 * window_size)
        return total_loss

    def solve(
        self,
        batch_pose_o_0,
        batch_dpts_o,
        subset_o=None,
        # steps=100,
    ):
        # initialize optimizer
        self._pose_o = [
            torch.nn.Parameter(p.clone(), requires_grad=True) for p in batch_pose_o_0
        ]
        self._optimizer = torch.optim.Adam(self._pose_o, lr=self._lr)
        faces_o, _ = self._object_group_layer.get_f_from_inds(subset_o)

        num_frames = len(batch_dpts_o)
        # copy initial pose
        losses = []

        for step in tqdm(range(self._steps), ncols=60, colour="green"):
            t_s = time.time()

            self._optimizer.zero_grad()

            verts_o, _ = self._object_group_layer_forward(subset_o)
            # compute sdf loss
            # loss_sdf_o = []
            # for i in range(num_frames):
            #     dpts_o = batch_dpts_o[i]
            #     loss_sdf_o.append(self._loss_sdf(verts_o[i], faces_o, dpts_o))
            if self._w_sdf == 0:
                loss_sdf_o = self._zero
            else:
                loss_sdf_o = [None] * num_frames
                workers = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    for i in range(num_frames):
                        workers.append(
                            executor.submit(
                                runner_loss_sdf,
                                verts_o[i],
                                faces_o,
                                batch_dpts_o[i],
                                idx=i,
                            )
                        )
                    for worker in concurrent.futures.as_completed(workers):
                        l, i = worker.result()
                        loss_sdf_o[i] = l
                del workers

                loss_sdf_o = torch.mean(torch.stack(loss_sdf_o))
                loss_sdf_o *= self._w_sdf

            if self._w_reg == 0:
                loss_reg_o = self._zero
            else:
                loss_reg_o = self._loss_reg() / num_frames
                loss_reg_o *= self._w_reg

            if self._w_smooth == 0:
                loss_smooth_o = self._zero
            else:
                loss_smooth_o = self._loss_temporal_smooth(self._window_size)
                loss_smooth_o *= self._w_smooth

            if self._w_robust == 0:
                loss_robust_o = self._zero
            else:
                loss_robust_o = self._loss_temporal_robust(delta=1.0)
                loss_robust_o *= self._w_robust

            loss = loss_sdf_o + loss_reg_o + loss_smooth_o + loss_robust_o

            # Set grad to None to prevent Adam from updating the parameters even when
            # the grad are all zeros. An alternative is to place this before the
            # forward pass, but the currently implementation of group layers does not
            # allow this since the non-active poses will still always be used due to
            # concatenation and hence their grad will be 0.0 rather than None.
            for i, p in enumerate(self._pose_o):
                if i not in subset_o and p.grad is not None:
                    assert p.grad.sum() == 0.0
                    p.grad = None

            loss.backward()
            self._optimizer.step()

            t_e = time.time()

            losses.append(
                [
                    loss.item(),
                    loss_sdf_o.item(),
                    loss_reg_o.item(),
                    loss_smooth_o.item(),
                    loss_robust_o.item(),
                ]
            )

            if (step + 1) % 1 == 0:
                tqdm.write(
                    f"loss: {loss.item():11.8f} "
                    + f"| loss_sdf: {loss_sdf_o.item():11.8f} "
                    + f"| loss_reg: {loss_reg_o.item():11.8f} "
                    + f"| loss_smooth: {loss_smooth_o.item():11.8f} "
                    + f"| loss_robust: {loss_robust_o.item():11.8f} "
                    + f"| time: {t_e - t_s:.2f}s"
                )

        losses = np.array(losses, dtype=np.float32)
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
    return parser.parse_args()


if __name__ == "__main__":
    dpts_thresh = 100
    render_results = True
    debug = True

    # sequence_folder = (
    #     PROJ_ROOT / "data/HOT_Dataset/release_dataset/subject_1/20231025_165502"
    # )
    # device = "cuda"

    args = args_parser()
    sequence_folder = Path(args.sequence_folder).resolve()
    device = args.device

    # load sequence
    loader = SequenceLoader(sequence_folder, device=device)
    num_frames = loader.num_frames
    # num_frames = 100
    object_ids = loader.object_ids
    mesh_files = loader.object_cleaned_mesh_files

    # load object initial poses
    poses_o = np.load(sequence_folder / "fd_poses_new.npy")

    # initialize solver
    solver = ObjectPoseSolver(mesh_files, device=device)

    # maske save folder
    save_folder = sequence_folder / "processed" / "pose_solver" / "object_pose_smooth"
    save_folder.mkdir(parents=True, exist_ok=True)

    save_dpts_folder = save_folder / "dpts"
    save_dpts_folder.mkdir(parents=True, exist_ok=True)

    # >>>>>>>>>> prepare data for pose optimization >>>>>>>>>>
    batch_pose_o_0 = []
    batch_dpts_o = []
    subset_o = OPTIM_CONFIG["subset_o"]

    for frame_id in tqdm(range(num_frames), ncols=60, colour="green"):
        dpts_file = save_dpts_folder / f"dpts_{frame_id:06d}.npy"
        if dpts_file.exists():
            dpts_o = np.load(dpts_file)
            dpts_o = torch.from_numpy(dpts_o).to(device)
        else:
            # prepare dpts
            seg_masks = np.stack(
                loader.get_mask_image(frame_id, erode_kernel=5), axis=0
            )
            # get subset_o
            masks_o = np.zeros_like(seg_masks).astype(bool)
            for i in subset_o:
                m = seg_masks == (i + 1)
                masks_o = np.logical_or(masks_o, m)

            masks_o = torch.from_numpy(masks_o.reshape(-1)).to(device)
            loader.step_by_frame_id(frame_id)
            masks_o = torch.logical_and(masks_o, loader.masks.view(-1))
            pcd_points = loader.points.view(-1, 3)[masks_o]
            # process points
            dpts_o = process_points(
                pcd_points,
                voxel_size=0.003,
                # voxel_size=0.0,
                nb_neighbors=100,
                std_ratio=1.0,
            )

            # save dpts
            np.save(dpts_file, dpts_o.cpu().numpy())
        batch_dpts_o.append(dpts_o)

        pose_o_0 = np.stack([quat_to_rvt(ps) for ps in poses_o[:, frame_id]])
        pose_o_0 = torch.from_numpy(pose_o_0).unsqueeze(0).to(device)

        batch_pose_o_0.append(pose_o_0)

    batch_pose_o_0 = torch.cat(batch_pose_o_0, dim=0)
    batch_pose_o_0 = [batch_pose_o_0[:, i] for i in range(len(object_ids))]

    # >>>>>>>>>> solve object poses >>>>>>>>>>

    # solve object pose
    optim_poses_o, losses_o = solver.solve(
        batch_pose_o_0=batch_pose_o_0,
        batch_dpts_o=batch_dpts_o,
        subset_o=subset_o,
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
    optim_poses_o = np.swapaxes(optim_poses_o, 0, 1)
    optim_poses_o = np.stack([rvt_to_quat(ps) for ps in optim_poses_o])
    # smooth the poses
    # optim_poses_o = apply_kalman_filter(optim_poses_o)

    np.save(save_folder / "poses.npy", optim_poses_o)

    # optim_poses_o = np.load(save_folder / "poses.npy")

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
        optim_poses_o_mat = np.stack([quat_to_mat(ps) for ps in optim_poses_o])
        # draw rendered images
        tqdm.write("- Generating vis images...")
        rendered_images = [None] * num_frames
        tqbar = tqdm(range(num_frames), ncols=60, colour="green")
        workers = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            for frame_id in range(num_frames):
                workers.append(
                    executor.submit(
                        runner_draw_rendered_image_by_frame_id,
                        rs_width,
                        rs_height,
                        loader.get_rgb_image(frame_id),
                        textured_meshes,
                        optim_poses_o_mat[:, frame_id],
                        rs_intrinsics,
                        rs_extrinsics,
                        idx=frame_id,
                    )
                )
            for worker in concurrent.futures.as_completed(workers):
                img, idx = worker.result()
                rendered_images[idx] = img
                tqbar.update(1)
                tqbar.refresh()
        tqbar.close()

        del tqbar, workers, textured_meshes

        # # save rendered images
        # tqdm.write("- Saving vis images...")
        # save_render_folder = save_folder / "vis"
        # make_clean_folder(save_render_folder)
        # tqbar = tqdm(range(num_frames), ncols=60, colour="green")
        # workers = []
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     for frame_id in range(num_frames):
        #         workers.append(
        #             executor.submit(
        #                 write_rgb_image,
        #                 save_render_folder / f"vis_{frame_id:06d}.png",
        #                 rendered_images[frame_id],
        #             )
        #         )
        #     for worker in concurrent.futures.as_completed(workers):
        #         worker.result()
        #         tqbar.update(1)
        #         tqbar.refresh()
        # tqbar.close()

        # del tqbar, workers

        # save rendered video
        tqdm.write("- Creating vis video...")
        create_video_from_rgb_images(
            save_folder / "vis_optas_poses.mp4",
            # save_folder
            # / f"vis_lr_{OPTIM_CONFIG['learning_rate']}_winsize_{OPTIM_CONFIG['window_size']}_steps_{OPTIM_CONFIG['steps']}.mp4",
            rendered_images,
            fps=30,
        )

        del rendered_images

        tqdm.write(">>>>>>>>>> Done!!! <<<<<<<<<<")

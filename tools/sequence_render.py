import numpy as np
import trimesh
from tqdm import tqdm
import concurrent.futures
import argparse

from _init_paths import *
from lib.utils import *
from lib.utils.common import *
from lib.loaders import SequenceLoader
from lib.renderers.renderer_pyrd import SequenceRenderer
from lib.layers import MANOGroupLayer


def mano_group_layer_forward(poses, subset=None):
    p = torch.cat(poses, dim=1)
    v, j = mano_group_layer(p, inds=subset)
    if p.size(0) == 1:
        v = v.sequeeze(0)
        j = j.squeeze(0)
    return v, j


def get_hand_meshes(frame_id):
    pose_m = [
        torch.from_numpy(poses_m[0 if side == "right" else 1, frame_id])
        .unsqueeze(0)
        .to(device)
        for side in mano_sides
    ]
    verts_m, _ = mano_group_layer_forward(pose_m)
    mesh = trimesh.Trimesh(
        vertices=verts_m.cpu().numpy(),
        faces=mano_faces,
        vertex_colors=mano_colors,
        process=False,
    )
    return [mesh]


def runner_save_vis_image(vis_images, save_path):
    display_images(vis_images, facecolor="black", save_path=save_path)


def runner_render_frame_image(
    rgb_image,
    width,
    height,
    mano_meshes,
    object_meshes,
    object_poses,
    cam_K,
    cam_pose,
    idx=None,
):
    renderer = SequenceRenderer(width, height)
    vis = renderer.render_sequence(
        mano_meshes, object_meshes, object_poses, cam_K, cam_pose
    )
    vis = cv2.addWeighted(rgb_image, 0.3, vis, 0.7, 0)
    if idx is None:
        return vis
    return vis, idx


def args_parser():
    parser = argparse.ArgumentParser(description="Run sequence renderer")
    parser.add_argument(
        "--sequence_folder",
        type=str,
        required=True,
        help="Sequence folder path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # sequence_folder = (
    #     PROJ_ROOT / "data/HOT_Dataset/release_dataset/subject_1/20231025_165502"
    # )
    # device = "cuda"

    args = args_parser()
    sequence_folder = Path(args.sequence_folder).resolve()
    device = args.device

    poses_o_file = sequence_folder / "fd_poses_new.npy"
    poses_m_file = sequence_folder / "hamer_poses.npy"
    # poses_o_file = (
    #     sequence_folder
    #     / "processed"
    #     / "pose_solver"
    #     / "joint_pose_smooth"
    #     / "poses_o.npy"
    # )
    # poses_m_file = (
    #     sequence_folder
    #     / "processed"
    #     / "pose_solver"
    #     / "joint_pose_smooth"
    #     / "poses_m.npy"
    # )
    if any([not f.exists() for f in [poses_o_file, poses_m_file]]):
        print(f"Poses files not found in {sequence_folder}")
        exit()

    poses_o = np.load(poses_o_file)
    poses_m = np.load(poses_m_file)

    # mano_beta_files = sequence_folder.parent.glob("2023*/hamer_betas.npy")
    # mano_beta = np.mean([np.load(f) for f in mano_beta_files], axis=0)

    loader = SequenceLoader(sequence_folder, device=device)
    mano_beta = loader.mano_beta.cpu().numpy()
    rs_serials = loader.rs_serials
    rs_width = loader.rs_width
    rs_height = loader.rs_height
    intrinsics = loader.intrinsics.cpu().numpy()
    extrinsics = loader.extrinsics2world.cpu().numpy()
    mano_sides = loader.mano_sides
    object_ids = loader.object_ids
    num_frames = loader.num_frames
    object_meshes = [
        trimesh.load_mesh(f, process=False) for f in loader.object_textured_mesh_files
    ]

    save_folder = sequence_folder / "vis"
    make_clean_folder(save_folder)

    mano_group_layer = MANOGroupLayer(mano_sides, [mano_beta for _ in mano_sides]).to(
        device
    )
    mano_faces = mano_group_layer.f.cpu().numpy()
    for i in range(len(mano_sides)):
        mano_faces = np.concatenate(
            [mano_faces, np.array(NEW_MANO_FACES) + i * NUM_MANO_VERTS]
        )
    mano_colors = np.stack(
        [
            [HAND_COLORS[1].rgb if side == "right" else HAND_COLORS[2].rgb]
            * NUM_MANO_VERTS
            for side in mano_sides
        ]
    ).reshape(-1, 3)

    tqdm.write(f"- Generating rendering images...")
    render_images = [[None] * len(rs_serials) for _ in range(num_frames)]
    workers = []
    tqbar = tqdm(total=num_frames * len(rs_serials), ncols=60, colour="green")
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for frame_id in range(num_frames):
            mano_meshes = get_hand_meshes(frame_id)
            object_poses = quat_to_mat(poses_o[:, frame_id])
            for cam_id in range(len(rs_serials)):
                workers.append(
                    executor.submit(
                        runner_render_frame_image,
                        loader.get_rgb_image(frame_id, rs_serials[cam_id]),
                        rs_width,
                        rs_height,
                        mano_meshes,
                        object_meshes,
                        object_poses,
                        intrinsics[cam_id],
                        extrinsics[cam_id],
                        idx=(frame_id, cam_id),
                    )
                )
        for worker in concurrent.futures.as_completed(workers):
            color, (f_id, c_id) = worker.result()
            render_images[f_id][c_id] = color
            tqbar.update(1)
            tqbar.refresh()
    tqbar.close()
    del workers, tqbar

    # save vis images
    tqdm.write(f"- Saving vis images...")
    tqbar = tqdm(total=num_frames * len(rs_serials), ncols=60, colour="green")
    workers = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for cam_id, serial in enumerate(rs_serials):
            save_folder_cam = save_folder / serial
            make_clean_folder(save_folder_cam)
            for frame_id in range(num_frames):
                workers.append(
                    executor.submit(
                        write_rgb_image,
                        save_folder_cam / f"vis_{frame_id:06d}.png",
                        render_images[frame_id][cam_id],
                    )
                )
        for worker in concurrent.futures.as_completed(workers):
            tqbar.update(1)
            tqbar.refresh()
    tqbar.close()
    del workers, tqbar

    tqdm.write(f"- Saving vis video...")
    for cam_id in tqdm(range(len(rs_serials)), ncols=60, colour="green"):
        video_path = save_folder / f"vis_{rs_serials[cam_id]}.mp4"
        create_video_from_rgb_images(
            video_path,
            [render_images[frame_id][cam_id] for frame_id in range(num_frames)],
            fps=30,
        )

    tqdm.write(f"- Saving display images...")
    tqbar = tqdm(total=num_frames, ncols=60, colour="green")
    workers = []
    save_folder_display = sequence_folder / "vis" / "display"
    make_clean_folder(save_folder_display)
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for frame_id in range(num_frames):
            workers.append(
                executor.submit(
                    runner_save_vis_image,
                    render_images[frame_id],
                    save_folder_display / f"vis_{frame_id:06d}.png",
                )
            )
        for worker in concurrent.futures.as_completed(workers):
            tqbar.update(1)
            tqbar.refresh()
    tqbar.close()
    del workers, tqbar, render_images

    # create video
    tqdm.write(f"- Saving display video...")
    video_path = save_folder / "vis_display.mp4"
    image_files = sorted(save_folder_display.glob("vis_*.png"))
    video_images = [None] * len(image_files)
    workers = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for i, image_file in enumerate(image_files):
            workers.append(executor.submit(read_rgb_image, image_file, i))
        for worker in concurrent.futures.as_completed(workers):
            img, idx = worker.result()
            video_images[idx] = img

    create_video_from_rgb_images(video_path, video_images, fps=30)

    del workers, video_images

    tqdm.write(f">>>>>>>>>> Done!!! <<<<<<<<<<")

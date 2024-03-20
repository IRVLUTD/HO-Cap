from pathlib import Path
import shutil
import pickle as pkl
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import logging
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as mpcm
import matplotlib.lines as mplines
import torch
import concurrent.futures
from .colors import (
    OBJ_CLASS_COLORS,
    HAND_COLORS,
    HAND_BONE_COLORS,
    HAND_JOINT_COLORS,
    COLORS,
)
from .mano_info import HAND_BONES


# ==========> common functions <=========
def init_logger(log_level="info", log_file=None, name="root"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)-20s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level.upper()))
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def make_clean_folder(folder_path):
    folder = Path(folder_path)
    if folder.exists():
        shutil.rmtree(str(folder))
    folder.mkdir(parents=True)


def copy_file(src_path, dst_path):
    src = Path(src_path)
    dst = Path(dst_path)
    if not dst.parent.exists():
        dst.parent.mkdir(parents=True)
    shutil.copy(src, dst)


def copy_folder(src_path, dst_path):
    src = Path(src_path)
    dst = Path(dst_path)
    if not src.exists():
        return
    if not dst.parent.exists():
        dst.parent.mkdir(parents=True)
    shutil.copytree(src, dst)


def delete_file(file_path):
    file = Path(file_path)
    if file.exists():
        file.unlink()


def delete_folder(folder_path):
    folder = Path(folder_path)
    if folder.exists():
        shutil.rmtree(folder)


def move_file(src_path, dst_path):
    src = Path(src_path)
    dst = Path(dst_path)
    if not dst.parent.exists():
        dst.parent.mkdir(parents=True)
    shutil.move(src, dst)


def move_folder(src_path, dst_path):
    src = Path(src_path)
    dst = Path(dst_path)
    if not dst.parent.exists():
        dst.parent.mkdir(parents=True)
    shutil.move(src, dst)


def read_data_from_json(file_path, idx=None):
    with open(str(file_path), "r", encoding="utf-8") as f:
        data = json.load(f)
    if idx is not None:
        return data, idx
    return data


def write_data_to_json(file_path, data):
    with open(str(file_path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def read_data_from_pkl(file_path, idx=None):
    with open(str(file_path), "rb") as f:
        data = pkl.load(f)
    if idx is not None:
        return data, idx
    return data


def write_data_to_pkl(file_path, data):
    with open(str(file_path), "wb") as f:
        pkl.dump(data, f)


# ==========> cv functions <=========
def rvt_to_quat(rvt):
    """Convert rotation vector and translation vector to quaternion and translation vector.

    Args:
        rvt (np.ndarray): Rotation vector and translation vector, shape (6,).

    Returns:
        np.ndarray: Quaternion and translation vector, shape (7,), [qx, qy, qz, qw, tx, ty, tz].
    """
    if rvt.ndim == 2:
        rv = rvt[:, :3]
        t = rvt[:, 3:]
    else:
        rv = rvt[:3]
        t = rvt[3:]
    r = R.from_rotvec(rv)
    q = r.as_quat()
    return np.concatenate([q, t], dtype=np.float32, axis=-1)


def rvt_to_mat(rvt):
    """Convert rotation vector and translation vector to pose matrix.

    Args:
        rvt (np.ndarray): Rotation vector and translation vector, shape (6,).
    Returns:
        np.ndarray: Pose matrix, shape (4, 4).
    """
    p = np.eye(4)
    rv = rvt[:3]
    t = rvt[3:]
    r = R.from_rotvec(rv)
    p[:3, :3] = r.as_matrix()
    p[:3, 3] = t
    return p.astype(np.float32)


def mat_to_rvt(mat_4x4):
    """Convert pose matrix to rotation vector and translation vector.

    Args:
        mat_4x4 (np.ndarray): Pose matrix, shape (4, 4).
    Returns:
        np.ndarray: Rotation vector and translation vector, shape (6,).
    """

    r = R.from_matrix(mat_4x4[:3, :3])
    rv = r.as_rotvec()
    t = mat_4x4[:3, 3]
    return np.concatenate([rv, t], dtype=np.float32)


def mat_to_quat(mat_4x4):
    """Convert pose matrix to quaternion and translation vector.

    Args:
        mat_4x4 (np.ndarray): Pose matrix, shape (4, 4).
    Returns:
        np.ndarray: Quaternion and translation vector, shape (7,).
    """
    if mat_4x4.ndim == 3:
        r = R.from_matrix(mat_4x4[:, :3, :3])
        q = r.as_quat()
        t = mat_4x4[:, :3, 3]
    else:
        r = R.from_matrix(mat_4x4[:3, :3])
        q = r.as_quat()
        t = mat_4x4[:3, 3]
    return np.concatenate([q, t], dtype=np.float32)


# def quat_to_mat(quat):
#     """Convert quaternion and translation vector to pose matrix.

#     Args:
#         quat (np.ndarray): Quaternion and translation vector, shape (7,).
#     Returns:
#         np.ndarray: Pose matrix, shape (4, 4).
#     """
#     if quat.ndim == 2:
#         q = quat[:, :4]
#         t = quat[:, 4:]
#         p = np.repeat(np.eye(4)[np.newaxis, ...], len(quat), axis=0)
#         r = R.from_quat(q)
#         p[:, :3, :3] = r.as_matrix()
#         p[:, :3, 3] = t
#     else:
#         q = quat[:4]
#         t = quat[4:]
#         p = np.eye(4)
#         r = R.from_quat(q)
#         p[:3, :3] = r.as_matrix()
#         p[:3, 3] = t
#     return p.astype(np.float32)


def quat_to_mat(quat):
    """Convert quaternion and translation vector to pose matrix.

    This function supports converting a single quaternion or a batch of quaternions.

    Args:
        quat (np.ndarray): Quaternion and translation vector. Shape can be (7,) for a single quaternion
                           or (N, 7) for a batch of quaternions, where N is the batch size.
    Returns:
        np.ndarray: Pose matrix. Shape will be (4, 4) for a single quaternion or (N, 4, 4) for a batch of quaternions.
    """
    batch_mode = quat.ndim == 2
    q = quat[..., :4]
    t = quat[..., 4:]

    if batch_mode:
        p = np.tile(np.eye(4), (len(quat), 1, 1))
    else:
        p = np.eye(4)

    r = R.from_quat(q)
    p[..., :3, :3] = r.as_matrix()
    p[..., :3, 3] = t

    return p.astype(np.float32)


def quat_to_rvt(quat):
    """Convert quaternion and translation vector to rotation vector and translation vector.

    Args:
        quat (np.ndarray): Quaternion and translation vector, shape (7,).
    Returns:
        np.ndarray: Rotation vector and translation vector, shape (6,).
    """
    if quat.ndim == 2:
        q = quat[:, :4]
        t = quat[:, 4:]
    else:
        q = quat[:4]
        t = quat[4:]
    r = R.from_quat(q)
    rv = r.as_rotvec()
    return np.concatenate([rv, t], dtype=np.float32, axis=-1)


def read_rgb_image(image_path, idx=None):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if idx is not None:
        return image, idx
    return image


def write_rgb_image(image_path, image):
    cv2.imwrite(str(image_path), image[:, :, ::-1])


def read_depth_image(image_path, idx=None):
    image = cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH)
    if idx is not None:
        return image, idx
    return image


def write_depth_image(image_path, image):
    cv2.imwrite(str(image_path), image)


def read_mask_image(image_path, idx=None):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if idx is not None:
        return image, idx
    return image


def write_mask_image(image_path, image):
    cv2.imwrite(str(image_path), image)


def read_pose_from_txt(pose_path, idx=None):
    if not pose_path.exists():
        pose = None
    else:
        pose = np.loadtxt(str(pose_path)).astype(np.float32).flatten()
    if idx is not None:
        return pose, idx
    return pose


def write_pose_to_txt(pose_path, pose, header=""):
    np.savetxt(str(pose_path), pose, fmt="%.6f", header=header)


def create_video_from_rgb_images(video_path, rgb_images, fps=30):
    h, w, _ = rgb_images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    for image in rgb_images:
        video.write(image[:, :, ::-1])
    video.release()


def create_video_from_depth_images(video_path, depth_images, fps=30):
    h, w = depth_images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    for image in depth_images:
        d_min = image.min()
        d_max = image.max()
        image = (image - d_min) / (d_max - d_min) * 255
        image = image.astype(np.uint8)
        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        video.write(image)
    video.release()


def create_video_from_image_files(video_path, image_files, fps=30, preload=False):
    def worker_read_image_file(image_file, idx=None):
        img = cv2.imread(str(image_file), cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:  # depth image
            d_min = img.min()
            d_max = img.max()
            img = (img - d_min) / (d_max - d_min) * 255
            img = img.astype(np.uint8)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        return img, idx

    img, _ = worker_read_image_file(image_files[0])
    h, w = img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

    if preload:
        images = [None] * len(image_files)
        workers = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i, image_file in enumerate(image_files):
                workers.append(executor.submit(worker_read_image_file, image_file, i))
            for worker in concurrent.futures.as_completed(workers):
                img, idx = worker.result()
                images[idx] = img
        for image in images:
            video.write(image)
        del images
    else:
        for image_file in image_files:
            img, _ = worker_read_image_file(image_file)
            video.write(img)
    video.release()


def erode_mask(mask, kernel_size=3, iterations=1):
    if kernel_size <= 1:
        return mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    m_dtype = mask.dtype
    mask = mask.astype(np.uint8)
    mask = cv2.erode(mask, kernel, iterations=iterations)
    mask = mask.astype(m_dtype)
    return mask


def dilate_mask(mask, kernel_size=3, iterations=1):
    if kernel_size <= 1:
        return mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    m_dtype = mask.dtype
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=iterations)
    mask = mask.astype(m_dtype)
    return mask


def draw_object_masks_overlay(
    image, mask, alpha=0.5, reduce_background=False, rgb_input=True
):
    overlay = np.zeros_like(image) if reduce_background else image.copy()
    for label in np.unique(mask):
        if label == 0:
            continue
        overlay[mask == label] = (
            OBJ_CLASS_COLORS[label].rgb if rgb_input else OBJ_CLASS_COLORS[label].bgr
        )
    overlay = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return overlay


def draw_mask_overlay(
    image,
    mask,
    alpha=0.5,
    rgb_color=(0, 255, 0),
    reduce_background=False,
    rgb_input=True,
):
    overlay = np.zeros_like(image) if reduce_background else image.copy()
    overlay[mask > 0] = rgb_color if rgb_input else rgb_color[::-1]
    overlay = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return overlay


def draw_image_overlay(image_0, image_1, alpha=0.5):
    overlay = image_0.copy()
    overlay = cv2.addWeighted(image_0, 1 - alpha, image_1, alpha, 0)
    return overlay


def draw_losses_curve(
    losses,
    loss_names=None,
    title="Loss Curve",
    figsize=(15, 5),
    dpi=100,
    save_path=None,
):
    if loss_names is None:
        loss_names = [f"loss_{i}" for i in range(len(losses[0]))]
    # draw loss curve
    plt.figure(figsize=figsize, dpi=dpi)
    for i, name in enumerate(loss_names):
        plt.plot([x[i] for x in losses], label=name)
    plt.legend()
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.tight_layout()
    # save or return figure
    if save_path is not None:
        plt.savefig(str(save_path), bbox_inches="tight", pad_inches=0)
    else:
        plt.show()
    plt.close()


def draw_hand_landmarks(image, landmarks, hand_side=None, box=None, rgb_input=True):
    img = image.copy()
    # draw bones
    for idx, bone in enumerate(HAND_BONES):
        if np.any(landmarks[bone[0]] == -1) or np.any(landmarks[bone[1]] == -1):
            continue
        cv2.line(
            img,
            landmarks[bone[0]],
            landmarks[bone[1]],
            HAND_BONE_COLORS[idx].rgb if rgb_input else HAND_BONE_COLORS[idx].bgr,
            2,
        )
    # draw joints
    for idx, mark in enumerate(landmarks):
        if np.any(mark == -1):
            continue
        cv2.circle(img, mark, 5, [255, 255, 255], -1)
        cv2.circle(
            img,
            mark,
            3,
            HAND_JOINT_COLORS[idx].rgb if rgb_input else HAND_JOINT_COLORS[idx].bgr,
            -1,
        )

    # draw hand box
    if box is not None:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # draw hand side text
    if hand_side is not None:
        text = hand_side.lower()
        text_x = np.min(landmarks[:, 0])
        text_y = np.min(landmarks[:, 1]) - 5  # add margin to top
        text_color = HAND_COLORS[1] if text == "right" else HAND_COLORS[2]
        cv2.putText(
            img,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            text_color.rgb if rgb_input else text_color.bgr,
            1,
            cv2.LINE_AA,
        )
    return img


def display_images(
    images,
    names=None,
    cmap="gray",
    figsize=(15, 5),
    dpi=100,
    max_cols=4,
    facecolor="white",
    save_path=None,
    return_array=False,
):
    num_images = len(images)
    num_cols = min(num_images, max_cols)
    num_rows = int(np.ceil(num_images / num_cols))
    if names is None:
        names = [None] * num_images

    # fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize, dpi=dpi)
    # Remove the gaps between subplots
    # plt.subplots_adjust(wspace=0, hspace=0)

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=facecolor)
    for i, (image, name) in enumerate(zip(images, names)):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(image, cmap=cmap)
        if name is not None:
            plt.title(name)
        plt.axis("off")
    plt.tight_layout()

    if return_array:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        img_array = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
            int(height), int(width), 3
        )

    if save_path is not None:
        plt.savefig(
            str(save_path),
            # bbox_inches="tight",
            # pad_inches=0,
        )
    else:
        plt.show()
    plt.close()

    if return_array:
        return img_array


def draw_loftr_matching_figure(
    imgA,
    imgB,
    mkptsA,
    mkptsB,
    mconf,
    kptsA=None,
    kptsB=None,
    text=[],
    figsize=(15, 5),
    dpi=100,
    save_path=None,
):
    assert mkptsA.shape[0] == mkptsB.shape[0], "Number of keypoints should be same"
    color = mpcm.jet(mconf)
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    axes[0].imshow(imgA, cmap="gray")
    axes[1].imshow(imgB, cmap="gray")
    for i in range(2):
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if kptsA is not None and kptsB is not None:
        for i in range(mkptsA.shape[0]):
            axes[0].scatter(kptsA[:, 0], kptsA[:, 1], c="w", s=2)
            axes[1].scatter(kptsB[:, 0], kptsB[:, 1], c="w", s=2)

    if mkptsA.shape[0] != 0 and mkptsB.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkptsA = transFigure.transform(axes[0].transData.transform(mkptsA))
        fkptsB = transFigure.transform(axes[1].transData.transform(mkptsB))
        fig.lines = [
            mplines.Line2D(
                (fkptsA[i, 0], fkptsB[i, 0]),
                (fkptsA[i, 1], fkptsB[i, 1]),
                transform=fig.transFigure,
                c=color[i],
                linewidth=1,
            )
            for i in range(len(mkptsA))
        ]
        axes[0].scatter(mkptsA[:, 0], mkptsA[:, 1], c=color, s=4)
        axes[1].scatter(mkptsB[:, 0], mkptsB[:, 1], c=color, s=4)
    # put txts
    txt_color = "k" if imgA[:100, :200].mean() > 200 else "w"
    fig.text(
        0.01,
        0.99,
        "\n".join(text),
        transform=fig.axes[0].transAxes,
        fontsize=15,
        va="top",
        ha="left",
        color=txt_color,
    )
    if save_path is not None:
        plt.savefig(str(save_path), bbox_inches="tight", pad_inches=0)
    else:
        plt.show()
    plt.close()


def read_rgb_image(image_path, idx=None):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if idx is not None:
        return img, idx
    return img


def read_depth_image(image_path, idx=None):
    img = cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH)
    if idx is not None:
        return img, idx
    return img


def read_mask_image(image_path, idx=None):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if idx is not None:
        return img, idx
    return img


def get_xyz_from_uvd(u, v, d, fx, fy, cx, cy):
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    z = d
    return np.array([x, y, z], dtype=np.float32)


def get_uv_from_xyz(x, y, z, fx, fy, cx, cy):
    u = x * fx / z + cx
    v = y * fy / z + cy
    return np.array([u, v], dtype=np.float32)


def get_bbox_from_landmarks(landmarks, width, height, margin=3):
    """
    Get bounding box from landmarks.

    Args:
        landmarks (np.ndarray): Landmarks, shape (num_points, 2)

    Returns:
        bbox (np.ndarray): Bounding box, shape (4,)
    """
    marks = np.array([m for m in landmarks if np.all(m != -1)], dtype=np.int64)
    x, y, w, h = cv2.boundingRect(marks)
    bbox = [x, y, x + w, y + h]
    bbox[0] = max(0, bbox[0] - margin)
    bbox[1] = max(0, bbox[1] - margin)
    bbox[2] = min(width - 1, bbox[2] + margin)
    bbox[3] = min(height - 1, bbox[3] + margin)
    return np.array(bbox, dtype=np.int64)


def get_bbox_from_mask(mask, margin=3):
    """
    Get bounding box from mask.

    Args:
        mask (np.ndarray): Mask, shape (height, width)

    Returns:
        bbox (np.ndarray): Bounding box, shape (4,)
    """
    height, width = mask.shape[:2]
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    bbox = [x, y, x + w, y + h]
    bbox[0] = max(0, bbox[0] - margin)
    bbox[1] = max(0, bbox[1] - margin)
    bbox[2] = min(width - 1, bbox[2] + margin)
    bbox[3] = min(height - 1, bbox[3] + margin)
    return np.array(bbox, dtype=np.int64)


def draw_debug_image(
    color_image,
    hand_mask=None,
    object_mask=None,
    prompt_points=None,
    prompt_labels=None,
    hand_marks=None,
    alpha=0.5,
    reduce_background=False,
    rgb_input=True,
    draw_boxes=False,
    draw_hand_sides=False,
):
    height, width = color_image.shape[:2]
    overlay = np.zeros_like(color_image) if reduce_background else color_image.copy()

    # draw hand mask
    if hand_mask is not None:
        for label in np.unique(hand_mask):
            if label == 0:
                continue
            mask = hand_mask == label
            color = HAND_COLORS[label]
            overlay[mask] = color.rgb if rgb_input else color.bgr

    # draw object mask
    if object_mask is not None:
        for label in np.unique(object_mask):
            if label == 0:
                continue
            mask = object_mask == label
            color = OBJ_CLASS_COLORS[label]
            overlay[mask] = color.rgb if rgb_input else color.bgr

    # draw boxes
    if draw_boxes:
        if hand_mask is not None:
            for label in np.unique(hand_mask):
                if label == 0:
                    continue
                mask = hand_mask == label
                color = HAND_COLORS[label]
                box = get_bbox_from_mask(mask)
                cv2.rectangle(
                    overlay,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    color.rgb if rgb_input else color.bgr,
                    2,
                )
        if object_mask is not None:
            for label in np.unique(object_mask):
                if label == 0:
                    continue
                mask = object_mask == label
                box = get_bbox_from_mask(mask)
                color = OBJ_CLASS_COLORS[label]
                cv2.rectangle(
                    overlay,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    color.rgb if rgb_input else color.bgr,
                    2,
                )

    # draw prompt points
    if prompt_points is not None and prompt_labels is not None:
        points = np.array(prompt_points, dtype=np.int32).reshape(-1, 2)
        labels = np.array(prompt_labels, dtype=np.int32).reshape(-1)
        for i, (point, label) in enumerate(zip(points, labels)):
            color = COLORS["dark_red"] if label == 0 else COLORS["dark_green"]
            cv2.circle(overlay, point, 3, color.rgb if rgb_input else color.bgr, -1)

    overlay = cv2.addWeighted(color_image, 1 - alpha, overlay, alpha, 0)

    # draw hand sides for hand mask
    if draw_hand_sides and hand_mask is not None and hand_marks is None:
        for label in np.unique(hand_mask):
            if label == 0:
                continue
            mask = hand_mask == label
            color = HAND_COLORS[label]
            text = "right" if label == 1 else "left"
            x, y, _, _ = cv2.boundingRect(mask.astype(np.uint8))
            text_x = x
            text_y = y - 5
            cv2.putText(
                overlay,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                color.rgb if rgb_input else color.bgr,
                1,
                cv2.LINE_AA,
            )

    # draw hand landmarks
    if hand_marks is not None:
        for idx, marks in enumerate(hand_marks):
            if np.all(marks == -1):
                continue

            # draw bones
            for bone_idx, bone in enumerate(HAND_BONES):
                if np.any(marks[bone[0]] == -1) or np.any(marks[bone[1]] == -1):
                    continue
                color = HAND_BONE_COLORS[bone_idx]
                cv2.line(
                    overlay,
                    marks[bone[0]],
                    marks[bone[1]],
                    color.rgb if rgb_input else color.bgr,
                    2,
                )
            # draw joints
            for i, mark in enumerate(marks):
                if np.any(mark == -1):
                    continue
                color = HAND_JOINT_COLORS[i]
                cv2.circle(overlay, mark, 5, (255, 255, 255), -1)
                cv2.circle(
                    overlay,
                    mark,
                    3,
                    color.rgb if rgb_input else color.bgr,
                    -1,
                )

            if draw_boxes:
                box = get_bbox_from_landmarks(marks, width, height)
                color = HAND_COLORS[1] if idx == 0 else HAND_COLORS[2]
                cv2.rectangle(
                    overlay,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    color.rgb if rgb_input else color.bgr,
                    2,
                )

            if draw_hand_sides:
                text = "right" if idx == 0 else "left"
                color = HAND_COLORS[1] if idx == 0 else HAND_COLORS[2]
                x, y, _, _ = cv2.boundingRect(
                    np.array([m for m in marks if np.all(m != -1)], dtype=np.int64)
                )
                text_x = x
                text_y = y - 5
                cv2.putText(
                    overlay,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    color.rgb if rgb_input else color.bgr,
                    1,
                    cv2.LINE_AA,
                )

    return overlay


def voxel_downsample_mask_th(points, voxel_size):
    """
    Creates a downsampling mask for a point cloud using voxel downsampling.

    Parameters:
    - points (torch.Tensor): The Nx3 tensor of points in the point cloud.
    - voxel_size (float): The size of the voxel used to downsample the point cloud.

    Returns:
    - torch.Tensor: A boolean mask indicating which points to keep.
    """

    # Compute voxel indices
    min_bound = torch.min(points, dim=0)[0]
    voxel_indices = ((points - min_bound) / voxel_size).floor().int()

    # Find unique voxels and their first occurrence
    _, indices = torch.unique(voxel_indices, dim=0, return_inverse=True)
    mask = torch.zeros_like(points[:, 0], dtype=torch.bool)

    # Mark the first occurrence of each unique voxel as True
    unique_indices = torch.unique(indices, sorted=False)
    mask.scatter_(0, unique_indices, True)

    return mask

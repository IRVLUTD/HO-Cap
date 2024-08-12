import pandas as pd
from hocap.utils import *

PCK_THRESH = [0.05, 0.1, 0.15, 0.2]  # Distance threshold for PCK calculation

r_or_l = ["left", "right"]


def calculate_mpjpe_3d(predicted, ground_truth):
    """
    Calculate the Mean Per Joint Position Error (MPJPE) between predicted and ground truth 3D joint positions.

    Parameters:
    predicted (numpy.ndarray): The predicted 3D joint positions with shape (N, J, 3).
    ground_truth (numpy.ndarray): The ground truth 3D joint positions with shape (N, J, 3).

    Returns:
    float: The MPJPE value.
    """
    # Calculate the Euclidean distance between the predicted and ground truth positions for each joint
    errors = np.linalg.norm(predicted - ground_truth, axis=2)

    # Calculate the mean distance across all joints for each sample
    sample_mpjpes = np.mean(errors, axis=1)

    # Calculate the mean MPJPE across all samples
    mpjpe = np.mean(sample_mpjpes)

    return mpjpe


# def calculate_pck(predicted, ground_truth, bboxes, threshold, normalize):
#     """
#     Calculate the Percentage of Correct Keypoints (PCK) for 2D hand pose estimation.

#     Parameters:
#     predicted (numpy.ndarray): The predicted 2D joint positions with shape (N, J, 2).
#     ground_truth (numpy.ndarray): The ground truth 2D joint positions with shape (N, J, 2).
#     bboxes (numpy.ndarray): Bounding boxes of the hands with shape (N, 4).
#     threshold (float): The distance threshold within which a predicted keypoint is considered correct.
#     normalize (numpy.ndarray): Normalization factors for distances with shape (N, 2).

#     Returns:
#     float: The PCK value (percentage of correct keypoints).
#     """
#     N, K, _ = predicted.shape

#     predicted = predicted.astype(np.float32)
#     ground_truth = ground_truth.astype(np.float32)
#     normalize = normalize.astype(np.float32)

#     box_s = np.zeros((N, 2), dtype=np.float32)
#     for i in range(N):
#         box_s[i, 0] = bboxes[i, 2] - bboxes[i, 0]
#         box_s[i, 1] = bboxes[i, 3] - bboxes[i, 1]

#     distances = np.linalg.norm((predicted - ground_truth) / box_s[:, None, :], axis=-1)

#     acc = np.array([acc_distance(d, threshold) for d in distances.T])
#     valid_acc = acc[acc >= 0]
#     cnt = len(valid_acc)
#     avg_acc = valid_acc.mean() if cnt > 0 else 0

#     return avg_acc * 100


def calculate_pck(predicted, ground_truth, bboxes, thresholds, normalize):
    """
    Calculate the Percentage of Correct Keypoints (PCK) for 2D hand pose estimation.

    Parameters:
    predicted (numpy.ndarray): The predicted 2D joint positions with shape (N, J, 2).
    ground_truth (numpy.ndarray): The ground truth 2D joint positions with shape (N, J, 2).
    bboxes (numpy.ndarray): Bounding boxes of the hands with shape (N, 4).
    thresholds (list[float]): A list of distance thresholds within which a predicted keypoint is considered correct.
    normalize (numpy.ndarray): Normalization factors for distances with shape (N, 2).

    Returns:
    dict: A dictionary where the keys are thresholds and the values are the PCK values for each threshold.
    """
    N, K, _ = predicted.shape

    predicted = predicted.astype(np.float32)
    ground_truth = ground_truth.astype(np.float32)
    normalize = normalize.astype(np.float32)

    box_s = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        box_s[i, 0] = bboxes[i, 2] - bboxes[i, 0]
        box_s[i, 1] = bboxes[i, 3] - bboxes[i, 1]

    # Normalize the predicted and ground truth keypoints
    distances = np.linalg.norm((predicted - ground_truth) / box_s[:, None, :], axis=-1)

    pck_results = []
    for threshold in thresholds:
        acc = np.array([acc_distance(d, threshold) for d in distances.T])
        valid_acc = acc[acc >= 0]
        cnt = len(valid_acc)
        avg_acc = valid_acc.mean() if cnt > 0 else 0
        pck_results.append(avg_acc)

    return pck_results


def acc_distance(distances, thr=0.5):
    """
    Return the percentage below the distance threshold, while ignoring
    distances values with -1.

    Parameters:
    distances (np.ndarray[N, ]): The normalized distances.
    thr (float): Threshold of the distances.

    Returns:
    float: Percentage of distances below the threshold.
           If all target keypoints are missing, return -1.
    """
    distance_valid = distances != -1
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < thr).sum() / num_distance_valid
    return -1


def get_hand_pose_evaluation(gt_result_file, pred_result_file):
    hand_json = read_data_from_json(gt_result_file)
    hamer_out_json = read_data_from_json(pred_result_file)

    all_pred_keypoints_3d = []
    all_gt_keypoints_3d_full = []
    all_gt_keypoints_2d_full = []
    all_pred_keypoints_2d_full = []
    all_gt_bboxes = []

    for out_id, out_data in hamer_out_json.items():
        if not out_data:
            continue
        gt_data = hand_json[out_id]
        is_right = np.array(out_data["is_right"], dtype=bool)
        pred_keypoints_2d_full = out_data["landmarks_2d"]
        pred_keypoints_3d = out_data["landmarks_3d"]

        gt_keypoints_2d_full = []
        gt_bboxes = []
        gt_keypoints_3d_full = []
        pred_keypoints_3d_r_and_l = []
        pred_keypoints_2d_r_and_l = []

        for n in range(is_right.shape[0]):
            rl = r_or_l[int(is_right[n])]

            gt_s_keypoints_2d_full = gt_data["landmarks_2d"][rl]
            gt_keypoints_2d_full.append(np.array(gt_s_keypoints_2d_full))
            gt_s_bboxes = np.array(gt_data["bbox"][rl])
            gt_bboxes.append(gt_s_bboxes)

            gt_s_keypoints_3d_full = gt_data["landmarks_3d"][rl]
            gt_keypoints_3d_full.append(np.array(gt_s_keypoints_3d_full))

            pred_keypoints_2d_r_and_l.append(np.array(pred_keypoints_2d_full[rl]))

        gt_keypoints_2d_full = np.stack(gt_keypoints_2d_full)
        gt_bboxes = np.stack(gt_bboxes)

        gt_keypoints_3d_full = np.stack(gt_keypoints_3d_full) * 1000

        for n in range(is_right.shape[0]):
            rl = r_or_l[int(is_right[n])]
            pred_keypoints_3d[rl] = np.array(pred_keypoints_3d[rl])
            if not is_right[n]:
                pred_keypoints_3d[rl][:, 0] = -pred_keypoints_3d[rl][:, 0]

            align = pred_keypoints_3d[rl][0] - gt_keypoints_3d_full[n][0]
            gt_keypoints_3d_full[n] += align

            pred_keypoints_3d_r_and_l.append(pred_keypoints_3d[rl])

        all_pred_keypoints_3d.append(np.stack(pred_keypoints_3d_r_and_l))
        all_gt_keypoints_3d_full.append(gt_keypoints_3d_full)

        all_pred_keypoints_2d_full.append(np.stack(pred_keypoints_2d_r_and_l))
        all_gt_keypoints_2d_full.append(gt_keypoints_2d_full)

        all_gt_bboxes.append(gt_bboxes)

    all_pred_keypoints_3d = np.concatenate(all_pred_keypoints_3d, axis=0)
    all_gt_keypoints_3d_full = np.concatenate(all_gt_keypoints_3d_full, axis=0)

    all_pred_keypoints_2d_full = np.concatenate(all_pred_keypoints_2d_full, axis=0)
    all_gt_keypoints_2d_full = np.concatenate(all_gt_keypoints_2d_full, axis=0)

    all_gt_bboxes = np.concatenate(all_gt_bboxes, axis=0)

    # Calculate PCK for each threshold in PCK_THRESH
    pcks = calculate_pck(
        all_pred_keypoints_2d_full,
        all_gt_keypoints_2d_full,
        all_gt_bboxes,
        PCK_THRESH,
        normalize=np.ones((len(all_pred_keypoints_2d_full), 2)),
    )

    # Calculate MPJPE
    mpjpe = calculate_mpjpe_3d(all_pred_keypoints_3d, all_gt_keypoints_3d_full)

    # Prepare data for the DataFrame
    pd_data = {}
    for i, thresh in enumerate(PCK_THRESH):
        pd_data[f"PCK ({thresh:.2f})"] = pcks[i] * 100

    pd_data["MPJPE (mm)"] = mpjpe

    # Convert the data to a DataFrame and print it
    df = pd.DataFrame([pd_data])
    result_str = df.to_string(index=False)

    print(result_str)

    # save to txt
    save_txt_file = pred_result_file.parent / f"{pred_result_file.stem}_PCK_MPJPE.txt"
    save_txt_file.write_text(result_str)
    tqdm.write(f"  * Results saved to {save_txt_file}")


if __name__ == "__main__":
    gt_result_file = PROJ_ROOT / "config/benchmarks/gt_hand_pose_results.json"
    pred_result_file = PROJ_ROOT / "config/benchmarks/demo_hand_pose_results.json"

    tqdm.write("- Evaluating Hand Pose Estimation results...")

    get_hand_pose_evaluation(gt_result_file, pred_result_file)

    tqdm.write("- Evaluation Done...")

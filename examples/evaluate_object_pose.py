import pandas as pd
from scipy.spatial import cKDTree
from hocap.utils import *


def to_homo(pts):
    """
    @pts: (N,3 or 2) will homogeneliaze the last dimension
    """
    assert len(pts.shape) == 2, f"pts.shape: {pts.shape}"
    homo = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)
    return homo


def add_err(pred, gt, model_pts):
    """
    Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).
    """
    pred_pts = (pred @ to_homo(model_pts).T).T[:, :3]
    gt_pts = (gt @ to_homo(model_pts).T).T[:, :3]
    e = np.linalg.norm(pred_pts - gt_pts, axis=1).mean()
    return e


def adi_err(pred, gt, model_pts):
    """
    @pred: 4x4 mat
    @gt:
    @model: (N,3)
    """
    pred_pts = (pred @ to_homo(model_pts).T).T[:, :3]
    gt_pts = (gt @ to_homo(model_pts).T).T[:, :3]
    nn_index = cKDTree(pred_pts)
    nn_dists, _ = nn_index.query(gt_pts, k=1, workers=-1)
    e = nn_dists.mean()
    return e


def compute_auc(rec, max_val=0.1):
    """
    Compute the Area Under Curve (AUC) for precision-recall curve up to a maximum recall value.

    This function calculates the AUC considering only the part of the precision-recall curve
    where the recall value is less than `max_val`. This is useful for scenarios where recall beyond
    a certain threshold is not relevant.

    Parameters:
    - rec (list or np.array): The recall values for different thresholds.
    - max_val (float): The maximum recall value to consider for AUC calculation.

    Returns:
    - float: The computed AUC value.

    Reference:
    - https://github.com/wenbowen123/iros20-6d-pose-tracking/blob/main/eval_ycb.py
    """
    if len(rec) == 0:
        return 0

    rec = np.sort(np.array(rec))
    n = len(rec)

    # Compute precision values based on the recall array
    prec = np.arange(1, n + 1) / n

    # Filter recall and precision arrays to include only recall values less than `max_val`
    valid_indices = np.where(rec < max_val)[0]
    rec = rec[valid_indices]
    prec = prec[valid_indices]

    # Prepare modified recall and precision arrays for AUC calculation
    mrec = np.concatenate(([0], rec, [max_val]))
    mpre = np.concatenate(([0], prec, [prec[-1] if len(prec) > 0 else 0]))

    # Ensure precision is non-decreasing
    for i in range(1, len(mpre)):
        mpre[i] = max(mpre[i], mpre[i - 1])

    # Calculate the differences in recall
    i = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i])

    return ap / max_val


def get_object_pose_evaluation(gt_pose_file, pred_pose_file):
    gt_poses = read_data_from_json(gt_pose_file)
    pred_poses = read_data_from_json(pred_pose_file)
    object_ids = sorted(pred_poses.keys())

    pd_data = {
        "Object_ID": [],
        "ADD-S_err (cm)": [],
        "ADD_err (cm)": [],
        "ADD-S_AUC (%)": [],
        "ADD_AUC (%)": [],
    }
    adi_errs = []
    add_errs = []

    for object_id in tqdm(object_ids, total=len(object_ids), ncols=60):
        if object_id not in gt_poses:
            continue

        object_mesh = trimesh.load(
            PROJ_ROOT / f"data/models/{object_id}/cleaned_mesh_10000.obj", process=False
        )
        vertices = object_mesh.vertices

        for key in sorted(pred_poses[object_id].keys()):
            if key not in gt_poses[object_id]:
                continue

            gt_ob_in_cam = np.array(gt_poses[object_id][key], dtype=np.float32)
            pred_ob_in_cam = np.array(pred_poses[object_id][key], dtype=np.float32)

            adi = adi_err(pred_ob_in_cam, gt_ob_in_cam, vertices.copy())
            add = add_err(pred_ob_in_cam, gt_ob_in_cam, vertices.copy())

            adi_errs.append(adi)
            add_errs.append(add)

        adi_dists = np.array(adi_errs)
        add_dists = np.array(add_errs)

        ADDS_AUC = compute_auc(adi_errs, max_val=0.1) * 100
        ADD_AUC = compute_auc(add_errs, max_val=0.1) * 100

        pd_data["Object_ID"].append(object_id)
        pd_data["ADD-S_err (cm)"].append(np.mean(adi_errs) * 100)
        pd_data["ADD_err (cm)"].append(np.mean(add_errs) * 100)
        pd_data["ADD-S_AUC (%)"].append(ADDS_AUC)
        pd_data["ADD_AUC (%)"].append(ADD_AUC)
    pd_data["Object_ID"].append("Average")
    pd_data["ADD-S_err (cm)"].append(np.mean(adi_dists) * 100)
    pd_data["ADD_err (cm)"].append(np.mean(add_dists) * 100)
    pd_data["ADD-S_AUC (%)"].append(np.mean(pd_data["ADD-S_AUC (%)"]))
    pd_data["ADD_AUC (%)"].append(np.mean(pd_data["ADD_AUC (%)"]))

    df = pd.DataFrame(pd_data)

    # save to txt
    result_str = df.to_string(index=False)
    save_txt_file = pred_pose_file.parent / f"{pred_pose_file.stem}_ADD_ADDS.txt"
    save_txt_file.write_text(result_str)

    print(result_str)

    tqdm.write(f"  * Results saved to {save_txt_file}")


if __name__ == "__main__":
    gt_pose_file = PROJ_ROOT / "config/benchmarks/gt_object_pose_results.json"
    dt_pose_file = PROJ_ROOT / "config/benchmarks/demo_object_pose_results.json"

    tqdm.write("- Evaluating Object Pose Estimation results...")

    get_object_pose_evaluation(gt_pose_file, dt_pose_file)

    tqdm.write("- Evaluation Done...")

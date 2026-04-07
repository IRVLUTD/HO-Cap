# HOCap Toolkit

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB.svg)](https://www.python.org/downloads/release/python-31015/) [![PyTorch 2.3.1](https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C.svg)](https://pytorch.org/) [![CUDA 11.8](https://img.shields.io/badge/CUDA-11.8-76B900.svg)](https://developer.nvidia.com/cuda-toolkit/) [![ROS Melodic](https://img.shields.io/badge/ROS1-Melodic-22314E.svg)](http://wiki.ros.org/melodic/) [![GPLv3.0 License](https://img.shields.io/badge/License-GPL--3.0-3DA639.svg)](./LICENSE)

The HOCap Toolkit is a Python package that provides evaluation and visualization tools for the HO-Cap dataset.

---

<div align=center>
  <H2>HO-Cap: A Capture System and Dataset for3D Reconstruction and Pose Tracking of Hand-Object Interaction</H2>
  <b>NeurIPS 2025, Datasets and Benchmarks Track</b>
  <p>Jikai Wang, Qifan Zhang, Yu-Wei Chao, Bowen Wen, Xiaohu Guo, Yu Xiang</p>
  <a src="https://img.shields.io/badge/project-website-green" href="https://irvlutd.github.io/HOCap/">
    <img src="https://img.shields.io/badge/project-website-green">
  </a>
  <a src="https://img.shields.io/badge/paper-arxiv-red" href="https://arxiv.org/abs/2406.06843">
    <img src="https://img.shields.io/badge/paper-arxiv-red">
  </a>
  <p><img src="./assets/ho-cap-demo-all-cameras.gif" width="80%" alt="HO-Cap Demo Video"></p>
</div>

<!-- ![hocap-demo-video](./assets/ho-cap-demo-all-cameras.gif) -->

---

<h2>Contents</h2>

- [HOCap Toolkit](#hocap-toolkit)
  - [Citation](#citation)
  - [License](#license)
  - [Installation](#installation)
  - [Download the HOCap Dataset](#download-the-hocap-dataset)
  - [Labels in the HOCap Dataset](#labels-in-the-hocap-dataset)
  - [Loading Dataset and Visualizing Samples](#loading-dataset-and-visualizing-samples)
  - [Evaluation](#evaluation)
    - [Hand Pose Estimation Evaluation](#hand-pose-estimation-evaluation)
    - [Object Pose Estimation Evaluation](#object-pose-estimation-evaluation)
    - [Object Detection Evaluation](#object-detection-evaluation)
  - [HOCap Dataset Split for Training and Testing](#hocap-dataset-split-for-training-and-testing)

## Citation

If HO-Cap helps your research, please consider citing the following:

```
@inproceedings{wang2025hocap,
title={{HO}-Cap: A Capture System and Dataset for 3D Reconstruction and Pose Tracking of Hand-Object Interaction},
author={Jikai Wang and Qifan Zhang and Yu-Wei Chao and Bowen Wen and Xiaohu Guo and Yu Xiang},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2025},
url={https://openreview.net/forum?id=hpu6r8oLw9}
}
```

## License

HOCap Toolkit is released under the [GNU General Public License v3.0](./LICENSE).

## Installation

This code is tested with [Python 3.10](https://docs.python.org/3.10) and [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) on [Ubuntu 20.04](https://releases.ubuntu.com/focal/). **Make sure CUDA 11.8 is installed on your system before running the code.**

1. Clone the HO-Cap repository from GitHub.

   ```bash
   git clone https://github.com/IRVLUTD/HO-Cap.git
   cd HO-Cap
   ```

2. Create conda environment

   ```bash
   conda create -n hocap-toolkit python=3.10
   ```

3. Activate conda environment

   ```bash
   conda activate hocap-toolkit
   ```

4. Install Pytorch and torchvision

   ```bash
   python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
   ```

5. Install hocap-toolkit package.

   ```bash
   python -m pip install --no-build-isolation -e .
   ```

6. Download MANO models and code (`mano_v1_2.zip`) from the [MANO website](https://mano.is.tue.mpg.de) and place the extracted `.pkl` files under `config/mano_models` directory. The directory should look like this:

   ```
   ./config/mano_models
   ├── MANO_LEFT.pkl
   └── MANO_RIGHT.pkl
   ```

## Download the HOCap Dataset

1. Run below code to download the whole dataset:

   ```
   python tools/hocap_downloader.py --subject_id all
   ```

2. Or you can download the dataset for a specific subject:

   ```
   python tools/hocap_downloader.py --subject_id subject_1
   ```

3. The downloaded `.zip` files will be extracted to the `./datasets` directory. And the directory should look like this:

   ```bash
   ./datasets
   ├── calibration
   ├── models
   ├── subject_1
   │   ├── 20231025_165502
   │   │   ├── 037522251142
   │   │   │   ├── color_000000.jpg
   │   │   │   ├── depth_000000.png
   │   │   │   ├── label_000000.npz
   │   │   │   └── ...
   │   │   ├── 043422252387
   │   │   ├── ...
   │   │   ├── hololens_kv5h72
   │   │   ├── meta.yaml
   │   │   ├── poses_m.npy
   │   │   ├── poses_o.npy
   │   │   └── poses_pv.npy
   │   ├── 20231025_165502
   │   └── ...
   ├── ...
   └── subject_9
   ```

## Labels in the HOCap Dataset

The HOCap dataset provides the following labels:

- 3d hand keypoints
- 2d hand keypoints
- hand bounding boxes
- hand sides
- hand MANO poses
- object 6OD poses
- segmentation masks

![vis_labels](./assets/vis_labels.png)

## Loading Dataset and Visualizing Samples

1. Below example shows how to visualize the pose annotations of one frame:

   ```bash
   python examples/sequence_pose_viewer.py
   ```

   ![sequence_pose_viewer](./assets/sequence_pose_viewer.png)

2. Below example shows how to visualize sequence by the interactive 3D viewer:

   ```bash
   python examples/sequence_3d_viewer.py
   ```

   ![sequence_3d_viewer](./assets/sequence_3d_viewer.gif)

   The 3D viewer provides the following functionalities:
   - `Background`: change the background color.
   - `Point Size`: change the point size.
   - `Show Skybox`: display/hide the skybox.
   - `Show Axes`: display/hide the axes of world coordinate.
   - `Crop Points`: crop the points outside the table area.
   - `Point Clouds`: display/hide the point clouds.
   - `Hand Mesh`: display/hide the hand mesh.
   - `Object Mesh`: display/hide the object mesh.
   - `Frame Slider`: change the frame index.
   - `Reset`: reset the camera view and the frame index.
   - `Pause/Play`: pause/play the sequence.
   - `Exit`: close the viewer.
   - `Help Tab`: show the help information.

3. Below example shows how to offline render the sequence:

   ```bash
   python examples/sequence_renderer.py
   ```

   This will render the color image and segmentation map for all the frames in the sequence. The rendered images will be saved in the `<sequence_folder>/renders/` directory.

   ![sequence_renderer_color](./assets/sequence_renderer_color.png)
   ![sequence_renderer_mask](./assets/sequence_renderer_mask.png)

4. Below example shows how to visualize the image labels:

   ```bash
   python examples/image_label_viewer.py
   ```

   ![image_label_viewer](./assets/image_label_viewer.png)

## Evaluation

HO-Cap provides the benchmark evaluation for three tasks:

- **Hand Pose Estimation (HPE)** (A2J-Transformer[^1], InterWild[^2] and HaMeR[^3])
- **Object Pose Estimation (OPE)** (MegaPose[^4] and FoundationPose[^5])
- **Object Detection (ODET)** (CNOS[^6], GroundingDINO[^7], YOLO11[^8] and RT-DETR[^9]).

Run below code to download the example evaluation results:

```bash
python config/benchmarks/benchmark_downloader.py
```

If the evaluation results are saved in the same format, the evaluation codes below can be used to evaluate the results.

### Hand Pose Estimation Evaluation

- Evaluate the hand pose estimation performance:

  ```bash
  python examples/evaluate_hand_pose.py
  ```

   <details>
   <summary> You should see the following output: </summary>

  ```
  PCK (0.05)  PCK (0.10)  PCK (0.15)  PCK (0.20)  MPJPE (mm)
   45.319048   81.247619   91.357143   95.080952   25.657379
  ```

   </details>

### Object Pose Estimation Evaluation

- Evaluate the novel object pose estimation performance:

  ```bash
  python examples/evaluate_object_pose.py
  ```

   <details>
   <summary> You should see the following output: </summary>

  ```
        Object_ID  ADD-S_err (cm)    ADD_err (cm)   ADD-S_AUC (%)     ADD_AUC (%)
  |-------------- |-------------- |-------------- |-------------- |-------------- |
            G01_1            0.42            0.72           95.79           92.82
            G01_2            0.37            0.69           96.39           93.38
            G01_3            0.45            0.82           95.72           92.08
            G01_4            0.61            2.73           94.14           74.19
          Average            0.46            1.24           95.43           88.04
  ```

   </details>

### Object Detection Evaluation

- Evaluate the object detection performance:

  ```bash
  python examples/evaluate_object_detection.py
  ```

   <details>
   <summary> You should see the following output: (click to expand) </summary>

  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.016
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.023
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.018
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.002
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.018
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.014
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.036
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.036
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.036
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.005
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.037
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.017
  AP: 0.016 | AP_50: 0.023 | AP_75: 0.018 | AP_s: 0.002 | AP_m: 0.018 | AP_l: 0.014
  ```

   </details>

## HOCap Dataset Split for Training and Testing

The train/valid/test split is defined separately for each task (HPE, ODET, OPE) by files `config/hocap_hpt.json`, `config/hocap_odt.json`, and `config/hocap_ope.json`. Each configuration file has the following structure:

```json
{
  "train": [[0, 0, 0, 0], ...],
  "valid": [...],
  "test": [...]
}
```

Each item is in format `[subject_index, sequence_index, camera_index, frame_index]`. For example, `[0, 0, 0, 0]` refers to `subject_1/20231022_190534/105322251564` folder and frame `color_000000.jpg`/ `depth_000000.png`.

To save time, we provide the pre-defined splits for each task, the split datasets could be downloaded [here](https://utdallas.box.com/s/dt19tcvhwitz223cjqa5riot6zcf6yba).

Or run below code to split the HOCap dataset manually, the split dataset will be saved in the `./datasets` directory.

- Hand Pose Estimation (HPE) task:

  ```bash
  python tools/hocap_dataset_split.py --task hpe
  ```

- Object Pose Estimation (OPE) task:

  ```bash
  python tools/hocap_dataset_split.py --task ope
  ```

- Object Detection (ODET) task:
  - COCO annotation type:
    ```bash
    python tools/hocap_dataset_split.py --task odet --anno_type coco
    ```
  - YOLO annotation type:
    ```bash
    python tools/hocap_dataset_split.py --task odet --anno_type yolo
    ```

[^1]: [A2J-Transformer: Anchor-to-Joint Transformer Network for 3D Interacting Hand Pose Estimation from a Single RGB Image](https://arxiv.org/abs/2304.03635)

[^2]: [Bringing Inputs to Shared Domains for 3D Interacting Hands Recovery in the Wild](https://arxiv.org/abs/2303.13652)

[^3]: [Reconstructing Hands in 3D with Transformers](https://arxiv.org/abs/2312.05251)

[^4]: [MegaPose: 6D Pose Estimation of Novel Objects via Render & Compare](https://arxiv.org/abs/2212.06870)

[^5]: [FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects](https://arxiv.org/abs/2312.08344)

[^6]: [CNOS: A Strong Baseline for CAD-based Novel Object Segmentation](http://arxiv.org/abs/2307.11067)

[^7]: [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)

[^8]: [YOLOv11: An Overview of the Key Architectural Enhancements](https://arxiv.org/html/2410.17725v1)

[^9]: [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)

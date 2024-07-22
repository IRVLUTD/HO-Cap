# HOCap Toolkit

HOCap Toolkit is a Python package that provides evaluation and visualization tools for the HOCap dataset.

**HO-Cap: A Capture System and Dataset for 3D Reconstruction and Pose Tracking of Hand-Object Interaction**

Jikai Wang, Qifan Zhang, Yu-Wei Chao, Bowen Wen, Xiaohu Guo, Yu Xiang

[ [arXiv](https://arxiv.org/abs/2406.06843) ] [ [Project page](https://irvlutd.github.io/HOCap/) ]

---

## Contents

- [HOCap Toolkit](#hocap-toolkit)
  - [Contents](#contents)
  - [News](#news)
  - [BibTeX Citation](#bibtex-citation)
    - [License](#license)
  - [Installation](#installation)
  - [Download the HO-Cap Dataset](#download-the-ho-cap-dataset)
  - [Loading Dataset and Visualizing Samples](#loading-dataset-and-visualizing-samples)
  - [Evaluation](#evaluation)
    - [Hand Pose Estimation Evaluation](#hand-pose-estimation-evaluation)
    - [Novel Object Pose Estimation Evaluation](#novel-object-pose-estimation-evaluation)
    - [Novel Object Detection Evaluation](#novel-object-detection-evaluation)


## News

- **2024-06-24**: The HO-Cap dataset is released! Please check the [project page](https://irvlutd.github.io/HOCap/) for more details.

<!-- ![hocap-demo-video](https://irvlutd.github.io/HOCap/assets/videos/ho-cap-demo.mp4) -->
![hocap-demo-video](./assets/ho-cap-demo-all-cameras.gif)

## BibTeX Citation

If HO-Cap helps your research, please consider citing the following citations:

```
@misc{wang2024hocap,
      title={HO-Cap: A Capture System and Dataset for 3D Reconstruction and Pose Tracking of Hand-Object Interaction}, 
      author={Jikai Wang and Qifan Zhang and Yu-Wei Chao and Bowen Wen and Xiaohu Guo and Yu Xiang},
      year={2024},
      eprint={2406.06843},
      archivePrefix={arXiv},
      primaryClass={id='cs.CV' full_name='Computer Vision and Pattern Recognition' is_active=True alt_name=None in_archive='cs' is_general=False description='Covers image processing, computer vision, pattern recognition, and scene understanding. Roughly includes material in ACM Subject Classes I.2.10, I.4, and I.5.'}
}
```

### License

HOCap Toolkit is released under the [GNU General Public License v3.0](LICENSE).



## Installation

This code is tested with [Python 3.10](https://docs.python.org/3.10) and CUDA 11.8 on Ubuntu 20.04. Make sure CUDA 11.8 is installed on your system before running the code.

1. Clone the HO-Cap repository from GitHub.

   ```bash
   git clone --rescursive git@github.com:IRVLUTD/HO-Cap.git
   ```

1. Change the current directory to the cloned repository.

   ```bash
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

4. Install hocap-toolkit package and dependencies

   ```bash
   # Install dependencies
   python -m pip install --no-cache-dir -r requirements.txt

   # Build meshsdf_loss
   bash build.sh

   # Install hocap-toolkit
   python -m pip install -e .
   ```

5. Download models for external libraries

   ```
   bash download_models.sh
   ```

6. Download MANO models and code (`mano_v1_2.zip`) from the [MANO website](https://mano.is.tue.mpg.de) and place the extracted `.pkl` files under `config/ManoModels` directory. The directory should look like this:

   ```
   ./config/ManoModels
   ├── MANO_LEFT.pkl
   └── MANO_RIGHT.pkl
   ```


## Download the HO-Cap Dataset

1. Download the HO-Cap dataset from [box](https://utdallas.box.com/v/ho-cap-release).
2. Extract the dataset to the `./data` directory. And the directory should look like this:
   
   ```
   ./data
   ├── calibration
   ├── models
   ├── subject_1
   │   ├── 20231025_165502
   │   ├── ...
   ├── ...
   └── subject_9
      ├── 20231027_123403
      ├── ...
   ```

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

3. Below example shows how to offline render the sequence:
   
   ```bash
   python examples/sequence_renderer.py
   ```

   ![sequence_renderer_color](./assets/sequence_renderer_color.png)
   ![sequence_renderer_mask](./assets/sequence_renderer_mask.png)

4. You can list all the sequence folders (70 in total) with:
   ```
   ls -d data/subject_*/2023* | sort
   ```

   <details>

   <summary> You should see the following output: (click to expand)</summary>

   ```
   data/subject_1/20231025_165502
   data/subject_1/20231025_165807
   data/subject_1/20231025_170105
   data/subject_1/20231025_170231
   data/subject_1/20231025_170532
   data/subject_1/20231025_170650
   data/subject_1/20231025_170959
   data/subject_1/20231025_171117
   data/subject_1/20231025_171314
   data/subject_1/20231025_171417
   data/subject_2/20231022_200657
   data/subject_2/20231022_201316
   data/subject_2/20231022_201449
   data/subject_2/20231022_201556
   data/subject_2/20231022_201942
   data/subject_2/20231022_202115
   data/subject_2/20231022_202617
   data/subject_2/20231022_203100
   data/subject_2/20231023_163929
   data/subject_2/20231023_164242
   data/subject_2/20231023_164741
   data/subject_2/20231023_170018
   data/subject_3/20231024_154531
   data/subject_3/20231024_154810
   data/subject_3/20231024_155008
   data/subject_3/20231024_161209
   data/subject_3/20231024_161306
   data/subject_3/20231024_161937
   data/subject_3/20231024_162028
   data/subject_3/20231024_162327
   data/subject_3/20231024_162409
   data/subject_3/20231024_162756
   data/subject_3/20231024_162842
   data/subject_4/20231026_162155
   data/subject_4/20231026_162248
   data/subject_4/20231026_163223
   data/subject_4/20231026_164131
   data/subject_4/20231026_164812
   data/subject_4/20231026_164909
   data/subject_4/20231026_164958
   data/subject_5/20231027_112303
   data/subject_5/20231027_113202
   data/subject_5/20231027_113535
   data/subject_6/20231025_110646
   data/subject_6/20231025_110808
   data/subject_6/20231025_111118
   data/subject_6/20231025_111357
   data/subject_6/20231025_112229
   data/subject_6/20231025_112332
   data/subject_6/20231025_112546
   data/subject_7/20231022_190534
   data/subject_7/20231022_192832
   data/subject_7/20231022_193506
   data/subject_7/20231022_193630
   data/subject_7/20231022_193809
   data/subject_7/20231023_162803
   data/subject_7/20231023_163653
   data/subject_8/20231024_180111
   data/subject_8/20231024_180651
   data/subject_8/20231024_180733
   data/subject_8/20231024_181413
   data/subject_9/20231027_123403
   data/subject_9/20231027_123725
   data/subject_9/20231027_123814
   data/subject_9/20231027_124057
   data/subject_9/20231027_124926
   data/subject_9/20231027_125019
   data/subject_9/20231027_125315
   data/subject_9/20231027_125407
   data/subject_9/20231027_125457
   ```

   </details>



## Evaluation


HO-Cap provides the benchmark evaluation for three tasks:
- **Hand Pose Estimation** (A2J-Transformer[^1] and HaMeR[^2])
- **Novel Object Pose Estimation** (MegaPose[^3] and FoundationPose[^4])
- **Novel Object Detection** (CNOS[^5] and GroundingDINO[^6]).

The benchmark evaluation example results are stored under `./config/benchmarks` directory.

### Hand Pose Estimation Evaluation

- Evaluate the hand pose estimation performance using A2J-Transformer and HaMeR:
   
   ```bash
   python examples/evaluate_hand_pose.py
   ```

   <details>

   <summary> You should see the following output: </summary>

   ```
   Evaluation results for A2J-Transformer:

   Evaluation results for HaMeR:
   ```
   </details>


### Novel Object Pose Estimation Evaluation

- Evaluate the novel object pose estimation performance using MegaPose and FoundationPose:
   
   ```bash
   python examples/evaluate_novel_object_pose.py
   ```

   <details>

   <summary> You should see the following output: </summary>

   ```
   Evaluation results for MegaPose:

   Evaluation results for FoundationPose:
   ```
   </details>


### Novel Object Detection Evaluation

- Evaluate the novel object detection performance using CNOS and GroundingDINO:
   
   ```bash
   python examples/evaluate_novel_object_detection.py
   ```

   <details>

   <summary> You should see the following output: (click to expand) </summary>

   ```
   Evaluation results for CNOS:

   Evaluation results for GroundingDINO:
   ```
   </details>





[^1]: [A2J-Transformer: Anchor-to-Joint Transformer Network for 3D Interacting Hand Pose Estimation from a Single RGB Image](https://arxiv.org/abs/2304.03635)
[^2]: [Reconstructing Hands in 3D with Transformers](https://arxiv.org/abs/2312.05251)
[^3]: [MegaPose: 6D Pose Estimation of Novel Objects via Render & Compare](https://arxiv.org/abs/2212.06870)
[^4]: [FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects](https://arxiv.org/abs/2312.08344)
[^5]: [CNOS: A Strong Baseline for CAD-based Novel Object Segmentation](http://arxiv.org/abs/2307.11067)
[^6]: [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)
```
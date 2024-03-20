# HO-Cap

## Environment Setup

1. Clone the repository
```bash
git clone --recurse-submodules git@github.com:IRVLUTD/HO-Cap.git
```

2. Create conda environment
```bash
conda create -n ho-cap python=3.10
```

3. Activate conda environment
```bash
conda activate ho-cap
```

4. Install dependencies ([CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) needed)
```bash
# install dependencies from requirements.txt
python -m pip install --no-cache-dir -r requirements.txt

# install chumpy for MANO model
python -m pip install --quiet --no-cache-dir git+https://github.com/gobanana520/chumpy.git

# install modified version pyopengl for PyRender
python -m pip install --quiet --no-cache-dir git+https://github.com/mmatl/pyopengl.git

# install PyTorch3D
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt211/download.html

# install SAM
python -m pip install --quiet --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git

# install MobileSAM
python -m pip install --quiet --no-cache-dir git+https://github.com/ChaoningZhang/MobileSAM.git

# build manopath and meshsdf_loss
bash build.sh
```
5. Download models for external libraries
```bash
# download model for LoFTR
bash config/LoFTR/download_loftr_model.sh

# download model for MediaPipe
bash config/Mediapipe/download_mediapipe_model.sh

# download model for SAM
bash config/SAM/download_sam_model.sh

# download model for XMem
bash config/XMem/download_xmem_model.sh
```

# HO-Cap

## Environment Setup

1. Create conda environment
```bash
conda create -n ho-cap python=3.10
```

2. Activate conda environment
```bash
conda activate ho-cap
```

3. Install dependencies
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
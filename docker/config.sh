#!/bin/bash

DOCKER_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJ_ROOT=$(dirname "$DOCKER_DIR")

CONTAINER_NAME="hopipe/release"
CONTAINER_DISPLAY_NAME="hopipe"
CONTAINER_TAG="latest"

#####################
# DOCKER BUILD ARGS
#####################

### GPU Compute Capability for RTX 1080,2080,3090 (ref to https://developer.nvidia.com/cuda-gpus)
CUDA_ARCH="6.0 6.1 7.0 7.5 8.0 8.6 8.9"
USER_NAME="my_user"
USER_ID=$(id -u)
GROUP_ID=$(id -g)
CONDA_ENV_NAME="HO101"

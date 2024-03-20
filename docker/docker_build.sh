#!/bin/bash

source $(dirname $0)/config.sh

# build the docker image
time docker build \
    --build-arg USERNAME=${USER_NAME} \
    --build-arg UID=${USER_ID} \
    --build-arg GID=${GROUP_ID} \
    --build-arg CONDA_ENV_NAME=${CONDA_ENV_NAME} \
    --build-arg CUDA_ARCH="${CUDA_ARCH}" \
    --network host \
    --file ${DOCKER_DIR}/Dockerfile \
    --tag ${CONTAINER_NAME}:${CONTAINER_TAG} \
    ${DOCKER_DIR}

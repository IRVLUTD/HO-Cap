#!/bin/bash

source $(dirname $0)/config.sh

# give permissions to the Docker client to connect to your X server
xhost +local:${USER_NAME}

# run the container
docker run \
    --detach \
    --tty \
    --gpus 'all,"capabilities=compute,graphics,utility,video,display"' \
    --ipc=host \
    --ulimit memlock=-1 \
    --network=host \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    --env DISPLAY=${DISPLAY} \
    --env GIT_INDEX_FILE \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --volume ${HOME}/.ssh:/home/${USER_NAME}/.ssh \
    --volume "${PROJ_ROOT}":/home/${USER_NAME}/code \
    --volume /metadisk:/metadisk \
    --volume /tmp:/tmp \
    --volume /etc/localtime:/etc/localtime:ro \
    --volume /dev:/dev \
    --device-cgroup-rule "c 81:* rmw" \
    --device-cgroup-rule "c 189:* rmw" \
    --name ${CONTAINER_DISPLAY_NAME} \
    ${CONTAINER_NAME}:latest

sleep 1

docker ps -a

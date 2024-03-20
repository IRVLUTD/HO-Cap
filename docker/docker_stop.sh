#!/bin/bash

source $(dirname $0)/config.sh

# stop the container
docker kill ${CONTAINER_DISPLAY_NAME}
sleep 1
docker rm -f ${CONTAINER_DISPLAY_NAME}

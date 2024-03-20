#!/bin/bash

source $(dirname $0)/config.sh

docker exec -it ${CONTAINER_DISPLAY_NAME} zsh --login

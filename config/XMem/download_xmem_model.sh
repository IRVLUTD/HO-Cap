#!/bin/bash

CURR_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Download the model XMem.pth
echo "Downloading 'XMem.pth' ..."
rm -rf ${CURR_DIR}/XMem.pth && \
wget -q --show-progress https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth -P ${CURR_DIR}/

# Download the model XMem-s012.pth
echo "Downloading 'XMem-s012.pth' ..."
rm -rf ${CURR_DIR}/XMem-s012.pth && \
wget -q --show-progress https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth -P ${CURR_DIR}/

# Download the model XMem-no-sensory.pth
echo "Downloading 'XMem-no-sensory.pth' ..."
rm -rf ${CURR_DIR}/XMem-no-sensory.pth && \
wget -q --show-progress https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-no-sensory.pth -P ${CURR_DIR}/
#!/bin/bash

CURR_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Download the SAM models.
echo "Downloading 'vit_h'..."
rm -rf ${CURR_DIR}/sam_vit_h.pth && \
wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O ${CURR_DIR}/sam_vit_h.pth

echo "Downloading 'vit_l'..."
rm -rf ${CURR_DIR}/sam_vit_l.pth && \
wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -O ${CURR_DIR}/sam_vit_l.pth

echo "Downloading 'vit_b'..."
rm -rf ${CURR_DIR}/sam_vit_b.pth && \
wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O ${CURR_DIR}/sam_vit_b.pth

# Download the MobileSAM model.
echo "Downloading 'vit_t'..."
rm -rf ${CURR_DIR}/sam_vit_t.pth && \
wget -q --show-progress https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt -O ${CURR_DIR}/sam_vit_t.pth

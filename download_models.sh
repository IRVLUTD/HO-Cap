#!/bin/bash

# Get the project root directory
PROJ_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Function to download using wget
wget_download() {
    local url=$1
    local output=$2

    target_file="${PROJ_ROOT}/config/${output}"
    mkdir -p "$(dirname "${target_file}")"
    if [ -f "${target_file}" ]; then
        echo "File '${output}' already exists. Skipping download."
    else
        wget -q --show-progress ${url} -O ${target_file}
    fi
}

# Function to download using gdown
gdown_download() {
    local url=$1
    local output=$2

    file_id=$(echo "$url" | sed -n 's|.*file/d/\(.*\)/view.*|\1|p')
    target_file="${PROJ_ROOT}/config/${output}"
    mkdir -p "$(dirname "${target_file}")"
    if [ -f "${target_file}" ]; then
        echo "File '${output}' already exists. Skipping download."
    else
        gdown "https://drive.google.com/uc?id=$file_id" -O ${target_file}
    fi
}

echo ">>>>>>>>>> Start downloading models <<<<<<<<<<"
start_time=$(date +%s)

# Download MediaPipe models
echo "Downloading the MediaPipe Hand Landmark model..."
wget_download \
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" \
    "mediapipe/hand_landmarker.task"

# Download SAM models
# echo "Downloading the SAM model 'vit_h'..."
# wget_download "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" "sam/sam_vit_h.pth"

# echo "Downloading the SAM model 'vit_l'..."
# wget_download "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth" "sam/sam_vit_l.pth"

# echo "Downloading the SAM model 'vit_b'..."
# wget_download "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" "sam/sam_vit_b.pth"

echo "Downloading the Mobile-SAM model 'vit_t'..."
wget_download \
    "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt" \
    "sam/sam_vit_t.pth"

# Download XMem models
echo "Downloading the XMem model 'XMem.pth'..."
wget_download \
    "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth" \
    "xmem/XMem.pth"

echo "Downloading the XMem model 'XMem-s012.pth'..."
wget_download \
    "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth" \
    "xmem/XMem-s012.pth"

echo "Downloading the XMem model 'XMem-no-sensory.pth'..."
wget_download \
    "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-no-sensory.pth" \
    "xmem/XMem-no-sensory.pth"

end_time=$(date +%s)
duration=$((end_time - start_time))
echo ">>>>>>>>>> Done!!! (${duration} seconds) <<<<<<<<<<"
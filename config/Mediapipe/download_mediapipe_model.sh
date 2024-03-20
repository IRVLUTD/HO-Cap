#!/bin/bash

CURR_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Download the Hand Landmark model and the palm detection model.
echo "Downloading the Hand Landmark model and the palm detection model..."
rm -rf ${CURR_DIR}/hand_landmarker.task && \
wget -q --show-progress https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task -O ${CURR_DIR}/hand_landmarker.task

# Download the Multi-class selfie segmentation model.
echo "Downloading the Multi-class selfie segmentation model..."
rm -rf ${CURR_DIR}/selfie_multiclass_256x256.tflite && \
wget -q --show-progress https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite -O ${CURR_DIR}/selfie_multiclass_256x256.tflite

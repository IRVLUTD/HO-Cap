#!/bin/bash

CURR_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Download the LoFTR model 'outdoor_ds.ckpt' from Google Drive
# https://drive.google.com/file/d/1M-VD35-qdB5Iw-AtbDBCKC7hPolFW9UY/view?usp=drive_link
gdown https://drive.google.com/uc?id=1M-VD35-qdB5Iw-AtbDBCKC7hPolFW9UY -O $CURR_DIR/outdoor_ds.ckpt

# Download the LoFTR model 'indoor_ds_new.ckpt' from Google Drive
# https://drive.google.com/file/d/19s3QvcCWQ6g-N1PrYlDCg-2mOJZ3kkgS/view?usp=drive_link
gdown https://drive.google.com/uc?id=19s3QvcCWQ6g-N1PrYlDCg-2mOJZ3kkgS -O $CURR_DIR/indoor_ds_new.ckpt

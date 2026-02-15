#!/usr/bin/env bash

WORK_DIR=$(cd "$(dirname "$0")"; pwd)
export PYTHONPATH=${WORK_DIR}/../

# ---- Train ---- #
rlaunch --cpu=1 --gpu=1 --memory=16384 -- python3 tools/train.py \
    -f configs/retinanet_res50_3x_800size_chongqigongmen.py \
    -b 8 \
    

# ---- Test ---- #
rlaunch --cpu=1 --gpu=1 --memory=16384 -- python3 tools/test.py \
    -f configs/retinanet_res50_3x_800size_chongqigongmen.py \
    -b 8 \

# inference
#python3 tools/inference.py \
#   -f configs/retinanet_res50_3x_800size_chongqigongmen.py  \
#   -i data/chongqigongmen/images/18516456,19ce1000ac177dec.jpg \
#   -w log-of-retinanet_res50_3x_800size_chongqigongmen/epoch_35.pkl

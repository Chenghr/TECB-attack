#!/bin/bash

# 基础路径
PROJECT_ROOT=$(dirname $(dirname $(readlink -f "$0")))
SCRIPT_PATH="${PROJECT_ROOT}/../pretest/permuted_training.py"

# DATASET="CIFAR100"
# DATA_DIR="${PROJECT_ROOT}/../data/"
# MODEL_DIR="${PROJECT_ROOT}/../results/models/TECB/cifar100/"

# DATASET="CINIC10L"
# DATA_DIR="${PROJECT_ROOT}/../data/cinic/"
# MODEL_DIR="${PROJECT_ROOT}/../results/models/TECB/cinic10"

DATASET="CIFAR10"
DATA_DIR="${PROJECT_ROOT}/../data/"
HALF=12
MODEL_DIR="${PROJECT_ROOT}/../results/models/TECB/cifar10/half_${HALF}/1_saved_models"
# UPDATE_MODE=('bottom_only' 'top_only' 'both')
UPDATE_MODE=('bottom_only')

for update_mode in "${UPDATE_MODE[@]}"; do
    python ${SCRIPT_PATH} \
        --dataset ${DATASET} \
        --data_dir ${DATA_DIR} \
        --model_dir ${MODEL_DIR} \
        --half ${HALF} \
        --epochs 30 \
        --lr 1e-1 \
        --batch_size 128 \  
        --update_mode $update_mode 
done
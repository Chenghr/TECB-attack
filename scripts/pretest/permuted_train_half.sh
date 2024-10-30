#!/bin/bash

# 基础路径
PROJECT_ROOT=$(dirname $(dirname $(readlink -f "$0")))
SCRIPT_PATH="${PROJECT_ROOT}/../pretest/permuted_training.py"

DATASET="CIFAR10"
DATA_DIR="${PROJECT_ROOT}/../data/"

# DATASET="CIFAR100"
# DATA_DIR="${PROJECT_ROOT}/../data/"
# MODEL_DIR="${PROJECT_ROOT}/../results/models/TECB/cifar100/"

# DATASET="CINIC10L"
# DATA_DIR="${PROJECT_ROOT}/../data/cinic/"
# MODEL_DIR="${PROJECT_ROOT}/../results/models/TECB/cinic10"

HALF=(1 4 8 12 24)
for half in "${HALF[@]}"; do
    MODEL_DIR="${PROJECT_ROOT}/../results/models/TECB/cifar10/half_${half}/1_saved_models"
    
    python ${SCRIPT_PATH} \
        --dataset ${DATASET} \
        --data_dir ${DATA_DIR} \
        --model_dir ${MODEL_DIR} \
        --half ${half} \
        --epochs 30 \
        --lr 1e-05 \
        --batch_size 256 
done
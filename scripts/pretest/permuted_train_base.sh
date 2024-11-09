#!/bin/bash

# 基础路径
PROJECT_ROOT=$(dirname $(dirname $(readlink -f "$0")))
# SCRIPT_PATH="${PROJECT_ROOT}/../pretest/permuted_training.py"

# Project configuration
# PROJECT_ROOT="$(pwd)"
DATA_DIR="${PROJECT_ROOT}/../data"
SCRIPT_PATH="${PROJECT_ROOT}/../pretest/permuted_train_for_tecb.py"

# DATASET="CIFAR100"
# MODEL_DIR="${PROJECT_ROOT}/../results/models/TECB/cifar100/"

# DATASET="CINIC10L"
# DATA_DIR="${PROJECT_ROOT}/../data/cinic/"
# MODEL_DIR="${PROJECT_ROOT}/../results/models/TECB/cinic10"

DATASET="CIFAR10"
HALF=16
MODEL_DIR="${PROJECT_ROOT}/../results/models/TECB/cifar10/half_${HALF}/1_saved_models"
log_file_name="test.log"

python ${SCRIPT_PATH} \
    --attack_method "TECB" \
    --dataset ${DATASET} \
    --data_dir ${DATA_DIR} \
    --model_dir ${MODEL_DIR} \
    --half ${HALF} \
    --epochs 10 \
    --lr 2e-4 \
    --batch_size 256 \
    --log_file_name $log_file_name \
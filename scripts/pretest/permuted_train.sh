#!/bin/bash

# 基础路径
PROJECT_ROOT=$(dirname $(dirname $(readlink -f "$0")))
SCRIPT_PATH="${PROJECT_ROOT}/../pretest/permuted_training.py"

# DATASET="CIFAR100"
# DATA_DIR="${PROJECT_ROOT}/../data/"
# MODEL_DIR="${PROJECT_ROOT}/../results/models/TECB/cifar100/"

DATASET="CINIC10L"
DATA_DIR="${PROJECT_ROOT}/../data/cinic/"
MODEL_DIR="${PROJECT_ROOT}/../results/models/TECB/cinic10"

HALF=16

python ${SCRIPT_PATH} \
    --dataset ${DATASET} \
    --data_dir ${DATA_DIR} \
    --model_dir ${MODEL_DIR} \
    --half ${HALF} \
    --epochs 20 \
    --lr 0.0005 \
    --batch_size 128 \
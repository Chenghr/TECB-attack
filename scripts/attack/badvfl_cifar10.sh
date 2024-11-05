#!/bin/bash

# run_experiments.sh

# 基础路径
PROJECT_ROOT=$(dirname $(dirname $(readlink -f "$0")))
DATASET="CIFAR10"
DATA_PATH="${PROJECT_ROOT}/../data"
BASE_SAVE_PATH="${PROJECT_ROOT}/../results/models/BadVFL/cifar10"
SCRIPT_PATH="${PROJECT_ROOT}/../attack/badvfl/badvfl_cifar10_training.py"

# half参数列表
HALF_VALUES=(14 16 18 20 22 24 26)

for half in "${HALF_VALUES[@]}"; do
    save_path="${BASE_SAVE_PATH}/half_${half}"
    
    python ${SCRIPT_PATH} \
        --dataset ${DATASET} \
        --data_dir ${DATA_PATH} \
        --half $half \
        --report_freq 5 \
        --workers 8 \
        --epochs 80 \
        --poison_budget 0.1 \
        --corruption_amp 5.0 \
        --optimal_sel \
        --saliency_map_injection \
        --pre_train_epochs 20 \
        --trigger_train_epochs 40 \
        --window_size 5 \
        --backdoor_start \
        --lr 0.02 \
        --alpha 0.05 \
        --eps 0.5 \
        --batch_size 64 \
        --save $save_path
done

#!/bin/bash

# run_experiments.sh
set -e

# 基础路径
PROJECT_ROOT=$(dirname $(dirname $(readlink -f "$0")))
DATASET="CIFAR100"
DATA_PATH="${PROJECT_ROOT}/../data"
BASE_SAVE_PATH="${PROJECT_ROOT}/../results/models/BadVFL/cifar100"
SCRIPT_PATH="${PROJECT_ROOT}/../attack/badvfl/badvfl.py"
DEVICE="cuda:0"

# half参数列表
HALF_VALUES=(4 8 12 16 20 24)
# HALF_VALUES=(8 )

for half in "${HALF_VALUES[@]}"; do
    save_path="${BASE_SAVE_PATH}/half_${half}"

    python ${SCRIPT_PATH} \
        --dataset ${DATASET} \
        --data_dir ${DATA_PATH} \
        --half $half \
        --save $save_path \
        --epochs 60 \
        --pre_train_epochs 20 \
        --trigger_train_epochs 40 \
        --backdoor_start_epoch 10 \
        --report_freq 5 \
        --workers 8 \
        --batch_size 64 \
        --lr 0.2 \
        --trigger_lr 0.001 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --grad_clip 5.0 \
        --gamma 0.97 \
        --decay_period 1 \
        --resume '' \
        --start_epoch 0 \
        --step_gamma 0.1 \
        --stone1 30 \
        --layers 18 \
        --u_dim 64 \
        --k 2 \
        --alpha 0.05 \
        --eps 0.75 \
        --corruption_amp 5.0 \
        --backdoor_start \
        --poison_budget 0.5 \
        --optimal_sel \
        --saliency_map_injection \
        --window_size 5
done
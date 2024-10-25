#!/bin/bash

# run_experiments.sh

# 基础路径
PROJECT_ROOT=$(dirname $(dirname $(readlink -f "$0")))
DATASET="CINIC10L"
DATA_PATH="${PROJECT_ROOT}/../data/cinic/"
BASE_SAVE_PATH="${PROJECT_ROOT}/../results/models/TECB/cinic"
SCRIPT_PATH="${PROJECT_ROOT}/../attack/tecb/vfl_cinic10.py"

# half参数列表
HALF_VALUES=(1 4 8 12 16 24)

for half in "${HALF_VALUES[@]}"; do
    save_path="${BASE_SAVE_PATH}/half_${half}"
    
    python ${SCRIPT_PATH} \
        --dataset ${DATASET} \
        --data_dir ${DATA_PATH} \
        --half $half \
        --report_freq 5 \
        --workers 8 \
        --backdoor 50 \
        --poison_epochs 80 \
        --target_class plane \
        --poison_num 4 \
        --corruption_amp 12.0 \
        --backdoor_start \
        --lr 0.8 \
        --alpha 0.05 \
        --eps 1.0 \
        --epochs 100 \
        --batch_size 2048 \
        --save $save_path
done
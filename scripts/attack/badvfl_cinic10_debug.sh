#!/bin/bash

# run_experiments.sh
set -e

# 基础路径
PROJECT_ROOT=$(dirname $(dirname $(readlink -f "$0")))
DATASET="CINIC10L"
DATA_PATH="${PROJECT_ROOT}/../data/cinic/"
BASE_SAVE_PATH="${PROJECT_ROOT}/../results/models/BadVFL/cinic10"
SCRIPT_PATH="${PROJECT_ROOT}/../attack/badvfl/badvfl.py"

# half参数列表
HALF_VALUES=(11)

for half in "${HALF_VALUES[@]}"; do
    save_path="${BASE_SAVE_PATH}/half_${half}"

    python ${SCRIPT_PATH} \
        --dataset ${DATASET} \
        --data_dir ${DATA_PATH} \
        --half $half \
        --epochs 2 \
        --backdoor_start_epoch 1 \
        --pre_train_epochs 1 \
        --trigger_train_epochs 1 \
        --seed_num 1 \
        --save $save_path
done

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
HALF_VALUES=(4 8 12 16 20 24 28)

for half in "${HALF_VALUES[@]}"; do
    save_path="${BASE_SAVE_PATH}/half_${half}"

    python ${SCRIPT_PATH} \
        --dataset ${DATASET} \
        --data_dir ${DATA_PATH} \
        --half $half \
        --save $save_path
done

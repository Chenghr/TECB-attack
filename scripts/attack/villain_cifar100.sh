#!/bin/bash
set -e

# 基础路径
PROJECT_ROOT=$(dirname $(dirname $(readlink -f "$0")))
DATASET="CIFAR100"
DATA_PATH="${PROJECT_ROOT}/../data"
BASE_SAVE_PATH="${PROJECT_ROOT}/../results/models/Villain/cifar100"
SCRIPT_PATH="${PROJECT_ROOT}/../attack/villain/villain.py"

# 实验参数
REPORT_FREQ=5
WORKERS=8
POISON_BUDGET=0.5
DROPOUT_RATIO=0.25
BETA=0.4
GAMMA_UP=1.2
GAMMA_LOW=0.6
TARGET_CLASS="boy"
LR=0.5
LOCAL_LR=0.75
EPS=0.06

# half参数列表
HALF_VALUES=(4 8 12 16 20 24 28)

for HALF in "${HALF_VALUES[@]}"; do
    SAVE_PATH="${BASE_SAVE_PATH}/half_${HALF}"

    python ${SCRIPT_PATH} \
        --dataset ${DATASET} \
        --data_dir ${DATA_PATH} \
        --half ${HALF} \
        --save ${SAVE_PATH} \
        --seed_num 3 \
        --report_freq ${REPORT_FREQ} \
        --workers ${WORKERS} \
        --poison_budget ${POISON_BUDGET} \
        --dropout_ratio ${DROPOUT_RATIO} \
        --beta ${BETA} \
        --gamma_up ${GAMMA_UP} \
        --gamma_low ${GAMMA_LOW} \
        --target_class ${TARGET_CLASS} \
        --lr ${LR} \
        --local_lr ${LOCAL_LR} \
        --eps ${EPS}
done

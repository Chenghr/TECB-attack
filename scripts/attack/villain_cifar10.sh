#!/bin/bash

# run_experiments.sh

# 基础路径
PROJECT_ROOT=$(dirname $(dirname $(readlink -f "$0")))
DATASET="CIFAR10"
DATA_PATH="${PROJECT_ROOT}/../data"
BASE_SAVE_PATH="${PROJECT_ROOT}/../results/models/Villain/cifar10"
SCRIPT_PATH="${PROJECT_ROOT}/../attack/villain/villain_cifar10_training.py"

# half参数列表
# HALF_VALUES=(14 16 18 20 22 24 26)

# for half in "${HALF_VALUES[@]}"; do
DROPOUT_RATIO_VALUE=(0.15 0.2 0.25)
POISON_BUDGET_VALUE=(0.05 0.1 0.15)
LOCAL_LR_VALUE=(0.2 0.5 0.8)
for dropout_ratio in "${DROPOUT_RATIO_VALUE[@]}"; do
    for poison_budget in "${POISON_BUDGET_VALUE[@]}"; do
        for local_lr in "${LOCAL_LR_VALUE[@]}"; do

            save_path="${BASE_SAVE_PATH}/half_${half}"

            python ${SCRIPT_PATH} \
                --dataset ${DATASET} \
                --data_dir ${DATA_PATH} \
                --half 16 \
                --report_freq 5 \
                --workers 8 \
                --epochs 80 \
                --poison_budget $poison_budget \
                --dropout_ratio $dropout_ratio \
                --beta 0.4 \
                --gamma_up 1.2 \
                --gamma_low 0.6 \
                --target_class cat \
                --local_lr $local_lr \
                --corruption_amp 5.0 \
                --pre_train_epoch 20 \
                --backdoor_start \
                --lr 0.1 \
                --alpha 0.05 \
                --eps 0.5 \
                --batch_size 1024 \
                --save $save_path
        done
    done
done

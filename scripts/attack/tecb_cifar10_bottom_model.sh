#!/bin/bash

# run_experiments.sh

set -e

# 基础路径
PROJECT_ROOT=$(dirname $(dirname $(readlink -f "$0")))
DATASET="CIFAR10"
DATA_PATH="${PROJECT_ROOT}/../data"
BASE_SAVE_PATH="${PROJECT_ROOT}/../results/models/TECB/cifar10"
SCRIPT_PATH="${PROJECT_ROOT}/../attack/tecb/vfl_cifar10.py"

# half参数列表
BOTTOM_MODEL_NAME_VALUES=("vgg16" "LeNet")

# 不进行攻击，寻找最优训练参数
for bottom_model_name in "${BOTTOM_MODEL_NAME_VALUES[@]}"; do
    for half in "${HALF_VALUES[@]}"; do
        save_path="${BASE_SAVE_PATH}/bottom_model_${bottom_model_name}/half_${half}"
        
        python ${SCRIPT_PATH} \
            --dataset ${DATASET} \
            --data_dir ${DATA_PATH} \
            --bottom_model_name $bottom_model_name \
            --half 16 \
            --epochs 100 \
            --batch_size 64 \
            --lr 0.01 \
            --alpha 0.05 \
            --eps 1.0 \
            --backdoor 50 \
            --poison_epochs 80 \
            --target_class airplane \
            --poison_num 4 \
            --corruption_amp 5.0 \
            --report_freq 5 \
            --workers 8 \
            --save $save_path
    done
done
# for bottom_model_name in "${BOTTOM_MODEL_NAME_VALUES[@]}"; do
#     for half in "${HALF_VALUES[@]}"; do
#         save_path="${BASE_SAVE_PATH}/bottom_model_${bottom_model_name}/half_${half}"
        
#         python ${SCRIPT_PATH} \
#             --dataset ${DATASET} \
#             --data_dir ${DATA_PATH} \
#             --bottom_model_name $bottom_model_name \
#             --half 16 \
#             --report_freq 5 \
#             --workers 8 \
#             --backdoor 50 \
#             --poison_epochs 80 \
#             --target_class airplane \
#             --poison_num 4 \
#             --corruption_amp 5.0 \
#             # --backdoor_start \
#             --lr 0.2 \
#             --alpha 0.05 \
#             --eps 1.0 \
#             --epochs 100 \
#             --batch_size 64 \
#             --save $save_path
#     done
# done
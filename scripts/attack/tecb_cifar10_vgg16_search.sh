#!/bin/bash

set -e

PROJECT_ROOT=$(dirname $(dirname $(readlink -f "$0")))
DATASET="CIFAR10"
DATA_PATH="${PROJECT_ROOT}/../data"
BASE_SAVE_PATH="${PROJECT_ROOT}/../results/models/TECB/cifar10"
SCRIPT_PATH="${PROJECT_ROOT}/../attack/tecb/vfl_cifar10.py"

# 定义参数搜索空间
BOTTOM_MODEL_NAME_VALUES=("vgg16" "LeNet")
LR_VALUES=(0.1 0.01 0.001)
BATCH_SIZE_VALUES=(32 64 128)
ALPHA_VALUES=(0.01 0.05 0.1)

for bottom_model_name in "${BOTTOM_MODEL_NAME_VALUES[@]}"; do
    for lr in "${LR_VALUES[@]}"; do
        for batch_size in "${BATCH_SIZE_VALUES[@]}"; do
            for alpha in "${ALPHA_VALUES[@]}"; do
                save_path="${BASE_SAVE_PATH}/param_search/${bottom_model_name}/lr_${lr}/batch_${batch_size}/alpha_${alpha}"
                
                echo "Running experiment with:"
                echo "Model: ${bottom_model_name}"
                echo "Learning rate: ${lr}"
                echo "Batch size: ${batch_size}"
                echo "Alpha: ${alpha}"
                
                python ${SCRIPT_PATH} \
                    --dataset ${DATASET} \
                    --data_dir ${DATA_PATH} \
                    --bottom_model_name $bottom_model_name \
                    --half 16 \
                    --epochs 100 \
                    --batch_size $batch_size \
                    --lr $lr \
                    --alpha $alpha \
                    --report_freq 5 \
                    --workers 8 \
                    --save $save_path
            done
        done
    done
done
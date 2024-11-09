#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

echo "Script started at $(date)"

# Project configuration
PROJECT_ROOT="$(pwd)"
DATA_DIR="${PROJECT_ROOT}/../../data/"
SCRIPT_PATH="${PROJECT_ROOT}/../../pretest/permuted_train_for_badvfl.py"

DATASET="CIFAR10"

# 28 的时候 windows size 大小不满足
declare -a HALF=(16)
declare -a SEED=(1)
declare -a UPDATE_MODES=('both')

# 定义要更新的层配置
declare -a UPDATE_TOP_LAYERS=(
    "all"                                           
)

# declare -a LRS=(0.1 0.01 0.001 1e-4 1e-5)
declare -a LRS=(1e-4 5e-5 2e-5 1e-5)
# Training parameters
ATTACK_METHOD='BadVFL'
EPOCHS=20
# LEARNING_RATE=2e-5
BATCH_SIZE=256

# 验证数组长度是否匹配
if [ ${#HALF[@]} -ne ${#SEED[@]} ]; then
    echo "Error: HALF and SEED arrays must have the same length"
    exit 1
fi

LOG_FILE="badvfl_cifar10_search_lr.log"

# 遍历所有配置并执行
for i in "${!HALF[@]}"; do
    current_half=${HALF[$i]}
    current_seed=${SEED[$i]}
    
    # 计算 attack feats
    attack_feats=$((32 - current_half))
    MODEL_DIR="${PROJECT_ROOT}/../../results/models/BadVFL/cifar10/half_${current_half}/${current_seed}_saved_models"
    
    for lr in "${LRS[@]}"; do
        for update_mode in "${UPDATE_MODES[@]}"; do
            if [ "$update_mode" = "both" ]; then
                # 如果是both模式，只运行一次，使用all作为layers
                echo "Processing: half: ${current_half}, seed: ${current_seed}, mode: ${update_mode}, layers: all"
                
                python "${SCRIPT_PATH}" \
                    --attack_method "${ATTACK_METHOD}" \
                    --dataset "${DATASET}" \
                    --data_dir "${DATA_DIR}" \
                    --model_dir "${MODEL_DIR}" \
                    --half "${current_half}" \
                    --epochs "${EPOCHS}" \
                    --lr "${lr}" \
                    --batch_size "${BATCH_SIZE}" \
                    --update_mode "${update_mode}" \
                    --update_top_layers "all" \
                    --log_file_name "${LOG_FILE}"
                    
                echo "Completed processing for half: ${current_half}, seed: ${current_seed}, mode: ${update_mode}, layers: all"
                
            else
                # 对于其他模式，遍历所有层配置
                for layers in "${UPDATE_TOP_LAYERS[@]}"; do
                    echo "Processing: half: ${current_half}, seed: ${current_seed}, mode: ${update_mode}, layers: ${layers}"

                    python "${SCRIPT_PATH}" \
                        --attack_method "${ATTACK_METHOD}" \
                        --dataset "${DATASET}" \
                        --data_dir "${DATA_DIR}" \
                        --model_dir "${MODEL_DIR}" \
                        --half "${current_half}" \
                        --epochs "${EPOCHS}" \
                        --lr "${lr}" \
                        --batch_size "${BATCH_SIZE}" \
                        --update_mode "${update_mode}" \
                        --update_top_layers ${layers} \
                        --log_file_name "${LOG_FILE}"
                
                    echo "Completed processing for half: ${current_half}, seed: ${current_seed}, mode: ${update_mode}, layers: ${layers}"
                done
            fi
        done
    done
done

echo "All configurations processed successfully"
echo "Script completed at $(date)"
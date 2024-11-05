#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

echo "Script started at $(date)"

# Project configuration
PROJECT_ROOT="$(pwd)"
DATA_DIR="${PROJECT_ROOT}/../../data/CIFAR10/"
SCRIPT_PATH="${PROJECT_ROOT}/../../attack/badvfl/badvfl_cifar10_training.py"
DATASET="CIFAR10"

# Base directories and logging
BASE_SAVE_DIR="${PROJECT_ROOT}/../../results/models/BadVFL/cifar10"
LOG_FILE="BadVFL_cifar10.log"

# Training parameters
POISON_BUDGET=0.1
CORRUPTION_AMP=5.0
BACKDOOR_START=true
OPTIMAL_SEL=true
SALIENCY_MAP_INJECTION=true
PRE_TRAIN_EPOCHS=20
TRIGGER_TRAIN_EPOCHS=40
ALPHA=0.05
EPOCHS=80

# Parameter combinations
declare -a WINDOW_SIZE=(5)
declare -a EPS=(0.3 0.5)
declare -a LEARNING_RATE=(0.05 0.1 0.2)
declare -a BATCH_SIZE=(64 128)

# Validate directories exist
mkdir -p "${BASE_SAVE_DIR}"
mkdir -p "$(dirname "${LOG_FILE}")"

# Function to run training with specific parameters
run_training() {
    local window_size=$1
    local eps=$2
    local lr=$3
    local batch_size=$4
    
    local save_dir="${BASE_SAVE_DIR}/ws=${window_size}_eps=${eps}_lr=${lr}_bs=${batch_size}"
    mkdir -p "${save_dir}"
    
    echo "Starting training with parameters:"
    echo "Window Size: ${window_size}, Epsilon: ${eps}, Learning Rate: ${lr}, Batch Size: ${batch_size}"
    
    python "${SCRIPT_PATH}" \
        --dataset "${DATASET}" \
        --data_dir "${DATA_DIR}" \
        --save "${save_dir}" \
        --log_file_name "${LOG_FILE}" \
        --poison_budget "${POISON_BUDGET}" \
        --corruption_amp "${CORRUPTION_AMP}" \
        --backdoor_start "${BACKDOOR_START}" \
        --optimal_sel "${OPTIMAL_SEL}" \
        --saliency_map_injection "${SALIENCY_MAP_INJECTION}" \
        --pre_train_epochs "${PRE_TRAIN_EPOCHS}" \
        --trigger_train_epochs "${TRIGGER_TRAIN_EPOCHS}" \
        --alpha "${ALPHA}" \
        --epochs "${EPOCHS}" \
        --window_size "${window_size}" \
        --eps "${eps}" \
        --lr "${lr}" \
        --batch_size "${batch_size}"
    
    echo "Completed training for configuration: ws=${window_size}, eps=${eps}, lr=${lr}, bs=${batch_size}"
}

# Main execution loop
for ws in "${WINDOW_SIZE[@]}"; do
    for eps in "${EPS[@]}"; do
        for lr in "${LEARNING_RATE[@]}"; do
            for bs in "${BATCH_SIZE[@]}"; do
                run_training "${ws}" "${eps}" "${lr}" "${bs}"
                echo "Waiting for process to complete..."
                wait
            done
        done
    done
done

echo "All configurations processed successfully"
echo "Script completed at $(date)"
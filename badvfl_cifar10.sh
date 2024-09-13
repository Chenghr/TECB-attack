#!/bin/bash

# 基础信息
dataset="CIFAR10"
data_dir="./data/CIFAR10/"
save="./results/models/BadVFL/cifar10/"
log_file_name="BadVFL_cifar10.log"

# 基础参数信息
poison_budget="0.1"
corruption_amp="5.0"
backdoor_start="True"
optimal_sel="True"
saliency_map_injection="True"
pre_train_epochs="20"
trigger_train_epochs="40"
alpha="0.05"
epochs="80"

# 超参组合
window_size_values=(5)
eps_values=(0.3 0.5)
lr_values=(0.05 0.1 0.2)
batch_size_values=(64 128)

for window_size in "${window_size_values[@]}"; do
    for eps in "${eps_values[@]}"; do
        for lr in "${lr_values[@]}"; do
            for batch_size in "${batch_size_values[@]}"; do
                # 以文件名区分 
                save="${save}/ws=${window_size}_eps=${eps}_lr=${lr}_bs=${batch_size}/"
                
                # 准备参数列表
                params=(
                    --save "$save"
                    --window_size "$window_size"
                    --eps "$eps"
                    --lr "$lr"
                    --batch_size "$batch_size"
                )

                # 运行 Python 脚本并传递参数
                python badvfl_cifar10_training.py "${params[@]}"

                # 等待 Python 脚本执行完毕
                wait
            done
        done
    done
done
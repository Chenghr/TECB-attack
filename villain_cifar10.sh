#!/bin/bash

# 基础信息
dataset="CIFAR10"
data_dir="./data/CIFAR10/"
save="./results/models/Villain/cifar10/"
log_file_name="Villain_cifar10.log"

# 基础参数信息
corruption_amp="5.0"
backdoor_start="True"
pre_train_epochs="20"
epochs="80"

# 超参组合
poison_budget_values=(0.05 0.1 0.15)
beta_values=(0.3 0.4 0.5)
lr_values=(0.05 0.1 0.2)
batch_size_values=(256 512 1024)

for poison_budget in "${poison_budget_values[@]}"; do
    for beta in "${beta_values[@]}"; do
        for lr in "${lr_values[@]}"; do
            for batch_size in "${batch_size_values[@]}"; do
                # 以文件名区分 
                save="${save}/poi=${poison_budget}_beta=${beta}_lr=${lr}_bs=${batch_size}/"
                
                # 准备参数列表
                params=(
                    --save "$save"
                    --poison_budget "$poison_budget"
                    --beta "$beta"
                    --lr "$lr"
                    --batch_size "$batch_size"
                )

                # 运行 Python 脚本并传递参数
                python villain_cifar10_training.py "${params[@]}"

                # 等待 Python 脚本执行完毕
                wait
            done
        done
    done
done

import os
import sys
import argparse

# 获取当前Python文件的绝对路径
current_file = os.path.abspath(__file__)
# 获取当前文件所在目录
current_dir = os.path.dirname(current_file)
# 获取上级目录
parent_dir = os.path.dirname(current_dir)

# 将上级目录添加到系统路径
sys.path.insert(0, parent_dir)

from pretest.utils import load_dataset, load_model_and_backdoor_data
import numpy as np
import torch
from torch import nn
# import wandb
import torch.nn.functional as F
import copy 
from fedml_core.trainer.permuted_trainer import PermutedTrainer

from fedml_core.utils.utils import (
    keep_predict_loss
)
from fedml_core.utils.logger import setup_logger
from tqdm import tqdm
import time


def permuted_validate(args, device, logger):
    # 记录关键参数
    logger.info(
        "Key Args - "
        f"Dataset: {args.dataset}, "
        f"Attack Method: {args.attack_method}, "
        f"Epochs: {args.epochs}, "
        f"LR: {args.lr}, "
        f"Batch Size: {args.batch_size}, "
        f"Attack Features: {32-args.half}, "
        f"Half: {args.half}, "
        f"Update Mode: {args.update_mode}, "
        f"Update Top Layers: {args.update_top_layers}"
    )
    
    train_dataloader, test_dataloader = load_dataset(args.dataset, args.data_dir, args.batch_size)
    model_list, backdoor_data = load_model_and_backdoor_data(args.dataset, args.model_dir)
    
    trainer = PermutedTrainer(model_list, args=args, attack_method=args.attack_method)
    
    criterion = nn.CrossEntropyLoss().to(device)
    bottom_criterion = keep_predict_loss
    optimizer_list = [
        torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        for model in trainer.perturbed_model
    ]
    
    if args.update_mode != "bottom_only":
        optimizer_list=trainer.update_optimizer_for_layers(trainer.perturbed_model, optimizer_list, args)
    
    baseline_clean_top1, baseline_clean_top5, baseline_asr_top1, baseline_asr_top5 = trainer.test_baseline_model(
        backdoor_data, test_dataloader, criterion, args,
    )
    
    main_task_acc, asr_acc = [baseline_clean_top1], [baseline_asr_top1]
    main_task_acc_top5, asr_acc_top5 = [baseline_clean_top5], [baseline_asr_top5]
    
    # 创建epoch进度条
    for epoch in tqdm(range(args.epochs), desc="Training Progress"):
        _ = trainer.train_perturbed(
            train_dataloader, criterion, bottom_criterion, optimizer_list, device, args
        )
        modified_clean_top1, modified_clean_top5, modified_asr_top1, modified_asr_top5 = trainer.test_modified_model(
            backdoor_data, test_dataloader, criterion, args,
        )
        main_task_acc.append(modified_clean_top1)
        main_task_acc_top5.append(modified_clean_top5)
        asr_acc.append(modified_asr_top1)
        asr_acc_top5.append(modified_asr_top5)
        
        # 使用tqdm.write避免进度条显示混乱
        tqdm.write(f"Epoch {epoch+1}: Acc={modified_clean_top1:.2f}%, ASR={modified_asr_top1:.2f}%")
    
    # 将所有数值保留两位小数
    main_task_acc = [f"{x:.2f}" for x in main_task_acc]
    main_task_acc_top5 = [f"{x:.2f}" for x in main_task_acc_top5]
    asr_acc = [f"{x:.2f}" for x in asr_acc]
    asr_acc_top5 = [f"{x:.2f}" for x in asr_acc_top5]
    
    # 记录完整的准确率列表
    FORMAT = "{:<25}: {}"
    logger.info("=" * 60)
    logger.info("Final Results")
    logger.info("=" * 60)
    logger.info(FORMAT.format("Main Task Accuracy", ", ".join(main_task_acc)))
    logger.info(FORMAT.format("Main Task Top5 Accuracy", ", ".join(main_task_acc_top5)))
    logger.info(FORMAT.format("ASR", ", ".join(asr_acc)))
    logger.info("=" * 60+"\n")
     

if __name__ == "__main__":
    # 获取当前文件的绝对路径和目录
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(os.path.dirname(current_file))  # 上上级目录
    
    # 设置默认路径
    default_data_path = os.path.join(project_root, "data")
    default_save_path = os.path.join(project_root, "results", "pretest", "permuted_training")
    default_model_dir = os.path.join(project_root, "results", "models", "TECB", "cifar10")
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--save', type=str, default=default_save_path)
    parser.add_argument('--log_file_name', type=str, default="test.log", help='log name')
    
    # 数据、模型、攻击方法
    parser.add_argument("--dataset", type=str, default="CIFAR10",  
                        choices=["CIFAR10", "CIFAR100","CINIC10L",], help="name of dataset")
    parser.add_argument("--data_dir", default=default_data_path, help="location of the data corpus")
    parser.add_argument('--half', type=int, default=16, help='feature num of the active party') 
    parser.add_argument('--model_dir', type=str, default="../results/models/TECB/cifar10/")
    parser.add_argument('--attack_method', type=str, default="TECB",
                        choices=["TECB", "BadVFL"])
    
    # permuted params
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--momentum", type=float, default=0.7, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--update_mode", type=str, default='both',
                        choices=['bottom_only', 'top_only', 'both'], help='Model update mode: bottom_only, top_only, or both')
    parser.add_argument("--update_top_layers", type=str, nargs='+', default=['all'])
    
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    # 创建一个logger
    logger = setup_logger(args)
    
    permuted_validate(args=args, device=device, logger=logger)


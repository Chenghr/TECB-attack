import copy
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
import argparse

import torch
import torch.nn as nn
from attack.villain.utils import (
    set_seed, 
    init_model_releated, 
    init_dataloader, 
    save_checkpoint,
)
from pretest.utils import load_dataset, load_model_and_backdoor_data
from fedml_core.trainer.permuted_trainer import PermutedTrainer
from fedml_core.utils.logger import setup_logger
from fedml_core.utils.utils import (keep_predict_loss,
                                    over_write_args_from_file)


def permuted_validate(args, logger):
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
    
    train_dataloader, test_dataloader = init_dataloader(
        args.dataset, args.data_dir, args.batch_size
    )
    model_list, backdoor_data = load_model_and_backdoor_data(
        args.dataset, args.model_dir
    )
    
    trainer = PermutedTrainer(model_list, args=args, attack_method=args.attack_method)
    
    criterion = nn.CrossEntropyLoss().to(args.device)
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
    
    baseline_clean_top1, _, baseline_asr_top1, _ = trainer.test_baseline_model(
        backdoor_data, test_dataloader, criterion, args,
    )
    
    main_task_acc, asr_acc = [baseline_clean_top1], [baseline_asr_top1]
    
    # 创建epoch进度条
    for epoch in tqdm(range(args.epochs), desc="Training Progress"):
        loss = trainer.train_perturbed(
            train_dataloader, criterion, bottom_criterion, optimizer_list, args.device, args
        )
        # print(f"epoch: {epoch}, loss: {loss}")
        modified_clean_top1, _, modified_asr_top1, _ = trainer.test_modified_model(
            backdoor_data, test_dataloader, criterion, args,
        )
        main_task_acc.append(modified_clean_top1)
        asr_acc.append(modified_asr_top1)
        
        # 使用tqdm.write避免进度条显示混乱
        tqdm.write(f"Epoch {epoch+1}: Acc={modified_clean_top1:.2f}%, ASR={modified_asr_top1:.2f}%")
    
    # 将所有数值保留两位小数
    main_task_acc = [f"{x:.2f}" for x in main_task_acc]
    asr_acc = [f"{x:.2f}" for x in asr_acc]
    
    # 记录完整的准确率列表
    FORMAT = "{:<25}: {}"
    logger.info("=" * 60)
    logger.info("Final Results")
    logger.info("=" * 60)
    logger.info(FORMAT.format("Main Task Accuracy", ", ".join(main_task_acc)))
    logger.info(FORMAT.format("ASR", ", ".join(asr_acc)))
    logger.info("=" * 60+"\n")
    

if __name__ == "__main__":
    
    default_data_path = os.path.abspath("../../data/")
    default_yaml_path = os.path.abspath("../villain/best_configs/cifar10_bestattack.yml")
    default_save_path = os.path.abspath("../../results/pretest/permuted_training/Villain")
    default_model_path = os.path.abspath("../../results/models/Villain/cifar10")

    parser = argparse.ArgumentParser("villain_cifar10")

    # 数据相关参数
    data_group = parser.add_argument_group('Data')
    data_group.add_argument("--data_dir", default=default_data_path, help="location of the data corpus")
    data_group.add_argument("--dataset", default="CIFAR10", type=str, choices=["CIFAR10", "CIFAR100", "CINIC10L"], help="name of dataset")

    # 实验相关参数
    experiment_group = parser.add_argument_group('Experiment')
    experiment_group.add_argument("--name", type=str, default="villain_cifar10", help="experiment name")
    experiment_group.add_argument("--save", default=default_save_path, type=str, metavar="PATH", help="path to save checkpoint")
    experiment_group.add_argument("--log_file_name", type=str, default="experiment.log", help="log name")
    experiment_group.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint")
    experiment_group.add_argument("--seed_num", type=int, default=3, help="repeat num.")
    experiment_group.add_argument("--yaml_path", type=str, default=default_yaml_path, help="attack yaml file")
    experiment_group.add_argument("--load_yaml", action="store_true", default=False, help="backdoor")
    experiment_group.add_argument("--device", type=str, default="cuda:0")
    experiment_group.add_argument("--attack_method", type=str, default="TECB")
    experiment_group.add_argument("--model_dir", type=str, default=default_model_path)
    
    # 模型相关参数
    model_group = parser.add_argument_group('Model')
    model_group.add_argument("--layers", type=int, default=18, help="total number of layers")
    model_group.add_argument("--u_dim", type=int, default=64, help="u layer dimensions")
    model_group.add_argument("--k", type=int, default=2, help="num of clients")
    model_group.add_argument("--parallel", action="store_true", default=False, help="data parallelism")
    model_group.add_argument("--half", type=int, default=16, help="half number of features")
    
    # 训练相关参数
    training_group = parser.add_argument_group('Training')
    # training_group.add_argument("--epochs", type=int, default=60, help="num of training epochs")
    training_group.add_argument("--start_epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
    training_group.add_argument("--backdoor_start_epoch", default=20, type=int, metavar="N", help="manual epoch number (useful on restarts)")
    # training_group.add_argument("--batch_size", type=int, default=64, help="batch size")
    # training_group.add_argument("--lr", type=float, default=0.02, help="init learning rate")
    # training_group.add_argument("--momentum", type=float, default=0.9, help="momentum")
    # training_group.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    training_group.add_argument("--decay_period", type=int, default=1, help="epochs between two learning rate decays")
    training_group.add_argument("--stone1", default=30, type=int, metavar="s1", help="stone1 for step scheduler")
    training_group.add_argument("--stone2", default=85, type=int, metavar="s2", help="stone2 for step scheduler")
    training_group.add_argument("--grad_clip", type=float, default=5.0, help="gradient clipping")
    training_group.add_argument("--gamma", type=float, default=0.97, help="learning rate decay")
    training_group.add_argument("--step_gamma", default=0.1, type=float, metavar="S", help="gamma for step scheduler")
    training_group.add_argument("--workers", type=int, default=8, help="num of workers")
    training_group.add_argument("--report_freq", type=float, default=10, help="report frequency")
    training_group.add_argument("--beta", type=float, default=0.4, help="controling trigger magnitude")
    
    # 后门相关参数
    backdoor_group = parser.add_argument_group('Backdoor')
    training_group.add_argument("--trigger_lr", type=float, default=0.001, help="init learning rate for trigger")
    backdoor_group.add_argument("--alpha", type=float, default=0.02, help="uap learning rate decay")
    backdoor_group.add_argument("--eps", type=float, default=16 / 255, help="uap clamp bound")
    backdoor_group.add_argument("--corruption_amp", type=float, default=5.0, help="amplification of corruption")
    backdoor_group.add_argument("--backdoor_start", action="store_true", default=False, help="backdoor")
    backdoor_group.add_argument("--poison_budget", type=float, default=0.1, help="poison sample fraction")
    backdoor_group.add_argument("--optimal_sel", action="store_true", default=True, help="optimal select tartget class")
    backdoor_group.add_argument("--saliency_map_injection", action="store_true", default=True, help="optimal select trigger loaction")
    backdoor_group.add_argument("--pre_train_epochs", default=20, type=int, metavar="N", help="")
    backdoor_group.add_argument("--trigger_train_epochs", default=40, type=int, metavar="N", help="")
    backdoor_group.add_argument("--window_size", default=5, type=int, metavar="N", help="")
    
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
    if args.load_yaml:
        over_write_args_from_file(args, args.yaml_path)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # 创建一个logger
    logger = setup_logger(args)
    
    permuted_validate(args=args, logger=logger)

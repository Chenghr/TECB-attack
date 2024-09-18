import copy
import os
import random
import sys
import torch
import torch.nn as nn
import argparse
import shutil
import torchvision.transforms as transforms
from torch.utils.data import Subset
import copy
import random
import numpy as np

from fedml_core.trainer.shuffle_trainer import ShuffleDefense
from fedml_core.utils.logger import setup_logger

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))


def main(logger, args):
    defense = ShuffleDefense(logger=logger, args=args)
    
    high_confidence_subset = defense.select_high_confidence_subset(args.confidence_threshold)
    
    print(high_confidence_subset)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser("defense")

    # 实验相关参数
    experiment_group = parser.add_argument_group('Experiment')
    experiment_group.add_argument('--name', type=str, default='defense_test', help='experiment name')
    experiment_group.add_argument('--save', default='./results/logs/defense/cifar10-test', 
                        type=str, metavar='PATH', help='path to save checkpoint (default: none)')
    experiment_group.add_argument('--log_file_name', type=str, default="experiment.log", help='log name')
    experiment_group.add_argument('--c', type=str, default='', help='config file')
    
    # 数据相关参数
    data_group = parser.add_argument_group('Data')
    data_group.add_argument("--dataset", default="CIFAR10", type=str, choices=[
        "CIFAR10", "CIFAR100", "TinyImageNet", "CINIC10L", "Yahoo", "Criteo", "BCW"
    ], help="name of dataset")
    data_group.add_argument("--data_dir", default="./data/CIFAR10/", help="location of the data corpus")
    data_group.add_argument('--half', type=int, default=16,
    )  # choices=[16, 14, 32, 1->party_num]. CIFAR10-16, Liver-14, TinyImageNet-32
    data_group.add_argument("--resume", default="", 
                                type=str, metavar="PATH", help="path to latest checkpoint")
    
    # 训练相关参数
    training_group = parser.add_argument_group('Training')
    training_group.add_argument("--batch_size", type=int, default=256, help="batch size")
    
    # 防御相关参数
    defense_group = parser.add_argument_group('Defense')
    defense_group.add_argument("--confidence_threshold", type=float, default=0.8, help="")
    training_group.add_argument("--workers", type=int, default=0, help="num of workers")
    
    args = parser.parse_args()
    
    args.device = device
    
    data_dir = os.path.abspath('./data')
    save = os.path.abspath('./results/logs/defense/cifar10-test')
    resume = os.path.abspath('./results/models/CIFAR10/base/0_saved_models')
    
    args.data_dir = data_dir
    args.save = save
    args.resume = resume

    print(data_dir, save, resume)
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # 创建一个logger
    logger = setup_logger(args)
    
    main(logger=logger, args=args)

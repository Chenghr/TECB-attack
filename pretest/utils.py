import logging
import os
import random
import sys

import numpy as np
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
import argparse
import glob
import logging
import shutil
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fedml_core.data_preprocessing.cifar10 import IndexedCIFAR10
from fedml_core.data_preprocessing.cifar100.dataset import IndexedCIFAR100
from fedml_core.data_preprocessing.CINIC10.dataset import CINIC10L
from fedml_core.model.baseline.vfl_models import (BottomModelForCifar10,
                                                  BottomModelForCifar100,
                                                  BottomModelForCinic10,
                                                  TopModelForCifar10,
                                                  TopModelForCifar100,
                                                  TopModelForCinic10)
from fedml_core.trainer.tecb_trainer import TECBTrainer
from fedml_core.utils.utils import (AverageMeter, image_format_2_rgb,
                                    keep_predict_loss,
                                    over_write_args_from_file)
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, CIFAR100


def split_data(dataset, data, half=16):
    if dataset in ["CIFAR10", "CIFAR100", "CINIC10L"]:
        x_a = data[:, :, :, 0 : half]
        x_b = data[:, :, :, half : 32]
    else:
        raise Exception("Unknown dataset name!")
    return x_a, x_b


def load_tecb_cifar10(batch_size=32, workers=1):
    data_dir = "../data/"
    save_model_dir = "../results/models/TECB/cifar10/"
    checkpoint_path = save_model_dir + "/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path)
    
    # 加载后门攻击信息
    backdoor_data = torch.load(save_model_dir + "/backdoor.pth")
    delta = backdoor_data.get("delta", None)
    target_label = backdoor_data.get("target_label", None)
    
    # load model
    model_list = []
    model_list.append(BottomModelForCifar10())
    model_list.append(BottomModelForCifar10())
    model_list.append(TopModelForCifar10())
    
    for i in range(len(model_list)):
        model_list[i].load_state_dict(checkpoint["state_dict"][i])

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = IndexedCIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    testset = IndexedCIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=batch_size, num_workers=workers
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=batch_size, num_workers=workers
    )
    
    return model_list, train_dataloader, test_dataloader, delta, target_label


def load_tecb_cifar100(batch_size=32, workers=1):
    data_dir = "../data/"
    save_model_dir = "../results/models/TECB/cifar100/"
    checkpoint_path = save_model_dir + "/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path)
    
    # 加载后门攻击信息
    backdoor_data = torch.load(save_model_dir + "/backdoor.pth")
    delta = backdoor_data.get("delta", None)
    target_label = backdoor_data.get("target_label", None)
    
    # load model
    model_list = []
    model_list.append(BottomModelForCifar100())
    model_list.append(BottomModelForCifar100())
    model_list.append(TopModelForCifar100())
    
    for i in range(len(model_list)):
        model_list[i].load_state_dict(checkpoint["state_dict"][i])

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = IndexedCIFAR100(
        root=data_dir, train=True, download=True, transform=transform
    )
    testset = IndexedCIFAR100(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=batch_size, num_workers=workers
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=batch_size, num_workers=workers
    )
    
    return model_list, train_dataloader, test_dataloader, delta, target_label


def load_tecb_cinic10(batch_size=32, workers=1):
    data_dir = "../data/cinic/"
    save_model_dir = "../results/models/TECB/cinic10/"
    checkpoint_path = save_model_dir + "/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path)
    
    # 加载后门攻击信息
    backdoor_data = torch.load(save_model_dir + "/backdoor.pth")
    delta = backdoor_data.get("delta", None)
    target_label = backdoor_data.get("target_label", None)
    
    # load model
    model_list = []
    model_list.append(BottomModelForCinic10())
    model_list.append(BottomModelForCinic10())
    model_list.append(TopModelForCinic10())
    
    for i in range(len(model_list)):
        model_list[i].load_state_dict(checkpoint["state_dict"][i])

    transform = transforms.Compose(
        [
            transforms.Lambda(image_format_2_rgb),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.47889522, 0.47227842, 0.43047404),
                (0.24205776, 0.23828046, 0.25874835),
            ),
        ]
    )
    trainset = CINIC10L(root=data_dir, split="/train", transform=transform)
    testset = CINIC10L(root=data_dir, split="/test", transform=transform)
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=batch_size, num_workers=workers
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=batch_size, num_workers=workers
    )
    
    return model_list, train_dataloader, test_dataloader, delta, target_label


if __name__ == "__main__":
    model_list, train_dataloader, test_dataloader, delta, target_label = load_tecb_cinic10()
    print(target_label)
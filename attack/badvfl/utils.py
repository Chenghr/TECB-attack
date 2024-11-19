import os
import sys
import random
import shutil

import numpy as np
from sklearn.utils import shuffle
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
import argparse
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import copy

import logging
import os
import random
import sys

import numpy as np
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fedml_core.data_preprocessing.cifar10 import IndexedCIFAR10
from fedml_core.data_preprocessing.cifar100.dataset import IndexedCIFAR100
from fedml_core.data_preprocessing.CINIC10.dataset import CINIC10L
from fedml_core.model.vfl_models import (BottomModelForCifar10,
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


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_recommended_num_workers():
    """获取推荐的num_workers值"""
    import multiprocessing as mp
    
    # 获取CPU核心数
    cpu_count = mp.cpu_count()
    
    # 一般建议设置为CPU核心数的2-4倍
    recommended = min(cpu_count * 4, 8)  # 设置上限为16
    
    return recommended
    
def init_dataloader(dataset, data_dir="./data", batch_size=128):
    """加载数据集
    
    Parameters:
    -----------
    dataset_name : str
        数据集名称 ('CIFAR10', 'CIFAR100', 或 'CINIC10L')
    data_dir : str, optional
        数据存储路径，默认为'./data'
    batch_size : int, optional
        批次大小，默认为128
    num_workers : int, optional
        数据加载的工作进程数，默认为4
        
    Returns:
    --------
    tuple
        (train_dataloader, test_dataloader)
        
    Raises:
    -------
    ValueError
        当数据集不支持时抛出
    """
    transform = _init_transform(dataset)
    trainset, testset = _load_dataset(dataset, transform, data_dir)

    # 创建数据加载器
    num_workers = get_recommended_num_workers()
    train_dataloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    full_train_dataloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=len(trainset),
        shuffle=False,
        num_workers=num_workers
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_dataloader, full_train_dataloader, test_dataloader

def _init_transform(dataset):
    # 定义数据转换
    if dataset == "CINIC10L":
        transform = transforms.Compose([
            transforms.Lambda(image_format_2_rgb),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=(0.47889522, 0.47227842, 0.43047404),
            #     std=(0.24205776, 0.23828046, 0.25874835)
            # )
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    return transform

def _load_dataset(dataset, transform, data_dir):
    # 加载数据集
    if dataset == "CIFAR10":
        trainset = IndexedCIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        testset = IndexedCIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
    elif dataset == "CIFAR100":
        trainset = IndexedCIFAR100(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        testset = IndexedCIFAR100(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
    elif dataset == "CINIC10L":
        trainset = CINIC10L(
            root=data_dir,
            split="train",
            transform=transform
        )
        testset = CINIC10L(
            root=data_dir,
            split="test",
            transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    return trainset, testset
    

def init_model_releated(dataset, lr, momentum, weight_decay, stone1, stone2, step_gamma):
    model_list = []
    if dataset == "CIFAR10":
        model_list.append(BottomModelForCifar10())
        model_list.append(BottomModelForCifar10())
        model_list.append(TopModelForCifar10())
        model_list.append(BottomModelForCifar10())  # set for pretrain
    elif dataset == "CIFAR100":
        model_list.append(BottomModelForCifar100())
        model_list.append(BottomModelForCifar100())
        model_list.append(TopModelForCifar100())
        model_list.append(BottomModelForCifar100())  # set for pretrain
    elif dataset == "CINIC10L":
        model_list.append(BottomModelForCinic10())
        model_list.append(BottomModelForCinic10())
        model_list.append(TopModelForCinic10())
        model_list.append(BottomModelForCinic10())  # set for pretrain
    else:
        raise ValueError
    
    # optimizer and stepLR
    optimizer_list = [
        torch.optim.SGD(
            model.parameters(),
            lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        for model in model_list
    ]
    
    stone1, stone2 = stone1, stone2 
    lr_scheduler_list = [
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[stone1, stone2], 
            gamma=step_gamma
        )
        for optimizer in optimizer_list
    ]
    
    return model_list, optimizer_list, lr_scheduler_list


def load_model_and_backdoor_data(dataset, save_dir):
    model_dir = os.path.join(save_dir, "model_best.pth.tar")
    backdoor_data_dir = os.path.join(save_dir, "backdoor.pth")
    # print(model_dir, backdoor_data_dir)
    
    model_list = []
    if dataset == "CIFAR10":
        model_list.append(BottomModelForCifar10())
        model_list.append(BottomModelForCifar10())
        model_list.append(TopModelForCifar10())
        model_list.append(BottomModelForCifar10())  # set for pretrain
    elif dataset == "CIFAR100":
        model_list.append(BottomModelForCifar100())
        model_list.append(BottomModelForCifar100())
        model_list.append(TopModelForCifar100())
        model_list.append(BottomModelForCifar100())  # set for pretrain
    elif dataset == "CINIC10L":
        model_list.append(BottomModelForCinic10())
        model_list.append(BottomModelForCinic10())
        model_list.append(TopModelForCinic10())
        model_list.append(BottomModelForCinic10())  # set for pretrain
    else:
        raise ValueError
    
    saved_model_list = torch.load(model_dir)
    for i in range(len(model_list)):
        model_list[i].load_state_dict(saved_model_list["state_dict"][i])

    backdoor_data = torch.load(backdoor_data_dir)
    
    return model_list, backdoor_data


def sample_poisoned_source_target_data(dataset, source_label, target_label, poison_budget):
    """
    从给定的数据集中选择源类别和目标类别的样本,并从目标类别中随机选择一定比例的样本进行污染。

    Args:
        dataset (torch.utils.data.Dataset): 输入数据集
        source_label (int): 源类别的标签
        target_label (int): 目标类别的标签
        poison_budget (float): 被污染目标类别样本的比例,范围为 [0, 1]

    Returns:
        tuple: 包含源类别样本索引、被污染的目标类别样本索引和一个只包含选中样本的数据加载器
    """
    targets = torch.tensor(dataset.targets)
    
    # 获取源类别和目标类别的样本索引
    source_indices = torch.nonzero(targets == source_label).squeeze()
    target_indices = torch.nonzero(targets == target_label).squeeze()
    
    # 计算目标类别样本总数和需要污染的样本数
    target_num = target_indices.numel()
    poison_num = int(poison_budget * target_num)
    
    # 随机选择源类别和被污染的目标类别样本索引
    selected_source_indices = source_indices[torch.randperm(len(source_indices))[:poison_num]]
    selected_source_indices, _ = torch.sort(selected_source_indices)
    selected_target_indices = target_indices[torch.randperm(len(target_indices))[:poison_num]]
    selected_target_indices, _ = torch.sort(selected_target_indices)
    
    # # 构建包含选中样本的数据子集
    # selected_indices = torch.cat((selected_source_indices, selected_target_indices))
    # selected_dataset = Subset(dataset, selected_indices)
    
    # # 创建数据加载器,每个批次包含所有选中的样本
    # batch_size = len(selected_indices)
    # selected_dataloader = torch.utils.data.DataLoader(selected_dataset, batch_size=batch_size, shuffle=False)
        
    return selected_source_indices, selected_target_indices

def construct_poison_train_dataloader(
    dataloader: DataLoader,
    dataset_name: str,
    selected_source_indices: List[int],
    selected_target_indices: List[int],
    delta: torch.Tensor,
    best_position: Tuple[int, int],
    args: object
) -> DataLoader:
    """
    创建带有投毒数据的训练数据加载器。

    Args:
        dataloader: 原始数据加载器
        dataset_name: 数据集名称
        selected_source_indices: 源数据索引
        selected_target_indices: 目标数据索引
        delta: 扰动值
        best_position: 最佳位置坐标 (y, x)
        args: 配置参数

    Returns:
        带有投毒数据的DataLoader
    """
    if not isinstance(dataloader, torch.utils.data.DataLoader):
        raise TypeError("dataloader must be an instance of torch.utils.data.DataLoader")
    
    if dataset_name not in ["CIFAR10", "CIFAR100", "CINIC10L"]:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    trainset = dataloader.dataset
    
    with torch.no_grad():
        if dataset_name in ["CIFAR10", "CIFAR100", "CINIC10L"]:
            window_size = args.window_size
            by, bx = best_position
            
            target_data = torch.from_numpy(trainset.data[selected_target_indices])
            
            delta_upd = (delta.permute(0, 2, 3, 1) * 255.0).to(torch.uint8)
            delta_exten = torch.zeros_like(target_data).to(args.device)
            delta_exten[:, by:by + window_size, bx + args.half:bx + window_size + args.half, :] = \
                delta_upd.expand(len(selected_target_indices), -1, -1, -1).detach().clone()
            
            poison_trainset = copy.deepcopy(trainset)
            poison_trainset.data[selected_target_indices] = np.clip(
                trainset.data[selected_source_indices] + delta.cpu().numpy(),
                0,
                255
            )
        else:
            raise ValueError
        
    return torch.utils.data.DataLoader(
        dataset=poison_trainset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=True
    )
        
def get_source_label_dataloader(
    dataloader: DataLoader,
    dataset_name: str,
    source_label: List[int],
    args: object
) -> DataLoader:
    """
    创建仅包含source_label的测试数据加载器。

    Args:
        dataloader: 原始测试数据加载器
        dataset_name: 数据集名称
        source_label: 源类别标签
        args: 配置参数

    Returns:
        仅包含source_label的测试DataLoader
    """
    if not isinstance(dataloader, torch.utils.data.DataLoader):
        raise TypeError("dataloader must be an instance of torch.utils.data.DataLoader")
    
    if dataset_name not in ["CIFAR10", "CIFAR100", "CINIC10L"]:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    testset = dataloader.dataset
    
    # 找到源类别的索引
    source_indices = []
    for idx, label in enumerate(testset.targets):
        if label in source_label:
            source_indices.append(idx)
    
    # 创建仅包含源类别数据的新数据集
    source_testset = copy.deepcopy(testset)
    
    # 提取源类别的数据和标签
    source_testset.data = testset.data[source_indices]
    source_testset.targets = [testset.targets[i] for i in source_indices]
            
    return torch.utils.data.DataLoader(
        dataset=source_testset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False
    )

def save_checkpoint(state, is_best, save, checkpoint):
    filename = os.path.join(save, checkpoint)
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, "model_best.pth.tar")
        shutil.copyfile(filename, best_filename)

def get_delta_exten(dataset_name, trainset, delta, best_position, selected_target_indices, window_size, half, device):
    delta_upd = delta.permute(0, 2, 3, 1)
    
    if dataset_name in ["CIFAR10", "CIFAR100", "CINIC10L"]:
        delta_upd = delta_upd * 255.0
    else:
        raise ValueError
    
    delta_upd = delta_upd.to(torch.uint8)
    by = best_position[0]
    bx = best_position[1]
    
    # 获取单个图片的形状
    if hasattr(trainset, 'data'):
        img_shape = trainset.data[selected_target_indices[0]].shape  # (H, W, C)
        delta_exten = torch.zeros(img_shape, dtype=torch.uint8).to(device)
        
        # 只取 delta_upd 的第一个元素 [0]，因为我们只需要一个模板
        delta_exten[by:by + window_size, bx + half:bx + window_size + half, :] = \
            delta_upd[0].detach().clone()
    else:
        # 对于 ImageFolder 类型的数据集
        img_path = trainset.image_paths[selected_target_indices[0]][0]
        img_shape = np.array(Image.open(img_path)).shape  # (H, W, C)
        delta_exten = torch.zeros(img_shape, dtype=torch.uint8).to(device)
        
        delta_exten[by:by + window_size, bx + half:bx + window_size + half, :] = \
            delta_upd[0].detach().clone()

    return delta_exten

def get_poison_train_dataloader(
    dataset_name, data_dir, batch_size, selected_source_indices, selected_target_indices, delta_exten,
):
    transform = _init_transform(dataset_name)
    trainset, _ = _load_dataset(dataset_name, transform, data_dir)

    delta_numpy = delta_exten.cpu().numpy()
    delta_numpy = np.tile(delta_numpy, (len(selected_source_indices), 1, 1, 1))
        
    if hasattr(trainset, 'data'):
        source_images = trainset.data[selected_source_indices]
        
        assert source_images.shape == delta_numpy.shape, \
            f"Shape mismatch: source {source_images.shape} vs delta {delta_numpy.shape}"
        
        modified_images = source_images + delta_numpy
        if dataset_name in ["CIFAR10", "CIFAR100", "CINIC10L"]:
            clipped_images = np.clip(modified_images, 0, 255).astype(np.uint8)
        trainset.modify_images(selected_target_indices, clipped_images)
    else:
        # 1. 获取源图像数据
        source_images = []
        for idx in selected_source_indices:
            img_path = trainset.image_paths[idx][0]
            img = np.array(Image.open(img_path).resize((32,32)))
            source_images.append(img)
        source_images = np.stack(source_images)
        modified_images = source_images + delta_numpy
        clipped_images = np.clip(modified_images, 0, 255).astype(np.uint8)
        
        # 5. 更新目标索引的图像
        trainset.modify_images(selected_target_indices, clipped_images)
        
    num_workers = get_recommended_num_workers()
    poison_train_dataloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle = True
    )
    
    return poison_train_dataloader

def get_delta_exten_pre(dataset_name, trainset, delta, best_position, selected_target_indices, window_size, half, device):
    delta_upd = delta.permute(0, 2, 3, 1)
    
    if dataset_name in ["CIFAR10", "CIFAR100", "CINIC10L"]:
        delta_upd = delta_upd * 255.0
    else:
        raise ValueError
    
    delta_upd = delta_upd.to(torch.uint8)
    by = best_position[0]
    bx = best_position[1]
    delta_exten = torch.zeros_like(torch.from_numpy(trainset.data[selected_target_indices])).to(device)
    delta_exten[:, by : by + window_size, bx + half : bx + window_size + half, :] = delta_upd.expand(len(selected_target_indices), -1, -1, -1).detach().clone()

    return delta_exten

def get_poison_train_dataloader_pre(
    dataset_name, data_dir, batch_size, selected_source_indices, selected_target_indices, delta_exten,
):
    transform = _init_transform(dataset_name)
    trainset, testset = _load_dataset(dataset_name, transform, data_dir)

    # 创建数据加载器
    num_workers = get_recommended_num_workers()
    if dataset_name in ["CIFAR10", "CIFAR100", "CINIC10L"]:
        trainset.data[selected_target_indices] = np.clip(trainset.data[selected_source_indices] + delta_exten.cpu().numpy(), 0, 255)
    else:
        raise ValueError
    poison_train_dataloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return poison_train_dataloader

def get_poison_test_dataloader(
    dataset_name, data_dir, batch_size, source_label
):
    transform = _init_transform(dataset_name)
    _, testset = _load_dataset(dataset_name, transform, data_dir)
    
    num_workers = get_recommended_num_workers()
    source_label = source_label.item() if torch.is_tensor(source_label) else source_label
    
    test_source_indices = np.where(np.array(testset.targets) == source_label)[0]
        
    source_poison_testset = Subset(testset, test_source_indices)
    
    poison_source_test_dataloader = torch.utils.data.DataLoader(
        dataset=source_poison_testset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return poison_source_test_dataloader

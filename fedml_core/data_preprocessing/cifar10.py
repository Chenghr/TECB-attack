import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np

class IndexedCIFAR10(CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class SelectedIndexedCIFAR10(Dataset):
    def __init__(self, dataset, selected_indices, label_mapping=None):
        """
        :param dataset: IndexedCIFAR10 的实例
        :param selected_indices: 要选择的样本索引列表
        :param label_mapping: 标签映射关系，用于标签扰动训练
        """
        self.dataset = dataset
        self.selected_indices = selected_indices
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        # 通过 selected_indices 获取原始数据集中的索引
        original_idx = self.selected_indices[idx]
        img, target, index = self.dataset[original_idx]
        if self.label_mapping is not None:  # 如果存在标签映射关系，则使用标签映射关系
            target = self.label_mapping.get(target, target)
        return img, target, index 

    def set_label_mapping(self, label_mapping):
        """设置标签映射, dataset 类不支持动态添加属性，需要通过函数实现
        """
        self.label_mapping = label_mapping

class split_dataset(Dataset):
    def __init__(self, data):
        super(split_dataset, self).__init__()
        self.Xa_data = data[0]
        self.labels = data[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = self.Xa_data[index]
        Y = self.labels[index]
        return X, Y



class cluster_dataset(CIFAR10):
    def __init__(self, *args, new_targets=None, **kwargs):
        super().__init__(*args, **kwargs)

        # 如果提供了新的标签，那么就替换原始标签
        if new_targets is not None:
            assert len(new_targets) == len(self.targets)
            self.targets = new_targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

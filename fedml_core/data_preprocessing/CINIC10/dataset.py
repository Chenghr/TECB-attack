import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100
from PIL import Image
import numpy as np
import torchvision


class CINIC10L_old(Dataset):
    def __init__(self, root, split='train', transform=None):
        super().__init__()
        # image_folder = torchvision.datasets.ImageFolder(root=root + '/' + split)
        image_folder = torchvision.datasets.ImageFolder(root=root + split)
        self.targets = image_folder.targets
        self.image_paths = image_folder.imgs
        self.transform = transform

    def __getitem__(self, index):
        file_path, label = self.image_paths[index]
        img = self.read_image(file_path)
        return img, label, index

    def __len__(self):
        return len(self.image_paths)

    def read_image(self, path):
        img = Image.open(path)
        return self.transform(img) if self.transform else img
    

class CINIC10L(Dataset):
    def __init__(self, root, split='train', transform=None):
        super().__init__()
        image_folder = torchvision.datasets.ImageFolder(root=root + split)
        self.targets = image_folder.targets
        self.image_paths = image_folder.imgs
        self.transform = transform
        self.modified_images = {}  # 用字典存储修改过的图像
        
    def __getitem__(self, index):
        file_path, label = self.image_paths[index]
        
        # 如果该索引的图像被修改过，则使用修改后的图像
        if index in self.modified_images:
            img = self.modified_images[index]
            # 如果存储的是numpy数组，转换为PIL Image
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img.astype('uint8'))
        else:
            img = Image.open(file_path)
            
        if self.transform:
            img = self.transform(img)
            
        return img, label, index

    def __len__(self):
        return len(self.image_paths)

    def modify_images(self, indices, new_images):
        """
        修改指定索引的图像
        
        参数:
        indices: list of int, 需要修改的图像索引列表
        new_images: numpy array (N, H, W, C) 或 list of numpy arrays, 新的图像数据
        """
        if isinstance(new_images, np.ndarray):
            assert len(indices) == len(new_images), "索引数量与图像数量不匹配"
            for idx, img in zip(indices, new_images):
                self.modified_images[idx] = img
        elif isinstance(new_images, list):
            assert len(indices) == len(new_images), "索引数量与图像数量不匹配"
            for idx, img in zip(indices, new_images):
                self.modified_images[idx] = img
        else:
            raise ValueError("new_images must be numpy array or list of numpy arrays")

    def reset_modifications(self, indices=None):
        """
        重置修改过的图像
        
        参数:
        indices: list of int or None, 要重置的索引列表。如果为None，重置所有修改
        """
        if indices is None:
            self.modified_images.clear()
        else:
            for idx in indices:
                self.modified_images.pop(idx, None)

    def get_original_image(self, index):
        """
        获取原始图像（未经修改和transform的图像）
        """
        file_path, _ = self.image_paths[index]
        return Image.open(file_path)

    def get_modified_indices(self):
        """
        返回所有被修改过的图像索引
        """
        return list(self.modified_images.keys())
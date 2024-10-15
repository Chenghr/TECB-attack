import os
import sys

import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
import torch

from fedml_core.trainer.tecb_trainer import TECBTrainer
from fedml_core.utils.utils import (AverageMeter, keep_predict_loss,
                                    over_write_args_from_file)
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10

from pretest.utils import load_tecb_cinic10, load_tecb_cifar10, load_tecb_cifar100

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap, hsv_to_rgb


def split_data(dataset, data, half=16):
    if dataset in ["CIFAR10", "CIFAR100", "CINIC10L"]:
        x_a = data[:, :, :, 0 : half]
        x_b = data[:, :, :, half : 32]
    else:
        raise Exception("Unknown dataset name!")
    return x_a, x_b

def prepare_data(device, dataset="cifar10", poison_ratio=0.1):
    # 1. load model and data
    if dataset == "CIFAR10":
        model_list, delta, target_label, dataloader = load_tecb_cifar10()
    elif dataset == "CIFAR100":
        model_list, delta, target_label, dataloader = load_tecb_cifar100()
    elif dataset == "CINIC10L":
        model_list, delta, target_label, dataloader = load_tecb_cinic10()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    model_list = [model.to(device).eval() for model in model_list]
    
    data = {i: [] for i in range(10 if dataset in ["CIFAR10", "CINIC10L"] else 100)}
    labels = {i: [] for i in range(10 if dataset in ["CIFAR10", "CINIC10L"] else 100)}
    
    with torch.no_grad():
        for batch_idx, (trn_X, trn_y, indices) in enumerate(dataloader):
            if dataset in ["CIFAR10", "CIFAR100", "CINIC10L"]:
                trn_X = trn_X.float().to(device)
                Xa, Xb = split_data(dataset, trn_X, half=16)
                target = trn_y.long().to(device)

            # 对非目标标签的样本应用 delta
            mask = (target != target_label) & (torch.rand(target.size(), device=device) < poison_ratio)
            if delta is not None and mask.any():
                Xb[mask] += delta
                    
            output_tensor_bottom_model_a = model_list[0](Xa)
            output_tensor_bottom_model_b = model_list[1](Xb)
            
            output = model_list[2](
                output_tensor_bottom_model_a, output_tensor_bottom_model_b
            )

            probs = F.softmax(output, dim=1)
            _, pred = probs.topk(1, 1, True, True)
            
            # Collect outputs based on predictions
            for i, p in enumerate(pred.squeeze()):
                data[p.item()].append(output[i].cpu().numpy())
                labels[p.item()].append(target[i].cpu().numpy())

    # Convert lists to numpy arrays
    for k in data:
        data[k] = np.array(data[k])
        labels[k] = np.array(labels[k])

    return data, labels, target_label       
    

def apply_tsne(embeddings, n_components=2, perplexity=30, n_iter=1000):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42)
    return tsne.fit_transform(embeddings)

def visualize_tsne_all_classes(tsne_results, labels, target_label, title, save_dir="../results/pics/pretest/detect_target_label"):
    def create_plot(include_target=True):
        plt.figure(figsize=(12, 10))
        
        # 获取唯一的类别标签
        unique_labels = np.unique(labels)
        
        # 创建一个不包含红色的自定义颜色映射，并降低饱和度
        num_classes = len(unique_labels)
        hues = np.linspace(0.1, 0.9, num_classes)
        saturation = 0.6  # 降低饱和度，可以根据需要调整（0-1之间）
        value = 0.9  # 保持较高的明度
        
        colors = [hsv_to_rgb((h, saturation, value)) for h in hues]
        
        # 确保目标类使用红色
        target_color = np.array([1, 0, 0, 1])  # 红色
        target_index = np.where(unique_labels == target_label)[0][0]
        colors[target_index] = target_color
        
        cmap = ListedColormap(colors)
        
        # 绘制每个类别的点
        for i, label in enumerate(unique_labels):
            if not include_target and label == target_label:
                continue
            mask = labels == label
            plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                        c=[colors[i]], label=f'Class {label}', 
                        alpha=0.7, edgecolors='none')
        
        plt.title(f"{title}{' (without target class)' if not include_target else ''}", fontsize=18)
        plt.xlabel('t-SNE feature 1', fontsize=16)
        plt.ylabel('t-SNE feature 2', fontsize=16)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=16)
        
        # 添加说明文字在右上角
        # if include_target:
        #     plt.text(0.98, 0.98, f"Target Label: Class {target_label}", 
        #              transform=plt.gca().transAxes, fontsize=12, 
        #              verticalalignment='top', horizontalalignment='right',
        #              bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'))
            
        plt.tight_layout()
        fig_name = os.path.join(save_dir, f"{title.replace(' ', '_')}{'_without_target' if not include_target else ''}.pdf")
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        plt.close()

    # 生成包含目标标签的图像
    create_plot(include_target=True)

    # # 生成不包含目标标签的图像
    # create_plot(include_target=False)
    

def main(device, dataset="cifar10", poison_ratio=0.1):
    # 准备数据
    data, labels, target_label = prepare_data(device, dataset, poison_ratio)
    print(data.keys())
    print(len(data.values()), target_label)
    
    # 合并所有类别的数据
    all_data = np.concatenate(list(data.values()), axis=0)
    all_labels = np.concatenate([np.full(len(data[k]), k) for k in data.keys()])
    print(all_data.shape, all_labels.shape)
    
    # 应用t-SNE
    tsne_results = apply_tsne(all_data)

    # 可视化所有类别
    visualize_tsne_all_classes(tsne_results, all_labels, target_label,
                               f"t-SNE for all classes in {dataset} (Poison Ratio: {poison_ratio})")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets = ["CIFAR10", "CIFAR100", "CINIC10L"]
    # datasets = ["CIFAR10"]
    for dataset in datasets:
        main(device, dataset, poison_ratio=0.1)
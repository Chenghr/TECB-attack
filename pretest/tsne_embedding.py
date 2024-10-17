import os
import sys

import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
import torch

from pretest.utils import load_tecb_cinic10, load_tecb_cifar10, load_tecb_cifar100, split_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap, hsv_to_rgb


def main(device, dataset="cifar10", poison_ratio=0.01, u=2.0, save_dir="../results/pics/pretest/detect_target_label"):
    # 准备数据
    tsne_data, target_label = prepare_data(device, dataset, poison_ratio)
    
    # TSNE 可视化
    visualize_tsne(tsne_data, target_label, 
                   title=f"t-SNE in {dataset} (Poison Ratio: {poison_ratio})",
                   save_dir=save_dir,
                )


def visualize_tsne(tsne_data, target_label, title, save_dir="../results/pics/pretest/detect_target_label",
                   n_components=2, perplexity=30, n_iter=1000):
    """根据预测结果的label分别进行聚类
    """
    # 合并所有类别的数据
    embeddings = np.concatenate(list(tsne_data.values()), axis=0)
    predict_labels = np.concatenate([np.full(len(tsne_data[k]), k) for k in tsne_data.keys()])
    
    # Apply t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))
    
    # 获取唯一的类别标签
    unique_labels = np.unique(predict_labels)
    
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
        mask = predict_labels == label
        plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                    c=[colors[i]], label=f'Class {label}', 
                    alpha=0.7, edgecolors='none')
    
    plt.title(title, fontsize=18)
    plt.xticks([])  # 移除 x 轴刻度
    plt.yticks([])  # 移除 y 轴刻度
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=16)

    plt.tight_layout()
    fig_name = os.path.join(save_dir, f"{title.replace(' ', '_')}.pdf")
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.close()
    

def prepare_data(device, dataset="cifar10", poison_ratio=0.1):
    """根据被投毒的模型，输出对应的 embedding

    Args:
        device (torch.device): 计算设备（CPU 或 GPU）
        dataset (str, optional): 数据集名称. Defaults to "cifar10".
        poison_ratio (float, optional): 投毒比例. Defaults to 0.1.

    Raises:
        ValueError: 当提供的数据集名称不受支持时抛出

    Returns:
        tuple: 包含以下四个元素的元组:
            tsne_data: key: 预测标签, value: embedding 列表
            target_label: int, 目标标签
    """
    # 1. 加载模型和数据
    if dataset.upper() == "CIFAR10":
        model_list, delta, target_label, dataloader = load_tecb_cifar10()
    elif dataset.upper() == "CIFAR100":
        model_list, delta, target_label, dataloader = load_tecb_cifar100()
    elif dataset.upper() == "CINIC10L":
        model_list, delta, target_label, dataloader = load_tecb_cinic10()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    model_list = [model.to(device).eval() for model in model_list]
    
    num_classes = 10 if dataset.upper() in ["CIFAR10", "CINIC10L"] else 100
    tsne_data = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        for batch_idx, (trn_X, trn_y, indices) in enumerate(dataloader):
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
            
            # 收集输出基于预测和真实标签
            for i, (p, t) in enumerate(zip(pred.squeeze(), target)):
                tsne_data[p.item()].append(output[i].cpu().numpy())

    # 将列表转换为 numpy 数组
    for k in tsne_data:
        tsne_data[k] = np.array(tsne_data[k])

    return tsne_data, target_label


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # datasets = ["CIFAR10", "CIFAR100", "CINIC10L"]
    datasets = ["CIFAR10"]
    for dataset in datasets:
        main(device, dataset, poison_ratio=0.1, u=2.0, save_dir="../results/pics/pretest/detect_target_label")
import os
import sys

import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from scipy.stats import multivariate_normal
import torch
import numpy as np
from pretest.utils import load_tecb_cifar10, load_tecb_cifar100, load_tecb_cinic10, split_data
from pretest.scan import SCAn

        
def prepare_data(device, dataset="cifar10", poison_ratio=0.1):
    # 1. 加载模型和数据
    if dataset.upper() == "CIFAR10":
        model_list, train_dataloader, test_dataloader, delta, target_label = load_tecb_cifar10()
    elif dataset.upper() == "CIFAR100":
        model_list, train_dataloader, test_dataloader, delta, target_label = load_tecb_cifar100()
    elif dataset.upper() == "CINIC10L":
        model_list, train_dataloader, test_dataloader, delta, target_label = load_tecb_cinic10()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    model_list = [model.to(device).eval() for model in model_list]
    
    poison_data_dict = {
        "embeddings": [],
        "probs": [],
        "preds": [],
        "labels": [],
    }
    with torch.no_grad():
        for batch_idx, (trn_X, trn_y, indices) in enumerate(train_dataloader):
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
            poison_data_dict["embeddings"].append(output.cpu().numpy())
            poison_data_dict["probs"].append(probs.cpu().numpy())
            poison_data_dict["preds"].append(pred.cpu().numpy())
            poison_data_dict["labels"].append(target.cpu().numpy())
    
    clean_data_dict = {
        "embeddings": [],
        "probs": [],
        "preds": [],
        "labels": [],
    }
    with torch.no_grad():
        for batch_idx, (trn_X, trn_y, indices) in enumerate(test_dataloader):
            trn_X = trn_X.float().to(device)
            Xa, Xb = split_data(dataset, trn_X, half=16)
            target = trn_y.long().to(device)
                    
            output_tensor_bottom_model_a = model_list[0](Xa)
            output_tensor_bottom_model_b = model_list[1](Xb)
            
            output = model_list[2](
                output_tensor_bottom_model_a, output_tensor_bottom_model_b
            )

            probs = F.softmax(output, dim=1)
            _, pred = probs.topk(1, 1, True, True)
            
            # 填充 clean data 中的内容
            clean_data_dict["embeddings"].append(output.cpu().numpy())
            clean_data_dict["probs"].append(probs.cpu().numpy())
            clean_data_dict["preds"].append(pred.cpu().numpy())
            clean_data_dict["labels"].append(target.cpu().numpy())

    # 将列表转换为numpy数组
    for key in poison_data_dict:
        poison_data_dict[key] = np.concatenate(poison_data_dict[key])
    
    for key in clean_data_dict:
        clean_data_dict[key] = np.concatenate(clean_data_dict[key])
        
    return poison_data_dict, clean_data_dict, target_label


def identify_potential_target_classes(poison_data_dict, clean_data_dict, num_classes, target_label):
    scan = SCAn()
    
    gb_model = scan.build_global_model(clean_data_dict["embeddings"], clean_data_dict["labels"], num_classes)
    lc_model = scan.build_local_model(poison_data_dict["embeddings"], poison_data_dict["labels"], gb_model, num_classes) 

    # 3. 计算最终异常分数
    anomaly_scores = scan.calc_final_score(lc_model)
    
     # 创建标签和分数的对应关系并排序
    label_scores = list(enumerate(anomaly_scores))
    sorted_label_scores = sorted(
        label_scores, 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # 打印排序结果
    print("\nAnomaly Scores (sorted from high to low):")
    print("----------------------------------------")
    print("Label\tScore\t\tStatus")
    print("----------------------------------------")
    for label, score in sorted_label_scores:
        status = "*** TARGET ***" if label == target_label else ""
        print(f"{label}\t{score:.4f}\t\t{status}")
    print("----------------------------------------")
    
    return anomaly_scores


def main(device, dataset="CIFAR10", poison_ratio=0.1, u=2.0, save_dir="../results/pics/pretest/detect_target_label"):
    # 准备数据
    num_classes = {
            'CIFAR10': 10,
            'CIFAR100': 100,
            'CINIC10L': 10
    }.get(dataset)
    poison_data_dict, clean_data_dict, target_label = prepare_data(device, dataset, poison_ratio)
    anomaly_scores = identify_potential_target_classes(poison_data_dict, clean_data_dict, num_classes, target_label)
    

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # datasets = ["CIFAR10", "CIFAR100", "CINIC10L"]
    datasets = ["CIFAR10"]
    for dataset in datasets:
        main(device, dataset, poison_ratio=0.1, u=2.0, save_dir="../results/pics/pretest/detect_target_label")
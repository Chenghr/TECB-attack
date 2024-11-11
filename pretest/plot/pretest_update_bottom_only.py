import matplotlib.pyplot as plt
import numpy as np
import os


def get_data(attack_method="TECB", dataset="cifar10", defense_features=16):
    if attack_method == "TECB":
        if dataset == "cifar10":
            if defense_features == 16:
                main_task_acc = [81.50, 76.95, 60.28, 55.40, 52.30, 49.81, 46.61, 44.34, 44.66, 42.85, 43.10, 42.42, 41.50, 42.78, 41.68, 40.95, 39.95, 39.31, 38.66, 38.81, 38.17, 38.40, 36.79, 36.49, 36.16, 35.69, 35.66, 36.44, 34.86, 34.80, 34.82]
                asr = [100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 99.99, 99.99, 100.00, 99.99, 99.99, 99.99, 99.97, 99.96, 99.97, 99.95, 99.92, 99.93, 99.87, 99.85, 99.85, 99.82, 99.95, 99.91, 99.93, 99.89]
            elif defense_features == 20:
                main_task_acc = [80.81, 49.36, 27.93, 23.04, 18.75, 17.88, 17.24, 16.84, 17.64, 17.56, 17.20, 18.02, 17.00, 17.02, 16.18, 15.69, 17.05, 15.83, 15.45, 16.07, 14.79, 15.41, 14.41, 14.58, 14.58, 14.47, 13.49, 14.66, 13.91, 13.63, 12.92]
                asr = [100.00, 100.00, 99.99, 99.91, 99.68, 99.60, 99.68, 99.50, 99.29, 99.64, 99.51, 99.63, 99.51, 99.79, 98.13, 99.70, 99.58, 99.67, 99.80, 99.26, 99.64, 99.76, 99.64, 99.76, 99.67, 99.25, 99.65, 99.45, 99.59, 99.47, 99.59]
            elif defense_features == 24:
                main_task_acc = []
                asr = []
            else:
                raise ValueError
        else:
            raise ValueError
    elif attack_method == "BadVFL":
        if dataset == "cifar10":
            if defense_features == 16:
                main_task_acc = [ 80.20, 80.10, 79.66, 79.21, 78.28, 76.97, 74.70, 71.70, 68.21, 63.83, 58.89, 54.67, 50.66, 46.68, 43.96, 41.02, 38.56, 36.42, 34.72, 33.09, 32.63, 30.76, 30.34, 28.41, 28.76, 26.84, 26.94, 25.84, 25.37, 24.75, 24.34]
                asr = [93.20, 94.00, 94.30, 93.90, 94.30, 94.10, 94.00, 93.80, 93.00, 91.40, 89.40, 86.30, 84.70, 82.80, 80.40, 78.80, 79.40, 76.20, 77.10, 77.10, 76.60, 74.10, 75.00, 72.00, 72.80, 71.30, 69.20, 67.70, 63.90, 62.70, 62.30]
            elif defense_features == 20:
                main_task_acc = [81.34, 81.40, 81.21, 81.25, 81.30, 81.18, 81.06, 81.10, 80.89, 80.91, 80.83, 80.68, 80.36, 80.37, 80.12, 79.71, 79.27, 78.99, 78.54, 77.89, 77.53, 76.46, 75.50, 74.24, 73.30, 72.59, 71.43, 70.42, 68.71, 67.52, 65.64]
                asr = [96.20, 97.30, 97.20, 97.40, 97.40, 97.50, 97.40, 97.50, 97.50, 97.40, 97.40, 97.50, 97.40, 97.20, 97.20, 97.40, 97.40, 97.40, 97.40, 97.30, 97.20, 97.00, 97.20, 97.40, 97.50, 97.20, 96.80, 97.20, 97.00, 96.80, 97.50]
            elif defense_features == 24:
                main_task_acc = [82.73, 82.82, 82.79, 82.67, 82.63, 82.69, 82.49, 82.38, 82.22, 82.13, 81.80, 81.60, 81.26, 80.68, 80.11, 79.30, 78.30, 77.05, 75.78, 74.19, 72.25, 70.58, 68.18, 65.92, 63.54, 60.88, 58.09, 54.85, 52.45, 51.03, 48.74]
                asr = [96.90, 98.90, 98.80, 98.90, 98.70, 98.80, 98.70, 98.60, 98.80, 98.60, 98.00, 98.30, 98.10, 97.90, 97.50, 97.70, 97.60, 97.70, 97.50, 97.10, 96.90, 96.30, 95.90, 96.00, 95.10, 94.80, 92.80, 91.40, 91.40, 88.70, 89.70]
            else:
                raise ValueError
        else:
            raise ValueError
    elif attack_method == "Villain":
        if dataset == "cifar10":
            if defense_features == 16:
                main_task_acc = [77.07, 77.34, 77.02, 77.02, 76.82, 76.58, 75.97, 75.08, 73.64, 70.85, 67.24, 62.01, 55.86, 51.00, 47.53, 45.52, 43.69, 42.35, 41.07, 40.54, 39.87, 39.39, 38.84, 37.83, 37.71, 37.53, 37.17, 36.35, 36.49, 36.05, 35.37]
                asr = [96.70, 98.41, 98.26, 98.09, 98.31, 98.13, 98.33, 98.08, 97.93, 96.43, 96.92, 95.94, 93.58, 85.47, 76.64, 73.35, 71.16, 65.10, 66.84, 65.33, 65.31, 61.88, 63.83, 65.00, 60.40, 59.82, 61.51, 60.94, 65.79, 64.94, 68.85]
            elif defense_features == 20:
                main_task_acc = [74.41, 74.49, 74.44, 73.80, 73.05, 71.51, 67.96, 63.61, 58.47, 53.52, 47.98, 43.57, 39.52, 36.10, 34.11, 32.31, 31.17, 30.46, 28.89, 28.48, 28.13, 27.40, 26.94, 26.58, 26.35, 25.72, 25.88, 25.31, 24.74, 24.03, 23.55]
                asr = [95.49, 96.73, 96.31, 96.20, 96.65, 96.01, 96.69, 96.30, 96.20, 95.73, 94.62, 60.86, 60.11, 55.52, 58.92, 41.36, 33.57, 31.07, 23.08, 22.92, 25.98, 11.41, 12.13, 9.48, 7.87, 5.16, 3.40, 3.04, 1.84, 1.55, 1.04]
            elif defense_features == 24:
                main_task_acc = [77.35, 77.43, 77.52, 77.20, 77.17, 76.69, 75.99, 74.64, 72.97, 68.61, 62.36, 54.90, 46.92, 38.71, 31.52, 26.39, 23.35, 21.05, 19.37, 17.61, 16.34, 14.95, 14.16, 13.76, 13.54, 13.26, 13.07, 12.92, 13.15, 12.80, 12.72]
                asr = [78.55, 96.98, 97.53, 97.07, 96.68, 97.23, 97.00, 97.38, 97.69, 98.91, 98.20, 97.42, 95.26, 92.82, 89.39, 87.13, 85.50, 82.46, 80.37, 79.99, 75.75, 73.52, 67.03, 66.23, 62.69, 66.58, 67.30, 68.92, 71.40, 75.64, 76.65]
            else:
                raise ValueError
        else:
            raise ValueError
    else:
        raise ValueError
    
    return main_task_acc, asr


def plot_acc(main_task_acc, asr, 
             attack_method="TECB", dataset="cifar10", defense_features=16,
             save_dir="../", pic_type = "png"):
    # 样式参数设置
    linewidth = 1.8
    labelsize = 18
    bwith = 2
    lwith = 2
    markevery=2
    markersize = 8
    ticksize = 12
    x_ticks_gap=5
    legendsize = 16
    legend_loc = "lower right"

    # 创建图形
    fig, ax = plt.subplots(1, 1)# figsize=(16, 9)
   
    # 设置轴的样式
    for spine in ax.spines.values():
        spine.set_linewidth(bwith)
    ax.grid(which="major", ls="--", lw=lwith, c="gray")

    # 颜色映射
    # marker 类型: o: 圆形, *: 五角星, v: 三角形, s: 正方形
    # google 配色: #f4433c 红色, #ffbc32 黄色, #0aa858 绿色, #2d85f0 蓝色
    colors = ['#0aa858', '#f4433c', '#2d85f0', '#ffbc32']
    markers = ['v', 'o', 's', '*']
    
    epochs = list(range(len(main_task_acc)))
    main_task_acc = [acc/100 for acc in main_task_acc]
    asr = [acc/100 for acc in asr]
    
    # 绘制主任务准确率
    ax.plot(
        epochs,
        main_task_acc,
        label=f'Main Task Acc',
        ls="--",
        linewidth=linewidth,
        c=colors[0],
        marker=markers[0],
        markersize=markersize,
        markevery=markevery,
    )
    # 绘制ASR
    ax.plot(
        epochs,
        asr,
        label=f'ASR',
        ls="-",
        linewidth=linewidth,
        c=colors[1],
        marker=markers[1],
        markersize=markersize,
        markevery=markevery,
    )
    
    # 设置x轴
    x_ticks = list(range(0, len(epochs) + 1, x_ticks_gap))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x}" for x in x_ticks], fontsize=ticksize)

    # 设置y轴
    ax.set_ylim(-0.05, 1.05)
    y_ticks = np.arange(0, 1.05, 0.2)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:.1f}" for y in y_ticks], fontsize=ticksize)

    # 设置标签和标题
    ax.set_xlabel("Epochs", fontsize=labelsize)
    ax.set_ylabel("Accuracy", fontsize=labelsize)
    
    # 添加Attack Method和Update Mode信息
    # title = f"Attack Method: TECB; Update Mode: bottom_only; Attack Feaure: {attack_features}"
    title = f"Update Mode: bottom only; Defense Feaure: {defense_features}"
    plt.title(title, fontsize=labelsize-2)
    
    ax.legend(fontsize=legendsize, loc=legend_loc,)
    plt.tight_layout()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_fig = os.path.join(save_dir, f"{attack_method}_{dataset}_defense_feat_{defense_features}.{pic_type}")
    plt.savefig(save_fig, bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    attack_methods = ["TECB", "BadVFL", "Villain"]
    datasets = ["cifar10"]
    defense_features = [16, 20, 24]
    
    for attack_method in attack_methods:
        for dataset in datasets:
            for defense_feat in defense_features:
                main_task_acc, asr = get_data(attack_method, dataset, defense_feat)
                plot_acc(main_task_acc, asr, 
                         attack_method, dataset, defense_feat, 
                         save_dir="../../results/pretest/permuted_train/update_bottom_only", pic_type = "png")

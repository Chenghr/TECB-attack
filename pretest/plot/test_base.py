import matplotlib.pyplot as plt
import numpy as np
import os

def plot_acc_pre(main_task_acc, asr, attack_features=16, dataset="CIFAR100",
             save_dir="../", pic_type = "png"):
    # 样式参数设置
    linewidth = 1.8
    labelsize = 18
    bwith = 2
    lwith = 2
    markevery=2
    markersize = 8
    ticksize = 12
    x_ticks_gap=2
    legendsize = 16
    legend_loc = "lower left" # upper right, lower right, lower left

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
    title = f"AC: {attack_features}; Dataset: {dataset}"
    plt.title(title, fontsize=labelsize-2)
    
    ax.legend(fontsize=legendsize, loc=legend_loc,)

    plt.tight_layout()
    save_fig = os.path.join(save_dir, f"{dataset}.{pic_type}")
    print(save_fig)
    plt.savefig(save_fig, bbox_inches='tight')
    plt.close()

def get_base_data(dataset="CINIC10L", attack_features=31):
    if dataset =="CINIC10L":
        if attack_features == 31:
            main_task_acc = [72.62, 72.49, 72.43, 71.95, 69.65, 66.83, 64.13, 61.30, 58.77, 56.56, 53.32, 50.21, 47.88, 43.20, 38.91, 36.04, 31.24, 27.78, 24.54, 23.32, 21.86, 20.53, 19.47, 19.06, 16.71, 14.52, 13.00, 11.96, 11.58, 10.68, 9.52]
            asr = [99.98, 99.97, 99.97, 99.97, 99.96, 99.95, 99.91, 99.87, 99.75, 99.47, 98.52, 97.86, 97.30, 96.95, 92.13, 80.27, 68.78, 32.59, 0.79, 0.27, 0.30, 0.69, 2.13, 6.90, 26.95, 50.15, 60.16, 67.25, 69.14, 72.25, 68.98]
        elif attack_features == 28:
            main_task_acc = [71.50, 71.84, 71.75, 71.54, 71.44, 71.10, 70.13, 69.04, 67.73, 66.98, 65.42, 64.01, 62.95, 61.48, 60.00, 58.25, 57.27, 55.79, 54.81, 53.96, 51.55, 49.28, 47.05, 44.92, 42.72, 40.64, 38.39, 36.81, 34.75, 33.14, 31.95]
            asr = [99.78, 99.57, 99.43, 99.37, 99.34, 99.20, 99.17, 98.88, 98.34, 98.07, 97.77, 97.22, 96.44, 96.69, 94.34, 92.68, 92.81, 91.81, 89.09, 86.80, 78.11, 67.95, 43.47, 33.69, 17.75, 10.51, 4.78, 2.40, 0.31, 0.10, 0.02]
        elif attack_features == 24:
            main_task_acc = [ 71.28, 71.28, 71.19, 70.98, 70.23, 68.08, 65.52, 63.16, 60.93, 58.90, 56.13, 53.97, 53.20, 50.70, 48.24, 45.86, 42.90, 40.46, 37.90, 35.95, 33.38, 31.22, 29.33, 27.29, 26.43, 24.51, 24.10, 22.46, 22.14, 21.24, 20.73]
            asr = [99.89, 99.89, 99.87, 99.88, 99.83, 99.80, 99.73, 99.42, 99.08, 98.46, 97.17, 95.93, 95.25, 91.90, 91.82, 90.04, 87.25, 86.72, 84.96, 82.99, 81.38, 79.00, 76.39, 72.86, 71.60, 68.06, 67.74, 62.48, 58.76, 57.34, 52.87]
        elif attack_features == 20:
            main_task_acc = [67.41, 68.60, 68.35, 68.01, 67.39, 66.42, 64.76, 62.01, 57.89, 53.02, 48.90, 45.17, 42.56, 39.80, 37.27, 35.35, 33.50, 32.39, 30.26, 29.46, 26.20, 24.73, 23.72, 21.94, 21.33, 19.77, 18.29, 17.21, 17.05, 16.29, 15.71]
            asr = [94.83, 91.51, 91.02, 90.18, 89.43, 87.42, 84.52, 78.48, 68.88, 60.37, 54.16, 46.43, 48.41, 44.26, 39.70, 43.00, 37.58, 34.90, 33.38, 31.44, 29.26, 25.70, 22.34, 21.72, 21.04, 16.52, 13.04, 9.13, 9.21, 7.57, 5.34]
        elif attack_features == 16:
            main_task_acc = [64.36, 67.11, 66.39, 65.51, 64.00, 61.97, 59.27, 56.48, 53.35, 50.78, 48.49, 46.14, 44.45, 42.74, 39.89, 34.79, 30.09, 26.17, 23.79, 21.57, 19.65, 16.51, 13.98, 12.00, 10.72, 9.89, 9.65, 9.11, 8.14, 7.54, 6.61]
            asr = [85.48, 88.43, 87.87, 87.76, 87.77, 83.63, 78.58, 71.16, 62.99, 54.93, 49.68, 46.14, 43.27, 43.01, 46.64, 43.07, 44.02, 38.57, 34.99, 35.05, 30.55, 25.30, 24.53, 22.96, 23.40, 19.20, 19.52, 17.09, 15.35, 15.61, 14.16]
        elif attack_features == 8:
            main_task_acc = [70.07, 70.04, 69.85, 69.51, 68.98, 66.40, 62.86, 58.53, 53.51, 49.36, 45.26, 42.32, 39.32, 35.41, 32.34, 31.00, 28.94, 27.71, 26.76, 26.70, 25.38, 24.28, 22.50, 20.74, 17.64, 14.88, 12.82, 11.24, 9.56, 8.69, 7.42]
            asr = [77.64, 81.07, 86.44, 88.89, 91.71, 93.85, 93.57, 93.31, 92.67, 89.46, 83.13, 71.31, 61.03, 58.02, 60.01, 61.88, 61.11, 60.30, 58.53, 60.15, 57.87, 57.81, 57.84, 58.98, 56.96, 54.70, 56.06, 53.18, 52.99, 48.98, 43.16]
        else:
            raise ValueError
    else:
        raise ValueError
    
    return main_task_acc, asr


def plot_acc(main_task_acc, asr, dataset="CINIC10L", attack_features=31, update_mode="both",
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
    legend_loc = "lower left" # upper right, lower right, lower left

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
    title = f"Update Mode: {update_mode}; AC: {attack_features}"
    plt.title(title, fontsize=labelsize-2)
    
    ax.legend(fontsize=legendsize, loc=legend_loc,)

    plt.tight_layout()
    save_fig = os.path.join(save_dir, f"AC_{ac}.{pic_type}")
    print(save_fig)
    plt.savefig(save_fig, bbox_inches='tight')
    plt.close()



if __name__ == "__main__":
    # main_task_acc = [47.4, 46.84, 45.2, 41.74, 36.05, 29.46, 24.17, 20.02, 17.6, 15.42, 13.63]
    # asr =  [98.89, 90.7, 90.65, 56.8, 38.24, 8.73, 0.79, 2.06, 0.45, 0.11, 0.79] 
    # plot_acc(main_task_acc, asr, attack_features=16, save_dir=f"../../results/pretest/permuted_training/", pic_type = "png")
    
    dataset = "CINIC10L"
    acs = [8, 16, 20, 24, 28, 31]
    for ac in acs:
        main_task_acc, asr = get_base_data(dataset=dataset, attack_features=ac)
        plot_acc(main_task_acc, asr, dataset=dataset, attack_features=ac, save_dir=f"../../results/pretest/permuted_training/test_base/CINIC10L/", pic_type = "png")

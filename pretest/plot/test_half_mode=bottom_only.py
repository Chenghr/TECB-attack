import matplotlib.pyplot as plt
import numpy as np
import os


def get_data(attack_features=31):
    if attack_features == 31:
        main_task_acc = [85.33, 85.28, 85.28, 85.31, 85.29, 85.38, 85.26, 85.11, 85.32, 85.29, 85.29, 85.30, 85.28, 85.32, 85.15, 85.30, 85.30, 85.33, 85.24, 85.21, 85.31, 85.26, 85.16, 85.19, 85.33, 85.32, 85.33, 85.34, 85.32, 85.34, 85.30]
        asr = [100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00]
    elif attack_features == 28:
        main_task_acc = [84.60, 84.59, 84.57, 84.57, 84.52, 84.49, 84.47, 84.50, 84.57, 84.58, 84.51, 84.51, 84.60, 84.50, 84.51, 84.58, 84.58, 84.54, 84.37, 84.51, 84.61, 84.56, 84.54, 84.57, 84.58, 84.55, 84.51, 84.48, 84.48, 84.43, 84.46]
        asr = [100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00]
    elif attack_features == 24:
        main_task_acc = [83.35, 83.14, 83.28, 83.35, 83.27, 83.34, 83.18, 83.24, 83.39, 83.20, 83.25, 83.31, 83.37, 83.11, 83.23, 83.32, 83.33, 83.25, 83.32, 83.26, 83.39, 83.24, 83.12, 83.11, 83.22, 83.24, 83.36, 83.37, 83.11, 83.33, 83.17]
        asr = [100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00]
    elif attack_features == 20:
        main_task_acc = [81.55, 80.78, 78.24, 76.65, 76.21, 75.91, 75.61, 75.51, 75.30, 75.13, 74.49, 74.65, 73.70, 73.91, 73.15, 73.13, 72.83, 72.90, 72.36, 72.66, 72.96, 72.19, 72.45, 71.79, 72.11, 71.90, 71.43, 71.76, 71.61, 71.22, 71.33]
        asr = [100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00]
    elif attack_features == 18:
        main_task_acc = [81.13, 77.55, 67.45, 62.76, 61.62, 60.94, 59.85, 59.52, 59.02, 56.82, 57.17, 56.84, 55.88, 57.67, 56.77, 55.33, 58.16, 57.23, 55.80, 56.25, 56.08, 55.77, 55.44, 56.21, 56.43, 55.51, 54.90, 56.43, 57.17, 55.78, 55.58]
        asr = [100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 99.99, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00]
    elif attack_features == 16:
        main_task_acc = [81.50, 76.95, 60.28, 55.40, 52.30, 49.81, 46.61, 44.34, 44.66, 42.85, 43.10, 42.42, 41.50, 42.78, 41.68, 40.95, 39.95, 39.31, 38.66, 38.81, 38.17, 38.40, 36.79, 36.49, 36.16, 35.69, 35.66, 36.44, 34.86, 34.80, 34.82]
        asr = [100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 99.99, 99.99, 100.00, 99.99, 99.99, 99.99, 99.97, 99.96, 99.97, 99.95, 99.92, 99.93, 99.87, 99.85, 99.85, 99.82, 99.95, 99.91, 99.93, 99.89]
    elif attack_features == 14:
        main_task_acc = [81.83, 69.18, 50.04, 38.39, 29.72, 28.58, 26.50, 25.56, 24.68, 22.85, 23.34, 23.21, 24.68, 25.22, 26.50, 26.15, 25.99, 26.04, 25.67, 24.94, 24.94, 23.39, 23.32, 23.55, 22.40, 21.52, 21.78, 21.42, 21.52, 21.50, 22.12]
        asr = [100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 99.99, 100.00, 99.99, 99.97, 99.99, 100.00, 99.94, 99.98, 99.84, 99.91, 99.84, 99.99, 99.77, 99.97, 99.84, 99.92, 99.82, 99.85]
    elif attack_features == 12:
        main_task_acc = [80.81, 49.36, 27.93, 23.04, 18.75, 17.88, 17.24, 16.84, 17.64, 17.56, 17.20, 18.02, 17.00, 17.02, 16.18, 15.69, 17.05, 15.83, 15.45, 16.07, 14.79, 15.41, 14.41, 14.58, 14.58, 14.47, 13.49, 14.66, 13.91, 13.63, 12.92]
        asr = [100.00, 100.00, 99.99, 99.91, 99.68, 99.60, 99.68, 99.50, 99.29, 99.64, 99.51, 99.63, 99.51, 99.79, 98.13, 99.70, 99.58, 99.67, 99.80, 99.26, 99.64, 99.76, 99.64, 99.76, 99.67, 99.25, 99.65, 99.45, 99.59, 99.47, 99.59]
    else:
        raise ValueError
    
    return main_task_acc, asr


def plot_acc(main_task_acc, asr, attack_features, 
             save_dir="../../results/pretest/permuted_training/", pic_type = "png"):
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
    title = f"Update Mode: bottom_only; Attack Feaure: {attack_features}"
    plt.title(title, fontsize=labelsize-2)
    
    ax.legend(fontsize=legendsize, loc=legend_loc,)

    plt.tight_layout()
    save_fig = os.path.join(save_dir, f"AC={attack_features}.{pic_type}")
    print(save_fig)
    plt.savefig(save_fig, bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    # attack_feats = [12, 14]
    attack_feats = [16, 18, 20, 24, 28, 31]
    for attack_feat in attack_feats:
        main_task_acc, asr = get_data(attack_features=attack_feat)
        plot_acc(main_task_acc, asr, attack_feat, save_dir="../../results/pretest/permuted_training/test_half_mode=bottom_only/", pic_type = "png")

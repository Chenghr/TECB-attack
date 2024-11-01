import matplotlib.pyplot as plt
import numpy as np
import os

def plot_acc(main_task_acc, asr, attack_features=16, dataset="CIFAR100",
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
    
if __name__ == "__main__":
    main_task_acc = [47.4, 46.84, 45.2, 41.74, 36.05, 29.46, 24.17, 20.02, 17.6, 15.42, 13.63]
    asr =  [98.89, 90.7, 90.65, 56.8, 38.24, 8.73, 0.79, 2.06, 0.45, 0.11, 0.79] 
    plot_acc(main_task_acc, asr, attack_features=16, save_dir=f"../../results/pretest/permuted_training/", pic_type = "png")

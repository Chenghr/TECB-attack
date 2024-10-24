import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def gen_soft_label_map(labels, n):
    # 获取最小值和最大值
    min_val = min(labels)
    max_val = max(labels)
    R = max_val - min_val
    
    # 获取唯一标签数量
    unique_labels = sorted(set(labels))
    C = len(unique_labels)
    
    # 生成n*C个均匀分布的值
    total_values = n * C
    values = np.linspace(min_val, max_val, total_values)
    
    # 创建结果字典
    Y_soft_map = {label: [] for label in unique_labels}
    
    # 交错分配值给不同的标签
    for i in range(total_values):
        label_idx = i % C
        label = unique_labels[label_idx]
        Y_soft_map[label].append(float(values[i]))  # 转换为float类型
    
    return Y_soft_map

def result_map(Y_hat, Y_soft_map):
    """
    将模型输出映射到最近的标签
    Args:
        Y_hat: 模型预测的软化标签值，numpy数组
        Y_soft_map: 软化标签映射字典 {label: [soft_values]}
    Returns:
        Y_pred: 预测的原始标签值，numpy数组
    """
    Y_pred = []
    
    # 将所有软化值和对应的标签组织成数组
    all_soft_values = []  # 所有软化值
    corresponding_labels = []  # 对应的原始标签
    
    for label, soft_values in Y_soft_map.items():
        all_soft_values.extend(soft_values)
        corresponding_labels.extend([label] * len(soft_values))
    
    all_soft_values = np.array(all_soft_values)
    corresponding_labels = np.array(corresponding_labels)
    
    # 对每个预测值找到最近的软化值
    for y_hat in Y_hat:
        # 计算与所有软化值的绝对距离
        distances = np.abs(all_soft_values - y_hat)
        # 找到最小距离的索引
        min_idx = np.argmin(distances)
        # 获取对应的原始标签
        pred_label = corresponding_labels[min_idx]
        Y_pred.append(pred_label)
    
    return np.array(Y_pred)

# 测试代码
def test_result_map():
    print("测试 result_map 函数:")
    
    # 测试用例1: 标准三分类
    Y_soft_map1 = {
        0: [0.0, 0.6, 1.2],
        1: [0.3, 0.9, 1.5],
        2: [0.6, 1.2, 1.8]
    }
    Y_hat1 = np.array([0.1, 0.8, 1.6, 0.4])
    Y_pred1 = result_map(Y_hat1, Y_soft_map1)
    print("\n测试用例1 - 三分类:")
    print(f"预测的软化值: {Y_hat1}")
    print(f"映射后的标签: {Y_pred1}")
    
    # 测试用例2: 二分类
    Y_soft_map2 = {
        0: [0.0, 0.25, 0.5],
        1: [0.5, 0.75, 1.0]
    }
    Y_hat2 = np.array([0.1, 0.6, 0.9, 0.3])
    Y_pred2 = result_map(Y_hat2, Y_soft_map2)
    print("\n测试用例2 - 二分类:")
    print(f"预测的软化值: {Y_hat2}")
    print(f"映射后的标签: {Y_pred2}")
    
    # 测试用例3: 边界情况
    Y_hat3 = np.array([-0.1, 2.0])  # 超出范围的值
    Y_pred3 = result_map(Y_hat3, Y_soft_map1)
    print("\n测试用例3 - 边界情况:")
    print(f"预测的软化值: {Y_hat3}")
    print(f"映射后的标签: {Y_pred3}")
    
    
def gen_bins(min_val, max_val, n):
    """
    生成分箱边界
    Args:
        min_val: 最小值
        max_val: 最大值 
        n: 箱子的数量
    Returns:
        bins: 分箱边界列表(长度为n+1)
    """
    bins = np.linspace(min_val, max_val, n+1)
        
    return bins.tolist()

def get_bin_indices(values, bins):
    """
    批量获取数据点所属的箱子下标
    Args:
        values: 数据点数组
        bins: 分箱边界列表
    Returns:
        indices: 箱子下标数组
    """
    return np.digitize(values, bins) - 1

def gen_Y_soft(X_client, X_host, Y, n_soft_labels=5, 
                  rand_max_num=200, client_num=2):
    """
    训练SplitNN模型
    Args:
        X_client: 客户端数据
        X_host: 主机端数据
        Y: 真实标签
        n_soft_labels: 软化标签数量（同时也是分箱数量）
    """
    Y_soft_map = gen_soft_label_map(Y, n_soft_labels)
    
    #  2. 获取所有软化标签值
    all_soft_values = []
    for values in Y_soft_map.values():
        all_soft_values.extend(values)
    all_soft_values = np.array(all_soft_values)
    
    # 3. 生成分箱边界，分箱数量等于软化标签数量
    min_val = 0
    max_val = client_num * rand_max_num
    bins = gen_bins(min_val, max_val, n_soft_labels)
    
    client_rand = np.random.randint(0, rand_max_num, size=X_client.shape[0])
    host_rand = np.random.randint(0, rand_max_num, size=X_host.shape[0])
    
    rand_sum = client_rand + host_rand
    bin_indices = get_bin_indices(rand_sum, bins)
    
    # 5. 根据真实标签和箱子索引生成软化标签
    Y_soft = []
    for y, bin_idx in zip(Y, bin_indices):
        # 确保bin_idx在有效范围内
        bin_idx = min(bin_idx, n_soft_labels - 1)
        bin_idx = max(bin_idx, 0)
        
        # 从Y_soft_map中获取对应标签的软化值
        y_soft = Y_soft_map[y][bin_idx]
        Y_soft.append(y_soft)
    
    return np.array(Y_soft)


# 测试代码
def test_result_map():
    print("测试 result_map 函数:")
    
    # 测试用例1: 标准三分类
    Y_soft_map1 = {
        0: [0.0, 0.6, 1.2],
        1: [0.3, 0.9, 1.5],
        2: [0.6, 1.2, 1.8]
    }
    Y_hat1 = np.array([0.1, 0.8, 1.6, 0.4])
    Y_pred1 = result_map(Y_hat1, Y_soft_map1)
    print("\n测试用例1 - 三分类:")
    print(f"预测的软化值: {Y_hat1}")
    print(f"映射后的标签: {Y_pred1}")
    
    # 测试用例2: 二分类
    Y_soft_map2 = {
        0: [0.0, 0.25, 0.5],
        1: [0.5, 0.75, 1.0]
    }
    Y_hat2 = np.array([0.1, 0.6, 0.9, 0.3])
    Y_pred2 = result_map(Y_hat2, Y_soft_map2)
    print("\n测试用例2 - 二分类:")
    print(f"预测的软化值: {Y_hat2}")
    print(f"映射后的标签: {Y_pred2}")
    
    # 测试用例3: 边界情况
    Y_hat3 = np.array([-0.1, 2.0])  # 超出范围的值
    Y_pred3 = result_map(Y_hat3, Y_soft_map1)
    print("\n测试用例3 - 边界情况:")
    print(f"预测的软化值: {Y_hat3}")
    print(f"映射后的标签: {Y_pred3}")


def test_gen_soft_label_map():
    """测试软化标签映射生成函数"""
    print("\n测试 gen_soft_label_map 函数:")
    
    # 测试用例1: 标准三分类
    labels1 = [0, 1, 2]
    n1 = 3
    result1 = gen_soft_label_map(labels1, n1)
    print(f"三分类 (n=3): {result1}")
    
    # 测试用例2: 二分类
    labels2 = [0, 1]
    n2 = 4
    result2 = gen_soft_label_map(labels2, n2)
    print(f"二分类 (n=4): {result2}")
    
    # 测试用例3: 非标准标签值
    labels3 = [2, 4, 6]
    n3 = 2
    result3 = gen_soft_label_map(labels3, n3)
    print(f"非标准标签值 (n=2): {result3}")

def test_gen_bins():
    """测试分箱边界生成函数"""
    print("\n测试 gen_bins 函数:")
    
    # 测试用例1: 标准范围
    bins1 = gen_bins(0, 100, 5)
    print(f"标准范围 [0,100], 5个箱子: {bins1}")
    
    # 测试用例2: 负值范围
    bins2 = gen_bins(-50, 50, 4)
    print(f"负值范围 [-50,50], 4个箱子: {bins2}")
    
    # 测试用例3: 小范围
    bins3 = gen_bins(0, 1, 3)
    print(f"小范围 [0,1], 3个箱子: {bins3}")

def test_get_bin_indices():
    """测试箱子索引获取函数"""
    print("\n测试 get_bin_indices 函数:")
    
    bins = [0, 25, 50, 75, 100]
    
    # 测试用例1: 标准值
    values1 = np.array([20, 40, 60, 80])
    indices1 = get_bin_indices(values1, bins)
    print(f"标准值 {values1}: {indices1}")
    
    # 测试用例2: 边界值
    values2 = np.array([0, 25, 50, 75, 100])
    indices2 = get_bin_indices(values2, bins)
    print(f"边界值 {values2}: {indices2}")
    
    # 测试用例3: 超出范围的值
    values3 = np.array([-10, 110])
    indices3 = get_bin_indices(values3, bins)
    print(f"超出范围值 {values3}: {indices3}")

def test_gen_Y_soft():
    """测试软化标签生成函数"""
    print("\n测试 gen_Y_soft 函数:")
    
    # 生成测试数据
    np.random.seed(42)  # 设置随机种子以确保可重复性
    n_samples = 10
    n_features = 5
    
    # 测试用例1: 三分类
    X_client1 = np.random.randn(n_samples, n_features)
    X_host1 = np.random.randn(n_samples, n_features)
    Y1 = np.array([0, 1, 2, 1, 0, 2, 1, 0, 2, 1])
    
    Y_soft1 = gen_Y_soft(X_client1, X_host1, Y1, n_soft_labels=5)
    print("三分类测试:")
    print(f"原始标签: {Y1}")
    print(f"软化标签: {Y_soft1}")
    print(f"软化标签范围: [{np.min(Y_soft1)}, {np.max(Y_soft1)}]")
    
    # 测试用例2: 二分类
    Y2 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    Y_soft2 = gen_Y_soft(X_client1, X_host1, Y2, n_soft_labels=3)
    print("\n二分类测试:")
    print(f"原始标签: {Y2}")
    print(f"软化标签: {Y_soft2}")
    print(f"软化标签范围: [{np.min(Y_soft2)}, {np.max(Y_soft2)}]")

def run_all_tests():
    """运行所有测试"""
    print("开始运行测试...")
    
    test_gen_soft_label_map()
    test_gen_bins()
    test_get_bin_indices()
    test_gen_Y_soft()
    test_result_map()
    
    print("\n所有测试完成!")

if __name__ == "__main__":
    run_all_tests()

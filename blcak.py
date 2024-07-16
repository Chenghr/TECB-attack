import copy
import random

class DummyDataset:
    def __init__(self, targets):
        self.targets = targets

def set_shuffle_dataset(args, trainset):
    shuffle_trainset = copy.deepcopy(trainset)
    original_labels = shuffle_trainset.targets 

    if args.shuffle_label_way == "class_to_class":
        print("Randomly replace the label of a certain class with the label of another class")
        
        # 获取所有唯一的标签
        unique_labels = list(set(original_labels))
        num_labels = len(unique_labels)
        
        # 创建一个新的标签列表，确保新标签与原始标签不同
        new_labels = unique_labels[:]
        random.shuffle(new_labels)
        
        # 保证新标签和原标签不同
        for i in range(num_labels):
            if unique_labels[i] == new_labels[i]:
                swap_index = (i + 1) % num_labels
                new_labels[i], new_labels[swap_index] = new_labels[swap_index], new_labels[i]
        
        # 创建新的标签字典，将每个原始标签映射到一个新的唯一标签
        new_labels_mapping = {original: new for original, new in zip(unique_labels, new_labels)}
        
        # 将所有标签替换为新的标签
        shuffle_trainset.targets = [new_labels_mapping[label] for label in original_labels]
        
        # 记录扰乱前后标签的对应关系
        print("Original to new label mapping:")
        for original_label, new_label in new_labels_mapping.items():
            print(f"{original_label} -> {new_label}")

    else:
        print("Randomly replace the label of each sample with the label of other samples")
        shuffle_labels = shuffle_trainset.targets  
        random.shuffle(shuffle_labels)
        shuffle_trainset.targets = shuffle_labels

    return shuffle_trainset

# 定义测试函数
def test_set_shuffle_dataset():
    class Args:
        shuffle_label_way = "class_to_class"
    
    # 生成20个样本的数据集，标签从0到9循环
    original_labels = [i % 10 for i in range(20)]
    trainset = DummyDataset(targets=original_labels)
    
    # 记录打乱前的标签
    original_labels_str = "Original labels: {}".format(trainset.targets)
    
    # 调用函数打乱标签
    shuffled_trainset = set_shuffle_dataset(Args, trainset)
    
    # 显示打乱前的标签
    print(original_labels_str)
    # 显示打乱后的标签
    print("Shuffled labels: {}".format(shuffled_trainset.targets))

# 运行测试函数
test_set_shuffle_dataset()
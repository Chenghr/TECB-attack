import matplotlib.pyplot as plt


class ShuffleTrainMetric(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.shuffle_train_loss = []
        self.shuffle_test_loss = []
        self.main_task_loss = []
        self.backdoor_attack_loss = []
        self.main_task_acc = []
        self.main_task_top5_acc = []
        self.backdoor_task_acc = []

        self.train_top1_acc = []
        self.train_top5_acc = []
        self.test_top1_acc = []
        self.test_top5_acc = []

        self.asr_acc = []

    def update(self, shuffle_train_loss=None, shuffle_test_loss=None, main_task_loss=None, backdoor_task_loss=None,
               main_task_acc=None, main_task_top5_acc=None, backdoor_task_acc=None,
               train_top1_acc=None, train_top5_acc=None, test_top1_acc=None, test_top5_acc=None, asr_acc=None
        ):
        if shuffle_train_loss is not None:
            self.shuffle_train_loss.append(shuffle_train_loss)
        if shuffle_test_loss is not None:
            self.shuffle_test_loss.append(shuffle_test_loss)
        if main_task_loss is not None:
            self.main_task_loss.append(main_task_loss)
        if backdoor_task_loss is not None:
            self.backdoor_attack_loss.append(backdoor_task_loss)
        
        if main_task_acc is not None:
            self.main_task_acc.append(main_task_acc)
        if main_task_top5_acc is not None:
            self.main_task_top5_acc.append(main_task_top5_acc)
        if backdoor_task_acc is not None:
            self.backdoor_task_acc.append(backdoor_task_acc)

        if train_top1_acc is not None:
            self.train_top1_acc.append(train_top1_acc)
        if train_top5_acc is not None:
            self.train_top5_acc.append(train_top5_acc)
        if test_top1_acc is not None:
            self.test_top1_acc.append(test_top1_acc)
        if test_top5_acc is not None:
            self.test_top5_acc.append(test_top5_acc)
        
        if asr_acc is not None:
            self.asr_acc.append(asr_acc)    

    def plot(self, metrics):
        for metric in metrics:
            if hasattr(self, metric):
                values = getattr(self, metric)
                if values:  # Only plot if there are values in the list
                    plt.plot(values, label=metric)
        
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend()
        plt.title('Metrics Over Epochs')
        plt.show()

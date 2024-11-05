import torch.nn as nn
import torch.nn.functional as F

from fedml_core.model.utils import TorchModuleType, make_nn_module



class LeNet(nn.Module):
    def __init__(
        self,
        in_channel,
        num_classes=10,
        activation: TorchModuleType = "ReLU",
    ):
        super(LeNet, self).__init__()
        self.activation = make_nn_module(activation)
        self.num_classes = num_classes
        # input size should be [in_channel, 32, 32]
        self.conv1 = nn.Conv2d(in_channel, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(80, 120)  # half image
        # self.fc1 = nn.Linear(400, 120)   # full image
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.activation((self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = self.activation((self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.activation((self.fc1(out)))
        out = self.activation((self.fc2(out)))
        out = self.fc3(out)  # logits
        return out
    

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

import torch.nn.functional as F
from fedml_core.model.modelzoo.resnet import resnet20
from fedml_core.model.modelzoo.vgg import VGG16
from fedml_core.model.modelzoo.lenet import LeNet


def weights_init(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class BottomModelForCifar100(nn.Module):
    def __init__(self, model_name="resnet20"):
        super(BottomModelForCifar100, self).__init__()
        if model_name == "resnet20":
            self.model = resnet20(in_channel=3, num_classes=10)
        elif model_name == "vgg16":
            self.model = VGG16(in_channel=3, num_classes=10)
        elif model_name == "LeNet":
            self.model = LeNet(in_channel=3, num_classes=10)

    def forward(self, x):
        x = self.model(x)
        return x


class TopModelForCifar100(nn.Module):
    def __init__(self):
        super(TopModelForCifar100, self).__init__()
        self.fc1top = nn.Linear(200, 200)
        self.fc2top = nn.Linear(200, 100)
        self.fc3top = nn.Linear(100, 100)
        self.fc4top = nn.Linear(100, 100)
        self.bn0top = nn.BatchNorm1d(200)
        self.bn1top = nn.BatchNorm1d(200)
        self.bn2top = nn.BatchNorm1d(100)
        self.bn3top = nn.BatchNorm1d(100)

        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)


class BottomModelForCinic10(nn.Module):
    def __init__(self, model_name="resnet20"):
        super(BottomModelForCinic10, self).__init__()
        if model_name == "resnet20":
            self.model = resnet20(in_channel=3, num_classes=10)
        elif model_name == "vgg16":
            self.model = VGG16(in_channel=3, num_classes=10)
        elif model_name == "LeNet":
            self.model = LeNet(in_channel=3, num_classes=10)

    def forward(self, x):
        x = self.model(x)
        return x


class TopModelForCinic10(nn.Module):
    def __init__(self):
        super(TopModelForCinic10, self).__init__()
        self.fc1top = nn.Linear(20, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 10)
        self.bn0top = nn.BatchNorm1d(20)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)

        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)


class BottomModelForCifar10(nn.Module):
    def __init__(self, model_name="resnet20"):
        super(BottomModelForCifar10, self).__init__()
        if model_name == "resnet20":
            self.model = resnet20(in_channel=3, num_classes=10)
        elif model_name == "vgg16":
            self.model = VGG16(in_channel=3, num_classes=10)
        elif model_name == "LeNet":
            self.model = LeNet(in_channel=3, num_classes=10)

    def forward(self, x):
        x = self.model(x)
        return x

class TopModelForCifar10(nn.Module):
    def __init__(self):
        super(TopModelForCifar10, self).__init__()
        self.fc1top = nn.Linear(20, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 10)
        self.bn0top = nn.BatchNorm1d(20)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)

        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)


class LocalClassifierForCifar10(nn.Module):
    def __init__(self):
        super(TopModelForCifar10, self).__init__()
        self.fc1top = nn.Linear(10, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 10)
        self.bn0top = nn.BatchNorm1d(10)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)

        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)


class VanillaBottomModelForCifar10(nn.Module):
    def __init__(self):
        super(VanillaBottomModelForCifar10, self).__init__()
        self.resnet20 = resnet20(num_classes=20)

    def forward(self, x):
        x = self.resnet20(x)
        return x


class VanillaTopModelForCifar10(nn.Module):
    def __init__(self):
        super(VanillaTopModelForCifar10, self).__init__()
        self.fc1top = nn.Linear(20, 10)
        self.fc2top = nn.Linear(10, 10)
        self.fc3top = nn.Linear(10, 10)
        self.bn0top = nn.BatchNorm1d(20)
        self.bn1top = nn.BatchNorm1d(10)
        self.bn2top = nn.BatchNorm1d(10)

        self.apply(weights_init)

    def forward(self, input_tensor_top_model):
        x = input_tensor_top_model
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        return F.log_softmax(x, dim=1)


class active_model(nn.Module):
    def __init__(self, input_dim, num_classes, k=2):
        super(active_model, self).__init__()
        self.classifier = nn.Linear(in_features=input_dim * k, out_features=num_classes)
    def forward(self, input, U_B):
        #out = torch.cat([input] + [U for U in U_B], dim=1)
        out = torch.cat((input, U_B), dim=1)
        #out = F.leaky_relu(out)
        out = F.relu(out)
        logits = self.classifier(out)
        return logits


class kmeans_classifier(nn.Module):
    def __init__(self, input_dim, num_classes,  use_bn=False):
        super(kmeans_classifier, self).__init__()
        self.use_bn = use_bn
        self.classifier = nn.Linear(in_features=input_dim, out_features=num_classes)
        self.bn_final = nn.BatchNorm1d(input_dim)
    def forward(self, U_B):
        if self.use_bn:
            U_B = self.bn_final(U_B)
        out = F.leaky_relu(U_B)
        logits = self.classifier(out)
        return logits


class passive_model(nn.Module):
    def __init__(self, input_dim, intern_dim, output_dim):
        super(passive_model, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, intern_dim),
            #nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(intern_dim, output_dim),
            #nn.LeakyReLU(),
            nn.ReLU(),
            #nn.Linear(in_features=intern_dim, out_features=output_dim)
        )

    def forward(self, input):
        out = self.mlp(input)
        return out


class AE_model(nn.Module):
    def __init__(self, input_dim, intern_dim, output_dim):
        super(AE_model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, intern_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=intern_dim, out_features=output_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, intern_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=intern_dim, out_features=input_dim)
        )

    def forward(self, input):
        latent = self.encoder(input)
        out = self.decoder(latent)
        return out, latent


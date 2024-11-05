import torch.nn as nn

from fedml_core.model.utils import TorchModuleType, make_nn_module



class VGG(nn.Module):
    # fmt: off
    cfg = {
        "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M_final"],
        "VGG13": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M_final'],  # noqa: E501
        "VGG16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M_final'],  # noqa: E501
        "VGG19": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M_final'],  # noqa: E501
    }
    # fmt: on

    def __init__(
        self,
        vgg_name,
        in_channel,
        num_classes=10,
        activation: TorchModuleType = "ReLU",
        loc=-1.0,
        scale=1.0,
        passport_mode="multi",
    ):
        super(VGG, self).__init__()
        self.in_channel = in_channel
        self.vgg_name = vgg_name
        self.num_classes = num_classes
        self.activation = make_nn_module(activation)
        self.loc = loc
        self.scale = scale
        self.passport_mode = passport_mode

        self.feature_extractor = self._make_layers(VGG.cfg[vgg_name])
        self.classifier = nn.Linear(512, self.num_classes)

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channel
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == "M_final":
                layers += [nn.AdaptiveMaxPool2d(output_size=(1, 1))]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11(in_channel=3, num_classes=10, activation: TorchModuleType = "ReLU"):
    return VGG(
        vgg_name="VGG11",
        in_channel=in_channel,
        num_classes=num_classes,
        activation=activation,
    )


def VGG13(in_channel=3, num_classes=10, activation: TorchModuleType = "ReLU"):
    return VGG(
        vgg_name="VGG13",
        in_channel=in_channel,
        num_classes=num_classes,
        activation=activation,
    )


def VGG16(in_channel=3, num_classes=10, activation: TorchModuleType = "ReLU"):
    return VGG(
        vgg_name="VGG16",
        in_channel=in_channel,
        num_classes=num_classes,
        activation=activation,
    )


def VGG19(in_channel=3, num_classes=10, activation: TorchModuleType = "ReLU"):
    return VGG(
        vgg_name="VGG19",
        in_channel=in_channel,
        num_classes=num_classes,
        activation=activation,
    )

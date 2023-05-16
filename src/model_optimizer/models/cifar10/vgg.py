# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""VGG7 in Pytorch."""
import math
from torch import nn
from torch.utils import model_zoo

__all__ = [
    'VGG', 'cifar10_vggsmall'
]

model_urls = {
    'vgg7': 'https://github.com/rhhc/zxd_releases/releases/download/Re/cifar10-vggsmall-zxd-93.4-8943fa3.pth',
}


def cifar10_vggsmall(pretrained=False, **kwargs):
    """VGG small model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG('VGG7')
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg7']))
    return model


cfg = {
    'VGG7': [128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
}


class VGG(nn.Module):
    """vgg model on cifar10"""
    def __init__(self, vgg_name):
        super().__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512 * 16, 10)
        self._initialize_weights()

    def forward(self, x):  # pylint: disable=missing-function-docstring
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    @staticmethod
    def _make_layers(config):
        layers = []
        in_channels = 3
        for x in config:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

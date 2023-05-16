# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""LeNet in PyTorch."""
from torch import nn
import torch.nn.functional as F
from torch.utils import model_zoo

__all__ = ['LeNet', 'mnist_lenet']
model_urls = {
    'lenet': 'https://github.com/rhhc/zxd_releases/releases/download/Re/mnist_lenet-eff-99.23-3eb8b274.pth'
}


def mnist_lenet(pretrained=False):  # pylint: disable=missing-function-docstring
    model = LeNet()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['lenet'], map_location='cpu'))
    return model


class LeNet(nn.Module):  # pylint: disable=missing-class-docstring
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):  # pylint: disable=missing-function-docstring
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

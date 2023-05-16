# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
model utilities on cifar10 dataset
"""
from .vgg import cifar10_vggsmall  # noqa: F401
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152  # noqa: F401
from .resnet import wide_resnet50_2, wide_resnet101_2  # noqa: F401

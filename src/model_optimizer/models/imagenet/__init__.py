# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
models on imagenet dataset
"""
from .MobileNetV2 import mobilenetv2  # noqa: F401
from .mobilenetv1_sparse import mobilenet_distiller, mobilenet_025, mobilenet_050, mobilenet_075  # noqa: F401
from .yolo import Model, yolov5x, yolov5l, yolov5m, yolov5s, yolov5_backbone, yolov5s_backbone  # noqa: F401
from .yolo import yolov5l_backbone, yolov5m_backbone, yolov5n_backbone, yolov5x_backbone  # noqa: F401

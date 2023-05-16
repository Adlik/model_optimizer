# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
yolo utils
"""
from ..utils import load_subnet, deploy_subnet
from ..models import imagenet as imagenet_extra_models


def get_yolov5_backbone_layers(arch: str, subnet_path: str):
    """
    get yolov5 backbone layers, no classifier
    Args:
        arch: the supernet arch of this subnet, one of "yolov5x","yolov5l","yolov5m","yolov5s","yolov5n"
        subnet_path: the path of the subnet config file

    Returns: yolov5 backbone layers

    """
    yolov5_temp = imagenet_extra_models.__dict__[arch]()
    channel_config = load_subnet(subnet_path)
    del channel_config['3']  # delete classifier
    channel_config_temp = {}
    for key, value in channel_config.items():
        key = key.replace("0.", "", 1)
        channel_config_temp[key] = value

    deploy_subnet(yolov5_temp, channel_config_temp)

    yolov5mtemp_list = list(yolov5_temp.children())
    yolov5m_backbone_layers = yolov5mtemp_list[:10]
    return yolov5m_backbone_layers


def get_yolov5_backbone_layers_width(subnet_path: str, width_multiple=1.0, depth_multiple=1.0):
    """
    get yolov5 backbone layers, no classifier
    Args:
        subnet_path: the path of the subnet config file

    Returns: yolov5 backbone layers

    """
    yolov5_temp = imagenet_extra_models.__dict__["yolov5_backbone"](width_multiple=width_multiple,
                                                                    depth_multiple=depth_multiple)[0]
    print("********** yolov5_temp", yolov5_temp)
    channel_config = load_subnet(subnet_path)
    del channel_config['3']  # delete classifier
    channel_config_temp = {}
    for key, value in channel_config.items():
        key = key.replace("0.", "", 1)
        channel_config_temp[key] = value

    deploy_subnet(yolov5_temp, channel_config_temp)

    yolov5temp_list = list(yolov5_temp.children())
    yolov5_backbone_layers = yolov5temp_list[:10]
    return yolov5_backbone_layers

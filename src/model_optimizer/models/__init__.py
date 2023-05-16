# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
models utilities
"""
import json
from collections import OrderedDict
import torch
from torch import nn
import torchvision
import timm
from pytorchcv.model_provider import get_model as ptcv_get_model
from ..proto import model_optimizer_torch_pb2 as eppb
from ..algorithms.autoslim.autoslim import AutoSlim
from ..pruners.ratio_pruning import RatioPruner
from ..models import imagenet as imagenet_extra_models
from ..utils import deploy_net_by_ratio


def load_pre_state_dict(model, original_state_dict, key_map=None):  # pylint: disable=missing-function-docstring
    if not isinstance(key_map, OrderedDict):
        with open(f'models/weight_keys_map/{key_map}', encoding='utf-8') as file:
            key_map = json.load(file)
    for key, value in key_map.items():
        if 'num_batches_tracked' in key:
            continue
        else:
            print(f'{key} <== {value}')
            model.state_dict()[key].copy_(original_state_dict[value])


def get_model_from_source(arch, source, pretrained=True, width_mult=1.0, depth_mult=1.0,
                          load_quantize_model=False, is_subnet=False, channel_config_path=None):
    """

    Args:
        arch (str): for example "resnet50"
        source (enum): eppb.HyperParam.ModelSource.[TorchVision|PyTorchCV|Timm|Local]
        pretrained (boolean): default True
        num_classes(int): default 1000
        width_mult (float): Width multiplier - some model like mobilenet_v2 adjusts number of channels in
                    each layer by this amount
        depth_mult(float): depth multiplier - some model like yolov5 adjusts number of depth in by this amount
        load_quantize_model (boolean): A boolean indicating if we load quantize defined model or not.
        is_subnet(boolean): if load model arch for autoslim subnet
        channel_config_path (str): used when is_subnet is True

    Returns:
        model

    """
    if source == eppb.HyperParam.ModelSource.TorchVision:
        if arch == "mobilenet_v2":
            if load_quantize_model:
                model = torchvision.models.quantization.__dict__[arch](pretrained=pretrained,
                                                                       width_mult=width_mult)
            else:
                model = torchvision.models.__dict__[arch](pretrained=pretrained,
                                                          width_mult=width_mult)
        else:
            if load_quantize_model:
                model = torchvision.models.quantization.__dict__[arch](pretrained=pretrained)
            else:
                model = torchvision.models.__dict__[arch](pretrained=pretrained)
    elif source == eppb.HyperParam.ModelSource.PyTorchCV:
        model = ptcv_get_model(arch, pretrained=pretrained)
    elif source == eppb.HyperParam.ModelSource.Timm:
        model = timm.create_model(arch, pretrained=pretrained)
    elif source == eppb.HyperParam.ModelSource.Local:
        if arch == "yolov5_backbone":
            model = imagenet_extra_models.__dict__[arch](pretrained=pretrained,
                                                         width_multiple=width_mult,
                                                         depth_multiple=depth_mult)
        else:
            model = imagenet_extra_models.__dict__[arch](pretrained=pretrained)

    if is_subnet:
        print("=> channel_config_path", channel_config_path)
        channel_config = AutoSlim.load_subnet(channel_config_path)
        RatioPruner.deploy_subnet(model, channel_config)
    return model


def get_model_from_source_ratio(arch, source, pretrained=True, width_mult=1.0, depth_mult=1.0,
                                channel_config_path=None, input_layer="conv1", output_layer="fc"):
    """Load model by width_mult and channel_config_path for those models like resnet50

    Args:
        arch (str): for example "resnet50"
        source (enum): eppb.HyperParam.ModelSource.[TorchVision|PyTorchCV|Timm|Local]
        pretrained (boolean): default True
        width_mult (float): Width multiplier - some model like mobilenet_v2 adjusts number of channels in
                    each layer by this amount
        depth_mult (float): depth multiplier - some model like yolov5 adjusts number of depth in by this amount
        channel_config_path (str): yaml configuration file path to configure channel information
        output_layer:

    Returns:
        model

    """
    model = get_model_from_source(arch, source, pretrained, width_mult, depth_mult)
    if arch not in ["mobilenet_v2", "yolov5_backbone"]:
        print("=> channel_config_path", channel_config_path)
        channel_config = AutoSlim.load_subnet(channel_config_path)

        deploy_net_by_ratio(model, channel_config, ratio=width_mult, input_layer=input_layer, output_layer=output_layer)

    return model


def get_teacher_model(models_arch, models_source):
    """ Loads an ensemble of models or a single model.

    Args:
        models_arch: list, for example,models_arch=["resnet50", "resnet101d"] or ["resnet50"]
        models_source: list, for example, models_source=["TorchVision", "Timm"] or ["TorchVision"]

    Returns:
        Ensemble model
    """

    assert len(models_arch) == len(models_source), "The number of models_arch and models_source is not equal!"

    ensemble_model = Ensemble()
    for (arch, model_source) in zip(models_arch, models_source):
        teacher_model = get_model_from_source(arch, model_source)
        ensemble_model.append(teacher_model)

    return ensemble_model


class Ensemble(nn.ModuleList):
    """Ensemble of models"""

    def __init__(self, ensemble_mode="mean"):
        super().__init__()
        self.ensemble_mode = ensemble_mode

    def forward(self, x):
        """
        forward
        Args:
            x:

        Returns:

        """
        ensemble_model_output = []
        for module in self:
            ensemble_model_output.append(module(x))

        if self.ensemble_mode == "mean":
            ensemble_model_output = torch.stack(ensemble_model_output).mean(0)  # mean ensemble
        elif self.ensemble_mode == "max":
            ensemble_model_output = torch.stack(ensemble_model_output).max(0)[0]  # max ensemble

        return ensemble_model_output

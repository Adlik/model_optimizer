# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
function utils
"""
import os
from typing import Optional
import copy
import yaml
import torch
from torch import nn


def load_subnet(config_path):
    """
    load subnet
    Args:
        config_path:

    Returns:

    """
    file_format = config_path.split('.')[-1]
    if file_format in ["yaml", "yml"]:
        print(f"loading subnet config from {config_path}")
        with open(config_path, encoding="utf-8") as subnet_file:
            channel_config = yaml.load(subnet_file.read(), Loader=yaml.FullLoader)
            return channel_config
    else:
        raise NotImplementedError("Only yaml or yml file format of channel_config_path is supportted")


def align_channel(channel_config, round_nearest=8, except_align_keys=""):
    """

    Args:
        channel_config:
        round_nearest:
        except_align_keys: the list of layer name which is not aligned, e.g. input layer

    Returns:

    """
    for key, item in channel_config.items():
        if "in_channels" in item:
            if key not in except_align_keys:
                in_channels = item['in_channels']
                in_channels_align = make_divisible(in_channels, round_nearest)
                item['in_channels'] = in_channels_align
        if "out_channels" in item:
            out_channels = item['out_channels']
            out_channels_align = make_divisible(out_channels, round_nearest)
            item['out_channels'] = out_channels_align
    return channel_config


def align(config_path, except_align_keys):
    """

    Args:
        config_path:
        except_align_keys: the list of the layer name which is not aligned, usually input

    Returns:

    """
    align_file_path = os.path.splitext(config_path)[0] + "_align" + os.path.splitext(config_path)[-1]
    channel_config = load_subnet(config_path)
    aligned = align_channel(channel_config, 8, except_align_keys)
    with open(align_file_path, 'w+', encoding="utf-8") as outfile:
        print(f"saving aligned config to {align_file_path}")
        # print(f"aligned config: {aligned}")
        yaml.dump(aligned, outfile, default_flow_style=False)


def make_divisible(value: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _update_module_attr_with_out_channels(module, out_channels):
    if hasattr(module, 'out_channels'):
        module.out_channels = out_channels
    if hasattr(module, 'out_features'):
        module.out_features = out_channels
    if hasattr(module, 'num_features'):
        module.num_features = out_channels
    if hasattr(module, 'out_mask'):
        module.out_mask = module.out_mask[:, :out_channels]


def _update_module_attr_with_in_channels(module, weight, in_channels):
    if in_channels > 1:
        weight_sub = weight[:, :in_channels].data
    if hasattr(module, 'in_channels'):
        module.in_channels = in_channels
    if hasattr(module, 'in_features'):
        module.in_features = in_channels
    if hasattr(module, 'in_mask'):
        module.in_mask = module.in_mask[:, :in_channels]
    # TODO Seems not support GroupConv
    if getattr(module, 'groups', in_channels) > 1:
        module.groups = in_channels
    return weight_sub


def deploy_subnet(supernet, channel_cfg):
    """Deploy subnet according `channel_cfg`."""
    for name, module in supernet.named_modules():
        if name not in channel_cfg:
            continue

        channels_per_layer = channel_cfg[name]
        requires_grad = module.weight.requires_grad
        out_channels = channels_per_layer['out_channels']
        temp_weight = module.weight.data[:out_channels]

        _update_module_attr_with_out_channels(module, out_channels)

        if 'in_channels' in channels_per_layer:
            in_channels = channels_per_layer['in_channels']

            temp_weight = _update_module_attr_with_in_channels(module, temp_weight, in_channels)

        module.weight = nn.Parameter(temp_weight.data)
        module.weight.requires_grad = requires_grad

        if hasattr(module, 'bias') and module.bias is not None:
            module.bias = nn.Parameter(module.bias.data[:out_channels])
            module.bias.requires_grad = requires_grad

        if hasattr(module, 'running_mean'):
            module.running_mean = module.running_mean[:out_channels]

        if hasattr(module, 'running_var'):
            module.running_var = module.running_var[:out_channels]


def deploy_net_by_ratio(model, channel_cfg, ratio=1, round_nearest=8, input_layer="conv1", output_layer="fc"):
    """
    Deploy subnet according `channel_cfg`.
    Args:
        model:
        channel_cfg:
        ratio:
        round_nearest:
        input_layer:
        output_layer:

    Returns:

    """
    for name, module in model.named_modules():
        if name not in channel_cfg:
            continue

        channels_per_layer = channel_cfg[name]
        requires_grad = module.weight.requires_grad
        if name != output_layer:
            out_channels = make_divisible(channels_per_layer['raw_out_channels'] * ratio, round_nearest)
        else:
            out_channels = channels_per_layer['raw_out_channels']
        temp_weight = module.weight.data[:out_channels]

        _update_module_attr_with_out_channels(module, out_channels)

        if 'in_channels' in channels_per_layer:
            if name != input_layer:
                in_channels = make_divisible(channels_per_layer['raw_in_channels'] * ratio, round_nearest)
            else:
                in_channels = channels_per_layer['raw_in_channels']

            temp_weight = _update_module_attr_with_in_channels(module, temp_weight, in_channels)

        module.weight = nn.Parameter(temp_weight.data)
        module.weight.requires_grad = requires_grad

        if hasattr(module, 'bias') and module.bias is not None:
            module.bias = nn.Parameter(module.bias.data[:out_channels])
            module.bias.requires_grad = requires_grad

        if hasattr(module, 'running_mean'):
            module.running_mean = module.running_mean[:out_channels]

        if hasattr(module, 'running_var'):
            module.running_var = module.running_var[:out_channels]


def _parent_name(target):
    """
    Turn 'foo.bar' into ['foo', 'bar']
    Args:
        target: str

    Returns:

    """
    r = target.rsplit('.', 1)
    if len(r) == 1:
        return '', r[0]
    else:
        return r[0], r[1]


def deepcopy_graphmodule(gm: torch.fx.GraphModule):  # type: ignore[name-defined]
    """
    Rewrite the deepcopy of GraphModule. (Copy its 'graph'.)
    Args:
        gm (GraphModule):

    Returns:
        GraphModule: A deepcopied gm.
    """
    copied_gm = copy.deepcopy(gm)
    copied_gm.graph = copy.deepcopy(gm.graph)
    return copied_gm


def replace_module(new_model: nn.Module, model: nn.Module, old_module, new_module, prefix=''):
    '''
    Args:
        new_model: nn.Module, after changed model
        model: nn.Module, needed to change model
        old_module: the target module that need to be replaced
        new_module: the new module to replace old_module
        prefix:

    Returns:

    '''
    assert isinstance(new_module, type(new_module)), f'need to initialize the {new_module}'

    for name, module in model.named_children():
        if prefix == '':
            new_prefix = name
        else:
            new_prefix = prefix + '.' + name

        if len(list(module.named_children())) > 0:
            replace_module(new_model, module, old_module, new_module, new_prefix)

        if isinstance(module, old_module):
            print(f'replace {new_prefix} with {new_module}')
            modules = dict(new_model.named_modules())
            setattr(modules[prefix], name, new_module)

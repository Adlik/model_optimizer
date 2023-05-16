# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
quantizer utilities
"""
import copy
from typing import Dict, Any, Tuple
import math
import torch
from model_optimizer.proto import model_optimizer_torch_pb2 as eppb
from .histogram_extend import HistogramExtendObserver, PerChannelHistogramExtendObserver
from .fake_quantize import LearnableFakeQuantize
from .prepare_fx_with_backend import prepare_fx_with_backend  # noqa: F401
from .tensorrt_quantizer import prepare_tensorrt_quantizer  # noqa: F401
from .fusion_convert import fuse_prepared_model  # noqa: F401
from .deploy_fx_with_backend import convert_model_by_backend, get_convert_fx_model  # noqa: F401


observer_map = {
    "quantization_error": torch.quantization.observer.HistogramObserver,
    "moving_average_minmax": torch.quantization.observer.MovingAverageMinMaxObserver,
    "minmax": torch.quantization.observer.MinMaxObserver,
    "percentile": HistogramExtendObserver
}

observer_perchannel_map = {
    "moving_average_minmax": torch.quantization.observer.MovingAveragePerChannelMinMaxObserver,
    "minmax": torch.quantization.observer.PerChannelMinMaxObserver,
    "percentile": PerChannelHistogramExtendObserver
}

dtype_map = {
    "quint8": torch.quint8,
    "qint8": torch.qint8,
    "qint32": torch.qint32
}


def _get_qscheme(observer_args):
    if observer_args.per_channel:
        if observer_args.symmetric:
            qscheme = torch.per_channel_symmetric
        else:
            qscheme = torch.per_channel_affine
    else:
        if observer_args.symmetric:
            qscheme = torch.per_tensor_symmetric
        else:
            qscheme = torch.per_tensor_affine
    return qscheme


# pylint: disable=too-many-branches
def get_observer(observer_args, backend, post_training_quantize=True, restrict_to_8bit=False, out_channels=None):
    """

    Args:
        observer_args:
        post_training_quantize:
        restrict_to_8bit:
        out_channels:

    Returns:

    """
    if post_training_quantize and observer_args.quantization_method not in \
            ["quantization_error", "moving_average_minmax", "minmax", "percentile"]:
        raise NotImplementedError
    if (not post_training_quantize) and observer_args.quantization_method != "moving_average_minmax":
        if observer_args.fake_method != "lsq":
            raise NotImplementedError
    if observer_args.per_channel:
        observer_func = observer_perchannel_map[observer_args.quantization_method]
    else:
        observer_func = observer_map[observer_args.quantization_method]

    if observer_args.quantization_method == 'percentile':
        observer_func = observer_func.with_args(percentile=observer_args.percentile)

    if observer_args.per_channel:
        if observer_args.symmetric:
            qscheme = torch.per_channel_symmetric
        else:
            qscheme = torch.per_channel_affine
    else:
        if observer_args.symmetric:
            qscheme = torch.per_tensor_symmetric
        else:
            qscheme = torch.per_tensor_affine

    if restrict_to_8bit:
        nbits = 8
    else:
        nbits = observer_args.nbits
    if dtype_map[observer_args.dtype] == torch.qint8:
        quant_min = -2 ** (nbits - 1)
        quant_max = 2 ** (nbits - 1) - 1
    else:
        quant_min = 0
        quant_max = 2 ** nbits - 1

    if backend == eppb.InferenceBackend.TENSORRT:
        assert qscheme in [torch.per_channel_symmetric, torch.per_tensor_symmetric], "Tensorrt must use symmetric"
        observer_args.reduce_range = False
        if nbits == 8 and dtype_map[observer_args.dtype] == torch.qint8:
            quant_min = -127
            quant_max = 127

    if observer_args.fake_method == "basic":
        if observer_args.quantization_method == 'moving_average_minmax':
            observer = torch.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize.with_args(
                observer=observer_func,
                quant_min=quant_min,
                quant_max=quant_max,
                dtype=dtype_map[observer_args.dtype],
                qscheme=qscheme,
                reduce_range=observer_args.reduce_range)
        else:
            observer = torch.quantization.fake_quantize.FakeQuantize.with_args(
                observer=observer_func,
                quant_min=quant_min,
                quant_max=quant_max,
                dtype=dtype_map[observer_args.dtype],
                qscheme=qscheme,
                reduce_range=observer_args.reduce_range)
    else:  # lsq
        if observer_args.per_channel and out_channels is not None:
            _out_channels = out_channels
        else:
            _out_channels = 1
        observer = LearnableFakeQuantize.with_args(
            observer=observer_func,
            quant_min=quant_min,
            quant_max=quant_max,
            dtype=dtype_map[observer_args.dtype],
            qscheme=qscheme,
            reduce_range=observer_args.reduce_range,
            out_channels=_out_channels)

    return observer


# pylint: enable=too-many-branches
def get_lsq_qconfig(model, qargs, backend):  # pylint: disable=missing-function-docstring
    qconfig = get_qconfig(qargs, backend, weight_restrict_to_8bit=False, activation_restrict_to_8bit=False,
                          out_channels=None)
    qconfig_dict = {"": qconfig}
    module_name_qconfig = get_layers_restrict_to_8bit_qconfig(model, qargs, backend)
    qconfig_dict.update({'module_name': module_name_qconfig})
    return qconfig_dict


def get_layers_restrict_to_8bit_qconfig(model, qargs, backend):  # pylint: disable=missing-function-docstring
    module_name_qconfig = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear)):
            activation_restrict_to_8bit = False
            weight_restrict_to_8bit = False
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                out_channels = module.out_channels
            elif isinstance(module, torch.nn.modules.linear.Linear):
                out_channels = module.out_features
            if name in qargs.activation_quantization_observer.layers_restrict_to_8bit.split(','):
                activation_restrict_to_8bit = True
            if name in qargs.weight_quantization_observer.layers_restrict_to_8bit.split(','):
                weight_restrict_to_8bit = True
            qconfig = get_qconfig(qargs, backend, weight_restrict_to_8bit=weight_restrict_to_8bit,
                                  activation_restrict_to_8bit=activation_restrict_to_8bit,
                                  out_channels=out_channels)
            module_name_qconfig.append((name, qconfig))
    return module_name_qconfig


def get_qconfig_dict(model, qargs, backend, extra_qconfig_dict=None):
    """

    Args:
        model:
        qargs:
        backend:
        extra_qconfig_dict (dict): qconfig_dict is a dictionary with the following configurations:
            qconfig_dict = {
                # optional, used for module names
                "module_name": [
                  ("foo.bar", qconfig?)
                  ...,
                ],

    Returns:
        qconfig_dict (dict)
    """
    qconfig = get_qconfig(qargs, backend)
    qconfig_dict = {"": qconfig}
    module_name_qconfig = get_layers_restrict_to_8bit_qconfig(model, qargs, backend)
    qconfig_dict.update({'module_name': module_name_qconfig})

    if extra_qconfig_dict is not None:
        module_name = extra_qconfig_dict.get("module_name", None)
        if module_name:
            qconfig_dict["module_name"].extend(module_name)
    return qconfig_dict


def get_qconfig(qargs, backend, weight_restrict_to_8bit=False, activation_restrict_to_8bit=False, out_channels=None):
    """
    get qconfig
    Args:
        qargs:
        backend:
        weight_restrict_to_8bit:
        activation_restrict_to_8bit:
        out_channels:

    Returns:

    """
    activation_observer = get_observer(qargs.activation_quantization_observer, backend,
                                       qargs.post_training_quantize,
                                       activation_restrict_to_8bit)
    weight_observer = get_observer(qargs.weight_quantization_observer, backend,
                                   qargs.post_training_quantize,
                                   weight_restrict_to_8bit, out_channels)
    qconfig = torch.quantization.QConfig(activation=activation_observer,
                                         weight=weight_observer)

    return qconfig


def _parent_name(target: str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name


def replace_node_module(node: torch.fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    """

    Args:
        node:
        modules:
        new_module:

    Returns:

    """
    assert isinstance(node.target, str)
    parent_name, name = _parent_name(node.target)
    modules[node.target] = new_module
    setattr(modules[parent_name], name, new_module)


def replace_activation_with_observer(model, activation_observer):
    """

    Args:
        model:
        activation_observer:

    Returns:

    """
    new_graph = copy.deepcopy(model.graph)
    modules = dict(model.named_modules())
    list_nodes = list(new_graph.nodes)
    node_1 = list_nodes[1]
    replace_node_module(node_1, modules, activation_observer)
    new_model = torch.quantization.fx.graph_module.ObservedGraphModule(model, new_graph, model.preserved_attr_names)
    return new_model


def _get_weight_bit_info(observer):
    quant_max = max(observer.quant_max, 0)
    quant_min = min(observer.quant_min, 0)
    dtype = observer.dtype
    nbits = int(math.ceil(math.log2((-1.0 * quant_min) + quant_max + 1)))
    if dtype == torch.qint8:
        type_str = 'qint8'
    else:
        type_str = 'quint8'
    return nbits, type_str


# pylint: disable=protected-access
def _get_node_bit_info(model, node):
    """

    Args:
        model:
        node:

    Returns:

    """
    if node.op == "call_module":
        submodel = model.get_submodule(node.target)
        if isinstance(submodel, (torch.ao.quantization.observer._ObserverBase,
                                 torch.quantization.FakeQuantizeBase)):
            quant_max = max(submodel.quant_max, 0)
            quant_min = min(submodel.quant_min, 0)
            dtype = submodel.dtype
            nbits = int(math.ceil(math.log2((-1.0 * quant_min) + quant_max + 1)))
            if dtype == torch.qint8:
                type_str = 'qint8'
            else:
                type_str = 'quint8'
        else:
            nbits = 32
            type_str = 'float32'
    else:
        node_pre = node.prev
        nbits, type_str = _get_node_bit_info(model, node_pre)
    return nbits, type_str


def get_model_quantize_bit_config(model):
    """

    Args:
        model:

    Returns:

    """
    model_bit_config = {}
    input_graph = model.graph
    qconfig_map = model._qconfig_map
    for node in input_graph.nodes:
        if node.op == "call_module":
            submodel = model.get_submodule(node.target)
            if isinstance(submodel, (torch.nn.quantized._ConvNd, torch.nn.modules.conv._ConvNd,
                                     torch.nn.intrinsic.modules.fused._FusedModule,
                                     torch.nn.quantized.modules.linear.Linear, torch.nn.Linear)):
                module_bit_config = {}
                node_pre = node.args[0]
                node_next = node.next
                input_nbits, input_type_str = _get_node_bit_info(model, node_pre)
                output_nbits, output_type_str = _get_node_bit_info(model, node_next)

                # print(f'node_name:{node.name}, qconfig:{qconfig_map[node.name]}')
                if qconfig_map[node.name] is None:
                    continue
                weight_observer = qconfig_map[node.name].weight()
                weight_nbits, weight_type_str = _get_weight_bit_info(weight_observer)
                module_bit_config['input_bits'] = input_nbits
                module_bit_config['input_dtype'] = input_type_str
                module_bit_config['output_bits'] = output_nbits
                module_bit_config['output_dtype'] = output_type_str
                module_bit_config['weight_bits'] = weight_nbits
                module_bit_config['weight_dtype'] = weight_type_str
                if isinstance(submodel, (torch.nn.quantized.modules.linear.Linear, torch.nn.Linear)):
                    node_name = node.target + '._packed_params'
                else:
                    node_name = node.target
                model_bit_config.update({node_name: module_bit_config})
    return model_bit_config


# pylint: disable=too-many-branches
def clip_model_weight_in_quant_min_max(model, bit_config):
    """
    clip model weight range in [quant_min, quant_max]
    Args:
        model:
        bit_config:

    Returns:

    """
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.quantized._ConvNd, torch.nn.quantized.modules.linear.Linear)):
            if isinstance(module, torch.nn.quantized.modules.linear.Linear):
                _name = name + '._packed_params'
            else:
                _name = name
            weight_bits = bit_config[_name]['weight_bits']
            quant_min = -2 ** (weight_bits - 1)
            quant_max = 2 ** (weight_bits - 1) - 1
            if weight_bits < 8 and bit_config[_name]['weight_dtype'] == 'qint8':
                tensor = module.weight()
                per_channel = tensor.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric)
                bias = module.bias()
                if per_channel:
                    scale = tensor.q_per_channel_scales()
                    zero_point = tensor.q_per_channel_zero_points()
                else:
                    scale = tensor.q_scale()
                    zero_point = tensor.q_zero_point()
                int8_tensor = module.weight().int_repr()
                int8_tensor[int8_tensor > quant_max] = quant_max
                int8_tensor[int8_tensor < quant_min] = quant_min
                if per_channel:
                    sizes = int8_tensor.size()
                    int8_tensor = torch.transpose(int8_tensor.contiguous().view(sizes[0], -1), 0, 1)
                float_tensor = (int8_tensor - zero_point) * scale
                float_tensor = float_tensor.to(torch.float32)
                if per_channel:
                    float_tensor = torch.transpose(float_tensor, 0, 1).contiguous().view(sizes)
                if per_channel:
                    new_tensor = torch.torch.quantize_per_channel(float_tensor, scales=scale, zero_points=zero_point,
                                                                  axis=0, dtype=torch.qint8)
                else:
                    new_tensor = torch.torch.quantize_per_tensor(float_tensor, scale=scale, zero_point=zero_point,
                                                                 dtype=torch.qint8)
                module.set_weight_bias(new_tensor, bias)
# pylint: enable=too-many-branches

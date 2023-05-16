# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Define Linear and Conv module fusion functions.
"""
import torch
from torch import nn
from torch.fx import GraphModule, Node
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval
import torch.nn.intrinsic.qat as nniq
import torch.nn.intrinsic as nni
import torch.nn.qat as nnqat

from model_optimizer.utils import _parent_name, deepcopy_graphmodule
from model_optimizer.utils.registry import register_convert_fucntion, FUSED_MODULE_CONVERT_FUNCTION


def _is_qat_fused_module(model: GraphModule, fused_node: Node):
    qat_fused_module = [nniq.ConvBn1d, nniq.ConvBnReLU1d, nniq.ConvBn2d,
                        nniq.ConvBnReLU2d, nniq.ConvBn3d, nniq.ConvBnReLU3d]
    normal_fused_module = [nni.ConvBn1d, nni.ConvBnReLU1d, nni.ConvBn2d,
                           nni.ConvBnReLU2d, nni.ConvBn3d, nni.ConvBnReLU3d]
    modules = dict(model.named_modules())
    fused_module = modules[fused_node.target]
    if type(fused_module) in qat_fused_module:
        return True
    else:
        assert type(fused_module) in normal_fused_module, 'the module type should be torch.nn.intrinsic'
        return False


@register_convert_fucntion(nniq.LinearBn1d)
def convert_nniq_linearbn(model: GraphModule, fused_node: Node):  # pylint: disable=missing-function-docstring
    modules = dict(model.named_modules())
    fused_module = modules[fused_node.target]
    linear = nn.Linear(fused_module.in_features, fused_module.out_features, fused_module.bias is not None)
    linear.weight = fused_module.weight
    if fused_module.bias is not None:
        linear.bias = fused_module.bias
    fused_linear = fuse_linear_bn_eval(linear.eval(), fused_module.bn)
    fused_linear.qconfig = fused_module.qconfig
    fused_linear = nnqat.Linear.from_float(fused_linear)
    fused_linear.weight_fake_quant = fused_module.weight_fake_quant
    parent_name, name = _parent_name(fused_node.target)
    setattr(modules[parent_name], name, fused_linear)


@register_convert_fucntion(nni.LinearBn1d)
def convert_nni_linearbn(model: GraphModule, fused_node: Node):  # pylint: disable=missing-function-docstring
    modules = dict(model.named_modules())
    fused_module = modules[fused_node.target]
    fused_linear = fuse_linear_bn_eval(fused_module[0], fused_module[1])
    parent_name, name = _parent_name(fused_node.target)
    setattr(modules[parent_name], name, fused_linear)


@register_convert_fucntion(nniq.ConvBn1d)
@register_convert_fucntion(nniq.ConvBn2d)
@register_convert_fucntion(nniq.ConvBn3d)
def convert_nniq_convbn(model: GraphModule, fused_node: Node):  # pylint: disable=missing-function-docstring
    fused_module_class_map = {
        nniq.ConvBn1d: nn.Conv1d,
        nniq.ConvBnReLU1d: nn.Conv1d,
        nniq.ConvBn2d: nn.Conv2d,
        nniq.ConvBnReLU2d: nn.Conv2d,
        nniq.ConvBn3d: nn.Conv3d,
        nniq.ConvBnReLU3d: nn.Conv3d,
    }

    fused_qat_module_class_map = {
        nn.Conv1d: nnqat.Conv1d,
        nn.Conv2d: nnqat.Conv2d,
        nn.Conv3d: nnqat.Conv3d,
    }

    modules = dict(model.named_modules())
    fused_module = modules[fused_node.target]

    conv = fused_module_class_map[type(fused_module)](fused_module.in_channels, fused_module.out_channels,
                                                      fused_module.kernel_size, fused_module.stride,
                                                      fused_module.padding, fused_module.dilation,
                                                      fused_module.groups, fused_module.bias is not None,
                                                      fused_module.padding_mode)
    conv.weight = fused_module.weight
    if fused_module.bias is not None:
        conv.bias = fused_module.bias
    fused_conv = fuse_conv_bn_eval(conv.eval(), fused_module.bn)
    fused_conv.qconfig = fused_module.qconfig
    fused_conv = fused_qat_module_class_map[type(conv)].from_float(fused_conv)  # type: ignore[call-arg]
    fused_conv.weight_fake_quant = fused_module.weight_fake_quant
    parent_name, name = _parent_name(fused_node.target)
    setattr(modules[parent_name], name, fused_conv)


@register_convert_fucntion(nni.ConvBn1d)
@register_convert_fucntion(nni.ConvBn2d)
@register_convert_fucntion(nni.ConvBn3d)
def convert_nni_convbn(model: GraphModule, fused_node: Node):  # pylint: disable=missing-function-docstring
    modules = dict(model.named_modules())
    fused_module = modules[fused_node.target]
    fused_conv = fuse_conv_bn_eval(fused_module[0], fused_module[1])
    parent_name, name = _parent_name(fused_node.target)
    setattr(modules[parent_name], name, fused_conv)


@register_convert_fucntion(nniq.ConvBnReLU1d)
@register_convert_fucntion(nniq.ConvBnReLU2d)
@register_convert_fucntion(nniq.ConvBnReLU3d)
@register_convert_fucntion(nni.ConvBnReLU1d)
@register_convert_fucntion(nni.ConvBnReLU2d)
@register_convert_fucntion(nni.ConvBnReLU3d)
def convert_nniq_convbnrelu(model: GraphModule, fused_node: Node):  # pylint: disable=missing-function-docstring
    is_qat = _is_qat_fused_module(model, fused_node)
    if is_qat:
        convert_nniq_convbn(model, fused_node)
    else:
        convert_nni_convbn(model, fused_node)
    modules = dict(model.named_modules(remove_duplicate=False))
    fused_module = modules[fused_node.target]
    parent_name, _ = _parent_name(fused_node.target)

    # relu = torch.nn.ReLU(inplace=True).train(fused_module.training)
    # new_fused_model = qat_module_class_map[type(fused_module)](fused_module, relu)
    # setattr(modules[parent_name], name, new_fused_model)
    relu_name = 'relu'
    if not hasattr(modules[parent_name], relu_name):
        setattr(modules[parent_name], relu_name,
                torch.nn.ReLU(inplace=True).train(fused_module.training))

    modules = dict(model.named_modules(remove_duplicate=False))
    graph = model.graph
    nodes = list(model.graph.nodes)
    with graph.inserting_after(fused_node):
        relu_node_name = relu_name if parent_name == "" else f"{parent_name}.{relu_name}"
        assert relu_node_name in modules and isinstance(modules[relu_node_name], torch.nn.ReLU)
        inserted_node = graph.create_node("call_module", relu_node_name, (fused_node,), {})
        for _node in nodes:
            for i, _arg in enumerate(_node.args):
                if _arg == fused_node:
                    _tmp = list(_node.args)
                    _tmp[i] = inserted_node
                    _node.args = tuple(_tmp)
    model.recompile()
    model.graph.lint()


def fuse_prepared_model(model: GraphModule, **kwargs):  # pylint: disable=unused-argument
    '''
    Args:
        model:

    Returns:
        fused_model: GraphModule

    '''
    print('merge BN')
    fuse_model = deepcopy_graphmodule(model)
    nodes = list(fuse_model.graph.nodes)
    modules = dict(fuse_model.named_modules(remove_duplicate=False))
    for node in nodes:
        if (node.op in ['placeholder', 'output', 'call_function', 'call_method']) or \
                ('activation_post_process' in node.target):
            continue
        elif node.op == 'call_module':
            if type(modules[node.target]) in FUSED_MODULE_CONVERT_FUNCTION:
                FUSED_MODULE_CONVERT_FUNCTION[type(modules[node.target])](fuse_model, node)
    return fuse_model

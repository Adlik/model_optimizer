# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="attr-defined"
"""
Tensorrt quantizer
"""
import operator

import torch
from torch.quantization.fake_quantize import FakeQuantizeBase

from model_optimizer.utils.registry import register_backend_quantizer
from model_optimizer.proto import model_optimizer_torch_pb2 as eppb


# pylint: disable=too-many-boolean-expressions
@register_backend_quantizer(eppb.InferenceBackend.TENSORRT)
def prepare_tensorrt_quantizer(model, additional_node_name=None):
    """Prepare to insert the FakeQuantize model based on tensorrt backend

    Args:
        model (torch.fx.GraphModule): torch.fx.GraphModule model, must be in train mode
        additional_node_name (list): Module name of the node that is not quantified

    Returns:
        A GraphModule with fake quant modules based on tensorrt backend
    """
    graph = model.graph

    node_removed_fake_quant = find_not_quantize_node_input_fake_quant(model, additional_node_name=additional_node_name)

    for node in node_removed_fake_quant:

        if hasattr(model, node.target):
            delattr(model, node.target)

        node._remove_from_list()  # pylint: disable=protected-access
        orig_users = list(node.users.keys())
        for user_node in orig_users:
            user_node.replace_input_with(node, node.args[0])
        graph.erase_node(node)

    model.recompile()
    model.graph.lint()

    return model


def find_not_quantize_node_input_fake_quant(model, additional_node_name=None):
    """Find the input fake quant for the unquantized node

    Args:
        model (torch.fx.GraphModule): torch.fx.GraphModule model
        additional_node_name (list): Module name of the node that is not quantified

    Returns:
        List of fake quant node that need to be removed
    """
    modules = dict(model.named_modules(remove_duplicate=False))

    if additional_node_name is None:
        additional_node_name = []

    node_removed_fake_quant = []

    for node in model.graph.nodes:
        if ((node.op == "call_module" and isinstance(modules[node.target], _module_type_not_insert_input_fake_quant()))
                or ((node.op in ('call_function', 'call_method'))
                    and node.target in _function_type_not_insert_input_fake_quant())
                or node.name in additional_node_name):

            input_node_list = [_node for _node in _flatten_args(node.args) if isinstance(_node, torch.fx.node.Node)]

            for input_node in input_node_list:
                if input_node.op == "call_module" and isinstance(modules[input_node.target], FakeQuantizeBase):
                    node_removed_fake_quant.append(input_node)

        if node.target is operator.add:
            node_input_list = [_node for _node in _flatten_args(node.args) if isinstance(_node, torch.fx.node.Node)]
            merge_node = _find_add_merge_node(model, node_input_list)
            if merge_node is not None:
                node_removed_fake_quant.append(merge_node)

    return node_removed_fake_quant


def _find_add_merge_node(model, input_node_list):
    """Find the first input node which has only one successor from the last.
    This kind of node can be merge with add.
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    for input_node in input_node_list:
        if input_node.op == 'call_module' and isinstance(modules[input_node.target], FakeQuantizeBase):
            input_node_input = input_node.args[0]
            if input_node_input.op == 'call_module' and isinstance(modules[input_node_input.target], _merge_add_type()):
                succ = 0
                for _node in list(model.graph.nodes):
                    _node_input_list = _flatten_args(_node.args)
                    if input_node_input in _node_input_list:
                        succ += 1
                if succ == 1:
                    return input_node
    return None


def _merge_add_type():
    return (torch.nn.Conv2d,
            torch.nn.intrinsic.modules.fused.ConvBn2d,
            torch.nn.intrinsic.modules.fused.ConvBnReLU2d,
            torch.nn.Linear,
            torch.nn.intrinsic.modules.fused.LinearReLU
            )


def _function_type_not_insert_input_fake_quant():
    return (torch.flatten,
            torch.nn.functional.relu6)


def _module_type_not_insert_input_fake_quant():
    return (torch.nn.Flatten,
            torch.nn.ReLU6)


def _flatten_args(node):
    flattned_args = []
    if isinstance(node, dict):
        for v in node.values():
            flattned_args.extend(_flatten_args(v))
    elif isinstance(node, (tuple, list)):
        for n in node:
            flattned_args.extend(_flatten_args(n))
    else:
        flattned_args.extend([node])
    return flattned_args

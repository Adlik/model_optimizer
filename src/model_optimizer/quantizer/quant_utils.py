# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=protected-access
'''
quantization releated utils tools.
'''
import os
from collections import OrderedDict
import onnx  # pylint: disable=import-error
import torch
from torch.fx import GraphModule
from .onnx_utils import FAKE_QUANTIZE_OP


def get_specified_not_quant_module(fused_model: GraphModule, onnx_model_path):
    """
    get onnx graph node name with None qconfig.
    Args:
        fused_model:
        onnx_model_path:

    Returns:
        qconfig_module_with_none: List, the onnx node name
    """
    assert os.path.exists(onnx_model_path)

    onnx_model = onnx.load(onnx_model_path)
    qconfig_map = fused_model._qconfig_map

    match_pairs = get_fx_module_matched_with_onnx_graph(fused_model, onnx_model)

    qconfig_module_with_none = []
    for name, qconfig in qconfig_map.items():  # type: ignore[union-attr, operator]
        if name in match_pairs and qconfig is None:
            qconfig_module_with_none.append(match_pairs[name])
    return qconfig_module_with_none


def get_fx_module_matched_with_onnx_graph(fused_model, onnx_model):
    """
    get fx nodes and onnx nodes, and find match pairs.

    Args:
        fused_model:
        onnx_model:

    Returns:
        matched_pair: dict
    """
    fx_modules = dict(fused_model.named_modules())

    fx_graph = fused_model.graph
    onnx_graph = onnx_model.graph

    fx_node = []
    onnx_node = []

    # don't add input, output and activation fakequant node
    for node in fx_graph.nodes:
        if node.name in ['x', 'output'] or 'activation_post_process' in node.name or \
                                           node.op in ['placeholder']:
            continue
        if node.target in fx_modules:
            if isinstance(fx_modules[node.target], (torch.nn.modules.dropout._DropoutNd,
                                                    torch.nn.modules.activation.ReLU,
                                                    torch.nn.modules.activation.SiLU)):
                continue
            elif isinstance(fx_modules[node.target], (torch.nn.modules.activation.LeakyReLU,
                                                      torch.nn.modules.activation.Sigmoid,
                                                      torch.nn.modules.pooling._AdaptiveAvgPoolNd,
                                                      torch.nn.modules.pooling._AvgPoolNd)):
                mod_name = fx_modules[node.target]._get_name().lower()
                act_name = node.name + '-' + mod_name
                fx_node.append(act_name)
            continue
        fx_node.append(node)

    for node in onnx_graph.node:
        if node.op_type in ['Identity', 'Constant', 'Relu', 'QuantizeLinear', 'DequantizeLinear'] or \
                node.op_type in FAKE_QUANTIZE_OP:
            continue
        else:
            onnx_node.append(node)

    matched_pair = matched_pair_impl(fx_node, onnx_node)
    return matched_pair


def matched_pair_impl(fx_node_list, onnx_node_list):
    """
    find matched node pairs and save to dict.
    Args:
        fx_node_list:
        onnx_node_list:

    Returns:

    """
    fx_to_onnx = OrderedDict()

    fx_node_index, onnx_node_index, len_fx, len_onnx = 0, 0, len(fx_node_list), len(onnx_node_list)
    while fx_node_index < len_fx and onnx_node_index < len_onnx:
        if is_same_node(fx_node_list[fx_node_index], onnx_node_list[onnx_node_index]):
            fx_name = fx_node_list[fx_node_index].split('-')[0] if isinstance(fx_node_list[fx_node_index], str) \
                else fx_node_list[fx_node_index].name
            fx_to_onnx.update({fx_name: onnx_node_list[onnx_node_index].name})
            fx_node_index += 1
            onnx_node_index += 1
        else:
            fx_node_index, onnx_node_index = find_next_same_node_index(fx_node_list,
                                                                       onnx_node_list,
                                                                       fx_node_index,
                                                                       onnx_node_index,
                                                                       len_fx,
                                                                       len_onnx)

    return fx_to_onnx


def find_next_same_node_index(fx_node_list, onnx_node_list, fx_node_index, onnx_node_index, len_fx, len_onnx):
    """
    find the firstly matched node index in fx_node_list and onnx_node_list.
    We assume that a fx node that corresponding to a untraced custom module should contains less three nodes in onnx
    graph, otherwise, you should not directly replace the target module with custom module and need to redefine your
    model.

    Args:
        fx_node_list: [fx node list]
        onnx_node_list: [onnx node list]
        fx_node_index: index of current fx node
        onnx_node_index: index of current onnx node
        len_fx:: length of fx_node_list
        len_onnx: length of onnx_node_list

    Returns:
        the pair index that fx node and onnx node are same.
    """
    # Maximum search times
    limit_index = min(fx_node_index + 3, len_fx)
    cur_fx_index = fx_node_index
    cur_onnx_index = onnx_node_index

    while cur_fx_index < limit_index:

        cur_onnx_index = onnx_node_index + (cur_fx_index - fx_node_index)
        while cur_onnx_index < len_onnx:

            if is_same_node(fx_node_list[cur_fx_index], onnx_node_list[cur_onnx_index]):

                return cur_fx_index, cur_onnx_index

            cur_onnx_index += 1

        cur_fx_index += 1

    return cur_fx_index, cur_onnx_index


def is_same_node(fx_node, onnx_node):
    """
    Here firstly introduce the naming rules.
    In fx graph, For modules, node name can be named by modules name, such as
        module name correspond to node name, layer1.0.conv2 -> layer1_0_conv2,
        or for function, node name will be replaced by "func name"_"index", add_2.
        reference  https://pytorch.org/docs/stable/fx.html. For some modules,
        such as leakyerlu, sigmoid and averagepool, we will use ' "node.name"-"module name" '
        as new node name to match the onnx node.

    In onnx graph, all node name can be formed with "node type"_"index", Mul_31,
        Concat_5, Add_329.

    Args:
        fx_node: torch.fx.Node or name
        onnx_node: NodeProto
    Returns:

    """

    fx_node_name = fx_node if isinstance(fx_node, str) else fx_node.name

    if onnx_node.op_type in ['Mul', 'Add', 'Sigmoid', 'LeakyRelu']:
        onnx_node_name = onnx_node.name
        onnx_node_name = onnx_node_name.split('_')[0].lower()
        return onnx_node_name in fx_node_name
    elif onnx_node.op_type == 'Concat':
        fx_node_name = fx_node_name.split('_')[0]
        return fx_node_name in ['cat', 'concat']
    elif onnx_node.op_type in ['GlobalAveragePool', 'AveragePool']:
        for avg_name in ['avgpool', 'adaptiveavgpool']:
            return avg_name in fx_node_name

    return False

# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Gets a quantized model that can be deployed directly on tensorrt
"""
import json
import numpy as np
import onnx
from onnx import numpy_helper

from model_optimizer.quantizer.deploy.common import (
    get_input2node_output2node,
    get_params_data,
    get_name_initializer,
    get_constant_nodes
)
from model_optimizer.proto import model_optimizer_torch_pb2 as eppb


PERCHANNEL_FAKEQUANTIZER = ['LearnablePerChannelAffine',
                            'PerChannelAffine']
PERTENSOR_FAKEQUANTIZER = ['LearnablePerTensorAffine',
                           'PerTensorAffine',
                           'FusedMovingObserveFakeQuant']
ALL_FAKEQUANTIZER = PERCHANNEL_FAKEQUANTIZER + PERTENSOR_FAKEQUANTIZER


def _remove_weight_fake_quant(node, input2node):
    next_nodes = input2node[node.output[0]]
    assert len(next_nodes) == 1
    next_node, idx = next_nodes[0]
    assert next_node.op_type in ['Conv', 'Gemm']
    next_node.input[idx] = node.input[0]


def _remove_activation_fake_quant(node, input2node):
    next_nodes = input2node[node.output[0]]
    for next_node, idx in next_nodes:
        next_node.input[idx] = node.input[0]


def _remove_identity_node(graph):
    _, output2node = get_input2node_output2node(graph)
    name_initializer = get_name_initializer(graph)
    for node in graph.node:
        for input_name in node.input:
            prev_node = output2node.get(input_name, None)
            if prev_node is not None and not isinstance(prev_node, str) and prev_node.op_type == 'Identity':
                orig_zero_point = name_initializer[prev_node.input[0]]
                if input_name not in name_initializer:
                    new_zero_point = onnx.helper.make_tensor(name=input_name,
                                                             data_type=orig_zero_point.data_type,
                                                             dims=orig_zero_point.dims,
                                                             vals=orig_zero_point.raw_data,
                                                             raw=True)
                    graph.initializer.append(new_zero_point)
                    graph.node.remove(prev_node)
                    _, output2node = get_input2node_output2node(graph)
                    name_initializer = get_name_initializer(graph)


def _parse_qparams(node, params2data):
    input_name, scale = node.input[:2]
    scale = params2data[scale]
    if len(node.input) > 3:
        qmin, qmax = node.input[-2:]
        qmin, qmax = params2data[qmin], params2data[qmax]
    elif len(node.attribute) > 0:
        qparams = params2data(node.attribute)
        qmin = qparams['quant_min']
        qmax = qparams['quant_max']
    else:
        print(f'qmin and qmax are not found for <{node.name}>!')

    if qmax == 255 and qmin == 0:
        nbits = int(np.log2(qmax - qmin + 1))
        qmin = -2 ** (nbits - 1)
        qmax = 2 ** (nbits - 1) - 1

    return input_name, scale, qmin, qmax


def clip_weight(node, params2data, input2node, name_initializer):
    """
    Clip the weight value within the range of [clip_range_min, clip_range_max]
    """
    weight_name, scale, qmin, qmax = _parse_qparams(node, params2data)
    assert -qmin == qmax, "qmin and qmax of tensorrt with weight INT8 quantized must be [-127, 127]"
    weight_data = params2data[weight_name]
    clip_range_min = (qmin * scale).astype(weight_data.dtype)
    clip_range_max = (qmax * scale).astype(weight_data.dtype)
    if len(scale.shape) > 0 and scale.shape[0] > 1:
        clip_weight_data = []
        transposed = False
        next_node = input2node[node.output[0]]
        if len(next_node) == 1 and next_node[0][0].op_type == 'ConvTranspose':
            transposed = True
            weight_data = weight_data.transpose(1, 0, 2, 3)
        for out_channel in range(weight_data.shape[0]):
            clip_weight_data.append(np.clip(weight_data[out_channel], clip_range_min[out_channel],
                                            clip_range_max[out_channel]))
        clip_weight_data = np.array(clip_weight_data)
        if transposed:
            clip_weight_data = clip_weight_data.transpose(1, 0, 2, 3)
        print(f'Clip weights <{weight_name}> to per-channel ranges.')
    else:
        clip_weight_data = np.clip(weight_data, clip_range_min, clip_range_max)
        print(f'Clip weights <{weight_name}> to range [{clip_range_min}, {clip_range_max}].')

    clip_weight_data = numpy_helper.from_array(clip_weight_data)
    name_initializer[weight_name].raw_data = clip_weight_data.raw_data


def post_process_clip_ranges(clip_ranges, graph, input2node):
    """
    Increase the clip range of operators such as Flatten, Resize.
    """
    def find_the_closest_clip_range(node):
        if node.input[0] in clip_ranges:
            return node.input[0]
        elif node.op_type in ['Flatten', 'Resize'] and node.output[0] in input2node:
            return find_the_closest_clip_range(input2node[node.output[0]][0][0])
        else:
            return None

    for node in graph.node:
        if node.op_type in ['Flatten', 'Resize']:
            tensor_name = find_the_closest_clip_range(node)
            if tensor_name:
                clip_ranges[node.input[0]] = clip_ranges[tensor_name]
                print(f'Pass <{tensor_name}> clip range to <{node.name}> input <{node.input[0]}>.')
    return clip_ranges


# pylint: disable=too-many-branches
def remove_fake_quant_and_collect_scale(onnx_path, model_name, backend):
    """Remove all fake quant operators and save the clip range of quantization operators.

    Args:
        onnx_path: the onnx model with the fake quant operator
        model_name: name of saved onnx model
        backend: tensorrt

    Returns:
        None
    """
    model = onnx.load(onnx_path)
    graph = model.graph

    _remove_identity_node(graph)

    input2node, output2node = get_input2node_output2node(graph)
    params2data = get_params_data(graph)
    name_initializer = get_name_initializer(graph)

    clip_ranges = {}
    nodes_to_be_removed = []
    for node in graph.node:
        if node.op_type in ALL_FAKEQUANTIZER:
            nodes_to_be_removed.append(node)
            nodes_to_be_removed.extend(get_constant_nodes(node, output2node))

        if node.op_type in PERCHANNEL_FAKEQUANTIZER or \
                (node.op_type == "FusedMovingObserveFakeQuant" and params2data[node.input[3]] == 0):
            # fake quantize for weights, suppose per-channel quantize only for weight
            _remove_weight_fake_quant(node, input2node)
            clip_weight(node, params2data, input2node, name_initializer)

        elif node.op_type in PERTENSOR_FAKEQUANTIZER:
            if node.output[0] not in input2node:
                assert node.output[0] in [out.name for out in graph.output]
                input2node[node.output[0]] = []
            next_nodes = input2node[node.output[0]]
            if len(next_nodes) == 1 and next_nodes[0][1] == 1 and next_nodes[0][0].op_type in ['Gemm', 'Conv']:
                _remove_weight_fake_quant(node, input2node)
                clip_weight(node, params2data, input2node, name_initializer)
            else:
                # fake quantize for activations
                _remove_activation_fake_quant(node, input2node)
                tensor_name, scale, qmin, qmax = _parse_qparams(node, params2data)
                for out in graph.output:
                    if out.name == node.output[0]:
                        out.name = tensor_name

                if backend == eppb.InferenceBackend.TENSORRT:
                    clip_ranges[tensor_name] = float(scale * max(-qmin, qmax))

    for node in nodes_to_be_removed:
        graph.node.remove(node)
    # delete initializer
    input2node, output2node = get_input2node_output2node(graph)
    name_initializer = get_name_initializer(graph)
    for name, initial_data in name_initializer.items():
        if name in (output2node.keys() | input2node.keys()):
            continue
        graph.initializer.remove(initial_data)

    clip_ranges = post_process_clip_ranges(clip_ranges, graph, input2node)
    if backend == eppb.InferenceBackend.TENSORRT:
        context = {"tensorrt": {"blob_range": clip_ranges}}

    context_filename = f"{model_name}_clip_ranges.json"
    with open(context_filename, 'w', encoding="utf-8") as f:
        json.dump(context, f, indent=4)
    onnx_filename = f"{model_name}_deploy_model.onnx"
    onnx.save(model, onnx_filename)
    print(f"Successfully exported onnx quantification model that can be deployed on TensorRT! "
          f"The file required for deploying quantitative models on Tensorrt is stored in "
          f"{onnx_filename} and {context_filename}")

# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Common function
"""
from onnx import numpy_helper


def get_input2node_output2node(graph):
    """
    Get the input and output corresponding node.
    """
    input2node = {}
    output2node = {}
    for node in graph.node:
        for idx, input_name in enumerate(node.input):
            if input_name not in input2node:
                input2node[input_name] = []
            input2node[input_name].append([node, idx])
        for output in node.output:
            output2node[output] = node

    return input2node, output2node


def get_params_data(graph):
    """
    Get the values for all parameters.
    """
    params = {}
    for init in graph.initializer:
        params[init.name] = numpy_helper.to_array(init)
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    params[node.output[0]] = numpy_helper.to_array(attr.t)
    return params


def get_name_initializer(graph):
    """
    Get the name corresponding to all initializer.
    """
    name_initializer = {}
    for init in graph.initializer:
        name_initializer[init.name] = init
    return name_initializer


def get_constant_nodes(node, output2node):
    """
    Get all constant nodes
    """
    constant_nodes = []
    for input_name in node.input:
        if input_name in output2node and output2node[input_name].op_type == 'Constant':
            constant_nodes.append(output2node[input_name])
    return constant_nodes

# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=missing-function-docstring, import-error
'''
resolve onnx graph.
'''
import copy
import numpy as np
import onnx  # pylint: disable=import-error
from onnx import TensorProto  # pylint: disable=import-error, no-name-in-module
from onnx import numpy_helper  # pylint: disable=import-error


FAKE_QUANTIZE_OP = ['LearnablePerTensorAffine', 'LearnablePerChannelAffine',
                    'PerTensorAffine', 'PerChannelAffine',
                    'FusedMovingObserveFakeQuant']


SUPPORT_REPLACE_QUANT_OP = ['LearnablePerTensorAffine', 'LearnablePerChannelAffine',
                            'PerTensorAffine', 'PerChannelAffine',
                            'FusedMovingObserveFakeQuant', 'QuantizeLinear', 'DequantizeLinear']


class ONNXGraph():
    """
    rearrange the onnx graph and easily to process the nodes.
    """
    def __init__(self, onnx_model_path):
        """
        Describe onnx graph
        args:
            input_map[tensor_name] = node which input is tensor_name
            output_map[tensor_name] = node which output is tensor_name
        """
        self.model = onnx.load(onnx_model_path)
        self.graph = self.model.graph
        self.initializer = {}
        self.input_map = {}
        self.output_map = {}
        self.topologize_graph()
        self.prepare_initializer()
        self.remove_identity()

    def remove_identity(self):
        for node in self.graph.node:
            for _, input_name in enumerate(node.input):
                prev_node = self.get_tensor_producer(input_name)
                if not isinstance(prev_node, str) and prev_node.op_type == 'Identity':
                    referenced_name = prev_node.input[0]
                    referenced_data = self.initializer[referenced_name][0]

                    node_zero_point = onnx.helper.make_tensor(prev_node.output[0],
                                                              referenced_data.data_type,
                                                              referenced_data.dims,
                                                              referenced_data.raw_data,
                                                              raw=True)
                    self.graph.initializer.append(node_zero_point)
                    self.prepare_initializer()
                    self.remove_node_purely(prev_node)
                    self.topologize_graph()

    def prepare_initializer(self):
        self.initializer.clear()
        for idx, init in enumerate(self.graph.initializer):
            self.initializer[init.name] = (init, idx)

    # pylint: disable=inconsistent-return-statements
    def get_constant(self, name):
        for node in self.model.graph.node:
            if node.op_type == 'Constant':
                if node.output[0] == name:
                    return numpy_helper.to_array(node.attribute[0].t).tolist()

    def get_initializer(self, initializer_name):
        return numpy_helper.to_array(self.initializer[initializer_name][0])

    def set_initializer(self, initializer_name, value_tensor, raw=True):
        idx = None
        if initializer_name in self.initializer:
            idx = self.initializer[initializer_name][1]
        if raw:
            initializer = numpy_helper.from_array(value_tensor)
        else:
            if value_tensor.dtype == np.float32:
                data_type = TensorProto.FLOAT
            if value_tensor.dtype == np.int32:
                data_type = TensorProto.INT32
            if value_tensor.dtype == np.uint8:
                data_type = TensorProto.UINT8
            if value_tensor.dtype == np.int8:
                data_type = TensorProto.INT8
            initializer = onnx.helper.make_tensor(name=initializer_name,
                                                  data_type=data_type,
                                                  dims=[] if value_tensor.size == 1 else list(value_tensor.shape),
                                                  vals=value_tensor,
                                                  raw=False)
        initializer.name = initializer_name
        if idx is not None:
            self.graph.initializer.remove(self.graph.initializer[idx])
        self.graph.initializer.append(initializer)
        self.prepare_initializer()

    def topologize_graph(self):
        self.input_map.clear()
        self.output_map.clear()
        for node in self.graph.node:
            for output_name in node.output:
                self.output_map[output_name] = node
            for input_name in node.input:
                if input_name not in self.input_map:
                    self.input_map[input_name] = []
                self.input_map[input_name].append(node)

    def get_tensor_producer(self, output_name):
        if output_name not in self.output_map:
            return 'INPUT_TOKEN'
        return self.output_map[output_name]

    def get_tensor_consumer(self, input_name):
        if input_name not in self.input_map:
            return ['OUTPUT_TOKEN']
        return self.input_map[input_name]

    def save_onnx_model(self, model_path):
        onnx.save(self.model, model_path)

    def remove_node_purely(self, node):
        self.graph.node.remove(node)

    def insert_node_purely(self, node, idx=0):
        self.graph.node.insert(idx, node)

    def del_initializer(self, initializer_name):
        if initializer_name in self.initializer:
            del self.initializer[initializer_name]

    # pylint: disable=too-many-branches
    def optimize_model(self):
        # Delete redundant nodes.

        remove_node_list = []
        for node in self.model.graph.node:
            if len(node.input) == 0:
                not_be_used = True
                for output_name in node.output:
                    if output_name in self.input_map:
                        not_be_used = False
                        break
                if not_be_used:
                    remove_node_list.append(node)
        for node in remove_node_list:
            self.remove_node_purely(node)
        self.topologize_graph()
        # Delete redundant initializers.
        initializers = copy.deepcopy(self.initializer)
        for initializer_name in initializers:
            if initializer_name not in self.input_map:
                self.del_initializer(initializer_name)
        # Make node in topology order.
        exist_input = [input_node.name for input_node in self.model.graph.input]
        origin_node_num = len(self.model.graph.node)
        finished_node_name = []
        # O(n^2)
        while len(finished_node_name) < origin_node_num:
            node_detect = False
            for i in range(origin_node_num):
                node = self.model.graph.node[i]
                all_inputs_exist = True
                for input_name in node.input:
                    if input_name not in exist_input and input_name not in self.initializer:
                        all_inputs_exist = False
                        break
                if all_inputs_exist:
                    if node.name not in finished_node_name:
                        node_detect = True
                        finished_node_name.append(node.name)
                        self.model.graph.node.append(node)
                        for output_name in node.output:
                            exist_input.append(output_name)
            assert node_detect, "Graph is illegel, error occured!"
        for i in range(origin_node_num):
            self.model.graph.node.remove(self.model.graph.node[0])

    def set_opset_version(self, domain, version):
        opset_info = copy.deepcopy(self.model.opset_import[0])
        opset_info.domain = domain
        opset_info.version = version
        self.model.opset_import.insert(0, opset_info)


def search_and_replace_input(next_node, name, new_name):
    """
    replace node name with new_name.
    """
    for idx, _input_name in enumerate(next_node.input):
        if _input_name == name:
            next_node.input[idx] = new_name


# pylint: disable=too-many-public-methods
class ONNXQNNPass():
    """
    convert onnx graph with fakequant to onnx with quantized node.
    """
    def __init__(self, onnx_model_path, qconfig_map_with_none=None):
        self.onnx_model = ONNXGraph(onnx_model_path)
        self.qconfig_map_with_none = qconfig_map_with_none
        self.get_convert_map()

    @property
    def qlinear_op_type(self):
        return ['QuantizeLinear', 'QLinearConv', 'QLinearAdd', 'QLinearGemm',
                'QLinearGlobalAveragePool', 'QLinearAveragePool', 'QLinearConcat',
                'QLinearSigmoid', 'QLinearMul', 'QLinearMatMul', 'QLinearLeakyRelu']

    @staticmethod
    def attribute_to_kwarg(attribute):
        """
        Convert attribute to kwarg format for use with onnx.helper.make_node.
            :parameter attribute: attribute in AttributeProto format.
            :return: attribute in {key: value} format.
        """
        if attribute.type == 0:
            raise ValueError(f'attribute {attribute.name} does not have type specified.')

        # Based on attribute type definitions from AttributeProto
        # definition in https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
        if attribute.type == 1:
            value = attribute.f
        elif attribute.type == 2:
            value = attribute.i
        elif attribute.type == 3:
            value = attribute.s
        elif attribute.type == 4:
            value = attribute.t
        elif attribute.type == 5:
            value = attribute.g
        elif attribute.type == 6:
            value = attribute.floats
        elif attribute.type == 7:
            value = attribute.ints
        elif attribute.type == 8:
            value = attribute.strings
        elif attribute.type == 9:
            value = attribute.tensors
        elif attribute.type == 10:
            value = attribute.graphs
        else:
            raise ValueError(f'attribute {attribute.name} has unsupported type {attribute.type}.')

        return {attribute.name: value}

    def _quantize_weight(self, weight_name, scale_name, zero_point_name):
        weight = self.onnx_model.get_initializer(weight_name)
        scale = self.onnx_model.get_initializer(scale_name)

        if zero_point_name in self.onnx_model.output_map:
            node = self.onnx_model.output_map[zero_point_name]
            zero_point = self.onnx_model.get_initializer(node.input[0])
        else:
            zero_point = self.onnx_model.get_initializer(zero_point_name)

        data_type = np.int8

        if data_type == np.int8:
            min_value = -128
            max_value = 127
        else:
            min_value = 0
            max_value = 255

        if scale.size == 1 and zero_point.size == 1:
            int_weight = (weight / scale).round() + zero_point
            int_weight = np.clip(int_weight, min_value, max_value)
            return int_weight.astype(data_type)
        else:
            dim = []
            for i, _ in enumerate(weight.shape):
                if i == 0:
                    continue
                dim.append(i)
            new_scale = np.expand_dims(scale, dim)
            new_zero_point = np.expand_dims(zero_point, dim)
            int_weight = (weight / new_scale).round() + new_zero_point
            int_weight = np.clip(int_weight, min_value, max_value)
            return int_weight.astype(data_type)

    def _quantize_bias(self, bias, x_scale, w_scale):
        x_scale = self.onnx_model.get_initializer(x_scale)
        w_scale = self.onnx_model.get_initializer(w_scale)
        bias = self.onnx_model.get_initializer(bias)
        return (bias / (x_scale * w_scale)).astype(np.int32)

    @property
    def node_without_qparams(self):
        return ['Flatten']

    def _replace_conv(self, node, idx):
        if node.input[1] not in self.onnx_model.output_map:
            return

        # Input scale
        qlinear_conv_inputs = []
        input_fake_quant_node = self.onnx_model.get_tensor_producer(node.input[0])
        assert input_fake_quant_node.op_type in SUPPORT_REPLACE_QUANT_OP
        x_scale, x_zero_point = input_fake_quant_node.input[1], input_fake_quant_node.input[2]
        # Output scale
        qlinear_conv_output = node.output
        y_scale, y_zero_point = self.get_node_output_qparams(node)
        # Weight scale
        weight_fake_quant_node = self.onnx_model.get_tensor_producer(node.input[1])
        w_scale, w_zero_point = weight_fake_quant_node.input[1], weight_fake_quant_node.input[2]
        weight_name = weight_fake_quant_node.input[0]
        W = self._quantize_weight(weight_name, w_scale, w_zero_point)
        self.onnx_model.set_initializer(weight_name, W)
        qlinear_conv_inputs.extend([node.input[0], x_scale, x_zero_point,
                                    weight_name, w_scale, w_zero_point,
                                    y_scale, y_zero_point])
        # Bias
        if len(node.input) == 3:
            bias_name = node.input[2]
            B = self._quantize_bias(bias_name, x_scale, w_scale)
            self.onnx_model.set_initializer(bias_name, B)
            qlinear_conv_inputs.append(bias_name)
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(ONNXQNNPass.attribute_to_kwarg(attribute))
        node_type = "QLinearConv"
        qlinear_conv_node = onnx.helper.make_node(node_type,
                                                  qlinear_conv_inputs,
                                                  qlinear_conv_output,
                                                  node.name + '_quantized',
                                                  **kwargs)
        self.onnx_model.remove_node_purely(node)
        self.onnx_model.remove_node_purely(weight_fake_quant_node)
        self.onnx_model.insert_node_purely(qlinear_conv_node, idx)
        self.onnx_model.topologize_graph()

    def _replace_add_to_qlinearadd(self, node, idx):
        # First input
        qlinear_add_input = []
        qlinear_add_output = node.output
        first_input_node = self.onnx_model.get_tensor_producer(node.input[0])
        assert first_input_node.op_type in SUPPORT_REPLACE_QUANT_OP
        first_input_quantized = first_input_node.output[0]
        first_scale = first_input_node.input[1]
        first_zero_point = first_input_node.input[2]
        # Second input
        second_input_node = self.onnx_model.get_tensor_producer(node.input[1])
        assert second_input_node.op_type in SUPPORT_REPLACE_QUANT_OP
        second_input_quantized = second_input_node.output[0]
        second_scale = second_input_node.input[1]
        second_zero_point = second_input_node.input[2]
        # Output
        output_scale, output_zero_point = self.get_node_output_qparams(node)
        qlinear_add_input.extend([first_input_quantized, first_scale, first_zero_point,
                                  second_input_quantized, second_scale, second_zero_point,
                                  output_scale, output_zero_point])
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(ONNXQNNPass.attribute_to_kwarg(attribute))
        qlinear_add_node = onnx.helper.make_node("QLinearAdd",
                                                 qlinear_add_input,
                                                 qlinear_add_output,
                                                 node.name + '_quantized',
                                                 domain='com.microsoft',
                                                 **kwargs)
        self.onnx_model.insert_node_purely(qlinear_add_node, idx)
        self.onnx_model.remove_node_purely(node)
        self.onnx_model.topologize_graph()

    def _replace_pool_to_qlinearpool(self, node, idx, is_global):
        qlinear_pool_input = []
        prev_node = self.onnx_model.get_tensor_producer(node.input[0])
        assert prev_node.op_type in SUPPORT_REPLACE_QUANT_OP
        x_scale, x_zero_point = prev_node.input[1], prev_node.input[2]
        y_scale, y_zero_point = self.get_node_output_qparams(node)
        qlinear_pool_input.extend([node.input[0], x_scale, x_zero_point,
                                   y_scale, y_zero_point])
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(ONNXQNNPass.attribute_to_kwarg(attribute))
        qlinear_add_output = node.output
        node_type = "QLinearGlobalAveragePool" if is_global else "QLinearAveragePool"
        qlinear_pool_node = onnx.helper.make_node(node_type,
                                                  qlinear_pool_input,
                                                  qlinear_add_output,
                                                  node.name + '_quantized',
                                                  domain='com.microsoft',
                                                  **kwargs)
        self.onnx_model.insert_node_purely(qlinear_pool_node, idx)
        self.onnx_model.remove_node_purely(node)
        self.onnx_model.topologize_graph()

    def _replace_mul_to_qlinearmul(self, node, idx):
        qlinear_mul_input = []
        # First input
        first_input_node = self.onnx_model.get_tensor_producer(node.input[0])
        assert first_input_node.op_type in SUPPORT_REPLACE_QUANT_OP
        first_input_quantized = first_input_node.output[0]
        first_scale = first_input_node.input[1]
        first_zero_point = first_input_node.input[2]
        # Second input
        second_input_node = self.onnx_model.get_tensor_producer(node.input[1])
        assert second_input_node.op_type in SUPPORT_REPLACE_QUANT_OP
        second_input_quantized = second_input_node.output[0]
        second_scale = second_input_node.input[1]
        second_zero_point = second_input_node.input[2]
        output_scale, output_zero_point = self.get_node_output_qparams(node)
        qlinear_mul_input.extend([first_input_quantized, first_scale, first_zero_point,
                                  second_input_quantized, second_scale, second_zero_point,
                                  output_scale, output_zero_point])
        qlinear_mul_output = node.output
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(ONNXQNNPass.attribute_to_kwarg(attribute))
        qlinear_mul_node = onnx.helper.make_node("QLinearMul",
                                                 qlinear_mul_input,
                                                 qlinear_mul_output,
                                                 node.name + '_quantized',
                                                 domain='com.microsoft',
                                                 **kwargs)
        self.onnx_model.insert_node_purely(qlinear_mul_node, idx)
        self.onnx_model.remove_node_purely(node)
        self.onnx_model.topologize_graph()

    def _replace_sigmoid_to_qlinearsigmoid(self, node, idx):
        qlinear_sigmoid_input = []
        prev_node = self.onnx_model.get_tensor_producer(node.input[0])
        assert prev_node.op_type in SUPPORT_REPLACE_QUANT_OP
        x_scale, x_zero_point = prev_node.input[1], prev_node.input[2]
        y_scale, y_zero_point = self.get_node_output_qparams(node)
        qlinear_sigmoid_input.extend([node.input[0], x_scale, x_zero_point,
                                      y_scale, y_zero_point])
        qlinear_sigmoid_output = node.output
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(ONNXQNNPass.attribute_to_kwarg(attribute))
        qlinear_sigmoid_node = onnx.helper.make_node("QLinearSigmoid",
                                                     qlinear_sigmoid_input,
                                                     qlinear_sigmoid_output,
                                                     node.name + '_quantized',
                                                     domain='com.microsoft',
                                                     **kwargs)
        self.onnx_model.insert_node_purely(qlinear_sigmoid_node, idx)
        self.onnx_model.remove_node_purely(node)
        self.onnx_model.topologize_graph()

    def _replace_concat_to_qlinearconcat(self, node, idx):
        y_scale, y_zero_point = self.get_node_output_qparams(node)
        qlinear_concat_input = []
        qlinear_concat_input.extend([y_scale, y_zero_point])
        for input_node in node.input:
            cur_node = self.onnx_model.get_tensor_producer(input_node)
            assert cur_node in SUPPORT_REPLACE_QUANT_OP
            qlinear_concat_input.extend(cur_node.input[0])
            qlinear_concat_input.extend(cur_node.input[1])
            qlinear_concat_input.extend(cur_node.input[2])

        qlinear_concat_output = node.output
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(ONNXQNNPass.attribute_to_kwarg(attribute))
        qlinear_concat_node = onnx.helper.make_node("QLinearConcat",
                                                    qlinear_concat_input,
                                                    qlinear_concat_output,
                                                    node.name + '_quantized',
                                                    domain='com.microsoft',
                                                    **kwargs)
        self.onnx_model.insert_node_purely(qlinear_concat_node, idx)
        self.onnx_model.remove_node_purely(node)
        self.onnx_model.topologize_graph()

    def _replace_leakyrelu_to_qlinearleakyrelu(self, node, idx):
        qlinear_leakyrelu_input = []
        prev_node = self.onnx_model.get_tensor_producer(node.input[0])
        assert prev_node.op_type in SUPPORT_REPLACE_QUANT_OP
        x_scale, x_zero_point = prev_node.input[1], prev_node.input[2]
        y_scale, y_zero_point = self.get_node_output_qparams(node)
        qlinear_leakyrelu_input.extend([node.input[0], x_scale, x_zero_point,
                                       y_scale, y_zero_point])

        qlinear_leakyrelu_output = node.output
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(ONNXQNNPass.attribute_to_kwarg(attribute))
        qlinear_leakyrelu_node = onnx.helper.make_node("QLinearLeakyRelu",
                                                       qlinear_leakyrelu_input,
                                                       qlinear_leakyrelu_output,
                                                       node.name + '_quantized',
                                                       domain='com.microsoft',
                                                       **kwargs)
        self.onnx_model.insert_node_purely(qlinear_leakyrelu_node, idx)
        self.onnx_model.remove_node_purely(node)
        self.onnx_model.topologize_graph()

    def _replace_matmul_to_qlinearmatmul(self, node, idx):
        qlinear_matmul_input = []
        # First input
        first_input_node = self.onnx_model.get_tensor_producer(node.input[0])
        assert first_input_node.op_type in SUPPORT_REPLACE_QUANT_OP
        first_input_quantized = first_input_node.output[0]
        first_scale = first_input_node.input[1]
        first_zero_point = first_input_node.input[2]
        # Second input
        second_input_node = self.onnx_model.get_tensor_producer(node.input[1])
        assert second_input_node.op_type in SUPPORT_REPLACE_QUANT_OP
        second_input_quantized = second_input_node.output[0]
        second_scale = second_input_node.input[1]
        second_zero_point = second_input_node.input[2]
        output_scale, output_zero_point = self.get_node_output_qparams(node)
        qlinear_matmul_input.extend([first_input_quantized, first_scale, first_zero_point,
                                     second_input_quantized, second_scale, second_zero_point,
                                     output_scale, output_zero_point])
        qlinear_matmul_output = node.output
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(ONNXQNNPass.attribute_to_kwarg(attribute))
        qlinear_matmul_node = onnx.helper.make_node("QLinearMatMul",
                                                    qlinear_matmul_input,
                                                    qlinear_matmul_output,
                                                    node.name + '_quantized',
                                                    domain='com.microsoft',
                                                    **kwargs)
        self.onnx_model.insert_node_purely(qlinear_matmul_node, idx)
        self.onnx_model.remove_node_purely(node)
        self.onnx_model.topologize_graph()

    def get_node_output_qparams(self, node):
        fake_quantize_node = self.onnx_model.get_tensor_consumer(node.output[0])[0]
        while fake_quantize_node.op_type not in FAKE_QUANTIZE_OP:
            assert fake_quantize_node.op_type in self.node_without_qparams
            fake_quantize_node = self.onnx_model.get_tensor_consumer(fake_quantize_node.output[0])[0]
        return fake_quantize_node.input[1], fake_quantize_node.input[2]

    def _replace_gemm(self, node, idx):
        weight_node = self.onnx_model.get_tensor_producer(node.input[1])
        gemm_input = []
        gemm_input.append(node.input[0])
        gemm_input.append(weight_node.input[0])
        gemm_input.append(node.input[2])
        gemm_output = node.output
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(ONNXQNNPass.attribute_to_kwarg(attribute))
        gemm_node = onnx.helper.make_node(node.op_type,
                                          gemm_input,
                                          gemm_output,
                                          node.name,
                                          **kwargs)
        self.onnx_model.insert_node_purely(gemm_node, idx)
        self.onnx_model.remove_node_purely(weight_node)
        self.onnx_model.remove_node_purely(node)
        self.onnx_model.topologize_graph()

    def _is_completly_fakequant(self, node):
        is_input_all_fake = True
        is_output_all_fake = True
        for output in node.output:
            out_node_list = self.onnx_model.get_tensor_consumer(output)
            for out_node in out_node_list:
                if isinstance(out_node, str) or out_node.op_type not in SUPPORT_REPLACE_QUANT_OP:
                    is_output_all_fake = False
                    break
        if node.op_type in ['Conv', 'Gemm']:
            node_input = [node.input[0], ]
        else:
            node_input = node.input

        for input_ in node_input:
            in_node = self.onnx_model.get_tensor_producer(input_)
            if isinstance(in_node, str):
                break
            if in_node.op_type not in SUPPORT_REPLACE_QUANT_OP:
                is_input_all_fake = False

        return is_input_all_fake and is_output_all_fake

    def get_convert_map(self):
        self.convert_map = {
            'Conv': self._replace_conv,
            'Gemm': self._replace_gemm,
            'Add': self._replace_add_to_qlinearadd,
            'GlobalAveragePool': self._replace_pool_to_qlinearpool,
            'AveragePool': self._replace_pool_to_qlinearpool,
            'Concat': self._replace_concat_to_qlinearconcat,
            'LeakyRelu': self._replace_leakyrelu_to_qlinearleakyrelu,
            'MatMul': self._replace_matmul_to_qlinearmatmul,
            'Sigmoid': self._replace_sigmoid_to_qlinearsigmoid,
            'Mul': self._replace_mul_to_qlinearmul,
        }

    def replace_op_pass(self):

        # Replace Conv / Gemm / Add / AvgPool / Concat / LeakyRelu.
        for idx, node in enumerate(self.onnx_model.graph.node):

            if (self.qconfig_map_with_none is not None and node.name in self.qconfig_map_with_none) or \
                    node.op_type in FAKE_QUANTIZE_OP or node.op_type in ['Identity', 'Constant']:
                continue

            if not self._is_completly_fakequant(node):
                continue

            if node.op_type == 'GlobalAveragePool':
                self.convert_map[node.op_type](node, idx, is_global=True)
            elif node.op_type == 'AveragePool':
                self.convert_map[node.op_type](node, idx, is_global=False)
            else:
                if node.op_type in self.convert_map:
                    self.convert_map[node.op_type](node, idx)

    def _make_quant_dequant_node(self, node, prev_node, quant=True):
        if quant:
            quant_op_type = "QuantizeLinear"
            quant_name = "_quantized"
        else:
            quant_op_type = "DequantizeLinear"
            quant_name = '_dequantized'

        output_value_info = [f'{node.output[0]}_{quant_op_type}']
        quant_node = onnx.helper.make_node(quant_op_type,
                                           node.input[0:3],
                                           output_value_info,
                                           ('input' if prev_node == 'INPUT_TOKEN'
                                            else prev_node.name) + quant_name)
        self.onnx_model.insert_node_purely(quant_node)
        return quant_node

    # pylint: disable=too-many-branches
    def _process_normal_fakequant(self, node, prev_node, next_node_list):
        quantize_node = None
        dequantize_node = None
        for next_node in next_node_list:
            if prev_node != 'INPUT_TOKEN':
                if next_node != 'OUTPUT_TOKEN':
                    if prev_node.op_type in self.qlinear_op_type and \
                            (next_node.op_type in self.qlinear_op_type or
                             next_node.op_type in FAKE_QUANTIZE_OP):
                        search_and_replace_input(next_node, node.output[0], node.input[0])
                    elif prev_node.op_type in self.qlinear_op_type and \
                            next_node.op_type not in self.qlinear_op_type:
                        if dequantize_node is None:
                            dequantize_node = self._make_quant_dequant_node(node, prev_node, quant=False)
                        search_and_replace_input(next_node, node.output[0], dequantize_node.output[0])
                    elif prev_node.op_type not in self.qlinear_op_type and \
                            (next_node.op_type not in self.qlinear_op_type or
                             next_node.op_type in FAKE_QUANTIZE_OP):
                        search_and_replace_input(next_node, node.output[0], prev_node.output[0])
                    elif prev_node.op_type not in self.qlinear_op_type and \
                            next_node.op_type in self.qlinear_op_type:
                        if quantize_node is None:
                            quantize_node = self._make_quant_dequant_node(node, prev_node, quant=True)
                        search_and_replace_input(next_node, node.output[0], quantize_node.output[0])
                else:
                    if prev_node.op_type in self.qlinear_op_type:
                        if dequantize_node is None:
                            dequantize_node = self._make_quant_dequant_node(node, prev_node, quant=False)
                    else:
                        output_name = node.output[0]
                        prev_node.output[0] = output_name
            else:
                if next_node != 'OUTPUT_TOKEN':
                    if next_node.op_type in self.qlinear_op_type:
                        if quantize_node is None:
                            quantize_node = self._make_quant_dequant_node(node, prev_node, quant=True)
                        search_and_replace_input(next_node, node.output[0], quantize_node.output[0])
                    else:
                        search_and_replace_input(next_node, node.output[0], prev_node.output[0])
        self.onnx_model.remove_node_purely(node)
        self.onnx_model.topologize_graph()

    def replace_qlinear_layer_pass(self):

        for node in self.onnx_model.graph.node:
            if node.op_type in FAKE_QUANTIZE_OP:
                prev_node = self.onnx_model.get_tensor_producer(node.input[0])
                next_node_list = self.onnx_model.get_tensor_consumer(node.output[0])
                self._process_normal_fakequant(node, prev_node, next_node_list)
            elif node.op_type in ["QuantizeLinear", ]:
                next_node = self.onnx_model.get_tensor_consumer(node.output[0])
                assert next_node.op_type in ["DequantizeLinear", ], "Quant/DeQuant should be present at same time"
                prev_node = self.onnx_model.get_tensor_producer(node.input[0])
                next_next_node = self.onnx_model.get_tensor_consumer(next_node.output[0])

                if next_next_node not in self.qlinear_op_type:
                    search_and_replace_input(next_next_node, next_node.output[0], prev_node.output[0])
                    self.onnx_model.remove_node_purely(node)

                else:
                    search_and_replace_input(next_next_node, next_node.output[0], node.output[0])
                self.onnx_model.remove_node_purely(next_node)
                self.onnx_model.topologize_graph()

    def merge_relu_pass(self):
        for node in self.onnx_model.graph.node:
            if node.op_type == 'Relu':
                next_node = self.onnx_model.get_tensor_consumer(node.output[0])[0]
                if next_node.op_type in FAKE_QUANTIZE_OP:
                    # Input idx2 is zero point.
                    self.onnx_model.set_initializer(next_node.input[2], np.array([0], dtype=np.uint8), raw=False)
                    next_node.input[0] = node.input[0]
                    self.onnx_model.remove_node_purely(node)

        self.onnx_model.topologize_graph()

    def format_qlinear_dtype_pass(self):
        seted_initializer_set = []
        for node in self.onnx_model.graph.node:
            if node.op_type in FAKE_QUANTIZE_OP:
                zero_point_type = np.uint8
                if len(node.input) > 5:
                    scale, zero_point, qmin, qmax = node.input[1], node.input[2], node.input[4], node.input[5]
                    qmin = self.onnx_model.get_constant(qmin)
                    qmax = self.onnx_model.get_constant(qmax)
                    ch_axis = self.onnx_model.get_constant(node.input[3])
                    zero_point_type = np.int8
                    if node.op_type == 'FusedMovingObserveFakeQuant' and ch_axis == -1:
                        zero_point_type = np.uint8
                else:
                    scale, zero_point, qmin, qmax = node.input[1], node.input[2], node.input[3], node.input[4]
                    qmin = self.onnx_model.get_constant(qmin)
                    qmax = self.onnx_model.get_constant(qmax)

                if (scale not in seted_initializer_set) and (scale not in self.onnx_model.output_map):
                    scale_data = self.onnx_model.get_initializer(scale)
                    self.onnx_model.set_initializer(scale, scale_data.astype(np.float32), raw=False)
                    seted_initializer_set.append(scale)
                # when zero_point in output_map, that means zero_point is a output in identity node.
                if (zero_point not in seted_initializer_set) and (zero_point not in self.onnx_model.output_map):
                    zero_point_data = self.onnx_model.get_initializer(zero_point)
                    self.onnx_model.set_initializer(zero_point, zero_point_data.astype(zero_point_type), raw=False)
                    seted_initializer_set.append(zero_point)

    def run(self, model_name='deploy_model'):
        self.format_qlinear_dtype_pass()
        self.merge_relu_pass()
        self.replace_op_pass()
        self.replace_qlinear_layer_pass()
        self.onnx_model.optimize_model()
        self.onnx_model.set_opset_version('com.microsoft', 1)

        try:
            onnx.checker.check_model(self.onnx_model.model)
        except onnx.checker.ValidationError as e:
            print(f'The model is invalid: {e}')
        self.onnx_model.save_onnx_model(f'{model_name}.onnx')

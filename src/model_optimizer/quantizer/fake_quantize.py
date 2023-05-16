# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=missing-function-docstring, unused-argument
"""
learned step size quantization
https://arxiv.org/abs/1902.08153
"""
import math

import torch
from torch.quantization.observer import MinMaxObserver
from torch.onnx import register_custom_op_symbolic


def _is_per_channel(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_channel_symmetric, torch.per_channel_affine]


def _is_per_tensor(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]


def _is_symmetric_quant(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_tensor_symmetric, torch.per_channel_symmetric]


class LearnableFakeQuantize(torch.quantization.FakeQuantizeBase):
    r""" Simulate the quantize and dequantize operations in training time.
    The output of this module is given by

    x_out = (clamp(round(x/scale + zero_point), quant_min, quant_max)-zero_point)*scale



    * :attr:`scale` defines the scale factor used for quantization.

    * :attr:`zero_point` specifies the quantized value to which 0 in floating point maps to

    * :attr:`quant_min` specifies the minimum allowable quantized value.

    * :attr:`quant_max` specifies the maximum allowable quantized value.

    * :attr:`fake_quant_enable` controls the application of fake quantization on tensors, note that
      statistics can still be updated.

    * :attr:`observer_enable` controls statistics collection on tensors

    * :attr:`dtype` specifies the quantized dtype that is being emulated with fake-quantization,
                    allowable values are torch.qint8 and torch.quint8. The values of quant_min and
                    quant_max should be chosen to be consistent with the dtype


    Args:
        observer (module): Module for observing statistics on input tensors and calculating scale
                           and zero-point.
        quant_min (int): The minimum allowable quantized value.
        quant_max (int): The maximum allowable quantized value.
        observer_kwargs (optional): Arguments for the observer module

    Attributes:
        observer (Module): User provided module that collects statistics on the input tensor and
                           provides a method to calculate scale and zero-point.

    """

    def __init__(self, observer=MinMaxObserver, quant_min=0, quant_max=255, out_channels=None, **observer_kwargs):
        super().__init__()
        assert quant_min <= quant_max, \
            'quant_min must be less than or equal to quant_max'
        self.quant_min = quant_min
        self.quant_max = quant_max
        if quant_min is not None and quant_max is not None:
            assert quant_min <= quant_max, \
                'quant_min must be less than or equal to quant_max'
            dtype = observer_kwargs.get("dtype", torch.quint8)
            if hasattr(observer, "p"):
                # In case observer is _PartialWrapper, dtype can be stored in
                # observer.p.keywords["dtype"]
                dtype = getattr(getattr(observer, "p", {}), "keywords", {}).get(
                    "dtype", dtype
                )
            assert torch.iinfo(dtype).min <= quant_min, 'quant_min out of bound'
            assert quant_max <= torch.iinfo(dtype).max, 'quant_max out of bound'
            observer_kwargs.update({"quant_min": quant_min, "quant_max": quant_max})
        self.activation_post_process = observer(**observer_kwargs)
        assert torch.iinfo(self.activation_post_process.dtype).min <= quant_min, 'quant_min out of bound'
        assert quant_max <= torch.iinfo(self.activation_post_process.dtype).max, 'quant_max out of bound'
        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = 1
        self.register_parameter("scale", torch.nn.Parameter(torch.Tensor([1.0] * self.out_channels)))
        self.register_parameter("zero_point", torch.nn.Parameter(torch.Tensor([1.0] * self.out_channels)))
        self.register_buffer('zero_point_init', torch.tensor([0] * self.out_channels))
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = self.activation_post_process.ch_axis \
            if hasattr(self.activation_post_process, 'ch_axis') else -1
        assert _is_per_channel(self.qscheme) or \
            _is_per_tensor(self.qscheme), \
            'Only per channel and per tensor quantization are supported in fake quantize' + \
            ' got qscheme: ' + str(self.qscheme)
        self.is_per_channel = _is_per_channel(self.qscheme)
        self.is_symmetric_quant = _is_symmetric_quant(self.activation_post_process.qscheme)
        self.register_buffer(
            "eps", torch.tensor([torch.finfo(torch.float32).eps])
            )

    @torch.jit.export
    def calculate_qparams(self, **kwargs):  # pylint: disable=missing-function-docstring, unused-argument
        self.scale.data.clamp_(min=self.eps.item())  # type: ignore[operator]
        scale = self.scale.detach()
        zero_point = self.zero_point.detach().round().clamp(self.quant_min, self.quant_max).long()
        return scale, zero_point

    # pylint: disable=pointless-string-statement
    def forward(self, x):  # pylint: disable=missing-function-docstring, too-many-branches
        """
        if self.dtype == torch.quint8:
            symmetric_qmax = self.quant_max // 2
        else:
            symmetric_qmax = self.quant_max
        """
        if self.observer_enabled[0] == 1:
            """
            if self.dtype == torch.quint8:
                min_value = torch.min(x.detach())
                if min_value >= 0.0:
                    self.qscheme = torch.per_tensor_affine
                    self.activation_post_process.qscheme = torch.per_tensor_affine
                    self.is_symmetric_quant = _is_symmetric_quant(self.activation_post_process.qscheme)
            """
            self.activation_post_process(x.detach())
            if self.dtype == torch.quint8:
                if self.activation_post_process.min_val >= 0.0:
                    self.qscheme = torch.per_tensor_affine
                    self.activation_post_process.qscheme = torch.per_tensor_affine
                    self.is_symmetric_quant = _is_symmetric_quant(self.activation_post_process.qscheme)
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.is_per_channel:
                if x.dim() == 4:
                    _scale = 2 * x.detach().abs().mean(dim=(1, 2, 3)) / math.sqrt(self.quant_max)
                else:
                    _scale = 2 * x.detach().abs().mean(dim=(1,)) / math.sqrt(self.quant_max)
            else:
                _scale = 2 * x.detach().abs().mean() / math.sqrt(self.quant_max)
            self.scale.data.copy_(_scale)
            self.zero_point.data.copy_(_zero_point)
            self.zero_point_init.data.copy_(_zero_point)
        else:
            self.scale.data.abs_()
            self.scale.data.clamp_(min=self.eps.item())
            self.update_observer_min_max()

        if self.fake_quant_enabled[0] == 1:
            if self.qscheme in (torch.per_channel_symmetric, torch.per_tensor_symmetric):
                self.zero_point.data.copy_(self.zero_point_init)
            elif self.qscheme == torch.per_tensor_affine and self.dtype == torch.quint8:
                self.zero_point.data.copy_(self.zero_point_init)
            if self.is_per_channel:
                grad_factor = 1.0 / (x.detach().numel()/self.out_channels * self.quant_max) ** 0.5
            else:
                grad_factor = 1.0 / (x.detach().numel() * self.quant_max) ** 0.5
            # pylint: disable=protected-access
            if self.qscheme in (
                    torch.per_channel_symmetric, torch.per_channel_affine):
                x = torch._fake_quantize_learnable_per_channel_affine(
                    x, self.scale, self.zero_point, self.ch_axis,
                    self.quant_min, self.quant_max, grad_factor)
            else:
                x = torch._fake_quantize_learnable_per_tensor_affine(
                    x, self.scale, self.zero_point,
                    self.quant_min, self.quant_max, grad_factor)
            # pylint: enable=protected-access
            return x
        return x

    @torch.jit.export
    def extra_repr(self):  # pylint: disable=missing-function-docstring
        return f'fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, ' \
               f'quant_min={self.quant_min}, quant_max={self.quant_max}, dtype={self.dtype}, ' \
               f'qscheme={self.qscheme}, ch_axis={self.ch_axis}, ' \
               f'scale={self.scale}, zero_point={self.zero_point}, out_channels={self.out_channels}'

    def update_observer_min_max(self):  # pylint: disable=missing-function-docstring
        if self.is_symmetric_quant:
            max_val = self.scale.data * (self.quant_max-self.quant_min)/2.0
        else:
            max_val = self.scale.data * (self.quant_max - self.quant_min)
        # if self.qscheme == torch.per_tensor_affine and self.dtype == torch.quint8:
        #    min_val = 0.0
        # min_val = self.scale.data * (self.quant_min - self.zero_point.data + 1)
        min_val = torch.zeros_like(self.scale.data)
        self.activation_post_process.max_val.resize_(max_val.shape)
        self.activation_post_process.min_val.resize_(min_val.shape)
        self.activation_post_process.max_val.copy_(max_val)
        self.activation_post_process.min_val.copy_(min_val)


def _fake_quantize_learnable_per_tensor_affine(g, x, scale, zero_point, quant_min,
                                               quant_max, grad_factor):
    return g.op("custom::LearnablePerTensorAffine", x, scale, zero_point, quant_min, quant_max)


def _fake_quantize_learnable_per_channel_affine(g, x, scale, zero_point, ch_axis,
                                                quant_min, quant_max, grad_factor):
    return g.op("custom::LearnablePerChannelAffine", x, scale, zero_point, ch_axis, quant_min, quant_max)


def fake_quantize_per_channel_affine(g, x, scale, zero_point, ch_axis, quant_min,
                                     quant_max):
    return g.op("custom::PerChannelAffine", x, scale, zero_point, ch_axis, quant_min, quant_max)


def fake_quantize_per_tensor_affine(g, x, scale, zero_point, quant_min, quant_max):
    return g.op("custom::PerTensorAffine", x, scale, zero_point, quant_min, quant_max)


def fused_moving_avg_obs_fake_quant(g, x, obs_enable, fake_quant_enable, runing_min,
                                    runing_max, scale, zero_point, avg_const, quant_min,
                                    quant_max, ch_axis, is_per_channel, is_symmetric):
    return g.op("custom::FusedMovingObserveFakeQuant", x, scale, zero_point, ch_axis, quant_min, quant_max)


register_custom_op_symbolic('::_fake_quantize_learnable_per_tensor_affine',
                            _fake_quantize_learnable_per_tensor_affine, 11)
register_custom_op_symbolic('::_fake_quantize_learnable_per_channel_affine',
                            _fake_quantize_learnable_per_channel_affine, 11)
register_custom_op_symbolic('::fake_quantize_per_channel_affine',
                            fake_quantize_per_channel_affine, 11)
register_custom_op_symbolic('::fake_quantize_per_tensor_affine',
                            fake_quantize_per_tensor_affine, 11)
register_custom_op_symbolic('::fused_moving_avg_obs_fake_quant',
                            fused_moving_avg_obs_fake_quant, 11)

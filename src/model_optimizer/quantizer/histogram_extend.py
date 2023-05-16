# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
quantization get min max with percentile
"""
import torch


class PerChannelHistogramExtendObserver(torch.quantization.observer._ObserverBase):  # pylint: disable=protected-access
    """
    per channel quantization get min max with percentile
    """
    def __init__(
            self,
            ch_axis=0,
            bins: int = 2048,
            dtype: torch.dtype = torch.qint8,
            qscheme=torch.per_channel_symmetric,
            reduce_range=False,
            quant_min=None,
            quant_max=None,
            factory_kwargs=None,
            method='percentile',
            percentile=99.99
    ):
        self.ch_axis = ch_axis
        quant_min = None
        quant_max = None
        super().__init__(dtype, qscheme, reduce_range, quant_min, quant_max)

        self.method = method
        self.percentile = percentile
        self.bins = bins
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer("calib_hist", torch.zeros(self.bins))
        self.register_buffer("calib_bin_edges", torch.zeros(self.bins + 1))
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def _compute_amax_percentile(self):
        """Returns amax that clips the percentile fraction of collected data"""

        if self.percentile < 0 or self.percentile > 100:
            raise ValueError("Invalid percentile. Must be in range 0 <= percentile <= 100.")

        # If calibrator hasn't collected any data, return none
        if self.init_state == 0:
            return None
        calib_hist_chs = self.calib_hist
        calib_bin_edges_chs = self.calib_bin_edges
        channel_size = calib_hist_chs.shape[0]
        percentile = self.percentile
        calib_amax = None
        for i in range(channel_size):
            calib_hist = calib_hist_chs[i]
            calib_bin_edges = calib_bin_edges_chs[i]
            total = calib_hist.sum()
            cdf = torch.cumsum(calib_hist / total, dim=0)
            idx = torch.searchsorted(cdf, percentile / 100)
            if calib_amax is None:
                calib_amax = calib_bin_edges[idx]
            else:
                calib_amax = torch.hstack((calib_amax, calib_bin_edges[idx]))
        calib_amax = torch.tensor(calib_amax)

        return calib_amax

    @torch.jit.export
    def calculate_qparams(self, **kwargs):  # pylint: disable=missing-function-docstring,disable=unused-argument
        is_uninitialized = self.min_val.shape == torch.Size([]) and self.max_val.shape == torch.Size([])
        if is_uninitialized:
            print(
                "warning, must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
            return torch.tensor([1.0], device=self.min_val.device.type), \
                torch.tensor([0], device=self.min_val.device.type)
        new_max = self._compute_amax_percentile()
        new_min = -1.0 * new_max

        return self._calculate_qparams(new_min, new_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        if x.numel() == 0:
            return x
        _x = x.detach()
        x_flat = torch.flatten(_x, 1)
        min_val, max_val = torch._aminmax(x_flat, 1)  # pylint: disable=protected-access
        self.min_val.resize_(min_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.resize_(max_val.shape)
        self.max_val.copy_(max_val)

        if torch.min(_x) < 0.:
            x_flat = x_flat.abs()

        x_flat_max, _ = x_flat.max(1)

        if self.init_state == 0:
            # first time it uses num_bins to compute histogram.
            channel_size = _x.shape[0]  # x shape [out_ch, in_ch, k_h, k_w]
            hists = []
            edges = []
            for i in range(channel_size):
                hists.append(torch.histc(torch.index_select(x_flat, dim=0, index=torch.tensor(i).to(_x.device)),
                                         bins=self.bins, min=0, max=x_flat_max[i]))
                edges.append(torch.linspace(0, x_flat_max[i], self.bins + 1))
            _hists = torch.stack(hists, 0)
            _edges = torch.stack(edges, 0)
            self.calib_hist.resize_(_hists.shape)
            self.calib_bin_edges.resize_(_edges.shape)
            self.calib_hist.data.copy_(_hists)
            self.calib_bin_edges.data.copy_(_edges)
            self.init_state.fill_(1)
        else:
            pass
        return x

    @torch.jit.export
    def reset_min_max_vals(self):  # pylint: disable=missing-function-docstring
        """Resets the min/max values."""
        self.min_val = torch.tensor([])  # pylint: disable=attribute-defined-outside-init
        self.max_val = torch.tensor([])  # pylint: disable=attribute-defined-outside-init


class HistogramExtendObserver(torch.quantization.observer._ObserverBase):  # pylint: disable=protected-access
    """
    per channel quantization get min max with percentile
    """
    def __init__(
            self,
            bins: int = 2048,
            dtype: torch.dtype = torch.quint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False,
            quant_min=None,
            quant_max=None,
            factory_kwargs=None,
            method='percentile',
            percentile=99.99
    ):
        quant_min = None
        quant_max = None

        super().__init__(dtype, qscheme, reduce_range, quant_min, quant_max)

        self.method = method
        self.percentile = percentile
        self.bins = bins
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer("calib_hist", torch.zeros(self.bins))
        self.register_buffer("calib_bin_edges", torch.zeros(self.bins + 1))
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def _compute_amax_percentile(self):
        """Returns amax that clips the percentile fraction of collected data"""

        if self.percentile < 0 or self.percentile > 100:
            raise ValueError("Invalid percentile. Must be in range 0 <= percentile <= 100.")

        # If calibrator hasn't collected any data, return none
        if self.init_state == 0:
            return None
        calib_hist = self.calib_hist
        calib_bin_edges = self.calib_bin_edges
        percentile = self.percentile
        total = calib_hist.sum()
        cdf = torch.cumsum(calib_hist / total, dim=0)
        idx = torch.searchsorted(cdf, percentile / 100)
        calib_amax = calib_bin_edges[idx]
        calib_amax = torch.tensor(calib_amax.item())

        return calib_amax

    @torch.jit.export
    def calculate_qparams(self, **kwargs):  # pylint: disable=missing-function-docstring,disable=unused-argument
        is_uninitialized = self.min_val == float("inf") and self.max_val == float(
            "-inf"
        )
        if is_uninitialized:
            print(
                "warning, must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
            return torch.tensor([1.0], device=self.min_val.device.type),\
                torch.tensor([0], device=self.min_val.device.type)
        assert self.bins == len(self.calib_hist), (
            "The number of bins in histogram should be equal to the number of bins "
            "supplied while making this observer"
        )

        new_max = self._compute_amax_percentile()
        new_min = -1.0 * new_max

        return self._calculate_qparams(new_min, new_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        if x.numel() == 0:
            return x
        _x = x.detach()
        min_val, max_val = torch._aminmax(_x)  # pylint: disable=protected-access
        self.min_val.resize_(min_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.resize_(max_val.shape)
        self.max_val.copy_(max_val)

        if torch.min(_x) < 0.:
            _x = _x.abs()
        x_max = _x.max()

        if self.init_state == 0:
            # first time it uses num_bins to compute histogram.
            self.calib_hist.data.copy_(torch.histc(_x, bins=self.bins, min=0, max=x_max))
            self.calib_bin_edges.data.copy_(torch.linspace(0, x_max, self.bins + 1))
            self.init_state.fill_(1)
        else:
            if x_max > self.calib_bin_edges[-1]:
                width = self.calib_bin_edges[1] - self.calib_bin_edges[0]
                self.bins = int((x_max / width).ceil().item())

                self.calib_bin_edges.resize_(self.bins + 1)
                self.calib_bin_edges.copy_(
                        torch.arange(0, x_max + width, width, device=_x.device))

            hist = torch.histc(_x, bins=self.bins, min=0, max=self.calib_bin_edges[-1])
            hist[:self.calib_hist.numel()] += self.calib_hist
            self.calib_hist.resize_(hist.shape)
            self.calib_hist.data.copy_(hist)
        return x

    @torch.jit.export
    def reset_min_max_vals(self):  # pylint: disable=missing-function-docstring
        """Resets the min/max values."""
        self.min_val = torch.tensor([])  # pylint: disable=attribute-defined-outside-init
        self.max_val = torch.tensor([])  # pylint: disable=attribute-defined-outside-init

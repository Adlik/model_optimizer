# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
A random ratio pruner.
"""

import numpy as np
import torch
from .structure_pruning import StructurePruner


class RatioPruner(StructurePruner):
    """A random ratio pruner.

    Each layer can adjust its own width ratio randomly and independently.

    Args:
        ratios (list | tuple): Width ratio of each layer can be
            chosen from `ratios` randomly. The width ratio is the ratio between
            the number of reserved channels and that of all channels in a
            layer. For example, if `ratios` is [0.25, 0.5], there are 2 cases
            for us to choose from when we sample from a layer with 12 channels.
            One is sampling the very first 3 channels in this layer, another is
            sampling the very first 6 channels in this layer. Default to None.
    """

    def __init__(self, ratios, **kwargs):
        super().__init__(**kwargs)
        ratios = list(ratios)
        ratios.sort()
        self.ratios = ratios
        self.min_ratio = ratios[0]

    def sample_subnet(self):
        """Random sample subnet by random mask.

        Returns:
            dict: Record the information to build the subnet from the supernet,
                its keys are the properties ``space_id`` in the pruner's search
                spaces, and its values are corresponding sampled out_mask.
        """
        subnet_dict = {}
        for space_id, out_mask in self.channel_spaces.items():
            subnet_dict[space_id] = self.get_channel_mask(out_mask)
        return subnet_dict

    def get_channel_mask(self, out_mask):
        """Randomly choose a width ratio of a layer from ``ratios``"""
        out_channels = out_mask.size(1)
        random_ratio = np.random.choice(self.ratios)
        new_channels = int(round(out_channels * random_ratio))
        assert new_channels > 0, \
            'Output channels should be a positive integer.'
        new_out_mask = torch.zeros_like(out_mask)
        new_out_mask[:, :new_channels] = 1

        return new_out_mask

    def set_min_channel(self):
        """Set the number of channels each layer to minimum."""
        subnet_dict = {}
        for space_id, out_mask in self.channel_spaces.items():
            out_channels = out_mask.size(1)
            random_ratio = self.min_ratio
            new_channels = int(round(out_channels * random_ratio))
            assert new_channels > 0, \
                'Output channels should be a positive integer.'
            new_out_mask = torch.zeros_like(out_mask)
            new_out_mask[:, :new_channels] = 1

            subnet_dict[space_id] = new_out_mask

        self.set_subnet(subnet_dict)

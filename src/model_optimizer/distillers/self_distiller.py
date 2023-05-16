# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
transfer knowledge inside a single model.
"""
from torch import nn


class SelfDistiller(nn.Module):
    """Transfer knowledge inside a single model.

    Args:
        criterion: nn.Module instance of loss criterion
    """

    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

        self.student_outputs = []
        self.teacher_outputs = []

    def compute_distill_loss(self, student_outputs, teacher_outputs):
        """Compute the distillation loss."""
        loss = self.criterion(student_outputs, teacher_outputs)
        return loss

    def forward(self, x):  # pylint: disable=missing-function-docstring
        pass

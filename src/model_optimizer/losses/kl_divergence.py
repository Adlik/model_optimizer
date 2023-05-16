# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KLDivergence
"""
from torch import nn
import torch.nn.functional as F


class KLDivergence(nn.Module):
    """A measure of how one probability distribution Q is different from a
    second, reference probability distribution P.
    """

    def __init__(
        self,
        temperature=1.0,
        reduction='batchmean',
        loss_weight=1.0,
    ):

        """

        Args:
            temperature (float): Temperature coefficient. Defaults to 1.0.
            reduction (str): Specifies the reduction to apply to the loss:
                              none | batchmean | sum | mean
                              none:  no reduction will be applied
                              batchmean:  the sum of the output will be divided by the batchsize,
                              sum: the output will be summed,
                              mean: the output will be divided by the number of elements in the output.
            loss_weight (float): Weight of loss. Defaults to 1.0.
        """
        super().__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight

        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction

    def forward(self, preds_student, preds_teacher):
        """Forward computation.

        Args:
            preds_student (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_teacher (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        softmax_pred_teacher = F.softmax(preds_teacher / self.temperature, dim=1)
        logsoftmax_preds_student = F.log_softmax(preds_student / self.temperature, dim=1)
        loss = (self.temperature**2) * F.kl_div(
            logsoftmax_preds_student, softmax_pred_teacher, reduction=self.reduction)
        return self.loss_weight * loss

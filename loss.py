# -*- coding:utf-8 -*-
# @Time   : 2021/12/17 12:10
# @Author : xinhongyang
# @File   : loss
import torch
from torch import nn


class LabelSmoothingLoss(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.01):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        # log_probs = nn.F.log_softmax(x, dim=-1)
        log_probs = torch.log_softmax(x, dim=-1)

        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

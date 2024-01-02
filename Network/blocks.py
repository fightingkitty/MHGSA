# -*- coding:utf-8 -*-
"""
@Time: 2022/10/28 1:33
@Author: Shuting Liu & Baochang Zhang
@IDE: PyCharm
@File: blocks.py
@Comment: #Enter some comments at here
"""
import torch
import torch.nn as nn


def InitWeights_He(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def eye_like(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    Return a tensor with same batch size as x, that has a nxn eye matrix in each sample in batch.

    Args:
        x: tensor of shape (B, *).

    Returns:
        tensor of shape (B, n, n) that has the same dtype and device as x.
    """
    return torch.eye(n, n, dtype=x.dtype, device=x.device).unsqueeze(0).repeat(x.shape[0], 1, 1)

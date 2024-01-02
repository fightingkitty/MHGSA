# -*- coding:utf-8 -*-
"""
@Time: 2022/10/28 18:17
@Author: Shuting Liu & Baochang Zhang
@IDE: PyCharm
@File: loss_func.py
@Comment: #Enter some comments at here
"""
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.autograd import Variable

class BCE(nn.Module):
    """
    does not requires one hot encoded target. c=1.
    N is the batch-size
    """
    def __init__(self):
        super(BCE, self).__init__()
        self.bce = nn.BCELoss(reduction='mean')
        self.eps = 1e-15

    def forward(self, input_data, target_data):
        # input_data = torch.clamp(input_data, self.eps, 1.0-self.eps)
        input_data = input_data.view(-1)
        target_data = target_data.view(-1)
        return self.bce(input_data, target_data)


class Weighted_BCE(nn.Module):
    """
    does not requires one hot encoded target. c=1.
    N is the batch-size
    """
    def __init__(self, alpha):
        super(Weighted_BCE, self).__init__()
        self.weight_0 = alpha
        self.weight_1 = 1-alpha
        self.eps = 1e-15

    def forward(self, input_data, target_data, reduction='mean'):
        # input_data = torch.clamp(input_data, self.eps, 1.0 - self.eps)
        input_data = input_data.view(-1)
        target_data = target_data.view(-1)
        class_1 = torch.eq(target_data, 1).float()
        class_0 = torch.eq(target_data, 0).float()
        weights = class_0 * self.weight_0 + class_1 * self.weight_1
        return F.binary_cross_entropy(input_data, target_data, weights, reduction=reduction)


class Binary_Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(Binary_Focal_loss, self).__init__()
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')
        self.wbce = Weighted_BCE(alpha)
        self.eps = 1e-15
        # self.drop = nn.Dropout(p=0.2)

    def forward(self, input_data, target_data):
        input_data = input_data.view(-1)
        target_data = target_data.view(-1)
        bce_loss = self.bce(input_data, target_data)
        pt = Variable(torch.exp(-bce_loss))
        # pt = torch.exp(-bce_loss)
        wbce = self.wbce(input_data, target_data)
        focal_loss = (1-pt)**self.gamma * wbce
        # focal_loss = self.drop(focal_loss)
        return focal_loss.mean()

class Binary_Focal_loss_v2(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(Binary_Focal_loss_v2, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input_data, target_data):
        input_data = input_data.view(-1)
        target_data = target_data.view(-1)
        bce_loss = F.binary_cross_entropy(input_data,target_data,reduction='none')
        p_t = input_data * target_data + (1 - input_data) * (1 - target_data)
        focal_loss = bce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * target_data + (1 - self.alpha) * (1 - target_data)
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class"""

    def __init__(self,):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, output, target):
        output = torch.clamp(output, self.eps, 1.0 - self.eps)
        output = output.contiguous().view(output.shape[0], -1)
        target = target.contiguous().view(output.shape[0], -1).float()
        num = 2 * torch.sum(torch.mul(output, target), dim=1)
        den = torch.sum(output + target, dim=1)+torch.tensor(0.1)
        loss = torch.mean(1-num/den)
        return loss


# -*- coding:utf-8 -*-
"""
@Time: 2022/10/27 16:52
@Author: Shuting Liu & Baochang Zhang
@IDE: PyCharm
@File: TabularNets.py
@Comment: #Enter some comments at here
"""
import torch.nn as nn
import torch
from thop import profile
from .blocks import InitWeights_He


class MIR(nn.Module):
    def __init__(self, in_channels=105, hidden_channels=64):
        super(MIR, self).__init__()
        self.offset_layer1 = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_channels, in_channels))
        # self.offset_layer1 =nn.Linear(in_channels, in_channels)
        self.offset_layer2 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.base_layer1 = nn.Linear(in_channels, hidden_channels)
        self.act_func = nn.LeakyReLU(inplace=True)
        InitWeights_He(self)

    def forward(self, x, vec):
        miss_pred = self.offset_layer1(x) * vec
        pred_offset = self.offset_layer2(miss_pred)
        x = self.base_layer1(x) + pred_offset
        return self.act_func(x)


class MLPs_mir(nn.Module):
    def __init__(self, in_channels=105, n_outputs=1, mlp_layers=3, hidden_channels=64):
        super(MLPs_mir, self).__init__()

        self.mir_block = MIR(in_channels, hidden_channels)
        MLP_blocks = []
        for i in range(mlp_layers-1):
            MLP_blocks += [nn.Linear(hidden_channels, hidden_channels),
                           nn.LeakyReLU(inplace=True)]

        self.MLPs = nn.Sequential(*MLP_blocks)

        self.fc = nn.Linear(hidden_channels, n_outputs)
        self.sigmoid = torch.nn.Sigmoid()
        InitWeights_He(self)

    def forward(self, x, vec):
        x = self.mir_block(x, vec)
        x = self.MLPs(x)
        x = self.fc(x)
        return self.sigmoid(x)


if __name__ == '__main__':
    model = MLPs_mir(hidden_channels=64)
    input_x = torch.randn(5, 105)
    input_vec = torch.randn(5, 105)
    print(model)
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())

    # count flops and parameters
    flops, params = profile(model, inputs=(input_x, input_vec,), verbose=False)
    print("flops: %e" % flops)
    print("params: %e" % params)
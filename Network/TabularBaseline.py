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


class Baseline_MLPs(nn.Module):
    def __init__(self, in_channels=105, n_outputs=1, mlp_layers=3, hidden_channels=64):
        super(Baseline_MLPs, self).__init__()

        MLP_blocks = [nn.Linear(in_channels, hidden_channels),
                      nn.LeakyReLU(inplace=True)]
        for i in range(mlp_layers-1):
            MLP_blocks += [nn.Linear(hidden_channels, hidden_channels),
                           nn.LeakyReLU(inplace=True)]

        self.MLPs = nn.Sequential(*MLP_blocks)
        self.fc = nn.Linear(hidden_channels, n_outputs)
        self.sigmoid = torch.nn.Sigmoid()
        InitWeights_He(self)

    def forward(self, x):
        x = self.MLPs(x)
        x = self.fc(x)
        return self.sigmoid(x)


if __name__ == '__main__':
    model = Baseline_MLPs(hidden_channels=64)
    input_temp = torch.randn(5, 105)
    print(model)
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())

    # count flops and parameters
    flops, params = profile(model, inputs=(input_temp,), verbose=False)
    print("flops: %e" % flops)
    print("params: %e" % params)
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

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, mlp_layers=2):
        super(MLP, self).__init__()
        if mlp_layers == 0:
            print('error: mlp_layers must be a positive integer, not incuding zero')
            return

        if mlp_layers >= 2:
            MLP_blocks = [nn.Linear(in_dim, hidden_dim),
                          nn.LeakyReLU(inplace=True)]
            for i in range(mlp_layers - 2):
                MLP_blocks += [nn.Linear(hidden_dim, hidden_dim),
                               nn.LeakyReLU(inplace=True)]
            MLP_blocks += [nn.Linear(hidden_dim, out_dim),
                           nn.LeakyReLU(inplace=True)]
            self.MLPs = nn.Sequential(*MLP_blocks)
        else:
            MLP_blocks = [nn.Linear(in_dim, out_dim),
                          nn.LeakyReLU(inplace=True)]
            self.MLPs = nn.Sequential(*MLP_blocks)

    def forward(self, X):
        for mlplayer in self.MLPs:
            X = mlplayer(X)
        return X


class Baseline_MLPs(nn.Module):
    def __init__(self,in_channels=47, n_outputs=1, hidden_channels=32, num_node=131, mlp_layers=2, out_mlpfeats=8):
        super(Baseline_MLPs, self).__init__()

        self.mlp = MLP(in_channels, out_mlpfeats, hidden_channels, mlp_layers=mlp_layers)
        self.fc1 = nn.Sequential(
            nn.Linear(out_mlpfeats*num_node, hidden_channels),
            nn.LeakyReLU(inplace=True))
        self.fc2 = nn.Linear(hidden_channels, n_outputs)
        self.sigmoid = torch.nn.Sigmoid()
        InitWeights_He(self)

    def forward(self, x):
        x = self.mlp(x)
        shape = x.shape
        x = x.view(shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.sigmoid(x)

class Baseline_RNN(nn.Module):
    def __init__(self, node_channels=47, n_outputs=1, rnn_layers=3, hidden_channels=64, dropout_rate=0):
        super(Baseline_RNN, self).__init__()

        self.rnn = nn.RNN(node_channels, hidden_channels, rnn_layers, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_channels, n_outputs)
        self.sigmoid = torch.nn.Sigmoid()
        InitWeights_He(self)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return self.sigmoid(x)

class Baseline_LSTM(nn.Module):
    def __init__(self, node_channels=47, n_outputs=1, rnn_layers=3, hidden_channels=64, dropout_rate=0):
        super(Baseline_LSTM, self).__init__()

        self.lstm = nn.LSTM(node_channels, hidden_channels, rnn_layers, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_channels, n_outputs)
        self.sigmoid = torch.nn.Sigmoid()
        InitWeights_He(self)

    def forward(self, x):
        x, (_, _) = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return self.sigmoid(x)

class Baseline_GRU(nn.Module):
    def __init__(self, node_channels=47, n_outputs=1, rnn_layers=3, hidden_channels=64, dropout_rate=0):
        super(Baseline_GRU, self).__init__()

        self.gru = nn.GRU(node_channels, hidden_channels, rnn_layers, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_channels, n_outputs)
        self.sigmoid = torch.nn.Sigmoid()
        InitWeights_He(self)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])
        return self.sigmoid(x)


if __name__ == '__main__':
    model = Baseline_MLPs(mlp_layers=4, hidden_channels=64, mlpout_channel=8)
    # model = Baseline_GRU(hidden_channels=32)
    input_x = torch.zeros(5, 131, 47)
    # base_layers = list(model.children())
    # print('base_layer:',base_layers)
    print(model)
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())

    # count flops and parameters
    flops, params = profile(model, inputs=(input_x,), verbose=False)
    print("flops: %e" % flops)
    print("params: %e" % params)


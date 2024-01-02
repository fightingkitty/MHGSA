# -*- coding:utf-8 -*-
"""
@Time: 2023/1/25 0:41
@Author: Shuting Liu & Baochang Zhang
@IDE: PyCharm
@File: Tabular_Graph_Nets.py
@Comment: #Enter some comments at here
"""
import torch.nn as nn
import torch
from thop import profile
from .blocks import InitWeights_He
from .GraphNets import Dynamic_GCN,GCN,Embeding
from .TabularNets import MIR

class Tabular_Feature_Extractor(nn.Module):
    def __init__(self, in_channels_T=105, hidden_channels_T=64, mlp_layers=2):
        super(Tabular_Feature_Extractor, self).__init__()

        self.mir_block = MIR(in_channels_T, hidden_channels_T)

        MLP_blocks = []
        for i in range(mlp_layers):
            MLP_blocks += [nn.Linear(hidden_channels_T, hidden_channels_T),
                           nn.LeakyReLU(inplace=True)]

        self.MLPs = nn.Sequential(*MLP_blocks)
        InitWeights_He(self)

    def forward(self, x, vec):
        x = self.mir_block(x, vec)
        x = self.MLPs(x)
        return x


class TGNN_Fusions_Baseline(nn.Module):
    def __init__(self, in_channels_T=105, hidden_channels_T=64, mlp_layers=2, gcnmodel= None, in_channels_G=47,
                 hidden_channels_G=32, out_graphfeats=8, num_node=131, gcn_layers=3, n_outputs=1):
        super(TGNN_Fusions_Baseline, self).__init__()

        # self.input_embeding = Embeding(in_channels_G, num_node)
        self.tnn = Tabular_Feature_Extractor(in_channels_T, hidden_channels_T, mlp_layers)
        self.gnn = gcnmodel(in_channels_G, out_graphfeats, hidden_channels_G, gcn_layers=gcn_layers)
        self.gnn_fc = nn.Sequential(
            nn.Linear(out_graphfeats * num_node, hidden_channels_G * 8),
            nn.LeakyReLU(inplace=True))

        self.fc = nn.Linear(hidden_channels_G * 8+hidden_channels_T, n_outputs)
        self.sigmoid = torch.nn.Sigmoid()
        InitWeights_He(self)

    def forward(self, Tx, vec, Gx, adj_matrix):
        tabular_feature = self.tnn(Tx, vec)

        graph_feature = self.gnn(Gx, adj_matrix)
        shape = graph_feature.shape
        graph_feature = graph_feature.view(shape[0], -1)
        graph_feature = self.gnn_fc(graph_feature)

        out_feature = torch.cat((tabular_feature, graph_feature), -1)
        out = self.fc(out_feature)
        return self.sigmoid (out)

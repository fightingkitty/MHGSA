# -*- coding:utf-8 -*-
"""
@Time: 2023/1/25 3:46
@Author: Shuting Liu & Baochang Zhang
@IDE: PyCharm
@File: Tabular_Graph_FusionNet.py
@Comment: #Enter some comments at here
"""
import torch.nn as nn
import torch
from thop import profile
from .blocks import InitWeights_He
from .GraphNets import preprocess_adj
from .TabularNets import MIR
import torch.nn.functional as F

class Dynamic_GCN_FusionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, norm=False):
        super(Dynamic_GCN_FusionLayer, self).__init__()

        self.G_v_proj = nn.Sequential(nn.Linear(in_dim, out_dim),
                                    nn.LeakyReLU(inplace=True))

        self.G_k_proj = nn.Linear(in_dim, out_dim)

        self.spatial_scale = out_dim ** -0.5
        self.edge_q_proj =  nn.Sequential(nn.Linear(in_dim, out_dim),
                                          nn.ReLU(inplace=True))
        self.edge_k_proj =  nn.Sequential(nn.Linear(in_dim, out_dim),
                                          nn.ReLU(inplace=True))

        self.T_q_proj = nn.Linear(in_dim, out_dim)
        self.T_v_proj = nn.Sequential(nn.Linear(in_dim, out_dim),
                                      nn.LeakyReLU(inplace=True))

        if norm:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = None


    def forward(self, tabular_feats, node_feats, adj_matrix):
        G_X = self.G_v_proj(node_feats)  # B N C/ feature-wise
        T_X = self.T_v_proj(tabular_feats)  # B N C/ feature-wise

        G_k_norm = self.G_k_proj(node_feats)  # B N C/ feature-wise & normalize
        T_q_norm = self.T_q_proj(tabular_feats)  # B 1 C/ feature-wise & normalize
        # G_k_norm = F.normalize(self.G_k_proj(node_feats),dim=-1)  # B N C/ feature-wise & normalize
        # T_q_norm = F.normalize(self.T_q_proj(tabular_feats),dim=-1)  # B 1 C/ feature-wise & normalize
        # node_att = torch.relu(torch.einsum("bxh,byh->bxy", T_q_norm, G_k_norm))  # B 1 N
        node_att = torch.einsum("bxh,byh->bxy", T_q_norm, G_k_norm).softmax(dim=-1)  # B 1 N

        # confidence = (self.sig(self.f((self.sig(self.f(T_X))-0.5)/0.5*T_X))-0.5)/0.5 # B 1 1
        # node_att = torch.transpose(confidence * node_att + 1, -1, -2) # B N 1
        node_att = torch.transpose(node_att+1, -1, -2) # B N 1

        G_X = node_att * G_X

        q = self.edge_q_proj(node_feats)  # B N C  / feature-wise
        k = self.edge_k_proj(node_feats)  # B N C
        att = torch.einsum("bxh,byh->bxy", q, k) * self.spatial_scale  # B N N
        # att = torch.sigmoid(att)
        att = torch.sigmoid(att + torch.transpose(att, -1, -2))

        A = preprocess_adj(adj_matrix*att)
        G_X = torch.bmm(A, G_X)
        if self.norm:
            G_X = self.norm(G_X)
        return T_X,G_X

class Dynamic_GCN_Fusion(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, gcn_layers=3):
        super(Dynamic_GCN_Fusion, self).__init__()
        if gcn_layers == 0:
            print('error: gcn_layers must be a positive integer, not incuding zero')
            return

        if gcn_layers > 1:
            GCN_blocks = [Dynamic_GCN_FusionLayer(in_dim, hidden_dim)]
            for i in range(gcn_layers - 2):
                GCN_blocks += [Dynamic_GCN_FusionLayer(hidden_dim, hidden_dim)]
            GCN_blocks += [Dynamic_GCN_FusionLayer(hidden_dim, out_dim)]
            self.GCNs = nn.ModuleList(GCN_blocks)
        else:
            GCN_blocks = [Dynamic_GCN_FusionLayer(in_dim, out_dim)]
            self.GCNs = nn.ModuleList(GCN_blocks)

    def forward(self, tabular_feats, node_feats, adj_matrix):
        for gcnlayer in self.GCNs:
            tabular_feats, node_feats = gcnlayer(tabular_feats, node_feats, adj_matrix)
        return node_feats

class GCN_FusionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, norm=False):
        super(GCN_FusionLayer, self).__init__()

        self.G_v_proj = nn.Sequential(nn.Linear(in_dim, out_dim),
                                    nn.LeakyReLU(inplace=True))

        self.G_k_proj = nn.Linear(in_dim, out_dim)

        self.T_q_proj = nn.Linear(in_dim, out_dim)

        self.T_v_proj = nn.Sequential(nn.Linear(in_dim, out_dim),
                                      nn.LeakyReLU(inplace=True))

        if norm:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = None


    def forward(self, tabular_feats, node_feats, adj_matrix):
        G_X = self.G_v_proj(node_feats)  # B N C/ feature-wise

        G_k_norm = self.G_k_proj(node_feats)  # B N C/ feature-wise & normalize
        T_q_norm = self.T_q_proj(tabular_feats)  # B 1 C/ feature-wise & normalize
        node_att = torch.einsum("bxh,byh->bxy", T_q_norm, G_k_norm).softmax(dim=-1) # B 1 N

        node_att = torch.transpose(node_att+1, -1, -2) # B N 1

        G_X = node_att * G_X

        A = preprocess_adj(adj_matrix)
        G_X = torch.bmm(A, G_X)
        if self.norm:
            G_X = self.norm(G_X)

        T_X = self.T_v_proj(tabular_feats)  # B 1 C/ feature-wise
        return T_X, G_X

class GCN_Fusion(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, gcn_layers=3):
        super(GCN_Fusion, self).__init__()
        if gcn_layers == 0:
            print('error: gcn_layers must be a positive integer, not incuding zero')
            return

        if gcn_layers > 1:
            GCN_blocks = [GCN_FusionLayer(in_dim, hidden_dim)]
            for i in range(gcn_layers - 2):
                GCN_blocks += [GCN_FusionLayer(hidden_dim, hidden_dim)]
            GCN_blocks += [GCN_FusionLayer(hidden_dim, out_dim)]
            self.GCNs = nn.ModuleList(GCN_blocks)
        else:
            GCN_blocks = [GCN_FusionLayer(in_dim, out_dim)]
            self.GCNs = nn.ModuleList(GCN_blocks)

    def forward(self, tabular_feats, node_feats, adj_matrix):
        for gcnlayer in self.GCNs:
            tabular_feats, node_feats = gcnlayer(tabular_feats, node_feats, adj_matrix)
        return node_feats

class TGNN_Fusions(nn.Module):
    def __init__(self, in_channels_T=105, gcnmodel= None, in_channels_G=47,
                 hidden_channels_G=32, out_graphfeats=8, num_node=131, gcn_layers=3, n_outputs=1):
        super(TGNN_Fusions, self).__init__()

        self.T2G_mapping = MIR(in_channels_T, in_channels_G)

        self.gnn = gcnmodel(in_channels_G, out_graphfeats, hidden_channels_G, gcn_layers=gcn_layers)

        self.gnn_fc = nn.Sequential(
            nn.Linear(out_graphfeats * num_node, hidden_channels_G * 8),
            nn.LeakyReLU(inplace=True))

        self.fc = nn.Linear(hidden_channels_G * 8, n_outputs)
        self.sigmoid = torch.nn.Sigmoid()
        InitWeights_He(self)

    def forward(self, Tx, vec, Gx, adj_matrix):

        tabular_feature = self.T2G_mapping(Tx, vec)
        tabular_feature = torch.unsqueeze(tabular_feature,dim=1)
        graph_feature = self.gnn(tabular_feature, Gx, adj_matrix)
        shape = graph_feature.shape
        graph_feature = graph_feature.view(shape[0], -1)
        graph_feature = self.gnn_fc(graph_feature)
        out = self.fc(graph_feature)
        return self.sigmoid (out)


if __name__ == '__main__':
    model = TGNN_Fusions( in_channels_T=20, gcnmodel= GCN_Fusion, in_channels_G=18,
                          hidden_channels_G=32, out_graphfeats=8, num_node=131, gcn_layers=3, n_outputs=1)


    input_temp = torch.randn(4, 20)
    input_V_temp = torch.randn(4, 20)
    input_X = torch.randn(4, 138, 18)
    input_A = torch.randint(0, 2, (4, 138, 138), dtype=torch.float32)
    print(model)
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())

    # count flops and parameters
    flops, params = profile(model, inputs=(input_temp, input_V_temp, input_A, input_X), verbose=False)
    print("flops: %e" % flops)
    print("params: %e" % params)
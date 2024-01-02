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
# from monai.utils import optional_import

# Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")
from typing import Union

def preprocess_adj(A):
    '''
    Pre-process adjacency matrix
    :param A: adjacency matrix
    :return:
    '''
    D_hat_diag = torch.sum(A, dim=-1)
    D_hat_diag_inv_sqrt = torch.pow(D_hat_diag, -0.5)
    D_hat_diag_inv_sqrt[torch.isinf(D_hat_diag_inv_sqrt)] = 0.
    D_hat_inv_sqrt = torch.diag_embed(D_hat_diag_inv_sqrt)
    return torch.matmul(torch.matmul(D_hat_inv_sqrt, A), D_hat_inv_sqrt)

def preprocess_adj_average(A):
    '''
    Pre-process adjacency matrix
    :param A: adjacency matrix
    :return:
    pls use function preprocess_adj!
    '''
    D_hat_diag = torch.sum(A, dim=-1)
    D_hat_diag_inv = torch.pow(D_hat_diag, -1.0)
    D_hat_diag_inv[torch.isinf(D_hat_diag_inv)] = 0.
    D_hat_inv = torch.diag_embed(D_hat_diag_inv)
    return torch.matmul(D_hat_inv,A)

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, norm=False):
        super(GCNLayer, self).__init__()
        self.linear = nn.Sequential(nn.Linear(in_dim, out_dim),
                                    nn.LeakyReLU(inplace=True))
        if norm:
            self.norm = nn.LayerNorm(out_dim,elementwise_affine=False)
        else:
            self.norm = None

    def forward(self, node_feats, adj_matrix):
        A = preprocess_adj(adj_matrix)
        X = self.linear(node_feats)
        X = torch.bmm(A, X)
        if self.norm:
            X = self.norm(X)
        return X

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, gcn_layers=2):
        super(GCN, self).__init__()
        if gcn_layers == 0:
            print('error: gcn_layers must be a positive integer, not incuding zero')
            return

        if gcn_layers >1:
            GCN_blocks = [GCNLayer(in_dim, hidden_dim)]
            for i in range(gcn_layers - 2):
                GCN_blocks += [GCNLayer(hidden_dim, hidden_dim)]
            GCN_blocks += [GCNLayer(hidden_dim, out_dim)]
            self.GCNs = nn.ModuleList(GCN_blocks)
        else:
            GCN_blocks = [GCNLayer(in_dim, out_dim)]
            self.GCNs = nn.ModuleList(GCN_blocks)

    def forward(self, X, adj_matrix):
        for gcnlayer in self.GCNs:
            X = gcnlayer(X, adj_matrix)
        return X

class Dynamic_GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, norm=False):
        super(Dynamic_GCNLayer, self).__init__()

        self.linear = nn.Sequential(nn.Linear(in_dim, out_dim),
                                    nn.LeakyReLU(inplace=True))
        self.spatial_scale = out_dim ** -0.5
        self.q_proj =  nn.Sequential(nn.Linear(in_dim, out_dim,bias=False),
                                    nn.LeakyReLU(inplace=True))
        self.k_proj =  nn.Sequential(nn.Linear(in_dim, out_dim,bias=False),
                                    nn.LeakyReLU(inplace=True))

        if norm:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = None


    def forward(self, node_feats, adj_matrix):
        X = self.linear(node_feats)  # B N C/ feature-wise

        q = self.q_proj(node_feats)  # B N C  / feature-wise
        k = self.k_proj(node_feats)  # B N C

        att = torch.einsum("bxh,byh->bxy", q, k) * self.spatial_scale  # B N N
        att = torch.sigmoid(att)
        # att = torch.sigmoid(att + torch.transpose(att, -1, -2))

        A = preprocess_adj(adj_matrix*att)
        X = torch.bmm(A, X)
        # X = torch.bmm(A, X) + node_feats
        if self.norm:
            X = self.norm(X)
        return X


class Dynamic_GCN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, gcn_layers=3):
        super(Dynamic_GCN, self).__init__()
        if gcn_layers == 0:
            print('error: gcn_layers must be a positive integer, not incuding zero')
            return

        if gcn_layers >= 2:
            WGCN_blocks = [Dynamic_GCNLayer(in_dim, hidden_dim)]
            for i in range(gcn_layers - 2):
                WGCN_blocks += [Dynamic_GCNLayer(hidden_dim, hidden_dim)]
            WGCN_blocks += [Dynamic_GCNLayer(hidden_dim, out_dim)]
            self.WGCNs = nn.ModuleList(WGCN_blocks)
        else:
            WGCN_blocks = [Dynamic_GCNLayer(in_dim, out_dim)]
            self.WGCNs = nn.ModuleList(WGCN_blocks)

    def forward(self, X, adj_matrix):
        for wgcnlayer in self.WGCNs:
            X = wgcnlayer(X, adj_matrix)
        return X


class Dynamic_MHGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=2, norm=False):
        super(Dynamic_MHGCNLayer, self).__init__()

        self.linear = nn.Sequential(nn.Linear(in_dim, out_dim),
                                    nn.LeakyReLU(inplace=True))

        self.num_heads = num_heads

        self.head_dim = out_dim//num_heads

        self.spatial_scale = self.head_dim ** -0.5

        self.q_proj =  nn.Sequential(nn.Linear(in_dim, out_dim,bias=False),
                                    nn.LeakyReLU(inplace=True))
        self.k_proj =  nn.Sequential(nn.Linear(in_dim, out_dim,bias=False),
                                    nn.LeakyReLU(inplace=True))


        if norm:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = None

    def forward(self, node_feats, adj_matrix):

        batch_size, num_node, _ = node_feats.size()
        X = self.linear(node_feats).reshape(batch_size, num_node,self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B H N C/ feature-wise
        q = self.q_proj(node_feats).reshape(batch_size, num_node,self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B H N C  / feature-wise
        k = self.k_proj(node_feats).reshape(batch_size, num_node,self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B H N C

        att = torch.einsum("blxh,blyh->blxy", q, k) * self.spatial_scale  # B H N N
        att = torch.sigmoid(att)
        adj_heads = adj_matrix.unsqueeze(1)*att
        A = preprocess_adj(adj_heads)
        X = torch.einsum("bhxy,bhyd->bhxd", A, X).permute(0,2,1,3).reshape(batch_size, num_node, -1)
        if self.norm:
            X = self.norm(X)
        return X


class Dynamic_MHGCN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, gcn_layers=3):
        super(Dynamic_MHGCN, self).__init__()
        if gcn_layers == 0:
            print('error: gcn_layers must be a positive integer, not incuding zero')
            return

        if gcn_layers >= 2:
            WGCN_blocks = [Dynamic_MHGCNLayer(in_dim, hidden_dim)]
            for i in range(gcn_layers - 2):
                WGCN_blocks += [Dynamic_MHGCNLayer(hidden_dim, hidden_dim)]
            WGCN_blocks += [Dynamic_MHGCNLayer(hidden_dim, out_dim)]
            self.WGCNs = nn.ModuleList(WGCN_blocks)
        else:
            WGCN_blocks = [Dynamic_MHGCNLayer(in_dim, out_dim)]
            self.WGCNs = nn.ModuleList(WGCN_blocks)

    def forward(self, X, adj_matrix):
        for wgcnlayer in self.WGCNs:
            X = wgcnlayer(X, adj_matrix)
        return X


class GNN_Collections(nn.Module):
    def __init__(self, gcnmodel= None, in_channels=47, n_outputs=1, hidden_channels=32, num_node=131, gcn_layers=3, out_graphfeats=8):
        super(GNN_Collections, self).__init__()

        # self.input_embeding = Embeding(in_channels,num_node)
        self.gnn = gcnmodel(in_channels, out_graphfeats, hidden_channels, gcn_layers=gcn_layers)
        self.fc1 = nn.Sequential(
            nn.Linear(out_graphfeats * num_node, hidden_channels * 8),
            nn.LeakyReLU(inplace=True))
        self.fc2 = nn.Linear(hidden_channels * 8, n_outputs)

        self.sigmoid = torch.nn.Sigmoid()
        InitWeights_He(self)

    def forward(self, X, adj_matrix):
        X = self.gnn(X,adj_matrix)
        shape = X.shape
        X = X.view(shape[0], -1)
        X = self.fc1(X)
        X = self.fc2(X)
        return self.sigmoid(X)

class GNN_Collections_v2(nn.Module):
    def __init__(self, gcnmodel= None, in_channels=47, n_outputs=1, hidden_channels=32, num_node=131,key_node=1, gcn_layers=3, out_graphfeats=8):
        super(GNN_Collections_v2, self).__init__()

        # self.input_embeding = Embeding(in_channels,num_node)
        self.gnn = gcnmodel(in_channels, out_graphfeats, hidden_channels, gcn_layers=gcn_layers)

        self.fc1 = nn.Sequential(nn.Linear(num_node, key_node),
                                 nn.LeakyReLU(inplace=True))

        self.fc2 = nn.Linear(out_graphfeats * key_node, n_outputs)

        self.sigmoid = torch.nn.Sigmoid()
        InitWeights_He(self)

    def forward(self, X, adj_matrix):
        X = self.gnn(X,adj_matrix)
        X = X.permute(0, 2, 1)
        X = self.fc1(X)
        shape = X.shape
        X = X.view(shape[0], -1)
        X = self.fc2(X)
        return self.sigmoid(X)

class Embeding(nn.Module):
    def __init__(self,in_dim, num_node):
        super(Embeding, self).__init__()
        # pretrained
        self.mlp_layer1 = nn.Sequential(nn.Linear(in_dim, in_dim),
                                        nn.LeakyReLU(inplace=True))

        self.mlp_layer2 = nn.Sequential(nn.Linear(num_node, num_node),
                                        nn.LeakyReLU(inplace=True))

    def forward(self, node_feats):
        X = self.mlp_layer1(node_feats)
        Tx = torch.transpose(X, -1, -2)  # B C N
        Tx = self.mlp_layer2(Tx)
        X = torch.transpose(Tx, -1, -2)  # B N C
        return X



#==================================================

class Dynamic_Edge(nn.Module):
    def __init__(self,in_dim, out_dim):
        super(Dynamic_Edge, self).__init__()
        # pretrained
        self.q_proj = nn.Sequential(nn.Linear(in_dim, out_dim),
                                    nn.LeakyReLU(inplace=True))
        self.k_proj = nn.Sequential(nn.Linear(in_dim, out_dim),
                                    nn.LeakyReLU(inplace=True))
        self.spatial_scale = (out_dim) ** -0.5

    def forward(self, node_feats):
        q = self.q_proj(node_feats)  # B N C  / feature-wise
        k = self.k_proj(node_feats)  # B N C

        att = torch.einsum("bxh,byh->bxy", q, k)  # B N N
        att = torch.sigmoid(att)
        return att

class Dynamic_GCN_V2(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, gcn_layers=3):
        super(Dynamic_GCN_V2, self).__init__()
        if gcn_layers == 0:
            print('error: gcn_layers must be a positive integer, not incuding zero')
            return
        self.edge_weight = Dynamic_Edge(in_dim, out_dim)
        if gcn_layers >= 2:
            GCN_blocks = [GCNLayer(in_dim, hidden_dim)]
            for i in range(gcn_layers - 2):
                GCN_blocks += [GCNLayer(hidden_dim, hidden_dim)]
            GCN_blocks += [GCNLayer(hidden_dim, out_dim)]
            self.GCNs = nn.ModuleList(GCN_blocks)
        else:
            GCN_blocks = [GCNLayer(in_dim, out_dim)]
            self.GCNs = nn.ModuleList(GCN_blocks)

    def forward(self, X, adj_matrix):
        att =self.edge_weight(X)
        adj = att*adj_matrix
        for gcnlayer in self.GCNs:
            X = gcnlayer(X, adj)
        return X


class Residual_GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, norm=False):
        super(Residual_GCNLayer, self).__init__()
        self.linear = nn.Sequential(nn.Linear(in_dim, out_dim),
                                    nn.LeakyReLU(inplace=True))
        if norm:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = None

    def forward(self, node_feats, adj_matrix):
        A = preprocess_adj(adj_matrix)
        X = self.linear(node_feats)
        X = torch.bmm(A, X)+node_feats
        if self.norm:
            X = self.norm(X)
        return X

class Residual_GCN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_node=None, gcn_layers=3):
        super(Residual_GCN, self).__init__()
        if gcn_layers == 0:
            print('error: gcn_layers must be a positive integer, not incuding zero')
            return

        if gcn_layers >1:
            GCN_blocks = [GCNLayer(in_dim, hidden_dim)]
            for i in range(gcn_layers - 2):
                GCN_blocks += [Residual_GCNLayer(hidden_dim, hidden_dim)]
            GCN_blocks += [GCNLayer(hidden_dim, out_dim)]
            self.GCNs = nn.ModuleList(GCN_blocks)
        else:
            GCN_blocks = [GCNLayer(in_dim, out_dim)]
            self.GCNs = nn.ModuleList(GCN_blocks)

    def forward(self, X, adj_matrix):
        for gcnlayer in self.GCNs:
            X = gcnlayer(X, adj_matrix)
        return X

class Channel_Transformer_GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_node):
        super(Channel_Transformer_GCNLayer, self).__init__()
        self.linear = nn.Sequential(nn.Linear(in_dim, out_dim),
                                    nn.LeakyReLU(inplace=True))

        self.channel_scale = num_node ** -0.5
        self.q_proj = nn.Linear(num_node, num_node)
        self.k_proj = nn.Linear(num_node, num_node)

        self.v_proj = nn.Sequential(nn.Linear(num_node, num_node),
                                    nn.LeakyReLU(inplace=True))
        self.norm = nn.LayerNorm(num_node)

    def forward(self, node_feats, adj_matrix):
        A = preprocess_adj(adj_matrix)

        X = self.linear(node_feats)  # B N C/ feature-wise

        Tx = torch.transpose(X,-1,-2)  # B C N
        q = self.q_proj(Tx)  # B C N  / node-wise
        k = self.k_proj(Tx)  # B C N
        # v = self.v_proj(Tx)  # B C N
        att = (torch.einsum("bxh,byh->bxy", q, k) * self.channel_scale).softmax(dim=-1)  # B C C
        Tx = torch.einsum("bxy, byh->bxh", att, Tx)  # B C N
        # Tx = Tx + vx # B C N
        X = torch.transpose(Tx, -1,-2)  # B N C
        X = torch.bmm(A, X)
        return X

class CTGCN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_node, gcn_layers=2):
        super(CTGCN, self).__init__()
        if gcn_layers == 0:
            print('error: gcn_layers must be a positive integer, not incuding zero')
            return

        if gcn_layers >= 2:
            CTGCN_blocks = [Channel_Transformer_GCNLayer(in_dim, hidden_dim, num_node)]
            for i in range(gcn_layers - 2):
                CTGCN_blocks += [Channel_Transformer_GCNLayer(hidden_dim, hidden_dim, num_node)]
            CTGCN_blocks += [Channel_Transformer_GCNLayer(hidden_dim, out_dim, num_node)]
            self.CTGCNs = nn.ModuleList(CTGCN_blocks)
        else:
            CTGCN_blocks = [Channel_Transformer_GCNLayer(in_dim, out_dim, num_node)]
            self.CTGCNs = nn.ModuleList(CTGCN_blocks)

    def forward(self, X, adj_matrix):
        for tcgcnlayer in self.CTGCNs:
            X = tcgcnlayer(X, adj_matrix)
        return X


class Channel_Transformer_WGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_node):
        super(Channel_Transformer_WGCNLayer, self).__init__()
        self.linear = nn.Sequential(nn.Linear(in_dim, out_dim),
                                    nn.LeakyReLU(inplace=True))
        self.acti = nn.LeakyReLU(inplace=True)
        self.channel_scale = num_node ** -0.5
        self.q_proj = nn.Linear(num_node, num_node)
        self.k_proj = nn.Linear(num_node, num_node)

        self.v_proj = nn.Sequential(nn.Linear(num_node, num_node),
                                    nn.LeakyReLU(inplace=True))

        self.norm1 = nn.LayerNorm(num_node)
        self.norm2 = nn.LayerNorm(out_dim)
        self.edge_weight = nn.Parameter(torch.ones(num_node, num_node), requires_grad=True)

    def forward(self, node_feats, adj_matrix):
        W = torch.sigmoid(self.edge_weight)
        W_A = adj_matrix * W
        W_A = preprocess_adj(W_A)

        X = self.linear(node_feats)  # B N C/ feature-wise
        Tx = torch.transpose(X,-1,-2)  # B C N
        Tx = self.norm1(Tx)
        q = self.q_proj(Tx)  # B C N  / node-wise
        k = self.k_proj(Tx)  # B C N
        v = self.v_proj(Tx)  # B C N
        att = (torch.einsum("bxh,byh->bxy", q, k) * self.channel_scale).softmax(dim=-1)  # B C C
        vx = torch.einsum("bxy, byh->bxh", att, v)  # B C N
        Tx = Tx + vx # B C N
        X = torch.transpose(Tx, -1,-2)  # B N C
        X = torch.bmm(W_A, X)
        X = self.norm2(X)
        return X

class Graph_inference_layer(nn.Module):
    def __init__(self, in_dim, num_node, norm=False):
        super(Graph_inference_layer, self).__init__()
        self.channel_proj = nn.Sequential(nn.Linear(in_dim, in_dim),
                                          nn.LeakyReLU(inplace=True))
        self.spatial_proj = nn.Sequential(nn.Linear(num_node, num_node),
                                          nn.LeakyReLU(inplace=True))
        if norm:
            self.norm1 = nn.LayerNorm(num_node)
            self.norm2 = nn.LayerNorm(in_dim)
        else:
            self.norm1 = None
            self.norm2 = None

    def forward(self, node_feats):
        X = self.channel_proj(node_feats)
        X = torch.transpose(X,-1,-2)
        if self.norm1:
            X = self.norm1(X)
        X = self.spatial_proj(X)
        X = torch.transpose(X,-1,-2)
        if self.norm2:
            X = self.norm2(X)
        return X

class GCN_fusion_Layer_(nn.Module):
    def __init__(self, in_dim1, in_dim2, out_dim, num_node, num_graph_inference=2, norm=False):
        super(GCN_fusion_Layer_, self).__init__()

        self.linear_tabular = nn.Sequential(nn.Linear(in_dim2, out_dim),
                                    nn.LeakyReLU(inplace=True))

        self.linear_node = nn.Sequential(nn.Linear(in_dim1, out_dim),
                                         nn.LeakyReLU(inplace=True))

        Graph_inference_blocks = []
        for i in range(num_graph_inference):
            Graph_inference_blocks+=[Graph_inference_layer(out_dim,num_node)]
        self.graph_inference = nn.ModuleList(Graph_inference_blocks)
        self.z_out = nn.Sequential(nn.Linear(num_node, 1),
                                   nn.LeakyReLU(inplace=True))

        if norm:
            self.norm_tabular = nn.LayerNorm(out_dim)
            self.norm_graph = nn.LayerNorm(out_dim)
        else:
            self.norm_tabular = None
            self.norm_graph = None

    def forward(self, tabular_feats, node_feats, adj_matrix):
        A = preprocess_adj(adj_matrix)
        X = self.linear(node_feats)
        X = torch.bmm(A, X)
        if self.norm_graph:
            X = self.norm_graph(X)
        X = self.graph_inference(X)
        X = torch.transpose(X,-1,-2) # B C N
        Z = self.z_out(X)
        Y = self.linear_tabular(tabular_feats)
        # todo: add layernorm

        return Y,Z


class CTWGCN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_node, gcn_layers=2):
        super(CTWGCN, self).__init__()
        if gcn_layers == 0:
            print('error: gcn_layers must be a positive integer, not incuding zero')
            return

        if gcn_layers >= 2:
            CTWGCN_blocks = [Channel_Transformer_WGCNLayer(in_dim, hidden_dim,num_node)]
            for i in range(gcn_layers - 2):
                CTWGCN_blocks += [Channel_Transformer_WGCNLayer(hidden_dim, hidden_dim, num_node)]
            CTWGCN_blocks += [Channel_Transformer_WGCNLayer(hidden_dim, out_dim, num_node)]
            self.CTWGCNs = nn.ModuleList(CTWGCN_blocks)
        else:
            CTWGCN_blocks = [Channel_Transformer_WGCNLayer(in_dim, out_dim,num_node)]
            self.CTWGCNs = nn.ModuleList(CTWGCN_blocks)

    def forward(self, X, adj_matrix):
        for gcnlayer in self.CTWGCNs:
            X = gcnlayer(X, adj_matrix)
        return X




if __name__ == '__main__':
    model = GNN_Collections(gcnmodel=Dynamic_GCN, in_channels=47, n_outputs=1, hidden_channels=32, num_node=131, gcn_layers=3, out_graphfeats=8)
    input_x = torch.zeros(5, 131, 47)
    input_ad = torch.zeros(5, 131, 131)
    print(model)
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())

    # count flops and parameters
    flops, params = profile(model, inputs=(input_x,input_ad,), verbose=False)
    print("flops: %e" % flops)
    print("params: %e" % params)
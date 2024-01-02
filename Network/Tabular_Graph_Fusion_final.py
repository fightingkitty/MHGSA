# -*- coding:utf-8 -*-
"""
@Time: 2023/1/28 3:29
@Author: Shuting Liu & Baochang Zhang
@IDE: PyCharm
@File: Tabular_Graph_Fusion_final.py
@Comment: #Enter some comments at here
"""
import torch.nn as nn
import torch
from thop import profile
from .blocks import InitWeights_He
from .GraphNets import preprocess_adj
from .TabularNets import MIR
# import torch.nn.functional as F

class Dynamic_MHGCN_FusionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_node=131, num_heads=2, norm=False):
        super(Dynamic_MHGCN_FusionLayer, self).__init__()

        self.num_heads = num_heads
        self.head_dim = out_dim//num_heads

        self.edge_scale = self.head_dim ** -0.5
        self.node_scale = out_dim ** -0.5

        self.T_q_proj = nn.Sequential(nn.Linear(in_dim, out_dim,bias=False),
                                    nn.LeakyReLU(inplace=True))

        self.G_q_proj =  nn.Sequential(nn.Linear(in_dim, out_dim,bias=False),
                                    nn.LeakyReLU(inplace=True))
        self.G_k_proj =  nn.Sequential(nn.Linear(in_dim, out_dim,bias=False),
                                    nn.LeakyReLU(inplace=True))

        self.updata_Gx = nn.Sequential(nn.Linear(in_dim, out_dim),
                                       nn.LeakyReLU(inplace=True))

        self.updata_Tx = nn.Sequential(nn.Linear(in_dim, out_dim),
                                      nn.LeakyReLU(inplace=True))

        self.updata_Tx = nn.Sequential(nn.Linear(in_dim, out_dim),
                                      nn.LeakyReLU(inplace=True))

        self.Graph_summary = nn.Sequential(nn.Linear(num_node, 1),
                                           nn.LeakyReLU(inplace=True))
        self.updata_Zx = nn.Sequential(nn.Linear(2*out_dim, out_dim),
                                         nn.LeakyReLU(inplace=True))

        if norm:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = None

    def forward(self, tabular_feats, node_feats, adj_matrix):

        G_X = self.updata_Gx(node_feats)  # B N C/ feature-wise
        T_X = self.updata_Tx(tabular_feats)  # B N C/ feature-wise

        # node attention from Tabular and graph info
        G_k = self.G_k_proj(node_feats)  # B N C/ feature-wise & normalize
        T_q = self.T_q_proj(tabular_feats)  # B 1 C/ feature-wise & normalize
        node_att = (torch.einsum("bxh,byh->bxy", T_q, G_k)*self.node_scale).softmax(dim=-1)  # B 1 N
        node_att = torch.transpose(node_att+1, -1, -2) # B N 1
        G_X = node_att * G_X

        # Edge attention from graph info
        batch_size, num_node, _ = node_feats.size()
        G_X_mh = G_X.reshape(batch_size, num_node,self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B H N C/ feature-wise
        G_q_mh = self.G_q_proj(node_feats).reshape(batch_size, num_node,self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B H N C  / feature-wise
        G_k_mh = G_k.reshape(batch_size, num_node,self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B H N C

        edge_att = torch.einsum("blxh,blyh->blxy", G_q_mh, G_k_mh) * self.edge_scale  # B H N N
        edge_att = torch.sigmoid(edge_att)
        adj_heads = adj_matrix.unsqueeze(1)*edge_att
        A = preprocess_adj(adj_heads)
        G_X = torch.einsum("bhxy,bhyd->bhxd", A, G_X_mh).permute(0,2,1,3).reshape(batch_size, num_node, -1)

        if self.norm:
            G_X = self.norm(G_X)

        # Fusion->Z_X
        global_node_feature = self.Graph_summary(G_X.permute(0, 2, 1)).reshape(batch_size, -1)
        Z_x_features = torch.cat((global_node_feature, T_X.reshape(batch_size, -1)), -1)
        Z_x = self.updata_Zx(Z_x_features)
        return G_X, T_X, Z_x, node_att, edge_att

class Dynamic_MHGCN_Fusion(nn.Module):
    def __init__(self, in_dim, hidden_dim, gcn_layers=3, num_node=131):
        super(Dynamic_MHGCN_Fusion, self).__init__()
        if gcn_layers == 0:
            print('error: gcn_layers must be a positive integer, not incuding zero')
            return
        GCN_blocks = [Dynamic_MHGCN_FusionLayer(in_dim, hidden_dim,num_node=num_node)]
        if gcn_layers > 1:
            for i in range(gcn_layers - 1):
                GCN_blocks += [Dynamic_MHGCN_FusionLayer(hidden_dim, hidden_dim, num_node=num_node)]
        self.GCNs = nn.ModuleList(GCN_blocks)

    def forward(self, tabular_feats, node_feats, adj_matrix):
        hidden_states_out = []
        node_att_map_collection = []
        edge_att_map_collection = []
        for gcnlayer in self.GCNs:
            node_feats, tabular_feats, Z_x, node_att, edge_att = gcnlayer(tabular_feats, node_feats, adj_matrix)
            hidden_states_out.append(Z_x)
            node_att_map_collection.append(node_att)
            edge_att_map_collection.append(edge_att)
        return hidden_states_out, node_feats, tabular_feats, node_att_map_collection, edge_att_map_collection

class TGNN_Fusions(nn.Module):
    def __init__(self, in_channels_T=105, in_channels_G=47, hidden_channels_G=32, num_node=131, gcn_layers=3, n_outputs=1):
        super(TGNN_Fusions, self).__init__()

        self.T2G_mapping = MIR(in_channels_T, in_channels_G)
        self.gnn = Dynamic_MHGCN_Fusion(in_channels_G, hidden_channels_G, gcn_layers=gcn_layers, num_node=num_node)
        self.gcn_layers=gcn_layers
        hidden_channels_total = hidden_channels_G

        # self.fc = nn.Sequential(
        #     nn.Linear(hidden_channels_total, hidden_channels_G),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(hidden_channels_G, n_outputs),
        # )

        self.fc = nn.Linear(hidden_channels_total, n_outputs)
        self.sigmoid = torch.nn.Sigmoid()
        InitWeights_He(self)

    def forward(self, Tx, vec, Gx, adj_matrix,show_att=False):
        tabular_feature = self.T2G_mapping(Tx, vec)
        tabular_feature = torch.unsqueeze(tabular_feature,dim=1)
        hidden_states_out, graph_feature, tabular_feature, node_att_map_collection, edge_att_map_collection = self.gnn(tabular_feature, Gx, adj_matrix)
        hidden_feature = hidden_states_out[2]
        # for i in range(1, self.gcn_layers):
        #     hidden_feature = torch.cat((hidden_feature, hidden_states_out[i]), -1)
        out = self.fc(hidden_feature)
        if show_att:
            return self.sigmoid(out), node_att_map_collection, edge_att_map_collection
        return self.sigmoid (out)

if __name__ == '__main__':
    model = TGNN_Fusions( in_channels_T=20,  in_channels_G=18, hidden_channels_G=32,  num_node=131, gcn_layers=3, n_outputs=1)


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
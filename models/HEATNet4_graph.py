import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn import edge_softmax
from dgl.nn.pytorch.glob import GlobalAttentionPooling

from pooling import AvgPooling, SumPooling, MaxPooling

from dgl.nn import SAGEConv, AvgPooling, GATConv
"""
HEATNet (Heterogeneous Edge Attribute Transformer)
Maybe can consider initialize an equal dimension (1024) vector filled by scalar edge weight
"""

class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv1d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)
    def forward(self, l, g):
        l = torch.unsqueeze(l, dim=-1)
        g = torch.unsqueeze(g, dim=-1)
        N, C, W = l.size()
        
        c = self.op(l+g) # batch_sizex1xWxH
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,1)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)

        # return c.view(N,1,W,H), g
        return g


def apply_weights(edges):
    return {'t': edges.data['t'] * edges.data['v']}


import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionPooling(nn.Module):
    """
    Graph-level Attention Pooling: learnable query attends to all node features
    """
    def __init__(self, in_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        # Query 向量，用于全局图表示
        self.query = nn.Parameter(torch.randn(1, hidden_dim))
        # Key / Value 映射
        self.key_proj = nn.Linear(in_dim, hidden_dim)
        self.value_proj = nn.Linear(in_dim, hidden_dim)

    def forward(self, h):
        """
        h: 节点特征 (N, F_in)
        返回图级表示 hg: (1, hidden_dim)
        """
        keys = self.key_proj(h)       # (N, hidden_dim)
        values = self.value_proj(h)   # (N, hidden_dim)
        query = self.query            # (1, hidden_dim)
        # Attention 分数 (N,1)
        attn = torch.softmax(keys @ query.T / (keys.shape[1] ** 0.5), dim=0)
        # 加权求和得到图表示
        hg = (attn * values).sum(dim=0, keepdim=True)  # (1, hidden_dim)
        return hg


class HEATNet4(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, n_heads, node_dict, dropout, graph_pooling_type='mean'):
        super(HEATNet4, self).__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.gcs = nn.ModuleList([GATConv(hidden_dim, hidden_dim//n_heads, num_heads=n_heads, feat_drop=dropout, attn_drop=dropout) for _ in range(n_layers)])
        self.pool = GraphAttentionPooling(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, G, h=None):
        if h is None:
            h = G.ndata['feat']
        h = self.input_proj(h)
        # GATConv 传播
        for gnn in self.gcs:
            h = gnn(G, h)
            h = h.flatten(1)
            h = F.relu(h)
        # 图级 attention pooling
        hg = self.pool(h)  # (1, hidden_dim)
        out = self.head(hg)
        return out
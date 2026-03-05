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
import dgl.nn.pytorch as dglnn


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

    def forward(self, h, return_weight=False):
        """
        h: 节点特征 (N, F_in)
        返回图级表示 hg: (1, hidden_dim)
        return_attention=True 时，同时返回每个节点的注意力权重 (N,)
        """
        keys = self.key_proj(h)       # (N, hidden_dim)
        values = self.value_proj(h)   # (N, hidden_dim)
        query = self.query            # (1, hidden_dim)

        # Attention 分数 (N,1)
        attn = torch.softmax(keys @ query.T / (keys.shape[1] ** 0.5), dim=0)

        # 加权求和得到图表示
        hg = (attn * values).sum(dim=0, keepdim=True)  # (1, hidden_dim)

        if return_weight:
            return hg, attn.squeeze()  # 返回节点注意力 (N,)
        return hg
        

# # 超图网路 best设置
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import dgl
# from dgl.nn.pytorch import HeteroGraphConv, GATConv, GlobalAttentionPooling

# # --------------------------
# # HEATNet4 with node & hyperedge attention
# # --------------------------
# class HEATNet4(nn.Module):
#     # def __init__(self, in_dim, hidden_dim, out_dim, n_layers, n_heads, dropout, graph_pooling_type='mean'):
#     def __init__(self, in_dim_dict, hidden_dim, out_dim, n_layers, n_heads, dropout):
#         super().__init__()
#         # in_dim_dict: {'node': 1024, 'he': 512}  #由外部传入每类节点的特征维度

#         # 为每种节点建输入投影层
#         self.input_proj = nn.ModuleDict({
#             ntype: nn.Linear(in_dim_dict[ntype], hidden_dim) 
#             for ntype in in_dim_dict
#         })

#         # 异质图卷积：node ↔ he
#         self.gcs = nn.ModuleList([
#             dglnn.HeteroGraphConv({
#                 ('node', 'incidence', 'he'): dglnn.GATConv(hidden_dim, hidden_dim // n_heads, num_heads=n_heads),
#                 ('he', 'expand', 'node'): dglnn.GATConv(hidden_dim, hidden_dim // n_heads, num_heads=n_heads),
#             }, aggregate='mean')
#             for _ in range(n_layers)
#         ])

#         self.pool = GraphAttentionPooling(hidden_dim)
#         self.node_pool = GraphAttentionPooling(hidden_dim)
#         self.he_pool = GraphAttentionPooling(hidden_dim)

#         self.head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, out_dim)
#         )
#     def forward(self, G, return_attention=False):
#         """
#         G: DGL heterograph
#         return_attention: 是否返回节点注意力，用于 GraphSHA
#         """
#         # 1. 节点特征投影
#         h_dict = {ntype: self.input_proj[ntype](G.nodes[ntype].data['feat']) 
#                 for ntype in G.ntypes}

#         # 2. GNN 层传播
#         for gnn in self.gcs:
#             h_dict = gnn(G, h_dict)
        
#         # 3. ReLU + flatten
#         h_dict = {k: F.relu(v.flatten(1)) for k, v in h_dict.items()}

#         # ===========================
#         # 4. 节点级别 pooling
#         # ===========================
#         node_feats = h_dict['node']  # [N_node, hidden_dim]
#         if return_attention:
#             node_emb, node_attn = self.node_pool(node_feats, return_weight=True)
#         else:
#             node_emb = self.node_pool(node_feats)

#         # ===========================
#         # 5. 超边级别 pooling
#         # ===========================
#         # 对 he 节点做 pooling，得到超边 embedding
#         # 如果 he 节点为空，可以跳过
#         if G.num_nodes('he') > 0:
#             he_feats = h_dict['he']  # [N_he, hidden_dim]
#             if return_attention:
#                 he_emb, he_attn = self.he_pool(he_feats, return_weight=True)
#             else:
#                 he_emb = self.he_pool(he_feats)
#         else:
#             # 如果没有 he 节点，用全零 embedding 占位
#             he_emb = torch.zeros_like(node_emb)
#             if return_attention:
#                 he_attn = torch.zeros(G.num_nodes('he'), 1, device=node_emb.device)

#         # ===========================
#         # 6. 融合节点和超边 embedding
#         # ===========================
#         graph_emb = node_emb + he_emb  # 也可以尝试 concat 然后 FC
#         out = self.head(graph_emb)

#         if return_attention:
#             return out, node_attn, he_attn
#         else:
#             return out

#     @staticmethod
#     def normalize_score(scores: torch.Tensor, eps=1e-12):
#         # scores: [N] or [N,1]
#         s = scores.view(-1).float()
#         mn, mx = s.min(), s.max()
#         return (s - mn) / (mx - mn + eps)

#     def sampling_node_source_hetero(self, pseudo_label, n_positive, n_negative=None):
#         """
#         从伪标签中采样 src (高 attention) 与 dst (低 attention).
#         pseudo_label: tensor [N], 1 = high-att, 0 = low-att
#         n_positive: desired number of positive samples (or None to use all)
#         n_negative: desired negatives (or None -> match positive count)
#         返回 src_idx (positives), dst_idx (negatives) (both LongTensor)
#         """
#         pos_idx = (pseudo_label == 1).nonzero(as_tuple=True)[0]
#         neg_idx = (pseudo_label == 0).nonzero(as_tuple=True)[0]

#         if pos_idx.numel() == 0 or neg_idx.numel() == 0:
#             return None, None

#         if n_positive is None:
#             n_positive = pos_idx.numel()
#         if n_negative is None:
#             n_negative = n_positive

#         # 随机采样（可替换为按 attention 权重采样）
#         perm_pos = pos_idx[torch.randperm(pos_idx.numel(), device=pos_idx.device)][:n_positive]
#         perm_neg = neg_idx[torch.randperm(neg_idx.numel(), device=neg_idx.device)][:n_negative]

#         # 如果数量不匹配，循环填充 neg
#         if perm_neg.numel() < perm_pos.numel():
#             perm_neg = perm_neg.repeat((perm_pos.numel() // perm_neg.numel()) + 1)[:perm_pos.numel()]

#         return perm_pos, perm_neg[:perm_pos.numel()]

#     def saliency_mixup(self, feat: torch.Tensor, src_idx: torch.LongTensor, dst_idx: torch.LongTensor, lam: torch.Tensor):
#         """
#         简单的 mixup：new = lam * feat[src] + (1-lam) * feat[dst]
#         feat: [N, D]
#         src_idx, dst_idx: same length
#         lam: [K,1] or [K,D] （广播支持）
#         返回 new_feat [K, D]
#         """
#         src_feat = feat[src_idx]       # [K, D]
#         dst_feat = feat[dst_idx]       # [K, D]
#         if lam.dim() == 1:
#             lam = lam.view(-1, 1)
#         new = lam * src_feat + (1 - lam) * dst_feat
#         return new

#     def duplicate_neighbors_hetero(self, G, src_nodes: torch.LongTensor, new_nodes_idx: torch.LongTensor):
#         """
#         为每个 src_nodes 复制其 incident edges 到对应的新节点。
#         适用于你的异构关系：
#           ('node','incidence','he')  &  ('he','expand','node')
#         假设 src_nodes 和 new_nodes_idx 长度相同，顺序一一对应。
#         """
#         device = next(self.parameters()).device

#         # collect edges to add for each relation
#         src_to_he_u = []
#         src_to_he_v = []
#         he_to_node_u = []
#         he_to_node_v = []

#         # Get all edges (u,v) for ('node','incidence','he')
#         # We'll iterate per src to find its he neighbors
#         for s, new_idx in zip(src_nodes.tolist(), new_nodes_idx.tolist()):
#             # successors: he neighbors of s (via incidence)
#             try:
#                 he_neighbors = G.successors(s, etype=('node', 'incidence', 'he'))
#             except Exception:
#                 # fallback: use G.edges for etype and mask
#                 u, v = G.edges(etype=('node', 'incidence', 'he'))
#                 mask = (u == s)
#                 he_neighbors = v[mask]

#             # if no he neighbors, skip (but we might still want to connect to some dummy or skip)
#             if he_neighbors.numel() == 0:
#                 continue

#             # add edges new_node -> each he_neighbor for ('node','incidence','he')
#             src_to_he_u.append(torch.tensor([new_idx] * he_neighbors.numel(), dtype=torch.long, device=device))
#             src_to_he_v.append(he_neighbors.to(device))

#             # add reverse edges ('he','expand','node'): he_neighbor -> new_node
#             he_to_node_u.append(he_neighbors.to(device))
#             he_to_node_v.append(torch.tensor([new_idx] * he_neighbors.numel(), dtype=torch.long, device=device))

#         # concatenate and call add_edges once per etype if not empty
#         if len(src_to_he_u) > 0:
#             u_cat = torch.cat(src_to_he_u)
#             v_cat = torch.cat(src_to_he_v)
#             G.add_edges(u_cat, v_cat, etype=('node','incidence','he'))
#         if len(he_to_node_u) > 0:
#             u_cat = torch.cat(he_to_node_u)
#             v_cat = torch.cat(he_to_node_v)
#             G.add_edges(u_cat, v_cat, etype=('he','expand','node'))

#     def graphsha_augment_hetero(self, G, node_attn=None, he_attn=None, epoch=0,
#                                 attn_threshold=0.001, max_new_ratio=0.2):
#         """
#         使用 node_attn 和 he_attn 对高 attention 的节点/超边进行特征生成并加入图。
#         返回增强后的图 G_aug（会在原图上就地修改并返回）。
#         说明与假设：
#           - node_attn: tensor [N_node, 1] 或 [N_node]
#           - he_attn: tensor [N_he, 1] 或 [N_he]
#           - 我们只对 'node' 类型做 mixup 生成新的 'node' 节点并复制邻居；
#             对 'he'（超边）也可以类似处理（下面实现了 node 与 he 两种情况的处理）
#         """
#         device = next(self.parameters()).device

#         # ---------- Node-level augmentation ----------
#         if node_attn is not None:
#             node_attn_vec = node_attn.view(-1).to(device)
#             # 改成以平均值为阈值
#             mean_attn = node_attn_vec.mean()
#             pseudo_label = (node_attn_vec > mean_attn).long()  # > mean = 1, <= mean = 0

#             class_num_list = [(pseudo_label == c).sum().item() for c in [0, 1]]
#             # 需要两类都至少有 2 个，否则跳过
#             if all(num > 1 for num in class_num_list):
#                 # 采样数量：限制为当前 node 数的 max_new_ratio
#                 N_node = G.num_nodes('node')
#                 max_new = max(1, int(N_node * max_new_ratio))
#                 # choose k = min(pos_count, max_new)
#                 pos_total = (pseudo_label == 1).sum().item()
#                 k = min(pos_total, max_new)

#                 src_idx, dst_idx = self.sampling_node_source_hetero(pseudo_label, n_positive=k, n_negative=k)
#                 if src_idx is not None:
#                     lam = torch.rand(src_idx.numel(), 1, device=device)  # [K,1]
#                     node_feat = G.nodes['node'].data['feat']  # [N_node, D]
#                     new_feat = self.saliency_mixup(node_feat, src_idx, dst_idx, lam)  # [K, D]

#                     # add new node(s)
#                     old_num = G.num_nodes('node')
#                     G.add_nodes(new_feat.size(0), ntype='node')
#                     new_idx_range = torch.arange(old_num, old_num + new_feat.size(0), device=device, dtype=torch.long)
#                     # assign features
#                     # ensure 'feat' exists for node type
#                     if 'feat' not in G.nodes['node'].data:
#                         # initialize with zeros if missing
#                         G.nodes['node'].data['feat'] = torch.zeros(old_num, new_feat.size(1), device=device)
#                     # extend storage if needed (DGL handles ndata auto-resize on add_nodes)
#                     G.nodes['node'].data['feat'][old_num:] = new_feat

#                     # duplicate neighbors: copy incident edges of src to new nodes
#                     self.duplicate_neighbors_hetero(G, src_idx, new_idx_range)

#         # ---------- HE-level augmentation (可选) ----------
#         if he_attn is not None:
#             he_attn_vec = he_attn.view(-1).to(device)
#             mean_he_attn = he_attn_vec.mean()
#             he_pseudo = (he_attn_vec > mean_he_attn).long()
#             class_num_list = [(he_pseudo == c).sum().item() for c in [0,1]]

#             if all(num > 1 for num in class_num_list):
#                 N_he = G.num_nodes('he')
#                 max_new = max(1, int(N_he * max_new_ratio))
#                 pos_total = (he_pseudo == 1).sum().item()
#                 k = min(pos_total, max_new)

#                 he_src_idx, he_dst_idx = self.sampling_node_source_hetero(he_pseudo, n_positive=k, n_negative=k)
#                 if he_src_idx is not None:
#                     lam = torch.rand(he_src_idx.numel(), 1, device=device)
#                     he_feat = G.nodes['he'].data['feat']
#                     new_he_feat = self.saliency_mixup(he_feat, he_src_idx, he_dst_idx, lam)

#                     old_he_num = G.num_nodes('he')
#                     G.add_nodes(new_he_feat.size(0), ntype='he')
#                     new_he_idx_range = torch.arange(old_he_num, old_he_num + new_he_feat.size(0), device=device, dtype=torch.long)
#                     if 'feat' not in G.nodes['he'].data:
#                         G.nodes['he'].data['feat'] = torch.zeros(old_he_num, new_he_feat.size(1), device=device)
#                     G.nodes['he'].data['feat'][old_he_num:] = new_he_feat

#                     # 对应地，需要将 he 新节点连接回一些 node（复制 he_src 的邻居）
#                     # 找到 he_src 的 node neighbors via ('he','expand','node') edges
#                     # For each he_src, copy edges he_src -> node to new he node
#                     he_src_list = he_src_idx.tolist()
#                     added_u = []
#                     added_v = []
#                     for s, new_idx in zip(he_src_list, new_he_idx_range.tolist()):
#                         try:
#                             node_neighbors = G.successors(s, etype=('he','expand','node'))
#                         except Exception:
#                             u, v = G.edges(etype=('he','expand','node'))
#                             mask = (u == s)
#                             node_neighbors = v[mask]
#                         if node_neighbors.numel() == 0:
#                             continue
#                         added_u.append(torch.tensor([new_idx]*node_neighbors.numel(), device=device))
#                         added_v.append(node_neighbors.to(device))
#                     if len(added_u) > 0:
#                         u_cat = torch.cat(added_u)
#                         v_cat = torch.cat(added_v)
#                         G.add_edges(u_cat, v_cat, etype=('he','expand','node'))
#                         # also add reverse edges ('node','incidence','he'): node -> new_he
#                         G.add_edges(v_cat, u_cat, etype=('node','incidence','he'))

#         return G  # in-place modified




class HEATNet4(nn.Module):
    def __init__(self, in_dim_dict, hidden_dim, out_dim, n_layers, n_heads, dropout,
                 use_spd=True, use_rw=True):
        super().__init__()
        self.use_spd = use_spd
        self.use_rw = use_rw

        # ========================
        # 输入投影
        # ========================
        self.input_proj = nn.ModuleDict()

        # node 特征: feat + de_spd + de_rw
        node_in_dim = in_dim_dict['node']  # 先用原始 feat 维度
        if self.use_spd:
            node_in_dim += in_dim_dict.get('de_spd', 0)
        if self.use_rw:
            node_in_dim += in_dim_dict.get('de_rw', 0)

        self.input_proj['node'] = nn.Linear(node_in_dim, hidden_dim)
        self.input_proj['he']   = nn.Linear(in_dim_dict['he'], hidden_dim)

        # 异质 GNN
        self.gcs = nn.ModuleList([
            dglnn.HeteroGraphConv({
                ('node', 'incidence', 'he'): dglnn.GATConv(hidden_dim, hidden_dim // n_heads, num_heads=n_heads),
                ('he', 'expand', 'node'): dglnn.GATConv(hidden_dim, hidden_dim // n_heads, num_heads=n_heads),
            }, aggregate='mean')
            for _ in range(n_layers)
        ])

        # pooling
        self.node_pool = GraphAttentionPooling(hidden_dim)
        self.he_pool = GraphAttentionPooling(hidden_dim)

        # 分类头
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, G, return_attention=False):
        # ============ 拼接 Distance Encoding ============
        node_feats = [G.nodes['node'].data['feat']]
        # if self.use_spd and 'de_spd' in G.nodes['node'].data:
        #     node_feats.append(G.nodes['node'].data['de_spd'])
        # if self.use_rw and 'de_rw' in G.nodes['node'].data:
        #     node_feats.append(G.nodes['node'].data['de_rw'])
        node_feats = torch.cat(node_feats, dim=-1)  # [N, d_total]

        # 投影
        h_dict = {
            'node': self.input_proj['node'](node_feats),
            'he': self.input_proj['he'](G.nodes['he'].data['feat'])
        }

        # # # GNN 层传播
        # for gnn in self.gcs:
        #     h_dict = gnn(G, h_dict)
        #     h_dict = {k: v.flatten(1) for k, v in h_dict.items()}

        h_dict = {k: F.relu(v) for k, v in h_dict.items()}

        # pooling
        if return_attention:
            node_emb, node_attn = self.node_pool(h_dict['node'], return_weight=True)
            if G.num_nodes('he') > 0:
                he_emb, he_attn = self.he_pool(h_dict['he'], return_weight=True)
            else:
                he_emb = torch.zeros_like(node_emb)
                he_attn = torch.zeros(G.num_nodes('he'), 1, device=node_emb.device)
        else:
            node_emb = self.node_pool(h_dict['node'])
            he_emb = self.he_pool(h_dict['he']) if G.num_nodes('he') > 0 else torch.zeros_like(node_emb)

        graph_emb = node_emb + 0.1*he_emb
        # graph_emb = node_emb
        out = self.head(graph_emb)

        if return_attention:
            return out, node_attn, he_attn
        else:
            return out

    @staticmethod
    def normalize_score(scores: torch.Tensor, eps=1e-12):
        # scores: [N] or [N,1]
        s = scores.view(-1).float()
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + eps)

    def sampling_node_source_hetero(self, pseudo_label, n_positive, n_negative=None):
        """
        从伪标签中采样 src (高 attention) 与 dst (低 attention).
        pseudo_label: tensor [N], 1 = high-att, 0 = low-att
        n_positive: desired number of positive samples (or None to use all)
        n_negative: desired negatives (or None -> match positive count)
        返回 src_idx (positives), dst_idx (negatives) (both LongTensor)
        """
        pos_idx = (pseudo_label == 1).nonzero(as_tuple=True)[0]
        neg_idx = (pseudo_label == 0).nonzero(as_tuple=True)[0]

        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            return None, None

        if n_positive is None:
            n_positive = pos_idx.numel()
        if n_negative is None:
            n_negative = n_positive

        # 随机采样（可替换为按 attention 权重采样）
        perm_pos = pos_idx[torch.randperm(pos_idx.numel(), device=pos_idx.device)][:n_positive]
        perm_neg = neg_idx[torch.randperm(neg_idx.numel(), device=neg_idx.device)][:n_negative]

        # 如果数量不匹配，循环填充 neg
        if perm_neg.numel() < perm_pos.numel():
            perm_neg = perm_neg.repeat((perm_pos.numel() // perm_neg.numel()) + 1)[:perm_pos.numel()]

        return perm_pos, perm_neg[:perm_pos.numel()]

    def saliency_mixup(self, feat: torch.Tensor, src_idx: torch.LongTensor, dst_idx: torch.LongTensor, lam: torch.Tensor):
        """
        简单的 mixup：new = lam * feat[src] + (1-lam) * feat[dst]
        feat: [N, D]
        src_idx, dst_idx: same length
        lam: [K,1] or [K,D] （广播支持）
        返回 new_feat [K, D]
        """
        src_feat = feat[src_idx]       # [K, D]
        dst_feat = feat[dst_idx]       # [K, D]
        if lam.dim() == 1:
            lam = lam.view(-1, 1)
        new = lam * src_feat + (1 - lam) * dst_feat
        return new

    def duplicate_neighbors_hetero(self, G, src_nodes: torch.LongTensor, new_nodes_idx: torch.LongTensor):
        """
        为每个 src_nodes 复制其 incident edges 到对应的新节点。
        适用于你的异构关系：
          ('node','incidence','he')  &  ('he','expand','node')
        假设 src_nodes 和 new_nodes_idx 长度相同，顺序一一对应。
        """
        device = next(self.parameters()).device

        # collect edges to add for each relation
        src_to_he_u = []
        src_to_he_v = []
        he_to_node_u = []
        he_to_node_v = []

        # Get all edges (u,v) for ('node','incidence','he')
        # We'll iterate per src to find its he neighbors
        for s, new_idx in zip(src_nodes.tolist(), new_nodes_idx.tolist()):
            # successors: he neighbors of s (via incidence)
            try:
                he_neighbors = G.successors(s, etype=('node', 'incidence', 'he'))
            except Exception:
                # fallback: use G.edges for etype and mask
                u, v = G.edges(etype=('node', 'incidence', 'he'))
                mask = (u == s)
                he_neighbors = v[mask]

            # if no he neighbors, skip (but we might still want to connect to some dummy or skip)
            if he_neighbors.numel() == 0:
                continue

            # add edges new_node -> each he_neighbor for ('node','incidence','he')
            src_to_he_u.append(torch.tensor([new_idx] * he_neighbors.numel(), dtype=torch.long, device=device))
            src_to_he_v.append(he_neighbors.to(device))

            # add reverse edges ('he','expand','node'): he_neighbor -> new_node
            he_to_node_u.append(he_neighbors.to(device))
            he_to_node_v.append(torch.tensor([new_idx] * he_neighbors.numel(), dtype=torch.long, device=device))

        # concatenate and call add_edges once per etype if not empty
        if len(src_to_he_u) > 0:
            u_cat = torch.cat(src_to_he_u)
            v_cat = torch.cat(src_to_he_v)
            G.add_edges(u_cat, v_cat, etype=('node','incidence','he'))
        if len(he_to_node_u) > 0:
            u_cat = torch.cat(he_to_node_u)
            v_cat = torch.cat(he_to_node_v)
            G.add_edges(u_cat, v_cat, etype=('he','expand','node'))

    def graphsha_augment_hetero(self, G, node_attn=None, he_attn=None, epoch=0,
                                attn_threshold=0.001, max_new_ratio=0.2):
        """
        使用 node_attn 和 he_attn 对高 attention 的节点/超边进行特征生成并加入图。
        返回增强后的图 G_aug（会在原图上就地修改并返回）。
        说明与假设：
          - node_attn: tensor [N_node, 1] 或 [N_node]
          - he_attn: tensor [N_he, 1] 或 [N_he]
          - 我们只对 'node' 类型做 mixup 生成新的 'node' 节点并复制邻居；
            对 'he'（超边）也可以类似处理（下面实现了 node 与 he 两种情况的处理）
        """
        device = next(self.parameters()).device

        # ---------- Node-level augmentation ----------
        if node_attn is not None:
            node_attn_vec = node_attn.view(-1).to(device)
            # 改成以平均值为阈值
            mean_attn = node_attn_vec.mean()
            pseudo_label = (node_attn_vec > mean_attn).long()  # > mean = 1, <= mean = 0

            class_num_list = [(pseudo_label == c).sum().item() for c in [0, 1]]
            # 需要两类都至少有 2 个，否则跳过
            if all(num > 1 for num in class_num_list):
                # 采样数量：限制为当前 node 数的 max_new_ratio
                N_node = G.num_nodes('node')
                max_new = max(1, int(N_node * max_new_ratio))
                # choose k = min(pos_count, max_new)
                pos_total = (pseudo_label == 1).sum().item()
                k = min(pos_total, max_new)

                src_idx, dst_idx = self.sampling_node_source_hetero(pseudo_label, n_positive=k, n_negative=k)
                if src_idx is not None:
                    lam = torch.rand(src_idx.numel(), 1, device=device)  # [K,1]
                    node_feat = G.nodes['node'].data['feat']  # [N_node, D]
                    new_feat = self.saliency_mixup(node_feat, src_idx, dst_idx, lam)  # [K, D]

                    # add new node(s)
                    old_num = G.num_nodes('node')
                    G.add_nodes(new_feat.size(0), ntype='node')
                    new_idx_range = torch.arange(old_num, old_num + new_feat.size(0), device=device, dtype=torch.long)
                    # assign features
                    # ensure 'feat' exists for node type
                    if 'feat' not in G.nodes['node'].data:
                        # initialize with zeros if missing
                        G.nodes['node'].data['feat'] = torch.zeros(old_num, new_feat.size(1), device=device)
                    # extend storage if needed (DGL handles ndata auto-resize on add_nodes)
                    G.nodes['node'].data['feat'][old_num:] = new_feat

                    # duplicate neighbors: copy incident edges of src to new nodes
                    self.duplicate_neighbors_hetero(G, src_idx, new_idx_range)

        # ---------- HE-level augmentation (可选) ----------
        if he_attn is not None:
            he_attn_vec = he_attn.view(-1).to(device)
            mean_he_attn = he_attn_vec.mean()
            he_pseudo = (he_attn_vec > mean_he_attn).long()
            class_num_list = [(he_pseudo == c).sum().item() for c in [0,1]]

            if all(num > 1 for num in class_num_list):
                N_he = G.num_nodes('he')
                max_new = max(1, int(N_he * max_new_ratio))
                pos_total = (he_pseudo == 1).sum().item()
                k = min(pos_total, max_new)

                he_src_idx, he_dst_idx = self.sampling_node_source_hetero(he_pseudo, n_positive=k, n_negative=k)
                if he_src_idx is not None:
                    lam = torch.rand(he_src_idx.numel(), 1, device=device)
                    he_feat = G.nodes['he'].data['feat']
                    new_he_feat = self.saliency_mixup(he_feat, he_src_idx, he_dst_idx, lam)

                    old_he_num = G.num_nodes('he')
                    G.add_nodes(new_he_feat.size(0), ntype='he')
                    new_he_idx_range = torch.arange(old_he_num, old_he_num + new_he_feat.size(0), device=device, dtype=torch.long)
                    if 'feat' not in G.nodes['he'].data:
                        G.nodes['he'].data['feat'] = torch.zeros(old_he_num, new_he_feat.size(1), device=device)
                    G.nodes['he'].data['feat'][old_he_num:] = new_he_feat

                    # 对应地，需要将 he 新节点连接回一些 node（复制 he_src 的邻居）
                    # 找到 he_src 的 node neighbors via ('he','expand','node') edges
                    # For each he_src, copy edges he_src -> node to new he node
                    he_src_list = he_src_idx.tolist()
                    added_u = []
                    added_v = []
                    for s, new_idx in zip(he_src_list, new_he_idx_range.tolist()):
                        try:
                            node_neighbors = G.successors(s, etype=('he','expand','node'))
                        except Exception:
                            u, v = G.edges(etype=('he','expand','node'))
                            mask = (u == s)
                            node_neighbors = v[mask]
                        if node_neighbors.numel() == 0:
                            continue
                        added_u.append(torch.tensor([new_idx]*node_neighbors.numel(), device=device))
                        added_v.append(node_neighbors.to(device))
                    if len(added_u) > 0:
                        u_cat = torch.cat(added_u)
                        v_cat = torch.cat(added_v)
                        G.add_edges(u_cat, v_cat, etype=('he','expand','node'))
                        # also add reverse edges ('node','incidence','he'): node -> new_he
                        G.add_edges(v_cat, u_cat, etype=('node','incidence','he'))

        return G  # in-place modified


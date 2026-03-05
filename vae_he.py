import os
import csv
import yaml
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import dgl
import numpy as np

# -------------------- Utils: robust hypergraph builder --------------------
def build_hypergraph_from_incidence(node_features, coords, incidence_matrix):
    """
    node_features: Tensor [N, F] (cpu)
    coords: Tensor [N, 2] (cpu)
    incidence_matrix: Tensor [N, M] (cpu)  (节点 × 超边) (0/1)
    返回：DGL heterograph，edge types: ('node','incidence','he') 和 ('he','expand','node')
    """
    # ensure tensor
    if not torch.is_tensor(incidence_matrix):
        incidence_matrix = torch.tensor(incidence_matrix)

    # reshape fixes
    if incidence_matrix.dim() == 1:
        incidence_matrix = incidence_matrix.unsqueeze(1)  # [N] -> [N,1]
    elif incidence_matrix.dim() == 3 and incidence_matrix.size(0) == 1:
        incidence_matrix = incidence_matrix.squeeze(0)  # [1, N, M] -> [N, M]

    if incidence_matrix.dim() != 2:
        raise ValueError(f"Incidence matrix must be 2D [N,M], got {incidence_matrix.shape}")

    N, M = incidence_matrix.shape
    # print for debug
    print(f"[build_hypergraph_from_incidence] incidence shape: {incidence_matrix.shape}")

    # get edges (node_ids, he_ids)
    row_idx, col_idx = incidence_matrix.nonzero(as_tuple=True)  # 1D tensors
    # ensure long type
    row_idx = row_idx.to(torch.int64)
    col_idx = col_idx.to(torch.int64)

    # Build heterograph with explicit node counts to avoid "0 nodes" issue
    hg = dgl.heterograph({
        ('node', 'incidence', 'he'): (row_idx, col_idx),
        ('he', 'expand', 'node'): (col_idx, row_idx),
    }, num_nodes_dict={'node': int(N), 'he': int(M)})

    # assign features (must match N)
    if node_features.shape[0] != N:
        raise ValueError(f"node_features length ({node_features.shape[0]}) != N ({N})")
    hg.nodes['node'].data['feat'] = node_features
    hg.nodes['node'].data['coord'] = coords

    return hg


# -------------------- Dataset --------------------
class HypergraphFolderDataset(Dataset):
    def __init__(self, data_dir, labels_csv, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.case_list = []  # list of (case_id, class_name, path)
        label_map = {}
        with open(labels_csv, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                case_id, class_name = row[0], row[1]
                label_map[case_id] = class_name

        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith('.bin'):
                continue
            case_id = os.path.splitext(fname)[0]
            if case_id not in label_map:
                continue
            path = os.path.join(data_dir, fname)
            self.case_list.append((case_id, label_map[case_id], path))

        classes = sorted(list({c for _, c, _ in self.case_list}))
        self.class2idx = {c: i for i, c in enumerate(classes)}
        self.idx2class = {i: c for c, i in self.class2idx.items()}

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        case_id, class_name, path = self.case_list[idx]
        graphs, _ = dgl.load_graphs(path)
        hg = graphs[0]

        # support both naming 'he'/'incidence' or 'hyperedge' etc - try common keys
        # assume node features stored as 'feat'
        node_feats = hg.nodes['node'].data.get('feat', None)
        coords = hg.nodes['node'].data.get('coord', None)

        if node_feats is None:
            raise KeyError(f"Graph {path} has no node feat")
        if coords is None:
            # make placeholder coords
            coords = torch.zeros((node_feats.shape[0], 2), dtype=torch.float32)

        # try standard etype
        try:
            src, dst = hg.edges(etype=('node', 'incidence', 'he'))
            N = hg.num_nodes('node')
            M = hg.num_nodes('he')
            incidence = torch.zeros((N, M), dtype=torch.float32)
            incidence[src, dst] = 1.0
        except Exception:
            # fallback: collect edges by any hetetype naming
            # build an empty single he as fallback
            N = hg.num_nodes('node')
            incidence = torch.zeros((N, 1), dtype=torch.float32)
            # if there is any 'he' node data and expand edges, try to populate
            if ('he', 'expand', 'node') in hg.canonical_etypes:
                src, dst = hg.edges(etype=('he','expand','node'))
                # src: he idx, dst: node idx -> convert to incidence
                for he_idx, node_idx in zip(src.tolist(), dst.tolist()):
                    incidence[node_idx, he_idx] = 1.0

        sample = {
            'case_id': case_id,
            'node_feats': node_feats,    # Tensor [N, F]
            'coords': coords,            # Tensor [N,2]
            'incidence': incidence,      # Tensor [N, M]
            'class_idx': self.class2idx[class_name]
        }
        return sample


# -------------------- GNN-VAE model (Graph-level conditional VAE) --------------------
class GraphEncoder(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, graph_emb_dim):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.readout = nn.Linear(hidden_dim * 2, graph_emb_dim)

    def forward(self, node_feats):
        h = self.node_proj(node_feats)  # [N, H]
        mean_pool = h.mean(dim=0, keepdim=True)
        max_pool, _ = h.max(dim=0, keepdim=True)
        g = torch.cat([mean_pool, max_pool], dim=1).squeeze(0)  # [2H]
        g = self.readout(g)  # [graph_emb_dim]
        return g


class ConditionalVAEEncoder(nn.Module):
    def __init__(self, graph_emb_dim, class_emb_dim, latent_dim, num_classes=1000):
        super().__init__()
        self.cls_emb = nn.Embedding(num_classes, class_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(graph_emb_dim + class_emb_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.mean = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self, graph_emb, class_idx):
        if isinstance(class_idx, int):
            class_idx = torch.tensor([class_idx], device=graph_emb.device)
        cls = self.cls_emb(class_idx).squeeze(0)
        x = torch.cat([graph_emb, cls], dim=0).unsqueeze(0)  # [1, graph_emb+cls]
        h = self.net(x)
        mu = self.mean(h).squeeze(0)
        logvar = self.logvar(h).squeeze(0)
        return mu, logvar


class ConditionalGenerator(nn.Module):
    """
    Generator outputs:
      - node_feats: [N, node_feat_dim]
      - he_prototypes: [M, node_feat_dim]  (用于决定哪些节点属于该 hyperedge)
      - he_exist_logits: [M] (该超边是否存在的 logit，用于训练的 BCE)
    """
    def __init__(self, latent_dim, class_emb_dim, node_feat_dim, hidden_dim,
                 max_nodes=204800, max_classes=1000):
        super().__init__()
        self.cls_emb = nn.Embedding(max_classes, class_emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + class_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # node decoder
        self.node_pos_emb = nn.Embedding(max_nodes, 64)
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feat_dim)
        )

        # he prototype generator (per hyperedge a prototype vector in same dim as node feat)
        self.he_proto_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feat_dim)
        )
        # he existence logit
        self.he_exist = nn.Linear(hidden_dim, 1)

    def forward(self, z, class_idx, target_N, target_M, device):
        """
        z: [latent_dim]
        class_idx: int or tensor
        target_N: number of nodes
        target_M: number of hyperedges to generate (we recommend smallish M for memory)
        """
        if isinstance(class_idx, int):
            class_idx = torch.tensor([class_idx], device=device)
        cls = self.cls_emb(class_idx).squeeze(0)
        g = torch.cat([z, cls], dim=0).unsqueeze(0)
        global_dec = self.fc(g).squeeze(0)  # [H]

        # nodes
        node_idx = torch.arange(target_N, device=device).long()
        pos = self.node_pos_emb(node_idx)  # [N, pos_dim]
        global_expand = global_dec.unsqueeze(0).repeat(target_N, 1)
        node_in = torch.cat([global_expand, pos], dim=1)
        node_feats = self.node_decoder(node_in)  # [N, node_feat_dim]

        # hyperedge prototypes & exist logits
        he_global = global_dec.unsqueeze(0).repeat(target_M, 1)  # [M, H]
        he_protos = self.he_proto_fc(he_global)  # [M, node_feat_dim]
        he_logits_exist = self.he_exist(he_global).squeeze(-1)  # [M]

        return node_feats, he_protos, he_logits_exist


class ConditionalGraphVAE(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, graph_emb_dim, class_emb_dim, latent_dim, max_nodes=204800):
        super().__init__()
        self.encoder_g = GraphEncoder(node_feat_dim, hidden_dim, graph_emb_dim)
        self.encoder_cvae = ConditionalVAEEncoder(graph_emb_dim, class_emb_dim, latent_dim)
        self.generator = ConditionalGenerator(latent_dim, class_emb_dim, node_feat_dim, hidden_dim, max_nodes=max_nodes)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, node_feats, class_idx, target_M):
        g = self.encoder_g(node_feats)
        mu, logvar = self.encoder_cvae(g, class_idx)
        z = self.reparam(mu, logvar)
        recon_node_feats, he_protos, he_exist_logits = self.generator(z, class_idx, node_feats.shape[0], target_M, node_feats.device)
        return recon_node_feats, he_protos, he_exist_logits, mu, logvar


# -------------------- Loss --------------------
def vae_loss_with_he(recon_node_feats, target_node_feats, mu, logvar, he_exist_logits, target_incidence, kl_weight=1.0):
    # reconstruction (node features) + KL
    rec_loss = F.mse_loss(recon_node_feats, target_node_feats, reduction='mean')
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # hyperedge existence loss (per hyperedge label = whether that he has any incident node)
    # target_incidence: [N, M]
    he_labels = (target_incidence.sum(dim=0) > 0).float().to(he_exist_logits.device)  # [M]
    he_loss = F.binary_cross_entropy_with_logits(he_exist_logits, he_labels)

    total_loss = rec_loss + kl_weight * kld_loss + he_loss
    return total_loss, rec_loss.item(), kld_loss.item(), he_loss.item()


# -------------------- Training epoch (with optional hyperedge subsampling) --------------------
def train_epoch(model, dataloader, optim, device, kl_weight=1.0, max_he_per_graph=512):
    model.train()
    total_loss = 0.0
    for sample in tqdm(dataloader):
        node_feats = sample['node_feats'].to(device)  # [N, F]
        incidence = sample['incidence']              # keep on cpu for sampling
        class_idx = sample['class_idx']
        N = node_feats.shape[0]
        M = incidence.shape[1]

        # If M too large, subsample hyperedges to keep memory/compute reasonable.
        if M > max_he_per_graph:
            perm = torch.randperm(M)[:max_he_per_graph]
            incidence_sub = incidence[:, perm].to(device)  # [N, M_sub]
            target_M = incidence_sub.shape[1]
        else:
            incidence_sub = incidence.to(device)
            target_M = M

        optim.zero_grad()
        recon_node_feats, he_protos, he_exist_logits, mu, logvar = model(node_feats, class_idx, target_M)

        # For he existence loss we compare he_exist_logits vs incidence_sub
        loss, rec_l, kld_l, he_l = vae_loss_with_he(recon_node_feats, node_feats, mu, logvar, he_exist_logits, incidence_sub, kl_weight=kl_weight)
        loss.backward()
        optim.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# -------------------- Generate graphs (nodes + edges) --------------------
def generate_graphs(model, class_name, class2idx, idx2class, n, sample_prototype_means,
                    device, save_dir, coords_template, threshold=0.5, max_he=256):
    print(class_name)
    model.eval()
    class_idx = class2idx[class_name]
    generated = []
    os.makedirs(save_dir, exist_ok=True)

    for i in range(n):
        # sample z
        if class_idx in sample_prototype_means and sample_prototype_means[class_idx] is not None:
            mu = sample_prototype_means[class_idx].to(device)
            z = mu + torch.randn_like(mu).to(device)
        else:
            latent_dim = model.encoder_cvae.mean.out_features if hasattr(model.encoder_cvae, 'mean') else 128
            # fallback if above not available
            z = torch.randn(model.generator.he_proto_fc[0].in_features if hasattr(model.generator, 'he_proto_fc') else latent_dim, device=device)
            # better fallback: get latent dim from model (we assume stored in obj)
            try:
                z = torch.randn(model.encoder_cvae.mean.out_features, device=device)
            except Exception:
                z = torch.randn(128, device=device)

        coords = coords_template  # Tensor [N,2] on cpu
        N = coords.shape[0]
        M = min(max_he, max(1, N // 8))  # choose some M based on N but bounded

        # generator -> node_feats [N,F], he_protos [M,F], he_exist_logits [M]
        with torch.no_grad():
            recon_feats, he_protos, he_exist_logits = model.generator(z, class_idx, N, M, device)

            # normalize for cosine-sim
            nf = recon_feats
            nf_norm = nf / (nf.norm(dim=1, keepdim=True) + 1e-8)  # [N, F]
            he_p = he_protos
            he_p_norm = he_p / (he_p.norm(dim=1, keepdim=True) + 1e-8)  # [M, F]

            # compute membership probabilities by cosine similarity (N x M)
            sim = torch.matmul(nf_norm, he_p_norm.t())  # [N, M] in [-1,1]
            # sharpen and combine with existence prob
            exist_p = torch.sigmoid(he_exist_logits).unsqueeze(0)  # [1, M]
            membership_p = torch.sigmoid(sim * 10.0) * exist_p  # [N, M]

            incidence = (membership_p > threshold).float().cpu()  # [N, M] on cpu

            # ensure at least one edge per graph: if all zeros, connect first node to first he
            if incidence.sum() == 0:
                incidence[0, 0] = 1.0

            # build hypergraph (node_features and coords must be cpu tensors)
            hg = build_hypergraph_from_incidence(recon_feats.detach().cpu(), coords.cpu(), incidence)

            # he node features: average incident node feats (or zeros if none)
            num_he = hg.num_nodes('he')
            he_feats = torch.zeros((num_he, recon_feats.shape[1]), dtype=torch.float32)
            src, dst = hg.edges(etype=('node', 'incidence', 'he'))  # node_idx, he_idx
            if len(src) > 0:
                src = src.to(torch.long)
                dst = dst.to(torch.long)
                for he_idx in range(num_he):
                    nodes_idx = src[dst == he_idx]
                    if len(nodes_idx) > 0:
                        he_feats[he_idx] = recon_feats[nodes_idx].mean(dim=0).cpu()
            hg.nodes['he'].data['feat'] = he_feats

        # filename: use class_name base (or change format as you like)
        save_path = os.path.join(save_dir, f"DC0000022 {i:02d}.bin")
        dgl.save_graphs(save_path, [hg])
        generated.append(save_path)

    return generated


# -------------------- main() kept mostly same, expose max_he_per_graph param --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mode', type=str, default='train', choices=['train','generate'])
    parser.add_argument('--class_name', type=str, default=None)
    parser.add_argument('--n_samples', type=int, default=10)
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = HypergraphFolderDataset(cfg['data_dir'], cfg['labels_csv'])
    class2idx = dataset.class2idx
    idx2class = dataset.idx2class

    model = ConditionalGraphVAE(
        node_feat_dim=cfg['node_feat_dim'],
        hidden_dim=cfg['hidden_dim'],
        graph_emb_dim=cfg['graph_emb_dim'],
        class_emb_dim=cfg['class_emb_dim'],
        latent_dim=cfg['latent_dim']
    ).to(device)

    os.makedirs(cfg.get('save_dir','./vae_checkpoints_seed3'), exist_ok=True)

    if args.mode == 'train':
        dataloader = DataLoader(dataset, batch_size=cfg.get('batch_size',1), shuffle=True, collate_fn=lambda x: x[0])
        optim = torch.optim.Adam(model.parameters(), lr=float(cfg.get('lr', 1e-3)))
        best_loss = 1e9
        for epoch in range(1, cfg.get('epochs', 100)+1):
            avg_loss = train_epoch(model, dataloader, optim, device, kl_weight=cfg.get('kl_weight',1.0), max_he_per_graph=cfg.get('max_he_per_graph', 512))
            print(f"Epoch {epoch} avg loss: {avg_loss:.6f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({'model': model.state_dict(), 'cfg': cfg}, os.path.join(cfg['save_dir'], 'best_cond_vae.pth'))
                print('Saved best checkpoint.')
    else:
        print(os.path.join(cfg['save_dir'], 'best_cond_vae.pth'))
        ckpt = torch.load(os.path.join(cfg['save_dir'], 'best_cond_vae.pth'), map_location=device)
        model.load_state_dict(ckpt['model'])
        prototype_means = {c_idx: None for c_idx in range(len(idx2class))}

        sample0 = dataset[0]
        coords_template = sample0['coords']
        generated_paths = generate_graphs(model, args.class_name, class2idx, idx2class, args.n_samples, prototype_means, device, cfg.get('gen_save_dir','/sharefiles1/yaoshuilian/projects/CVPR2026/hypergraphs_mixed_k9/'), coords_template, threshold=cfg.get('threshold',0.5), max_he=cfg.get('gen_max_he',256))
        print('Generated files:')
        for p in generated_paths:
            print(p)


if __name__ == '__main__':
    main()

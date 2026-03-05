import os
import h5py
import torch
import numpy as np
import dgl
from collections import deque, defaultdict
from sklearn.neighbors import NearestNeighbors

"""
Hypergraph construction from patch coordinates & features.
- Node type: "node" (WSI patches)
- Hyperedge type: "he" (constructed groups)

We build two families of hyperedges per case:
  1) Spatial hyperedges: for each patch, collect its 8-neighborhood (Chebyshev radius = patch_size)
     including itself; deduplicate identical sets across nodes.
  2) Feature KNN hyperedges: for each patch, collect its top-k neighbors by cosine similarity
     (including itself); deduplicate identical sets.

The final heterograph is a bipartite incidence graph between nodes and hyperedges with two relations:
  ('node','incidence','he') and ('he','expand','node').

Hyperedge node features:
  - 'he_type': 0 for spatial, 1 for feature
  - 'size': number of member nodes
  - 'centroid': centroid of member patch coords (x, y)
  - 'mean_feat': mean of member features (optional; toggled by SAVE_MEAN_FEAT)

Node features:
  - 'feat': original patch features
  - 'coord': (x, y)
  - 'layer': distance-to-boundary layer via BFS on 8-neighborhood grid

Edge features (both directions share same weights):
  - 'w': membership weight; for spatial hyperedges use 1.0; for feature hyperedges use cosine similarity
         between member and the hyperedge's seed node (the node that originated the KNN set).

Saved to DGL .bin per case.
"""

# ==================== Config ====================
# patch_dir = "/public/home/jiaqi2/project/NIMM/patches"
# feat_dir = "/public/home/jiaqi2/project/NIMM/features"
# save_dir = "/public/home/jiaqi2/project/NIMM/hypergraphs_mixed_k9"
patch_dir = "/sharefiles2/yaoshuilian/TCGA-LUSC-feat/h5_files/"
feat_dir = "/sharefiles2/yaoshuilian/TCGA-LUSC-feat/pt_files/"
save_dir = "/sharefiles2/yaoshuilian/TCGA-LUSC-Hypergraph/"
os.makedirs(save_dir, exist_ok=True)

patch_size = 256                    # pixel stride of the sampling grid
k_feat = 9                          # K for feature-based hyperedges
BUILD_SPATIAL = True
BUILD_FEATURE = True
SAVE_MEAN_FEAT = True              # set True if memory allows
DEDUP_HYPEREDGES = True             # remove identical member sets

# ==================== Utilities ====================

def compute_layers(points, patch_size=256):
    point_to_idx = {tuple(p): i for i, p in enumerate(map(tuple, points))}
    visited = np.zeros(len(points), dtype=bool)
    layer_id = -np.ones(len(points), dtype=int)

    neighbors = [(dx, dy) for dx in [-patch_size, 0, patch_size]
                          for dy in [-patch_size, 0, patch_size]
                          if not (dx == 0 and dy == 0)]

    # boundary if missing at least one of the 8 neighbors
    boundary_idx = []
    for i, p in enumerate(points):
        cnt = sum(1 for dx, dy in neighbors if (p[0]+dx, p[1]+dy) in point_to_idx)
        if cnt < len(neighbors):
            boundary_idx.append(i)

    q = deque([(i, 0) for i in boundary_idx])
    for i in boundary_idx:
        visited[i] = True
        layer_id[i] = 0

    while q:
        idx, layer = q.popleft()
        p = points[idx]
        for dx, dy in neighbors:
            np_ = (p[0]+dx, p[1]+dy)
            if np_ in point_to_idx:
                j = point_to_idx[np_]
                if not visited[j]:
                    visited[j] = True
                    layer_id[j] = layer + 1
                    q.append((j, layer + 1))
    return layer_id


def build_spatial_hyperedges(coords, patch_size=256):
    """Return list of member index arrays and a list of seed indices (for weights)."""
    point_to_idx = {tuple(p): i for i, p in enumerate(map(tuple, coords))}
    neighbors = [(dx, dy) for dx in [-patch_size, 0, patch_size]
                          for dy in [-patch_size, 0, patch_size]]  # include (0,0)
    hedges = []
    seeds = []
    for i, p in enumerate(coords):
        members = []
        for dx, dy in neighbors:
            np_ = (p[0]+dx, p[1]+dy)
            if np_ in point_to_idx:
                members.append(point_to_idx[np_])
        if members:
            hedges.append(np.array(sorted(set(members)), dtype=np.int64))
            seeds.append(i)
    return hedges, seeds


def build_feature_hyperedges(features, k=8):
    """Use cosine distance KNN to form per-node hyperedges including the center itself."""
    feats = features.astype(np.float32)
    # NearestNeighbors with cosine metric returns distances in [0,2], actually [0, 2] if not normalized.
    # Using normalize L2 makes cosine distance in [0, 2], but we just compute similarity as 1 - dist.
    nbrs = NearestNeighbors(n_neighbors=min(k+1, len(feats)), metric='cosine').fit(feats)
    dists, idxs = nbrs.kneighbors(feats, return_distance=True)
    hedges = []
    seeds = []
    for i in range(len(feats)):
        neigh = idxs[i]
        # ensure self present; remove self then add back to front
        neigh = [j for j in neigh.tolist() if j != i]
        neigh = [i] + neigh[:k]  # include self + top-(k)
        hedges.append(np.array(sorted(set(neigh)), dtype=np.int64))
        seeds.append(i)
    # Also return similarity matrix for weighting to the seed
    sims = 1.0 - dists  # (N, k+1) similarity to each queried neighbor (includes self at 1.0)
    # Build a dict for quick seed->neighbor sim lookup
    seed_sim = {}
    for i in range(len(feats)):
        seed = i
        neighs = idxs[i].tolist()
        simvals = sims[i].tolist()
        seed_sim[seed] = {n: s for n, s in zip(neighs, simvals)}
        seed_sim[seed][seed] = 1.0
    return hedges, seeds, seed_sim


def deduplicate_hyperedges(hedges, seeds, prefer_first=True):
    """Remove duplicate member sets; keep the first or last seed for each set."""
    seen = {}
    new_hedges, new_seeds = [], []
    for h, s in zip(hedges, seeds):
        key = tuple(h.tolist())
        if key not in seen:
            seen[key] = (h, s)
        else:
            if not prefer_first:
                seen[key] = (h, s)
    for h, s in seen.values():
        new_hedges.append(h)
        new_seeds.append(s)
    return new_hedges, new_seeds


def build_hypergraph(features, coords, patch_size=256, k_feat=8,
                     build_spatial=True, build_feature=True,
                     dedup=True, save_mean_feat=False):
    N = features.shape[0]
    all_hedges = []          # list of np.ndarray member indices
    all_seeds = []           # seed node index per hyperedge
    all_types = []           # 0 spatial, 1 feature
    all_weights = []         # dict mapping member idx -> membership weight

    # Spatial hyperedges
    if build_spatial:
        s_hedges, s_seeds = build_spatial_hyperedges(coords, patch_size)
        if dedup:
            s_hedges, s_seeds = deduplicate_hyperedges(s_hedges, s_seeds)
        for h, seed in zip(s_hedges, s_seeds):
            all_hedges.append(h)
            all_seeds.append(seed)
            all_types.append(0)
            all_weights.append({int(m): 1.0 for m in h.tolist()})

    # Feature hyperedges
    if build_feature:
        f_hedges, f_seeds, seed_sim = build_feature_hyperedges(features, k=k_feat)
        if dedup:
            f_hedges, f_seeds = deduplicate_hyperedges(f_hedges, f_seeds)
        for h, seed in zip(f_hedges, f_seeds):
            all_hedges.append(h)
            all_seeds.append(seed)
            all_types.append(1)
            # weight by cosine similarity to seed (default 1.0 if missing)
            w = {int(m): float(seed_sim.get(seed, {}).get(int(m), 1.0)) for m in h.tolist()}
            all_weights.append(w)

    M = len(all_hedges)
    # Build bipartite incidence edges
    src_node_to_he = []   # (node -> he)
    dst_node_to_he = []
    src_he_to_node = []   # (he -> node)
    dst_he_to_node = []
    memb_weights = []     # parallel weights for both relations

    for he_id, (members, wdict) in enumerate(zip(all_hedges, all_weights)):
        for m in members.tolist():
            src_node_to_he.append(m)
            dst_node_to_he.append(he_id)
            src_he_to_node.append(he_id)
            dst_he_to_node.append(m)
            memb_weights.append(float(wdict.get(int(m), 1.0)))

    # Create heterograph
    hg = dgl.heterograph({
        ('node','incidence','he'): (torch.tensor(src_node_to_he, dtype=torch.int64),
                                    torch.tensor(dst_node_to_he, dtype=torch.int64)),
        ('he','expand','node'): (torch.tensor(src_he_to_node, dtype=torch.int64),
                                 torch.tensor(dst_he_to_node, dtype=torch.int64))
    }, num_nodes_dict={'node': N, 'he': M})

    # Assign node features
    hg.nodes['node'].data['feat'] = torch.tensor(features, dtype=torch.float32)
    hg.nodes['node'].data['coord'] = torch.tensor(coords, dtype=torch.float32)
    hg.nodes['node'].data['layer'] = torch.tensor(compute_layers(coords, patch_size), dtype=torch.int64)

    # Hyperedge features
    he_type = torch.tensor(all_types, dtype=torch.int64)
    he_size = torch.tensor([len(m) for m in all_hedges], dtype=torch.int64)
    # centroid of coords per hyperedge
    centroids = []
    for members in all_hedges:
        c = coords[members]
        centroids.append(c.mean(axis=0))
    he_centroid = torch.tensor(np.vstack(centroids), dtype=torch.float32)

    # ==================== Hyperedge features ====================
    hg.nodes['he'].data['he_type'] = he_type
    hg.nodes['he'].data['size'] = he_size
    hg.nodes['he'].data['centroid'] = he_centroid

    # Hyperedge feature: ensure 'feat' exists
    if save_mean_feat:
        mean_feats = []
        for members in all_hedges:
            mean_feats.append(features[members].mean(axis=0))
        mean_feat_tensor = torch.tensor(np.vstack(mean_feats), dtype=torch.float32)
    else:
        # 如果不保存 mean_feat，也初始化为0或使用成员均值
        mean_feat_tensor = torch.zeros(len(all_hedges), features.shape[1], dtype=torch.float32)

    hg.nodes['he'].data['feat'] = mean_feat_tensor


    # Edge weights (store on both relations with the same name)
    w = torch.tensor(memb_weights, dtype=torch.float32).unsqueeze(-1)
    hg.edges['incidence'].data['w'] = w
    hg.edges['expand'].data['w'] = w.clone()

    return hg


# ==================== Batch processing ====================

def load_case_coords(h5_path):
    with h5py.File(h5_path, 'r') as f:
        return f['coords'][:]


def load_case_features(pt_path):
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    if isinstance(data, dict) and 'feat' in data:
        t = data['feat']
        if isinstance(t, torch.Tensor):
            return t.cpu().numpy()
        return np.asarray(t)
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    else:
        return np.asarray(data)


def main():
    files = [f for f in os.listdir(patch_dir) if f.endswith('.h5')]
    for idx, fname in enumerate(sorted(files)):
        case_id = os.path.splitext(fname)[0]
        h5_path = os.path.join(patch_dir, fname)
        feat_path = os.path.join(feat_dir, case_id + '.pt')
        save_path = os.path.join(save_dir, case_id + '.bin')

        if not os.path.exists(feat_path):
            print(f"⚠️ 缺少特征文件: {case_id}")
            continue

        coords = load_case_coords(h5_path)
        features = load_case_features(feat_path)

        if coords.shape[0] != features.shape[0]:
            print(f"❌ 数量不匹配: {case_id} -> coords={coords.shape[0]}, feats={features.shape[0]}")
            continue

        print(f"→ 构建超图: {case_id} | N={coords.shape[0]} | k_feat={k_feat}")
        hg = build_hypergraph(
            features, coords,
            patch_size=patch_size,
            k_feat=k_feat,
            build_spatial=BUILD_SPATIAL,
            build_feature=BUILD_FEATURE,
            dedup=DEDUP_HYPEREDGES,
            save_mean_feat=SAVE_MEAN_FEAT,
        )

        dgl.save_graphs(save_path, [hg])
        print(f"✅ 已保存超图: {save_path} | nodes={hg.num_nodes('node')} he={hg.num_nodes('he')} "
              f"inc_edges={hg.num_edges('incidence')}")


if __name__ == '__main__':
    main()

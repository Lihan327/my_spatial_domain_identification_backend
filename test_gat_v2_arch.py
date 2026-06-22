"""Test improved GAT approach with multi-seed ensemble - clean version."""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import scanpy as sc
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score


def load_slice(sid):
    adata = sc.read_visium(path=f'DLPFC/{sid}', count_file='filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    ann_df = pd.read_csv(f'DLPFC/{sid}/metadata.tsv', sep='\t')
    adata.obs['Ground Truth'] = ann_df.loc[adata.obs_names, 'layer_guess'].values
    adata = adata[~pd.isnull(adata.obs['Ground Truth'])].copy()
    coords = adata.obsm['spatial'].astype(np.float32)
    X = adata.X.toarray()[:, adata.var['highly_variable'].values].astype(np.float32)
    gt_raw = adata.obs['Ground Truth'].astype(str).values
    gt_codes, _ = pd.factorize(gt_raw, sort=True)
    return X, coords, gt_codes


class GATRefiner(nn.Module):
    """Simple GAT-based feature refiner. Input -> hidden -> output."""
    def __init__(self, in_dim, hidden_dim, out_dim, heads=1, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.W1 = nn.Linear(in_dim, heads * hidden_dim, bias=False)
        self.a1_src = nn.Parameter(torch.empty(1, heads, hidden_dim))
        self.a1_dst = nn.Parameter(torch.empty(1, heads, hidden_dim))
        self.W2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.a2_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.a2_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        self.dropout = dropout
        for p in [self.a1_src, self.a1_dst, self.a2_src, self.a2_dst]:
            nn.init.xavier_uniform_(p)

    def _attn(self, x, ei, W, a_src, a_dst):
        N = x.size(0); H = self.heads
        D = W.weight.shape[0] // H
        Wh = F.linear(x, W.weight).view(N, H, D)
        src, dst = ei[0], ei[1]
        e = F.leaky_relu((Wh[src] * a_src).sum(-1) + (Wh[dst] * a_dst).sum(-1), 0.2)
        e_max = torch.full((N, H), float('-inf'), device=x.device)
        e_max = e_max.scatter_reduce(0, dst.unsqueeze(-1).expand(-1, H), e, reduce='amax', include_self=False)
        e_max = e_max[dst]
        alpha = (e - e_max).exp()
        denom = torch.zeros(N, H, device=x.device).scatter_add_(0, dst.unsqueeze(-1).expand(-1, H), alpha)
        denom = denom[dst] + 1e-16
        alpha = alpha / denom
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        msg = Wh[src] * alpha.unsqueeze(-1)
        out = torch.zeros(N, H, D, device=x.device).scatter_add_(0, dst.view(-1,1,1).expand_as(msg), msg)
        return out.reshape(N, H * D)

    def forward(self, x, ei):
        h = F.elu(self._attn(x, ei, self.W1, self.a1_src, self.a1_dst))
        z = self._attn(h, ei, self.W2, self.a2_src, self.a2_dst)
        return z


def train_gat(X, coords, seed=0, n_pca=50, hidden=64, out_dim=30, epochs=300,
              smooth_w=0.0, l2_w=1.0, heads=1, lr=1e-3):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(coords)
    _, knn_idx = nbrs.kneighbors(coords)
    knn_idx = knn_idx[:, 1:]
    rows = np.repeat(np.arange(n), 6)
    cols = knn_idx.reshape(-1)
    ei = torch.from_numpy(np.vstack((rows, cols)).astype(np.int64)).cuda()

    Z0 = PCA(n_components=n_pca).fit_transform(StandardScaler().fit_transform(X))
    Z_t = torch.from_numpy(Z0.astype(np.float32)).cuda()

    model = GATRefiner(n_pca, hidden, out_dim, heads=heads, dropout=0.0).cuda()
    # Init W1 as identity-like projection from PCA
    with torch.no_grad():
        take = min(hidden, n_pca)
        new_w1 = torch.zeros(heads * hidden, n_pca).cuda()
        new_w1[:take, :take] = torch.eye(take)
        model.W1.weight.copy_(new_w1)
        # W2: identity-like from hidden to out_dim
        take2 = min(out_dim, hidden)
        new_w2 = torch.zeros(heads * out_dim, hidden).cuda()
        new_w2[:take2, :take2] = torch.eye(take2)
        model.W2.weight.copy_(new_w2)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for ep in range(epochs):
        opt.zero_grad()
        z = model(Z_t, ei)
        sm = ((z[ei[0]] - z[ei[1]])**2).sum(-1).mean()
        l2 = (z - Z_t[:, :out_dim]).pow(2).mean()
        loss = l2_w * l2 + smooth_w * sm
        loss.backward()
        opt.step()
    return z.detach().cpu().numpy()


def test_method(name, fn, SLICES, K=7):
    print(f"\n=== {name} (K={K}) ===")
    aris = []
    for sid in SLICES:
        X, coords, gt = load_slice(sid)
        Z = fn(X, coords)
        g = GaussianMixture(n_components=K, covariance_type='full', n_init=10, random_state=0, reg_covar=1e-3).fit(Z).predict(Z)
        ari = adjusted_rand_score(gt, g)
        aris.append(ari)
    print(f"  ARIs: {[f'{a:.3f}' for a in aris]}")
    print(f"  median={np.median(aris):.4f}, mean={np.mean(aris):.4f}")
    return aris


SLICES = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']

# Try GAT with different l2 weights
for l2_w in [0.0, 0.01, 0.1, 1.0]:
    aris = test_method(f"GAT l2={l2_w}", lambda X, c, lw=l2_w: train_gat(X, c, l2_w=lw), SLICES, K=7)

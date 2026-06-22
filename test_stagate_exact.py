"""Implement STAGATE exact recipe."""
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
import scipy.sparse as sp
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


class GAT(nn.Module):
    """STAGATE-style GAT with attention."""
    def __init__(self, in_dim, hidden, out_dim, heads=1):
        super().__init__()
        self.heads = heads
        self.W1 = nn.Linear(in_dim, heads * hidden, bias=False)
        self.a1 = nn.Parameter(torch.empty(heads, 2 * hidden))
        self.W2 = nn.Linear(hidden, out_dim, bias=False)
        self.a2 = nn.Parameter(torch.empty(1, 2 * out_dim))
        for p in [self.a1, self.a2]:
            nn.init.xavier_uniform_(p)

    def _attn(self, x, ei, W, a, heads):
        N = x.size(0)
        D = a.shape[-1] // 2
        Wh = F.linear(x, W.weight).view(N, heads, D)
        src, dst = ei[0], ei[1]
        cat = torch.cat([Wh[src], Wh[dst]], dim=-1)  # (E, heads, 2D)
        e = (cat * a).sum(-1)  # (E, heads)
        e = F.leaky_relu(e, 0.2)
        e_max = torch.full((N, heads), float('-inf'), device=x.device)
        e_max = e_max.scatter_reduce(0, dst.unsqueeze(-1).expand(-1, heads), e, reduce='amax', include_self=False)
        e_max = e_max[dst]
        alpha = (e - e_max).exp()
        denom = torch.zeros(N, heads, device=x.device).scatter_add_(0, dst.unsqueeze(-1).expand(-1, heads), alpha)
        denom = denom[dst] + 1e-16
        alpha = alpha / denom
        msg = Wh[src] * alpha.unsqueeze(-1)
        out = torch.zeros(N, heads, D, device=x.device).scatter_add_(0, dst.view(-1,1,1).expand_as(msg), msg)
        if heads == 1:
            return out.squeeze(1)
        return out.reshape(N, heads * D)

    def forward(self, x, ei):
        h = F.elu(self._attn(x, ei, self.W1, self.a1, self.heads))
        z = self._attn(h, ei, self.W2, self.a2, 1)
        return z


def train_stagate_exact(X, coords, seed=0, hidden=32, out_dim=30, n_pca=3000,
                        epochs=300, lr=1e-3, smooth_w=0.5, l2_w=0.0,
                        pos_w=5.0, weighted_bce=True):
    """STAGATE exact: PCA init, single GAT, weighted BCE adj recon, smooth loss."""
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(coords)
    _, knn_idx = nbrs.kneighbors(coords)
    knn_idx = knn_idx[:, 1:]
    rows = np.repeat(np.arange(n), 6)
    cols = knn_idx.reshape(-1)
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A = A.maximum(A.T) + sp.eye(n)
    A_dense = A.toarray().astype(np.float32)
    ei = torch.from_numpy(np.vstack((rows, cols, np.roll(rows, 1), np.roll(cols, 1))).reshape(2, -1).astype(np.int64)).cuda()
    # Note: we need directed edge_index for proper GAT; let's also add reverse edges
    ei_sym = np.vstack((A.tocoo().row, A.tocoo().col)).astype(np.int64)
    ei_t = torch.from_numpy(ei_sym).cuda()
    A_t = torch.from_numpy(A_dense).cuda()

    # Init with PCA
    n_pca_use = min(n_pca, X.shape[1], n)
    Z0 = PCA(n_components=n_pca_use).fit_transform(X)
    Z_t = torch.from_numpy(Z0.astype(np.float32)).cuda()

    model = GAT(n_pca_use, hidden, out_dim, heads=1).cuda()
    # Init with identity projection
    with torch.no_grad():
        take1 = min(hidden, n_pca_use)
        new_w1 = torch.zeros(hidden, n_pca_use).cuda()
        new_w1[:take1, :take1] = torch.eye(take1)
        model.W1.weight.copy_(new_w1)
        take2 = min(out_dim, hidden)
        new_w2 = torch.zeros(out_dim, hidden).cuda()
        new_w2[:take2, :take2] = torch.eye(take2)
        model.W2.weight.copy_(new_w2)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for ep in range(epochs):
        opt.zero_grad()
        z = model(Z_t, ei_t)
        # logit for BCE with pos_weight
        logits = z @ z.t()
        if weighted_bce:
            l_adj = F.binary_cross_entropy_with_logits(logits, A_t, pos_weight=torch.tensor([pos_w]).cuda())
        else:
            l_adj = F.binary_cross_entropy_with_logits(logits, A_t)
        sm = ((z[ei_t[0]] - z[ei_t[1]])**2).sum(-1).mean()
        l2 = (z - Z_t[:, :out_dim]).pow(2).mean()
        loss = l_adj + smooth_w * sm + l2_w * l2
        loss.backward()
        opt.step()
    return z.detach().cpu().numpy()


def test(name, fn, SLICES, K=7):
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

# Test 1: STAGATE exact
for sm_w in [0.5]:
    for pw in [1.0, 5.0, 10.0]:
        for l2_w in [0.0, 0.1, 1.0]:
            try:
                test(f"STAGATE sm={sm_w} pw={pw} l2={l2_w}",
                     lambda X, c, sm=sm_w, pw=pw, l2=l2_w: train_stagate_exact(X, c, smooth_w=sm, pos_w=pw, l2_w=l2, weighted_bce=(pw > 1)),
                     SLICES, K=7)
            except Exception as e:
                print(f"  Failed: {e}")

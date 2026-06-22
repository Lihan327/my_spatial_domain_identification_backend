"""Test GNN variants for DLPFC."""
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def build_adj_norm(coords, k=6):
    n = coords.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
    _, idx = nbrs.kneighbors(coords)
    idx = idx[:, 1:]
    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1)
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A = A.maximum(A.T) + sp.eye(n)
    deg = np.asarray(A.sum(1)).flatten()
    d_inv_sqrt = np.power(deg, -0.5, where=deg>0)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    D = sp.diags(d_inv_sqrt)
    L = (D @ A @ D).tocoo()
    ei = np.vstack((L.row, L.col)).astype(np.int64)
    ew = L.data.astype(np.float32)
    return ei, ew


class GCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden)
        self.W2 = nn.Linear(hidden, out_dim)
    def forward(self, x, ei, ew):
        N = x.size(0)
        msg = ew.unsqueeze(-1) * x[ei[0]]
        agg = torch.zeros(N, x.size(1), device=x.device).scatter_add_(0, ei[1].view(-1,1).expand_as(msg), msg)
        deg = torch.zeros(N, device=x.device).scatter_add_(0, ei[1], ew)
        deg = deg.clamp(min=1e-6)
        h = agg / deg.unsqueeze(-1)
        h = F.elu(self.W1(h))
        msg2 = ew.unsqueeze(-1) * h[ei[0]]
        agg2 = torch.zeros(N, h.size(1), device=x.device).scatter_add_(0, ei[1].view(-1,1).expand_as(msg2), msg2)
        h2 = agg2 / deg.unsqueeze(-1)
        return self.W2(h2)


def test_gcn(sid, smooth_w=0.1, hidden=64, lr=1e-3, epochs=300):
    adata = sc.read_visium(path=f'DLPFC/{sid}', count_file='filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    ann_df = pd.read_csv(f'DLPFC/{sid}/metadata.tsv', sep='\t')
    adata.obs['Ground Truth'] = ann_df.loc[adata.obs_names, 'layer_guess'].values
    adata = adata[~pd.isnull(adata.obs['Ground Truth'])].copy()
    n = adata.shape[0]
    coords = adata.obsm['spatial'].astype(np.float32)
    X = adata.X.toarray()[:, adata.var['highly_variable'].values].astype(np.float32)
    gt_raw = adata.obs['Ground Truth'].astype(str).values
    gt_codes, _ = pd.factorize(gt_raw, sort=True)

    Z0 = PCA(n_components=30).fit_transform(StandardScaler().fit_transform(X))
    nbrs = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(coords)
    _, knn_idx = nbrs.kneighbors(coords)
    knn_idx = knn_idx[:, 1:]
    Y = Z0.copy()
    for _ in range(2):
        Y = 0.5 * Y + 0.5 * Y[knn_idx].mean(axis=1)

    ei, ew = build_adj_norm(coords, k=6)
    ei_t = torch.from_numpy(ei).cuda()
    ew_t = torch.from_numpy(ew).cuda()
    Y_t = torch.from_numpy(Y.astype(np.float32)).cuda()

    torch.manual_seed(0)
    gcn = GCN(30, hidden, 30).cuda()
    opt = torch.optim.Adam(gcn.parameters(), lr=lr, weight_decay=1e-5)

    for ep in range(epochs):
        opt.zero_grad()
        z = gcn(Y_t, ei_t, ew_t)
        sm = ((z[ei_t[0]] - z[ei_t[1]])**2).sum(-1).mean()
        recon = F.mse_loss(z, Y_t)
        loss = recon + smooth_w * sm
        loss.backward()
        opt.step()

    Z = z.detach().cpu().numpy()
    g = GaussianMixture(n_components=7, covariance_type='full', n_init=5, random_state=0, reg_covar=1e-3).fit(Z).predict(Z)
    ari = adjusted_rand_score(gt_codes, g)
    nmi = normalized_mutual_info_score(gt_codes, g)
    return ari, nmi


if __name__ == "__main__":
    SLICES = ['151507', '151508', '151509', '151510', '151669', '151670', '151671']
    print("\n=== GCN with various smooth weights ===")
    for sw in [0.0, 0.01, 0.1, 0.5]:
        aris = []
        for sid in SLICES:
            ari, nmi = test_gcn(sid, smooth_w=sw)
            aris.append(ari)
        print(f"  sw={sw}: ARIs = {[f'{a:.3f}' for a in aris]}, mean={np.mean(aris):.4f}")

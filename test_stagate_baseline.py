"""Test STAGATE-style approach: GAT with adj reconstruction."""
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


# Simple GATv2 layer with proper attention
class GATv2Layer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, concat=True, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.concat = concat
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if not concat else None
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x, ei):
        N = x.size(0)
        H = self.heads
        D = self.W.out_features // H
        Wh = self.W(x).view(N, H, D)
        src, dst = ei[0], ei[1]
        Wh_src, Wh_dst = Wh[src], Wh[dst]
        e_src = (Wh_src * self.a_src).sum(-1)
        e_dst = (Wh_dst * self.a_dst).sum(-1)
        e = F.leaky_relu(e_src + e_dst, negative_slope=0.2)
        e_max = torch.full((N, H), float('-inf'), device=x.device)
        e_max = e_max.scatter_reduce(0, dst.unsqueeze(-1).expand(-1, H), e, reduce='amax', include_self=False)
        e_max = e_max[dst]
        alpha = (e - e_max).exp()
        denom = torch.zeros(N, H, device=x.device).scatter_add_(0, dst.unsqueeze(-1).expand(-1, H), alpha)
        denom = denom[dst] + 1e-16
        alpha = alpha / denom
        msg = Wh_src * alpha.unsqueeze(-1)
        out = torch.zeros(N, H, D, device=x.device).scatter_add_(0, dst.view(-1,1,1).expand_as(msg), msg)
        if self.concat:
            out = out.reshape(N, H * D)
            if self.bias is not None: out = out + self.bias
        else:
            out = out.mean(dim=1)
            if self.bias is not None: out = out + self.bias
        return out


class STAGATEModel(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, heads=1, dropout=0.0):
        super().__init__()
        self.gat1 = GATv2Layer(in_dim, hidden // heads, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATv2Layer(hidden, out_dim, heads=1, concat=False, dropout=dropout)

    def forward(self, x, ei):
        h = F.elu(self.gat1(x, ei))
        z = self.gat2(h, ei)
        return z


def stagate_train(X, coords, k=6, hidden=32, out_dim=30, epochs=300, lr=1e-3,
                  pos_w=1.0, smooth_w=0.0, seed=0):
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
    _, knn_idx = nbrs.kneighbors(coords)
    knn_idx = knn_idx[:, 1:]
    rows = np.repeat(np.arange(n), k)
    cols = knn_idx.reshape(-1)
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A = A.maximum(A.T) + sp.eye(n)
    A_dense = A.toarray().astype(np.float32)
    ei = np.vstack((A.tocoo().row, A.tocoo().col)).astype(np.int64)

    # Init with PCA
    Xs = StandardScaler().fit_transform(X)
    Z0 = PCA(n_components=out_dim).fit_transform(Xs).astype(np.float32)

    ei_t = torch.from_numpy(ei).cuda()
    Z_t = torch.from_numpy(Z0).cuda()
    A_t = torch.from_numpy(A_dense).cuda()

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = STAGATEModel(out_dim, hidden, out_dim, heads=1, dropout=0.0).cuda()
    # Init W of gat1 with identity
    with torch.no_grad():
        if hidden == out_dim:
            model.gat1.W.weight.copy_(torch.eye(out_dim).cuda())
        else:
            model.gat1.W.weight.copy_(torch.eye(hidden, out_dim).cuda())
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for ep in range(epochs):
        opt.zero_grad()
        z = model(Z_t, ei_t)
        # Adjacency reconstruction
        a_hat = torch.sigmoid(z @ z.t())
        # BCE on adj
        l_adj = F.binary_cross_entropy(a_hat.clamp(1e-7, 1-1e-7), A_t, reduction='mean')
        # Smooth loss
        l_sm = ((z[ei_t[0]] - z[ei_t[1]])**2).sum(-1).mean()
        loss = l_adj + smooth_w * l_sm
        loss.backward()
        opt.step()

    return z.detach().cpu().numpy()


def test_stagate(sid, hidden=32, out_dim=30, epochs=300, pos_w=1.0, smooth_w=0.0, K=7):
    X, coords, gt = load_slice(sid)
    Z = stagate_train(X, coords, hidden=hidden, out_dim=out_dim,
                      epochs=epochs, pos_w=pos_w, smooth_w=smooth_w)
    g = GaussianMixture(n_components=K, covariance_type='full', n_init=10, random_state=0, reg_covar=1e-3).fit(Z).predict(Z)
    ari = adjusted_rand_score(gt, g)
    return ari


if __name__ == "__main__":
    SLICES = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']

    # Test 1: STAGATE-style with PCA init, identity-like W init
    print("\n=== STAGATE-style: PCA-init W, BCE on adj ===")
    for K in [6, 7]:
        aris = []
        for sid in SLICES:
            ari = test_stagate(sid, hidden=32, out_dim=30, epochs=300, K=K)
            aris.append(ari)
        print(f"  K={K}: ARIs={[f'{a:.3f}' for a in aris]}, median={np.median(aris):.4f}, mean={np.mean(aris):.4f}")

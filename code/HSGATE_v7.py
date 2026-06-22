"""Pipeline v7: multi-scale smoothing + GAT denoising + multi-K GMM + post-processing."""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             homogeneity_score, completeness_score,
                             silhouette_score)
import matplotlib.pyplot as plt

sc.settings.verbosity = 1


def load_slice(sid, data_root='DLPFC'):
    adata = sc.read_visium(path=os.path.join(data_root, sid), count_file='filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    ann_df = pd.read_csv(os.path.join(data_root, sid, 'metadata.tsv'), sep='\t')
    adata.obs['Ground Truth'] = ann_df.loc[adata.obs_names, 'layer_guess'].values
    adata = adata[~pd.isnull(adata.obs['Ground Truth'])].copy()
    return adata


def spatial_smooth(X, knn_idx, rounds=2, alpha=0.5):
    Y = X.copy()
    for _ in range(rounds):
        Y = (1 - alpha) * Y + alpha * Y[knn_idx].mean(axis=1)
    return Y


class GATv2(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, heads=1):
        super().__init__()
        self.heads = heads
        self.W1 = nn.Linear(in_dim, heads * hidden, bias=False)
        self.a1_src = nn.Parameter(torch.empty(1, heads, hidden))
        self.a1_dst = nn.Parameter(torch.empty(1, heads, hidden))
        self.W2 = nn.Linear(hidden, out_dim, bias=False)
        self.a2_src = nn.Parameter(torch.empty(1, 1, out_dim))
        self.a2_dst = nn.Parameter(torch.empty(1, 1, out_dim))
        for p in [self.a1_src, self.a1_dst, self.a2_src, self.a2_dst]:
            nn.init.xavier_uniform_(p)

    def _attn(self, x, ei, W, a_src, a_dst):
        N = x.size(0); H = a_src.size(1)
        D = a_src.size(2)
        Wh = F.linear(x, W.weight).view(N, H, D)
        src, dst = ei[0], ei[1]
        Wh_src, Wh_dst = Wh[src], Wh[dst]
        e = F.leaky_relu((Wh_src * a_src).sum(-1) + (Wh_dst * a_dst).sum(-1), 0.2)
        e_max = torch.full((N, H), float('-inf'), device=x.device)
        e_max = e_max.scatter_reduce(0, dst.unsqueeze(-1).expand(-1, H), e, reduce='amax', include_self=False)
        e_max = e_max[dst]
        alpha = (e - e_max).exp()
        denom = torch.zeros(N, H, device=x.device).scatter_add_(0, dst.unsqueeze(-1).expand(-1, H), alpha)
        denom = denom[dst] + 1e-16
        alpha = alpha / denom
        msg = Wh_src * alpha.unsqueeze(-1)
        out = torch.zeros(N, H, D, device=x.device).scatter_add_(0, dst.view(-1,1,1).expand_as(msg), msg)
        if self.heads == 1:
            return out.squeeze(1)
        return out.reshape(N, H * D)

    def forward(self, x, ei):
        h = F.elu(self._attn(x, ei, self.W1, self.a1_src, self.a1_dst))
        z = self._attn(h, ei, self.W2, self.a2_src, self.a2_dst)
        return z


def train_gat_denoise(Z0, ei, seed=0, hidden=64, out_dim=30, epochs=300,
                      l2_w=1.0, smooth_w=0.1, lr=1e-3):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    Z_t = torch.from_numpy(Z0.astype(np.float32)).cuda()
    ei_t = torch.from_numpy(ei.astype(np.int64)).cuda()
    in_dim = Z0.shape[1]
    n = Z0.shape[0]
    model = GATv2(in_dim, hidden, out_dim, heads=1).cuda()
    with torch.no_grad():
        take1 = min(hidden, in_dim)
        new_w1 = torch.zeros(hidden, in_dim).cuda()
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
        l2 = (z - Z_t[:, :out_dim]).pow(2).mean()
        sm = ((z[ei_t[0]] - z[ei_t[1]])**2).sum(-1).mean()
        loss = l2_w * l2 + smooth_w * sm
        loss.backward()
        opt.step()
    return z.detach().cpu().numpy()


def majority_vote(labels, knn_idx, k=6, min_consensus=5):
    out = labels.copy()
    for i in range(labels.shape[0]):
        nbrs = knn_idx[i, :k]
        nbr_labels = labels[nbrs]
        uniq, counts = np.unique(nbr_labels, return_counts=True)
        top = uniq[counts.argmax()]
        if top != labels[i] and counts.max() >= min_consensus:
            out[i] = top
    return out


def small_cluster_cleanup(labels, knn_idx, min_ratio=0.02):
    n = labels.shape[0]
    min_size = max(1, int(min_ratio * n))
    uniq, counts = np.unique(labels, return_counts=True)
    small = set(uniq[counts < min_size].tolist())
    if not small:
        return labels
    out = labels.copy()
    for i in np.where(np.isin(labels, list(small)))[0]:
        nbrs = knn_idx[i]
        nbr_labels = labels[nbrs]
        uniq2, counts2 = np.unique(nbr_labels, return_counts=True)
        keep = ~np.isin(uniq2, list(small))
        if keep.sum() == 0: continue
        uniq2 = uniq2[keep]; counts2 = counts2[keep]
        out[i] = uniq2[counts2.argmax()]
    return out


def hungarian_remap(pred, gt):
    from scipy.optimize import linear_sum_assignment
    p_uniq = np.unique(pred); g_uniq = np.unique(gt)
    cost = np.zeros((len(p_uniq), len(g_uniq)), dtype=np.int64)
    for i, p in enumerate(p_uniq):
        for j, g in enumerate(g_uniq):
            cost[i, j] = -((pred == p) & (gt == g)).sum()
    row, col = linear_sum_assignment(cost)
    remap = {int(p_uniq[r]): int(g_uniq[c]) for r, c in zip(row, col)}
    return np.array([remap.get(int(v), int(v)) for v in pred], dtype=np.int64)


def plot_spatial(adata, color_key, title, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    sc.pl.spatial(adata, img_key='hires', color=color_key, show=False, ax=ax,
                  legend_fontsize=11, frameon=False)
    plt.subplots_adjust(right=0.78)
    plt.title(title, fontsize=18)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_slice(sid, data_root='DLPFC', out_root='DLPFC/DLPFC_result',
                  K_list=(5, 6, 7), n_seeds_gmm=3, post_iter=2,
                  use_multiscale=True, use_gat=True, gat_epochs=200):
    print(f"\n========== {sid} ==========")
    t0 = time.time()
    adata = load_slice(sid, data_root)
    n = adata.shape[0]
    coords = adata.obsm['spatial'].astype(np.float32)
    X = adata.X.toarray()[:, adata.var['highly_variable'].values].astype(np.float32)
    gt_raw = adata.obs['Ground Truth'].astype(str).values
    gt_codes, gt_uniques = pd.factorize(gt_raw, sort=True)
    n_layers = len(gt_uniques)

    nbrs = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(coords)
    _, knn_idx = nbrs.kneighbors(coords)
    knn_idx = knn_idx[:, 1:]

    # Build edge_index
    rows = np.repeat(np.arange(n), 6)
    cols = knn_idx.reshape(-1)
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A = A.maximum(A.T) + sp.eye(n)
    ei = np.vstack((A.tocoo().row, A.tocoo().col)).astype(np.int64)

    # Multi-scale smoothing
    if use_multiscale:
        Y1 = spatial_smooth(X, knn_idx, rounds=2, alpha=0.3)
        Y2 = spatial_smooth(X, knn_idx, rounds=2, alpha=0.5)
        Y3 = spatial_smooth(X, knn_idx, rounds=3, alpha=0.7)
        Y = np.hstack([Y1, Y2, Y3])
    else:
        Y = spatial_smooth(X, knn_idx, rounds=2, alpha=0.5)
    Z0 = PCA(n_components=30).fit_transform(StandardScaler().fit_transform(Y))

    # Optional GAT denoising
    if use_gat:
        Z = train_gat_denoise(Z0, ei, seed=0, hidden=64, out_dim=30,
                              epochs=gat_epochs, l2_w=1.0, smooth_w=0.1)
    else:
        Z = Z0

    # Multi-K, multi-seed GMM
    best_score = -np.inf; best_labels = None; best_k = None
    for K in K_list:
        for s in range(n_seeds_gmm):
            gmm = GaussianMixture(n_components=K, covariance_type='full', n_init=3,
                                  random_state=s, reg_covar=1e-3, max_iter=200)
            gmm.fit(Z)
            labels = gmm.predict(Z)
            try:
                sil = silhouette_score(Z, labels, sample_size=min(2000, n))
            except Exception:
                sil = 0
            score = sil - 0.001 * abs(gmm.bic(Z)) / 1e6
            if score > best_score:
                best_score = score
                best_labels = labels
                best_k = K

    # Post-processing
    labels = best_labels
    for it in range(post_iter):
        labels = small_cluster_cleanup(labels, knn_idx, min_ratio=0.02)
        labels = majority_vote(labels, knn_idx, k=6, min_consensus=5)

    ari = adjusted_rand_score(gt_codes, labels)
    nmi = normalized_mutual_info_score(gt_codes, labels)
    hs = homogeneity_score(gt_codes, labels)
    cs = completeness_score(gt_codes, labels)

    labels_h = hungarian_remap(labels, gt_codes)
    adata.obs['Pred'] = pd.Categorical([f'd{c}' for c in labels_h])
    adata.obs['PredRaw'] = pd.Categorical([f'p{c}' for c in labels])
    adata.uns['gt_uniques'] = list(gt_uniques)
    adata.uns['n_layers'] = n_layers
    adata.uns['K'] = best_k

    out_dir = os.path.join(out_root, sid)
    plot_spatial(adata, 'Pred', f'{sid} Pred (K={best_k}, true={n_layers})',
                 os.path.join(out_dir, f'{sid}_pred.png'))

    elapsed = time.time() - t0
    print(f"  K={best_k} (true={n_layers}); ARI={ari:.4f} NMI={nmi:.4f} HS={hs:.4f} CS={cs:.4f}  ({elapsed:.1f}s)")
    return dict(section=sid, n_spots=n, K=best_k, true_layers=n_layers,
                ARI=ari, NMI=nmi, HS=hs, CS=cs, time_s=round(elapsed, 1))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='DLPFC')
    parser.add_argument('--out_root', default='DLPFC/DLPFC_result')
    parser.add_argument('--csv_path', default='results/HSGATE-v7_per_slice_metrics.csv')
    parser.add_argument('--summary_path', default='results/summary_mean_median.csv')
    parser.add_argument('--slices', default='all')
    parser.add_argument('--n_seeds_gmm', type=int, default=3)
    parser.add_argument('--gat_epochs', type=int, default=200)
    args = parser.parse_args()

    if args.slices == 'all':
        SLICES = ['151507', '151508', '151509', '151510',
                  '151669', '151670', '151671', '151672',
                  '151673', '151674', '151675', '151676']
    else:
        SLICES = [s.strip() for s in args.slices.split(',')]

    rows = []
    for sid in SLICES:
        try:
            row = process_slice(sid, args.data_root, args.out_root,
                                n_seeds_gmm=args.n_seeds_gmm, gat_epochs=args.gat_epochs)
            rows.append(row)
        except Exception as e:
            import traceback
            print(f"!! {sid} failed: {e}")
            traceback.print_exc()
            rows.append(dict(section=sid, n_spots=0, K=0, true_layers=0,
                             ARI=0.0, NMI=0.0, HS=0.0, CS=0.0, time_s=0.0))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)
    df.to_csv(args.csv_path, index=False)
    print("\n========== Per-slice metrics ==========")
    print(df.to_string(index=False))

    metric_cols = ['ARI', 'NMI', 'HS', 'CS']
    summary = {c: dict(mean=df[c].mean(), median=df[c].median(),
                       std=df[c].std(), min=df[c].min(), max=df[c].max())
               for c in metric_cols}
    summary_df = pd.DataFrame(summary).T
    summary_df.index.name = 'metric'
    summary_df.to_csv(args.summary_path)
    print("\n========== Summary ==========")
    print(summary_df.to_string())

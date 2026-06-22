"""Pipeline v4: smart K selection + iterative refinement + post-processing."""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
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
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def cluster_gmm_multi_seed(Z, K, n_seeds=5, random_state_offset=0):
    """Run GMM multiple times with different seeds, return labels of best by likelihood."""
    best_ll = -np.inf; best_labels = None
    for s in range(n_seeds):
        gmm = GaussianMixture(n_components=K, covariance_type='full', n_init=3,
                              random_state=s + random_state_offset, reg_covar=1e-3, max_iter=200)
        gmm.fit(Z)
        labels = gmm.predict(Z)
        ll = gmm.score(Z) * Z.shape[0]  # log likelihood
        if ll > best_ll:
            best_ll = ll
            best_labels = labels
    return best_labels, best_ll


def select_K_bic(Z, K_range=(4, 5, 6, 7, 8, 9)):
    """Select K by BIC."""
    best_bic = np.inf; best_k = None; best_labels = None
    for k in K_range:
        try:
            gmm = GaussianMixture(n_components=k, covariance_type='full', n_init=3,
                                  random_state=0, reg_covar=1e-3, max_iter=200)
            gmm.fit(Z)
            labels = gmm.predict(Z)
            bic = gmm.bic(Z)
            if bic < best_bic:
                best_bic = bic; best_k = k; best_labels = labels
        except Exception as e:
            print(f"  K={k} failed: {e}")
    return best_labels, best_k, best_bic


def process_slice(sid, data_root='DLPFC', out_root='DLPFC/DLPFC_result',
                  smooth_rounds=2, smooth_alpha=0.5, n_pca=30,
                  n_seeds_gmm=5, n_iter_refine=1, post_iter=2,
                  K_range=(5, 6, 7)):
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

    # Smooth
    Y = spatial_smooth(X, knn_idx, rounds=smooth_rounds, alpha=smooth_alpha)
    Z = PCA(n_components=n_pca).fit_transform(StandardScaler().fit_transform(Y))

    # Try K in range, pick by best likelihood (not BIC since BIC always picks highest)
    best_ll = -np.inf; best_k = None; best_labels = None
    for K in K_range:
        labels, ll = cluster_gmm_multi_seed(Z, K, n_seeds=n_seeds_gmm)
        if ll > best_ll:
            best_ll = ll
            best_k = K
            best_labels = labels

    # Iterative refinement: re-cluster with refined features
    for it in range(n_iter_refine):
        Y_ref = Y.copy()
        for i in range(n):
            same_cluster = (best_labels[knn_idx[i]] == best_labels[i])
            same_nbrs = knn_idx[i][same_cluster]
            if len(same_nbrs) > 0:
                Y_ref[i] = 0.5 * Y[i] + 0.5 * Y[same_nbrs].mean(axis=0)
        Y = Y_ref
        Y = spatial_smooth(Y, knn_idx, rounds=1, alpha=0.3)
        Z = PCA(n_components=n_pca).fit_transform(StandardScaler().fit_transform(Y))
        # Re-pick K and re-cluster
        best_ll2 = -np.inf; best_k2 = None; best_labels2 = None
        for K in K_range:
            labels, ll = cluster_gmm_multi_seed(Z, K, n_seeds=n_seeds_gmm)
            if ll > best_ll2:
                best_ll2 = ll; best_k2 = K; best_labels2 = labels
        best_labels = best_labels2
        best_k = best_k2

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
    parser.add_argument('--csv_path', default='results/HSGATE-v4_per_slice_metrics.csv')
    parser.add_argument('--summary_path', default='results/summary_mean_median.csv')
    parser.add_argument('--slices', default='all')
    parser.add_argument('--n_seeds_gmm', type=int, default=5)
    parser.add_argument('--n_iter_refine', type=int, default=1)
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
                                n_seeds_gmm=args.n_seeds_gmm,
                                n_iter_refine=args.n_iter_refine)
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

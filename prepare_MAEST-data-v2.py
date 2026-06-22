"""Stage 1: MAEST standard data preprocessing.

Strictly follows MAEST paper's preprocessing pipeline:
  1. HVG selection: 3000 highly variable genes (seurat_v3 flavor)
  2. Library normalization: total count = 1e4
  3. Log1p transform
  4. Scale with zero_center=False, max_value=10 (clip to [-10, 10])
  5. k=3 kNN graph (no self-loop in raw A, add self-loop in normalized)
  6. Symmetric normalization: D^-1/2 (A + I) D^-1/2

Also pre-computes v7 features (5-scale smoothing + scRNA + position + PCA(30))
for ensemble with MAEST embedding.
"""
from __future__ import annotations

import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

from code.utils import set_seed, load_visium_slice
from code.scrna_features import compute_cell_type_score
from code.multi_scale_smooth import multi_scale_smooth

set_seed(42)
SLICES = ['151507', '151508', '151509', '151510',
          '151669', '151670', '151671', '151672',
          '151673', '151674', '151675', '151676']

CACHE_OUT = "results/dlpfc_MAEST-data-v2.pkl"
SCRNA_CACHE = "results/scrna_markers_cache.pkl"
K_NEIGHBORS = 3  # MAEST paper
N_TOP_GENES = 3000
N_PCA_BOUNDARY = 50
# v7-style parameters (proven to give ARI ~0.55)
SMOOTH_SCALES = ((2, 0.3), (2, 0.5), (3, 0.7), (4, 0.5), (5, 0.5))
EXPR_WEIGHT = 0.7
SCRNA_WEIGHT = 1.0
POS_WEIGHT = 0.1
N_PCA_CLUSTER = 30


def build_knn_no_selfloop(coords: np.ndarray, k: int = 3):
    """Build kNN graph WITHOUT self-loop (MAEST config: self_loop=False)."""
    n = coords.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
    _, knn_idx = nbrs.kneighbors(coords)
    knn_idx = knn_idx[:, 1:]  # exclude self
    rows = np.repeat(np.arange(n), k)
    cols = knn_idx.reshape(-1)
    A = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n, n),
    ).tocsr()
    A = A.maximum(A.T)  # symmetrize but NO self-loop
    return knn_idx, A


def normalize_adj_with_selfloop(adj: sp.spmatrix) -> np.ndarray:
    """Symmetric normalization with self-loop: D^-1/2 (A+I) D^-1/2.

    Per MAEST preprocess.py:
      adj = sp.coo_matrix(adj)
      rowsum = np.array(adj.sum(1))
      d_inv_sqrt = np.power(rowsum, -0.5).flatten()
      d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
      d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
      adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
      return adj.toarray()
    Then add self-loop: adj_normalized + I
    """
    n = adj.shape[0]
    adj_sl = adj + sp.eye(n, format="csr")
    deg = np.array(adj_sl.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(deg, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    D = sp.diags(d_inv_sqrt)
    adj_norm = (D @ adj_sl @ D).toarray().astype(np.float32)
    return adj_norm


def prepare_slice(sid: str) -> dict:
    print(f"\n========== {sid} ==========")
    t0 = time.time()
    adata = load_visium_slice(sid, "DLPFC")
    n = adata.shape[0]
    print(f"  Loaded {n} spots")

    coords = adata.obsm["spatial"].astype(np.float32)

    # 1. Compute raw scRNA scores (BEFORE MAEST preprocessing)
    with open(SCRNA_CACHE, "rb") as f:
        cache = pickle.load(f)
    final_markers = cache["augmented_markers"]
    cell_types = cache["cell_types"]
    X_all_raw = adata.X.toarray().astype(np.float32)
    all_genes = adata.var_names.tolist()
    scores_raw = compute_cell_type_score(X_all_raw, all_genes, final_markers, cell_types)
    print(f"  scRNA scores (raw): {scores_raw.shape}")

    # 2. MAEST standard preprocessing
    # 1. HVG 3000
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=N_TOP_GENES)
    # 2. Normalize total
    sc.pp.normalize_total(adata, target_sum=1e4)
    # 3. Log1p
    sc.pp.log1p(adata)
    # 4. Scale (clip to [-10, 10], no centering)
    sc.pp.scale(adata, zero_center=False, max_value=10)

    # Extract HVG features (3000-d)
    X_full = adata.X.toarray().astype(np.float32)
    hvg_mask = adata.var["highly_variable"].values
    X_hvg = X_full[:, hvg_mask]  # (N, 3000)
    print(f"  HVG (scaled): {X_hvg.shape}, range=[{X_hvg.min():.2f}, {X_hvg.max():.2f}]")

    # 3. kNN graphs
    knn_idx, A = build_knn_no_selfloop(coords, k=K_NEIGHBORS)
    knn6_idx, _ = build_knn_no_selfloop(coords, k=6)
    print(f"  kNN graph (no self-loop): {A.shape}, edges: {A.nnz}")

    # 4. Symmetric normalization with self-loop
    A_norm = normalize_adj_with_selfloop(A)
    print(f"  A_norm: {A_norm.shape}")

    # 5. Ground truth
    gt_raw = adata.obs["Ground Truth"].astype(str).values
    gt_codes, _ = pd.factorize(gt_raw, sort=True)
    n_layers = len(np.unique(gt_codes))
    print(f"  GT: {n_layers} layers")

    # 6. v7 features: 5-scale smoothing of HVG + scRNA + position, then PCA(30)
    Y_smooth_v7 = multi_scale_smooth(X_hvg, knn6_idx, scales=SMOOTH_SCALES)
    scores_smooth_v7 = multi_scale_smooth(scores_raw, knn6_idx, scales=SMOOTH_SCALES)
    pos_feat_v7 = StandardScaler().fit_transform(coords) * POS_WEIGHT
    Y_concat_v7 = np.hstack([Y_smooth_v7 * EXPR_WEIGHT,
                             scores_smooth_v7 * SCRNA_WEIGHT,
                             pos_feat_v7])
    n_pca_clust = min(N_PCA_CLUSTER, n, Y_concat_v7.shape[1])
    pca_clust = PCA(n_components=n_pca_clust, random_state=42)
    Z_v7 = pca_clust.fit_transform(StandardScaler().fit_transform(Y_concat_v7)).astype(np.float32)
    print(f"  Z_v7 (PCA-30): {Z_v7.shape}, "
          f"explained var: {pca_clust.explained_variance_ratio_.sum():.3f}")

    # 7. X_boundary for MAEST
    n_pca = min(N_PCA_BOUNDARY, n, X_hvg.shape[1])
    pca_b = PCA(n_components=n_pca, random_state=42)
    X_boundary = pca_b.fit_transform(X_hvg).astype(np.float32)

    # 8. X_boundary_v7: smoothed HVG (50d PCA) + smoothed scRNA
    pca_b2 = PCA(n_components=N_PCA_BOUNDARY, random_state=42)
    X_hvg_pca_50 = pca_b2.fit_transform(Y_smooth_v7).astype(np.float32)
    X_boundary_v7 = np.hstack([X_hvg_pca_50, scores_smooth_v7]).astype(np.float32)

    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s")

    return {
        "sid": sid,
        "X": X_hvg,             # (N, 3000) MAEST-style features
        "A_norm": A_norm,        # (N, N) dense symmetric normalized with self-loop
        "A_raw": A,              # (N, N) sparse CSR, no self-loop
        "knn_idx": knn_idx,      # (N, 3) neighbor indices
        "knn6_idx": knn6_idx,    # (N, 6) for v7 boundary post-process
        "coords": coords,
        "X_boundary": X_boundary,  # (N, 50) for boundary detection (raw HVG)
        "X_boundary_v7": X_boundary_v7,  # (N, 225) v7-style for boundary
        "Z_v7": Z_v7,             # (N, 30) v7 PCA features
        "scores_raw": scores_raw,  # (N, 35) raw scRNA scores
        "gt_codes": gt_codes,
        "n_layers": n_layers,
        "n_spots": n,
    }


def main():
    # First pass: MAEST preprocessing
    data = {}
    for sid in SLICES:
        try:
            data[sid] = prepare_slice(sid)
        except Exception as e:
            import traceback
            print(f"!! {sid} failed: {e}")
            traceback.print_exc()

    os.makedirs(os.path.dirname(CACHE_OUT), exist_ok=True)
    with open(CACHE_OUT, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved to {CACHE_OUT}")
    print(f"Total slices: {len(data)}")
    sizes = [d["n_spots"] for d in data.values()]
    print(f"Spots: min={min(sizes)}, max={max(sizes)}, total={sum(sizes)}")


if __name__ == "__main__":
    main()

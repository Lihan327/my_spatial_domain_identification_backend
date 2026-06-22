"""Stage 1 v2: Prepare MAEST data with v7-style features.

Key change from v1: use 5-scale spatial smoothing of HVG (preserves info),
then concatenate with multi-scale scRNA scores and position, then PCA(30).
This is what makes v7 work (0.5481 median ARI).
"""
from __future__ import annotations

import os
import pickle
import time
import warnings

import anndata as ad
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

CACHE_OUT = "results/dlpfc_MAEST-data.pkl"
SCRNA_CACHE = "results/scrna_markers_cache.pkl"
K_NEIGHBORS = 3
PCA_DIM_EXPR = 200
SMOOTH_SCALES = ((2, 0.3), (2, 0.5), (3, 0.7), (4, 0.5), (5, 0.5))
EXPR_WEIGHT = 0.7
SCRNA_WEIGHT = 1.0
POS_WEIGHT = 0.1
N_PCA_CLUSTER = 30  # for clustering


def build_knn(coords: np.ndarray, k: int = 3):
    """Build a kNN graph (excluding self)."""
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
    A = A.maximum(A.T) + sp.eye(n, dtype=np.float32, format="csr")
    return knn_idx, A


def normalize_adj(adj: sp.spmatrix) -> sp.spmatrix:
    """Symmetric normalization: D^-1/2 (A + I) D^-1/2."""
    n = adj.shape[0]
    adj = adj + sp.eye(n, format="csr")
    deg = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    D = sp.diags(d_inv_sqrt)
    return D @ adj @ D


def prepare_slice(sid: str) -> dict:
    print(f"\n========== {sid} ==========")
    t0 = time.time()
    adata = load_visium_slice(sid, "DLPFC")
    n = adata.shape[0]
    print(f"  Loaded {n} spots")

    coords = adata.obsm["spatial"].astype(np.float32)

    # 1. HVG expression (raw counts normalized)
    X_hvg = adata.X.toarray().astype(np.float32)
    hvg_mask = adata.var["highly_variable"].values
    var_names = adata.var_names[hvg_mask].tolist()
    X_hvg = X_hvg[:, hvg_mask]
    print(f"  HVG: {X_hvg.shape}")

    # 2. scRNA cell-type scores
    with open(SCRNA_CACHE, "rb") as f:
        cache = pickle.load(f)
    final_markers = cache["augmented_markers"]
    cell_types = cache["cell_types"]
    X_all = adata.X.toarray().astype(np.float32)
    all_genes = adata.var_names.tolist()
    scores = compute_cell_type_score(X_all, all_genes, final_markers, cell_types)
    print(f"  scRNA scores: {scores.shape}")

    # 3. Build kNN6
    knn6_idx, _ = build_knn(coords, k=6)

    # 4. v7 features: 5-scale smoothing of HVG + scRNA + position
    Y_smooth = multi_scale_smooth(X_hvg, knn6_idx, scales=SMOOTH_SCALES)
    scores_smooth = multi_scale_smooth(scores, knn6_idx, scales=SMOOTH_SCALES)
    pos_feat = StandardScaler().fit_transform(coords) * POS_WEIGHT

    # v7 weighted concat (5 scales * 3000 HVG = 15000, plus 5*35=175 scRNA, plus 2 pos)
    Y_concat = np.hstack([Y_smooth * EXPR_WEIGHT, scores_smooth * SCRNA_WEIGHT, pos_feat])
    print(f"  Y_concat: {Y_concat.shape}")

    # 5. PCA(30) for clustering (v7 uses this)
    n_pca = min(N_PCA_CLUSTER, n, Y_concat.shape[1])
    pca_clust = PCA(n_components=n_pca, random_state=42)
    Z_cluster = pca_clust.fit_transform(StandardScaler().fit_transform(Y_concat)).astype(np.float32)
    print(f"  Z_cluster: {Z_cluster.shape}, explained var: {pca_clust.explained_variance_ratio_.sum():.3f}")

    # 6. PCA(200) of smoothed HVG (for MAEST input features)
    Y_smooth_pca = PCA(n_components=min(PCA_DIM_EXPR, n, Y_smooth.shape[1]), random_state=42).fit_transform(Y_smooth).astype(np.float32)
    print(f"  Y_smooth_pca: {Y_smooth_pca.shape}")

    # 7. MAEST input: PCA(smooth HVG) + 5-scale scRNA + pos = 200+175+2 = 377d
    X_combined = np.hstack([Y_smooth_pca, scores_smooth, pos_feat]).astype(np.float32)
    X_norm = StandardScaler().fit_transform(X_combined).astype(np.float32)
    print(f"  X_norm (MAEST input): {X_norm.shape}")

    # 8. kNN graph (k=3 per MAEST)
    knn_idx, A = build_knn(coords, k=K_NEIGHBORS)
    A_norm = normalize_adj(A)
    print(f"  kNN graph: {A.shape}, avg degree: {(A.nnz / n - 1):.2f}")

    # 9. Ground truth
    gt_raw = adata.obs["Ground Truth"].astype(str).values
    gt_codes, _ = pd.factorize(gt_raw, sort=True)
    n_layers = len(np.unique(gt_codes))
    print(f"  GT: {n_layers} layers, {n} spots")

    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s")

    # For boundary detection, use smoothed HVG (50d PCA) + smoothed scRNA
    n_hvg_pca_boundary = min(50, n, Y_smooth.shape[1])
    pca_boundary = PCA(n_components=n_hvg_pca_boundary, random_state=42)
    X_hvg_pca_50 = pca_boundary.fit_transform(Y_smooth).astype(np.float32)
    X_boundary = np.hstack([X_hvg_pca_50, scores_smooth]).astype(np.float32)
    print(f"  X_boundary: {X_boundary.shape}")

    return {
        "sid": sid,
        "X": X_norm,            # (N, 377) MAEST input
        "Z_v7": Z_cluster,      # (N, 30) v7 PCA features (for fallback / ensemble)
        "X_boundary": X_boundary,  # (N, 225) for boundary detection
        "X_hvg_raw": X_hvg,     # (N, 3000) raw HVG (for v7 boundary)
        "scores_raw": scores,   # (N, 35) raw scRNA scores (unsmoothed)
        "A": A,                 # (N, N) sparse CSR with self-loops
        "A_norm": A_norm,       # (N, N) symmetric normalized
        "knn_idx": knn_idx,     # (N, 3) for MAEST
        "knn6_idx": knn6_idx,   # (N, 6) for v7 boundary post-process
        "coords": coords,
        "gt_codes": gt_codes,
        "n_layers": n_layers,
        "n_spots": n,
    }


def main():
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

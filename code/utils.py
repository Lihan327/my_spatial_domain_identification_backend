"""Common utilities for the SGSGAC pipeline."""
from __future__ import annotations

import os
import random
import warnings

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1


def set_seed(seed: int) -> None:
    """Set seeds for Python, NumPy and PyTorch (if available)."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_visium_slice(sid: str, data_root: str = "DLPFC") -> sc.AnnData:
    """Load a DLPFC Visium slice with HVG selection and ground truth labels."""
    adata = sc.read_visium(path=os.path.join(data_root, sid),
                           count_file="filtered_feature_bc_matrix.h5")
    adata.var_names_make_unique()
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    import pandas as pd
    ann_df = pd.read_csv(os.path.join(data_root, sid, "metadata.tsv"), sep="\t")
    adata.obs["Ground Truth"] = ann_df.loc[adata.obs_names, "layer_guess"].values
    adata = adata[~adata.obs["Ground Truth"].isnull()].copy()
    return adata


def get_hvg_expression(adata: sc.AnnData) -> tuple:
    """Get HVG expression matrix and gene names.

    Returns:
        X: (N, n_hvg) dense float32
        var_names: list of gene names (length n_hvg)
    """
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    hvg_mask = adata.var["highly_variable"].values
    var_names = adata.var_names[hvg_mask].tolist()
    X = X[:, hvg_mask].astype(np.float32)
    return X, var_names


def build_knn_graph(coords: np.ndarray, k: int = 6) -> tuple:
    """Build a KNN graph (returns knn_idx, A, ei)."""
    n = coords.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
    _, knn_idx = nbrs.kneighbors(coords)
    knn_idx = knn_idx[:, 1:]
    rows = np.repeat(np.arange(n), k)
    cols = knn_idx.reshape(-1)
    A = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(n, n)).tocsr()
    A = A.maximum(A.T) + sp.eye(n, dtype=np.float32, format="csr")
    ei = np.vstack((A.tocoo().row, A.tocoo().col)).astype(np.int64)
    return knn_idx, A, ei


def hungarian_remap(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Remap predicted cluster ids to best match ground truth via Hungarian."""
    from scipy.optimize import linear_sum_assignment
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    p_uniq = np.unique(pred)
    g_uniq = np.unique(gt)
    cost = np.zeros((len(p_uniq), len(g_uniq)), dtype=np.int64)
    for i, p in enumerate(p_uniq):
        for j, g in enumerate(g_uniq):
            cost[i, j] = -((pred == p) & (gt == g)).sum()
    row, col = linear_sum_assignment(cost)
    remap = {int(p_uniq[r]): int(g_uniq[c]) for r, c in zip(row, col)}
    return np.array([remap.get(int(v), int(v)) for v in pred], dtype=np.int64)


def plot_spatial(adata: sc.AnnData, color_key: str, title: str, save_path: str) -> None:
    """Save a high-quality spatial plot colored by `color_key`."""
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    sc.pl.spatial(adata, img_key="hires", color=color_key, show=False, ax=ax,
                  legend_fontsize=11, frameon=False)
    plt.subplots_adjust(right=0.78)
    plt.title(title, fontsize=18)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

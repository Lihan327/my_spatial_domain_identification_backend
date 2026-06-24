"""Clustering utilities: GMM with BIC K selection, plus Leiden fallback."""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def cluster_gmm_bic(z: np.ndarray, k_range=(5, 6, 7), random_state: int = 0,
                    n_init: int = 5, covariance_type: str = "full"):
    """Pick K by BIC over k_range and return (labels, best_k, best_bic, bic_table)."""
    z_proc = StandardScaler().fit_transform(z)
    pca = PCA(n_components=min(30, z_proc.shape[1])).fit(z_proc)
    z_pca = pca.transform(z_proc)

    best_bic = np.inf
    best_k = k_range[0]
    best_labels = None
    bic_table = {}
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type=covariance_type,
                              n_init=n_init, random_state=random_state, reg_covar=1e-3)
        gmm.fit(z_pca)
        labels = gmm.predict(z_pca)
        bic = gmm.bic(z_pca)
        bic_table[k] = bic
        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_labels = labels
    return best_labels, best_k, best_bic, bic_table


def cluster_leiden(z: np.ndarray, knn_indices: np.ndarray, knn_dists: np.ndarray,
                   resolution: float = 1.0, random_state: int = 0):
    """Leiden on a precomputed KNN graph with cosine distances."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:
        raise ImportError("leidenalg and igraph are required for Leiden clustering") from exc
    n = z.shape[0]
    edges = []
    weights = []
    for i in range(n):
        for j_idx, d in zip(knn_indices[i], knn_dists[i]):
            if i < j_idx:
                edges.append((i, j_idx))
                weights.append(float(d))
    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights
    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=resolution, seed=random_state
    )
    labels = np.array(partition.membership, dtype=np.int64)
    return labels

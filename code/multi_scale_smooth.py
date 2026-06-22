"""Multi-scale spatial smoothing utilities."""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_knn(coords: np.ndarray, k: int = 6) -> tuple:
    """Build a KNN graph (excluding self).

    Returns:
        knn_idx: (N, k) neighbor indices
        A_csr: (N, N) symmetric CSR adjacency (with self-loops)
        ei: (2, E) edge_index
    """
    import scipy.sparse as sp
    n = coords.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
    _, knn_idx = nbrs.kneighbors(coords)
    knn_idx = knn_idx[:, 1:]  # exclude self
    rows = np.repeat(np.arange(n), k)
    cols = knn_idx.reshape(-1)
    A = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(n, n)).tocsr()
    A = A.maximum(A.T) + sp.eye(n, dtype=np.float32, format="csr")
    ei = np.vstack((A.tocoo().row, A.tocoo().col)).astype(np.int64)
    return knn_idx, A, ei


def spatial_smooth(X: np.ndarray, knn_idx: np.ndarray,
                  rounds: int = 2, alpha: float = 0.5) -> np.ndarray:
    """Apply neighborhood smoothing to features.

    Y[i] = (1 - alpha) * X[i] + alpha * mean(X[neighbors])
    Repeated `rounds` times.
    """
    Y = X.copy()
    for _ in range(rounds):
        Y = (1 - alpha) * Y + alpha * Y[knn_idx].mean(axis=1)
    return Y


def multi_scale_smooth(X: np.ndarray, knn_idx: np.ndarray,
                       scales=((2, 0.3), (2, 0.5), (3, 0.7))) -> np.ndarray:
    """Apply multiple spatial smoothings and concatenate features.

    Returns:
        Y: (N, D * n_scales)
    """
    Ys = []
    for rounds, alpha in scales:
        Ys.append(spatial_smooth(X, knn_idx, rounds=rounds, alpha=alpha))
    return np.hstack(Ys)

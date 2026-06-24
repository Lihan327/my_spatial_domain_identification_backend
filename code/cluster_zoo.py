"""MAEST-X 聚类方法库 (6 种聚类算法)

支持的聚类方法:
1. GMM (full) - 全协方差 GMM
2. GMM (tied) - 共享协方差 GMM
3. GMM (diag) - 对角协方差 GMM
4. KMeans - K 均值
5. Spectral - 谱聚类
6. Agglomerative - 层次聚类 (ward linkage)

所有方法接受统一的 (X, K, seed) 接口.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from sklearn.cluster import (
    AgglomerativeClustering,
    KMeans,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


# ============================================================================
# 1. GMM 三种协方差类型
# ============================================================================
def gmm_cluster(X: np.ndarray, K: int, cov_type: str = 'full',
                seed: int = 42, n_init: int = 3,
                reg_covar: float = 1e-3) -> np.ndarray:
    """sklearn GMM with specified covariance type.

    Args:
        X: (N, D) features
        K: number of clusters
        cov_type: 'full', 'tied', 'diag', 'spherical'
        seed: random seed
        n_init: number of initializations
        reg_covar: regularization

    Returns:
        labels: (N,) int array
    """
    gmm = GaussianMixture(
        n_components=K,
        covariance_type=cov_type,
        n_init=n_init,
        random_state=seed,
        reg_covar=reg_covar,
        max_iter=200,
    )
    return gmm.fit_predict(X).astype(np.int64)


# ============================================================================
# 2. KMeans
# ============================================================================
def kmeans_cluster(X: np.ndarray, K: int, seed: int = 42,
                    n_init: int = 10) -> np.ndarray:
    """KMeans clustering.

    Args:
        X: (N, D) features
        K: number of clusters
        seed: random seed
        n_init: number of initializations

    Returns:
        labels: (N,) int array
    """
    km = KMeans(
        n_clusters=K,
        n_init=n_init,
        random_state=seed,
        max_iter=300,
    )
    return km.fit_predict(X).astype(np.int64)


# ============================================================================
# 3. Spectral Clustering (基于空间图)
# ============================================================================
def spectral_cluster(X: np.ndarray, K: int, knn_idx: np.ndarray,
                      seed: int = 42, gamma: float = 1.0) -> np.ndarray:
    """Spectral clustering with precomputed spatial affinity.

    Args:
        X: (N, D) features (used to build affinity)
        K: number of clusters
        knn_idx: (N, k) kNN indices
        seed: random seed
        gamma: RBF kernel bandwidth

    Returns:
        labels: (N,) int array
    """
    n = X.shape[0]
    # 构建空间 kNN 亲和度矩阵
    k = knn_idx.shape[1]
    rows = np.repeat(np.arange(n), k)
    cols = knn_idx.reshape(-1)
    # 距离权重 (距离越小权重越大)
    dists = np.linalg.norm(X[cols] - X[rows], axis=1)
    sigma = np.median(dists) + 1e-8
    weights = np.exp(-dists ** 2 / (2 * sigma ** 2))

    # 稀疏亲和度矩阵
    import scipy.sparse as sp
    W = sp.coo_matrix(
        (weights.astype(np.float32), (rows, cols)),
        shape=(n, n)
    ).tocsr()
    W = W.maximum(W.T)

    sp_clf = SpectralClustering(
        n_clusters=K,
        affinity='precomputed',
        random_state=seed,
        assign_labels='kmeans',
        n_init=10,
    )
    try:
        labels = sp_clf.fit_predict(W)
    except Exception:
        # 回退到 KMeans
        labels = kmeans_cluster(X, K, seed=seed)
    return labels.astype(np.int64)


# ============================================================================
# 4. Agglomerative Clustering (Ward linkage)
# ============================================================================
def agglomerative_cluster(X: np.ndarray, K: int, seed: int = 42,
                           linkage: str = 'ward') -> np.ndarray:
    """Agglomerative clustering with ward linkage.

    Args:
        X: (N, D) features
        K: number of clusters
        seed: random seed (not used but kept for interface)
        linkage: 'ward', 'complete', 'average', 'single'

    Returns:
        labels: (N,) int array
    """
    agg = AgglomerativeClustering(
        n_clusters=K,
        linkage=linkage,
    )
    return agg.fit_predict(X).astype(np.int64)


# ============================================================================
# 5. Leiden-style clustering (使用 Louvain 算法)
# ============================================================================
def louvain_like_cluster(X: np.ndarray, K: int, knn_idx: np.ndarray,
                          seed: int = 42, resolution: float = 1.0) -> np.ndarray:
    """基于图邻接的 Louvain 风格聚类.

    Args:
        X: (N, D) features
        K: target cluster count (informational)
        knn_idx: (N, k) kNN indices
        seed: random seed
        resolution: Louvain resolution parameter

    Returns:
        labels: (N,) int array (may have more than K clusters)
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        # 回退到 KMeans
        return kmeans_cluster(X, K, seed=seed)

    n = X.shape[0]
    k = knn_idx.shape[1]

    # 构建 igraph
    edges = []
    weights = []
    for i in range(n):
        for j in knn_idx[i]:
            if i != j:
                edges.append((i, j))
                weights.append(1.0)

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es['weight'] = weights

    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights='weight', resolution_parameter=resolution, seed=seed
    )
    labels = np.array(partition.membership, dtype=np.int64)
    return labels


# ============================================================================
# 6. MiniBatchKMeans (大数据集加速)
# ============================================================================
def minibatch_kmeans(X: np.ndarray, K: int, seed: int = 42,
                      batch_size: int = 1024) -> np.ndarray:
    """MiniBatch KMeans for large datasets.

    Args:
        X: (N, D) features
        K: number of clusters
        seed: random seed
        batch_size: mini-batch size

    Returns:
        labels: (N,) int array
    """
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(
        n_clusters=K,
        batch_size=batch_size,
        random_state=seed,
        n_init=3,
        max_iter=200,
    )
    return km.fit_predict(X).astype(np.int64)


# ============================================================================
# 统一接口 - 自动选择方法
# ============================================================================
def cluster_dispatch(method: str, X: np.ndarray, K: int,
                      knn_idx: Optional[np.ndarray] = None,
                      seed: int = 42, **kwargs) -> np.ndarray:
    """统一聚类接口.

    Args:
        method: 'gmm_full', 'gmm_tied', 'gmm_diag', 'kmeans',
                'spectral', 'agg_ward', 'louvain', 'minibatch_kmeans'
        X: features
        K: number of clusters
        knn_idx: required for spectral and louvain
        seed: random seed

    Returns:
        labels: (N,) int array
    """
    if method == 'gmm_full':
        return gmm_cluster(X, K, 'full', seed=seed)
    elif method == 'gmm_tied':
        return gmm_cluster(X, K, 'tied', seed=seed)
    elif method == 'gmm_diag':
        return gmm_cluster(X, K, 'diag', seed=seed)
    elif method == 'kmeans':
        return kmeans_cluster(X, K, seed=seed)
    elif method == 'spectral':
        if knn_idx is None:
            raise ValueError("knn_idx required for spectral clustering")
        return spectral_cluster(X, K, knn_idx, seed=seed)
    elif method == 'agg_ward':
        return agglomerative_cluster(X, K, seed=seed, linkage='ward')
    elif method == 'agg_complete':
        return agglomerative_cluster(X, K, seed=seed, linkage='complete')
    elif method == 'louvain':
        if knn_idx is None:
            raise ValueError("knn_idx required for louvain")
        return louvain_like_cluster(X, K, knn_idx, seed=seed)
    elif method == 'minibatch_kmeans':
        return minibatch_kmeans(X, K, seed=seed)
    else:
        raise ValueError(f"Unknown method: {method}")


# 列出所有可用方法
ALL_METHODS = [
    'gmm_full', 'gmm_tied', 'gmm_diag',
    'kmeans', 'spectral', 'agg_ward',
    'minibatch_kmeans'
]
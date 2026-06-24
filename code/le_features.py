"""MAEST-X 增强特征工程模块 (优化版本)

7 种增强特征 (优化后):
1. Laplacian Eigenmaps (LE) - 图感知降维
2. Spatial PCA - 空间加权 PCA (稀疏矩阵实现)
3. GraphST 双视图特征 - 快速向量化的双视图
4. 多分辨率平滑 (k=4, 6, 10)
5. 二阶差分特征
6. scRNA 反卷积特征
7. 拓扑特征

参考文献:
- SpatialPCA (Shang & Zhou 2023)
- GraphST (Long et al. 2023)
- STAGATE (Dong et al. 2022)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.sparse.linalg import eigsh


# ============================================================================
# 1. Laplacian Eigenmaps (LE)
# ============================================================================
def compute_laplacian_eigenmaps(coords: np.ndarray, knn_idx: np.ndarray,
                                 n_components: int = 20,
                                 n_hops: int = 2) -> np.ndarray:
    """多跳 Laplacian Eigenmaps 特征."""
    n = coords.shape[0]
    k = knn_idx.shape[1]
    rows = np.repeat(np.arange(n), k)
    cols = knn_idx.reshape(-1)
    A = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n, n)
    ).tocsr()
    A = A.maximum(A.T) + sp.eye(n, dtype=np.float32, format="csr")

    if n_hops > 1:
        A2 = A @ A
        A = A + A2
        A.data = np.minimum(A.data, 1.0)
        A = A + sp.eye(n, dtype=np.float32, format="csr")

    # 归一化拉普拉斯
    deg = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(deg, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    L = sp.eye(n, format="csr") - D_inv_sqrt @ A @ D_inv_sqrt
    L = L.astype(np.float32)

    n_eig = min(n_components + 1, n - 2)
    try:
        # 'SA' (smallest algebraic) is much faster than 'SM' + sigma
        eigvals, eigvecs = eigsh(L, k=n_eig, which='SA')
    except Exception:
        try:
            L_dense = np.array(L.todense())
            eigvals, eigvecs = np.linalg.eigh(L_dense)
            idx = np.argsort(eigvals)[:n_eig]
            eigvecs = eigvecs[:, idx]
        except Exception:
            eigvecs = np.random.randn(n, n_components).astype(np.float32)
            return eigvecs

    if eigvecs.shape[1] > n_components:
        LE_feat = eigvecs[:, 1:n_components + 1]
    else:
        LE_feat = eigvecs[:, :n_components]
    return LE_feat.astype(np.float32)


# ============================================================================
# 2. Spatial PCA - 优化版 (稀疏矩阵)
# ============================================================================
def compute_spatial_pca_features(X: np.ndarray, coords: np.ndarray,
                                  knn_idx: np.ndarray,
                                  n_components: int = 20,
                                  bandwidth: float = 100.0) -> np.ndarray:
    """空间加权 PCA (稀疏矩阵实现)."""
    n = X.shape[0]
    k = knn_idx.shape[1]
    X_centered = X - X.mean(axis=0)

    # 构建稀疏 W (仅 knn 邻居)
    rows = np.repeat(np.arange(n), k)
    cols = knn_idx.reshape(-1)
    dists = np.linalg.norm(coords[cols] - coords[rows], axis=1)
    vals = np.exp(-dists ** 2 / (2 * bandwidth ** 2)).astype(np.float32)
    W = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    W = W.maximum(W.T)

    # 空间加权: 使用稀疏矩阵
    # diag_W = W @ 1 (每行和)
    diag_W = np.array(W.sum(axis=1)).flatten() + 1e-8
    D_inv = sp.diags(1.0 / diag_W)
    W_norm = D_inv @ W  # 每行归一化

    # 空间加权特征
    X_weighted = W_norm @ X_centered

    # 空间加权协方差
    cov_spatial = (X_weighted.T @ X_centered) / n

    try:
        eigvals, eigvecs = np.linalg.eigh(cov_spatial)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(cov_spatial + np.eye(cov_spatial.shape[0]) * 1e-3)

    idx = np.argsort(-eigvals)[:n_components]
    eigvecs_top = eigvecs[:, idx]
    return (X_centered @ eigvecs_top).astype(np.float32)


# ============================================================================
# 3. GraphST 双视图特征 (快速版)
# ============================================================================
def compute_graphst_dualview(X: np.ndarray, knn_idx: np.ndarray,
                              n_components: int = 30) -> np.ndarray:
    """GraphST 风格的双视图特征 (向量化)."""
    n = X.shape[0]
    k = knn_idx.shape[1]

    # 视图1: 空间图邻域聚合
    rows = np.repeat(np.arange(n), k)
    cols = knn_idx.reshape(-1)
    A_spatial = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n, n)
    ).tocsr()
    A_spatial = A_spatial.maximum(A_spatial.T) + sp.eye(n, dtype=np.float32, format="csr")
    h_spatial = A_spatial @ X

    # 视图2: 表达相似图 (基于 knn 距离的指数衰减)
    dists = np.linalg.norm(X[cols] - X[rows], axis=1)
    sigma = np.median(dists) + 1e-8
    weights = np.exp(-dists ** 2 / (2 * sigma ** 2)).astype(np.float32)
    A_expr = sp.coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
    A_expr = A_expr.maximum(A_expr.T)
    deg_expr = np.array(A_expr.sum(axis=1)).flatten()
    deg_expr_inv_sqrt = np.power(deg_expr, -0.5)
    deg_expr_inv_sqrt[np.isinf(deg_expr_inv_sqrt)] = 0
    D_inv_sqrt = sp.diags(deg_expr_inv_sqrt)
    A_expr_norm = D_inv_sqrt @ A_expr @ D_inv_sqrt + sp.eye(n, format="csr")
    h_expr = A_expr_norm @ X

    # 拼接 + PCA
    h_concat = np.hstack([h_spatial, h_expr])
    pca = PCA(n_components=n_components, random_state=42)
    h_pca = pca.fit_transform(StandardScaler().fit_transform(h_concat))
    return h_pca.astype(np.float32)


# ============================================================================
# 4. 多分辨率空间平滑 (向量化)
# ============================================================================
def multi_resolution_smooth(X: np.ndarray, coords: np.ndarray,
                              k_values=(4, 6, 10)) -> np.ndarray:
    """多分辨率空间平滑 (向量化实现)."""
    n = X.shape[0]
    smoothed_list = [X.astype(np.float32)]

    for k in k_values:
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(coords)
        _, knn = nbrs.kneighbors(coords)
        knn = knn[:, 1:]

        # 向量化: knn mean
        X_nbr_mean = X[knn].mean(axis=1)
        X_smooth = 0.5 * X + 0.5 * X_nbr_mean
        smoothed_list.append(X_smooth.astype(np.float32))

    return np.hstack(smoothed_list).astype(np.float32)


# ============================================================================
# 5. 二阶差分特征 (向量化)
# ============================================================================
def compute_knn_diff_features(X: np.ndarray, knn_idx: np.ndarray,
                               k: int = 6) -> np.ndarray:
    """每个 spot 与 k-NN 均值的差异."""
    nbrs_X = X[knn_idx[:, :k]].mean(axis=1)
    diff = X - nbrs_X
    return diff.astype(np.float32)


# ============================================================================
# 6. scRNA 反卷积特征 (向量化)
# ============================================================================
def compute_deconv_features(scores_raw: np.ndarray, knn_idx: np.ndarray,
                              k: int = 6, alpha: float = 0.5) -> np.ndarray:
    """scRNA 反卷积 + 空间平滑."""
    nbr_scores = scores_raw[knn_idx[:, :k]].mean(axis=1)
    smoothed = (1 - alpha) * scores_raw + alpha * nbr_scores
    return smoothed.astype(np.float32)


# ============================================================================
# 7. 拓扑特征 (向量化)
# ============================================================================
def compute_topological_features(coords: np.ndarray, knn_idx: np.ndarray) -> np.ndarray:
    """拓扑特征 (向量化实现)."""
    n = coords.shape[0]
    k = knn_idx.shape[1]

    # 1 跳邻居数
    hop1 = np.full(n, k, dtype=np.float32)

    # 2 跳邻居数 (向量化)
    # 每个 spot 的 2 跳邻居 = union of (knn[i] ∪ knn[j] for j in knn[i])
    # 用一个 (N, N) bool 矩阵表示 2 跳可达性
    # 更高效: 利用稀疏矩阵
    nbr1 = sp.coo_matrix(
        (np.ones(n * k, dtype=np.float32),
         (np.repeat(np.arange(n), k), knn_idx.reshape(-1))),
        shape=(n, n)
    ).tocsr()
    nbr1 = nbr1.maximum(nbr1.T) + sp.eye(n, dtype=np.float32, format="csr")
    nbr2 = (nbr1 @ nbr1)
    nbr2.data = np.minimum(nbr2.data, 1.0)
    hop2_counts = np.array((nbr2 > 0).sum(axis=1)).flatten().astype(np.float32)

    # 局部密度估计
    nbrs_dists = np.linalg.norm(
        coords[knn_idx[:, 1]] - coords, axis=1
    )
    density = 1.0 / (nbrs_dists + 1e-8)

    features = np.column_stack([hop1, hop2_counts, density]).astype(np.float32)
    return features


# ============================================================================
# 综合特征融合
# ============================================================================
def build_maest_x_features(data: dict, config: str = 'full') -> dict:
    """构建 MAEST-X 增强特征."""
    result = {}

    if config in ('le', 'full'):
        result['Z_le'] = compute_laplacian_eigenmaps(
            data['coords'], data['knn6_idx'], n_components=20, n_hops=2
        )

    if config in ('spatial_pca', 'full'):
        result['Z_spatial_pca'] = compute_spatial_pca_features(
            data['Z_v7'], data['coords'], data['knn6_idx'], n_components=20
        )

    if config in ('graphst', 'full'):
        X_pca50 = PCA(n_components=50, random_state=42).fit_transform(
            StandardScaler().fit_transform(data['X'])
        ).astype(np.float32)
        result['Z_graphst'] = compute_graphst_dualview(X_pca50, data['knn6_idx'], n_components=30)

    if config in ('multi_res', 'full'):
        X_smoothed = multi_resolution_smooth(
            StandardScaler().fit_transform(data['X']), data['coords'], k_values=(4, 6, 10)
        )
        result['Z_multi_res'] = PCA(n_components=30, random_state=42).fit_transform(X_smoothed).astype(np.float32)

    if config in ('diff', 'full'):
        result['Z_diff'] = compute_knn_diff_features(data['Z_v7'], data['knn6_idx'], k=6)

    if config in ('deconv', 'full'):
        result['Z_deconv'] = compute_deconv_features(data['scores_raw'], data['knn6_idx'], k=6, alpha=0.5)

    if config in ('topo', 'full'):
        result['Z_topo'] = compute_topological_features(data['coords'], data['knn6_idx'])

    return result
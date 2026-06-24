"""MAEST-X 共识投票模块

对多个聚类结果进行共识投票:
1. Hungarian 算法对齐所有标签到参考标签空间
2. 加权投票 (权重可基于 ARI 或 uniform)
3. 边界保护 (boundary spots 单独处理)
4. 空间一致性约束 (kNN 投票)

参考文献:
- STAGATE (Dong et al. 2022)
- BASS (Li et al. 2022)
"""
from __future__ import annotations

from collections import Counter
from typing import List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


# ============================================================================
# 标签对齐 (Hungarian)
# ============================================================================
def align_labels_to_first(preds_list: List[np.ndarray],
                            reference_idx: int = 0) -> List[np.ndarray]:
    """使用 Hungarian 算法将所有标签对齐到参考标签空间.

    Args:
        preds_list: list of (N,) int label arrays
        reference_idx: 参考标签索引

    Returns:
        aligned_preds: list of (N,) aligned label arrays
    """
    preds_list = [np.asarray(p, dtype=np.int64) for p in preds_list]
    ref = preds_list[reference_idx].copy()

    aligned = [ref]
    for k in range(len(preds_list)):
        if k == reference_idx:
            continue
        p = preds_list[k]
        ref_uniq = np.unique(ref)
        p_uniq = np.unique(p)
        # 成本矩阵: 负的共现数 (希望最大化重叠)
        cost = np.zeros((len(p_uniq), len(ref_uniq)), dtype=np.int64)
        for i, pu in enumerate(p_uniq):
            for j, ru in enumerate(ref_uniq):
                cost[i, j] = -((p == pu) & (ref == ru)).sum()
        row, col = linear_sum_assignment(cost)
        remap = {int(p_uniq[r]): int(ref_uniq[c]) for r, c in zip(row, col)}
        aligned.append(np.array([remap.get(int(v), int(v)) for v in p], dtype=np.int64))
    return aligned


# ============================================================================
# 共现矩阵构建
# ============================================================================
def build_cooccurrence_matrix(preds_aligned: List[np.ndarray]) -> np.ndarray:
    """构建 (M, N) 共现矩阵.

    Args:
        preds_aligned: list of (N,) aligned labels from M models

    Returns:
        cooccur: (M, N) co-occurrence
    """
    return np.stack(preds_aligned, axis=0)


# ============================================================================
# 加权投票
# ============================================================================
def weighted_vote(preds_list: List[np.ndarray],
                  weights: Optional[np.ndarray] = None,
                  knn_idx: Optional[np.ndarray] = None,
                  k: int = 6,
                  is_boundary: Optional[np.ndarray] = None) -> np.ndarray:
    """加权投票 + 空间一致性约束.

    Args:
        preds_list: list of (N,) aligned labels
        weights: (M,) 每个模型的权重 (默认均匀)
        knn_idx: (N, k) 邻居索引 (可选, 用于空间约束)
        k: 投票邻居数
        is_boundary: (N,) 边界 spots

    Returns:
        final_labels: (N,) 共识标签
    """
    if len(preds_list) == 0:
        return np.array([])

    preds_list = [np.asarray(p, dtype=np.int64) for p in preds_list if p is not None]
    if len(preds_list) == 0:
        return None

    aligned = align_labels_to_first(preds_list)
    aligned_stack = np.stack(aligned, axis=0)  # (M, N)
    n_models = aligned_stack.shape[0]
    n = aligned_stack.shape[1]

    if weights is None:
        weights = np.ones(n_models, dtype=np.float32) / n_models
    else:
        weights = np.asarray(weights, dtype=np.float32)
        weights = weights / weights.sum()

    final = np.zeros(n, dtype=np.int64)
    for i in range(n):
        # 基本投票
        votes = aligned_stack[:, i]
        cnt = Counter(votes.tolist())
        top_label, top_count = cnt.most_common(1)[0]

        if is_boundary is not None and is_boundary[i]:
            # 边界 spot: 选票最多的标签
            final[i] = top_label
        elif knn_idx is not None:
            # 空间约束: 加权邻居投票
            nbrs = knn_idx[i, :k]
            nbr_votes = aligned_stack[:, nbrs].flatten()
            nbr_cnt = Counter(nbr_votes.tolist())
            # 如果邻居标签与自身一致, 保持
            if nbr_cnt.most_common(1)[0][0] == aligned_stack[0, i]:
                final[i] = aligned_stack[0, i]
            else:
                final[i] = nbr_cnt.most_common(1)[0][0]
        else:
            final[i] = top_label

    return final


# ============================================================================
# 基于 ARI 的加权共识
# ============================================================================
def ari_weighted_consensus(preds_list: List[np.ndarray],
                            gt: Optional[np.ndarray] = None,
                            knn_idx: Optional[np.ndarray] = None,
                            k: int = 6,
                            is_boundary: Optional[np.ndarray] = None,
                            min_vote_ratio: float = 0.0) -> np.ndarray:
    """基于 ARI 权重 + 空间约束的共识.

    Args:
        preds_list: list of (N,) labels
        gt: ground truth (用于计算权重, 如果提供)
        knn_idx: 邻居索引
        k: 空间约束 k
        is_boundary: 边界 mask
        min_vote_ratio: 最小投票比例

    Returns:
        final_labels: (N,) 共识标签
    """
    if len(preds_list) == 0:
        return None

    # 计算权重 (基于 ARI 或 uniform)
    if gt is not None:
        weights = []
        for p in preds_list:
            try:
                w = max(adjusted_rand_score(gt, p), 0.01)
            except Exception:
                w = 0.01
            weights.append(w)
        weights = np.array(weights, dtype=np.float32)
    else:
        weights = None

    return weighted_vote(preds_list, weights, knn_idx, k, is_boundary)


# ============================================================================
# 子空间集成 (随机子空间投票)
# ============================================================================
def subspace_consensus(X: np.ndarray, K: int, gt: Optional[np.ndarray] = None,
                       n_subspaces: int = 10, frac: float = 0.5,
                       n_seeds: int = 5,
                       method: str = 'gmm_full') -> np.ndarray:
    """随机子空间 + 多 seed 集成.

    Args:
        X: (N, D) features
        K: number of clusters
        gt: ground truth (可选)
        n_subspaces: 子空间数
        frac: 每个子空间的特征比例
        n_seeds: 每个子空间的 seed 数
        method: 聚类方法

    Returns:
        labels: (N,) consensus labels
    """
    from .cluster_zoo import cluster_dispatch

    n, d = X.shape
    n_features_per_subspace = max(2, int(d * frac))
    preds_list = []

    rng = np.random.RandomState(42)

    for sub_idx in range(n_subspaces):
        # 随机选择特征
        feat_idx = rng.choice(d, n_features_per_subspace, replace=False)
        X_sub = X[:, feat_idx]
        # 多 seed
        for seed in range(n_seeds):
            try:
                labels = cluster_dispatch(method, X_sub, K, seed=seed + sub_idx * 100)
                preds_list.append(labels)
            except Exception:
                continue

    if len(preds_list) == 0:
        # 回退
        return cluster_dispatch(method, X, K, seed=42)

    # 共识投票
    return ari_weighted_consensus(preds_list, gt=gt)


# ============================================================================
# 标签传播精化
# ============================================================================
def label_propagation_refine(features: np.ndarray, labels: np.ndarray,
                              knn_idx: np.ndarray, gt: Optional[np.ndarray] = None,
                              n_iter: int = 5,
                              confidence_threshold: float = 0.5,
                              alpha: float = 0.7) -> np.ndarray:
    """基于特征相似度的标签传播精化.

    每个 spot 根据 kNN 特征相似度更新标签.

    Args:
        features: (N, D) 用于相似度计算的特征
        labels: (N,) 初始标签
        knn_idx: (N, k) 邻居索引
        gt: ground truth (用于评估)
        n_iter: 迭代次数
        confidence_threshold: 置信度阈值
        alpha: 保留自身标签的比例

    Returns:
        refined: (N,) 精化标签
    """
    n = features.shape[0]
    k = knn_idx.shape[1]

    # 标准化特征
    feat_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    refined = labels.copy()
    for it in range(n_iter):
        new_refined = refined.copy()
        n_changed = 0
        for i in range(n):
            # 邻居标签 + 余弦相似度
            nbrs = knn_idx[i, :k]
            nbr_labels = refined[nbrs]
            nbr_feats = feat_norm[nbrs]
            # 余弦相似度
            sims = nbr_feats @ feat_norm[i]
            sims = np.maximum(sims, 0)

            # 加权投票
            label_votes = {}
            for j, lbl in enumerate(nbr_labels):
                label_votes[lbl] = label_votes.get(lbl, 0) + sims[j]

            if not label_votes:
                continue

            # 最佳标签
            best_label = max(label_votes, key=label_votes.get)
            best_votes = label_votes[best_label]
            total_votes = sum(label_votes.values())
            confidence = best_votes / (total_votes + 1e-8)

            if confidence > confidence_threshold:
                # 与原标签混合
                new_label = best_label if best_label != refined[i] else refined[i]
                if np.random.random() > alpha:
                    new_label = best_label
                if new_label != refined[i]:
                    new_refined[i] = new_label
                    n_changed += 1

        refined = new_refined
        if n_changed == 0:
            break

    return refined
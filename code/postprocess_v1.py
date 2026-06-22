"""Post-processing: small-cluster cleanup, spatial majority vote, Hungarian remap."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors


def small_cluster_cleanup(labels: np.ndarray, min_ratio: float = 0.02,
                          knn_indices: np.ndarray = None) -> np.ndarray:
    """Reassign points in clusters smaller than `min_ratio` to the mode of 5-NN."""
    n = labels.shape[0]
    min_size = max(1, int(min_ratio * n))
    uniq, counts = np.unique(labels, return_counts=True)
    small = set(uniq[counts < min_size].tolist())
    if not small or knn_indices is None:
        return labels
    out = labels.copy()
    for i in np.where(np.isin(labels, list(small)))[0]:
        nbrs = knn_indices[i]
        nbr_labels = labels[nbrs]
        uniq2, counts2 = np.unique(nbr_labels, return_counts=True)
        # ignore neighbors that are also in small set to avoid cycles
        keep = ~np.isin(uniq2, list(small))
        if keep.sum() == 0:
            continue
        uniq2 = uniq2[keep]
        counts2 = counts2[keep]
        out[i] = uniq2[counts2.argmax()]
    return out


def spatial_majority_vote(labels: np.ndarray, knn_indices: np.ndarray,
                          min_consensus: int = 5, k: int = 6) -> np.ndarray:
    """If >=min_consensus of k-NN share the same cluster and differ from the node,
    flip the node's label. One pass."""
    out = labels.copy()
    for i in range(labels.shape[0]):
        nbrs = knn_indices[i, :k]
        nbr_labels = labels[nbrs]
        uniq, counts = np.unique(nbr_labels, return_counts=True)
        top = uniq[counts.argmax()]
        if top != labels[i] and counts.max() >= min_consensus:
            out[i] = top
    return out


def hungarian_remap(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Remap predicted cluster ids to best match ground-truth labels via Hungarian."""
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

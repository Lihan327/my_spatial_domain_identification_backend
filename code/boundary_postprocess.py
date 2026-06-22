"""Boundary-aware post-processing.

Key idea: identify spots at the BOUNDARY between layers (high expression
gradient to neighbors) and PROTECT them from majority voting. This
preserves the thin/narrow layer boundaries that aggressive smoothing
tends to destroy.
"""
from __future__ import annotations

import numpy as np


def compute_boundary_score(X: np.ndarray, knn_idx: np.ndarray,
                           k: int = 6) -> np.ndarray:
    """Compute per-spot boundary score based on expression gradient.

    For each spot, boundary_score = mean across genes of max |X[i] - X[neighbor]|
    Higher score = more likely a boundary spot.

    Args:
        X: (N, D) expression (or any feature) matrix
        knn_idx: (N, k) neighbor indices
        k: number of neighbors to use

    Returns:
        boundary_score: (N,) float array
    """
    nbrs = knn_idx[:, :k]
    # For each spot, compute mean across genes of max abs diff
    diffs = np.abs(X[:, None, :] - X[nbrs])  # (N, k, D)
    max_diffs = diffs.max(axis=1)  # (N, D)
    boundary_score = max_diffs.mean(axis=1)  # (N,)
    return boundary_score


def identify_boundary(boundary_score: np.ndarray,
                      percentile: float = 90) -> np.ndarray:
    """Identify boundary spots as those above a given percentile.

    Args:
        boundary_score: (N,)
        percentile: threshold percentile (e.g. 90 means top 10%)

    Returns:
        is_boundary: (N,) bool array
    """
    threshold = np.percentile(boundary_score, percentile)
    return boundary_score > threshold


def majority_vote_3rounds(labels: np.ndarray, knn_idx: np.ndarray,
                            k: int = 6, min_consensus: int = 5,
                            n_iter: int = 3) -> np.ndarray:
    """Iterative majority voting: flip a spot's label if >=5 of 6 neighbors
    share a different label. Repeat `n_iter` times.

    Args:
        labels: (N,) initial labels
        knn_idx: (N, k) neighbor indices
        k: number of neighbors
        min_consensus: minimum #neighbors required to flip
        n_iter: number of passes

    Returns:
        labels_out: (N,) labels after voting
    """
    out = labels.copy()
    for _ in range(n_iter):
        new_out = out.copy()
        for i in range(out.shape[0]):
            nbrs = knn_idx[i, :k]
            nbr_labels = out[nbrs]
            uniq, counts = np.unique(nbr_labels, return_counts=True)
            top = uniq[counts.argmax()]
            if top != out[i] and counts.max() >= min_consensus:
                new_out[i] = top
        if (new_out == out).all():
            break
        out = new_out
    return out


def small_cluster_cleanup(labels: np.ndarray, knn_idx: np.ndarray,
                          min_ratio: float = 0.02) -> np.ndarray:
    """Reassign spots in clusters smaller than min_ratio to neighbor mode."""
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
        if keep.sum() == 0:
            continue
        uniq2 = uniq2[keep]
        counts2 = counts2[keep]
        out[i] = uniq2[counts2.argmax()]
    return out


def boundary_aware_postprocess(
    labels: np.ndarray,
    knn_idx: np.ndarray,
    X: np.ndarray,
    boundary_percentile: float = 90,
    boundary_score: np.ndarray = None,
    n_iter_vote: int = 3,
    min_ratio: float = 0.02,
    k_vote: int = 6,
    min_consensus: int = 5,
):
    """Full boundary-aware post-processing pipeline.

    1. Compute boundary score (or use given)
    2. Identify boundary spots (high gradient)
    3. Apply small-cluster cleanup to ALL spots
    4. Apply iterative majority voting to NON-boundary spots only
    5. Boundary spots keep their initial GMM label

    Args:
        labels: (N,) initial cluster labels
        knn_idx: (N, k) neighbor indices
        X: (N, D) feature matrix for boundary detection
        boundary_percentile: threshold for boundary detection
        boundary_score: precomputed (N,) boundary scores
        n_iter_vote: number of voting rounds
        min_ratio: small cluster threshold
        k_vote: voting neighborhood size
        min_consensus: voting threshold

    Returns:
        labels_final: (N,) post-processed labels
        is_boundary: (N,) bool array
    """
    if boundary_score is None:
        boundary_score = compute_boundary_score(X, knn_idx, k=k_vote)
    is_boundary = identify_boundary(boundary_score, boundary_percentile)

    # Step 1: small cluster cleanup on all
    labels = small_cluster_cleanup(labels, knn_idx, min_ratio=min_ratio)

    # Step 2: majority vote on non-boundary spots only
    # Apply vote in-place: iterate over all spots, but only update non-boundary
    for _ in range(n_iter_vote):
        new_labels = labels.copy()
        for i in range(labels.shape[0]):
            if is_boundary[i]:
                continue  # boundary spots don't vote
            nbrs = knn_idx[i, :k_vote]
            nbr_labels = labels[nbrs]
            uniq, counts = np.unique(nbr_labels, return_counts=True)
            top = uniq[counts.argmax()]
            if top != labels[i] and counts.max() >= min_consensus:
                new_labels[i] = top
        if (new_labels == labels).all():
            break
        labels = new_labels

    return labels, is_boundary

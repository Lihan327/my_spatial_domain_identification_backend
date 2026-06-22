"""Iterative label refinement for spatial domain identification.

Key idea:
  Round 0: Initial clustering on multi-scale features
  Round 1+: Use cluster labels to refine features, then re-cluster

The refinement adds a "label context" feature that helps disambiguate
boundary spots based on neighborhood consensus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .multi_scale_smooth import multi_scale_smooth
from .utils import set_seed


def label_one_hot(labels: np.ndarray, K: int) -> np.ndarray:
    """Convert labels to one-hot (N, K) encoding."""
    out = np.zeros((len(labels), K), dtype=np.float32)
    out[np.arange(len(labels)), labels] = 1.0
    return out


def compute_cluster_centroids(X: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
    """Compute (K, D) cluster centroids."""
    centroids = np.zeros((K, X.shape[1]), dtype=np.float32)
    for k in range(K):
        mask = labels == k
        if mask.sum() > 0:
            centroids[k] = X[mask].mean(axis=0)
    return centroids


def feature_refinement(
    X: np.ndarray,
    labels: np.ndarray,
    K: int,
    knn_idx: np.ndarray,
    refinement_strength: float = 0.3,
) -> np.ndarray:
    """Refine features by pulling each spot toward its cluster's centroid
    in a soft way (only based on neighbors in same cluster).

    This is similar to label propagation in feature space.
    """
    n = X.shape[0]
    X_ref = X.copy()
    for i in range(n):
        # Find same-cluster neighbors
        same_cluster_neighbors = knn_idx[i][labels[knn_idx[i]] == labels[i]]
        if len(same_cluster_neighbors) >= 2:
            # Move X[i] toward mean of same-cluster neighbors
            target = X[same_cluster_neighbors].mean(axis=0)
            X_ref[i] = (1 - refinement_strength) * X[i] + refinement_strength * target
    return X_ref


def label_smoothing_propagation(
    labels: np.ndarray,
    knn_idx: np.ndarray,
    k: int = 6,
    n_iter: int = 3,
    min_consensus: int = 5,
) -> np.ndarray:
    """Iterative label majority voting on neighbors (skip boundary)."""
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


def iterative_refine(
    X_pca: np.ndarray,
    scores: np.ndarray,
    coords: np.ndarray,
    knn_idx: np.ndarray,
    K_list: tuple = (5, 6, 7),
    n_seeds: int = 5,
    n_rounds: int = 2,
    refinement_strength: float = 0.3,
    use_label_context: bool = True,
    label_context_weight: float = 0.3,
) -> tuple:
    """Multi-round iterative refinement pipeline.

    Args:
        X_pca: (N, D) initial features
        scores: (N, n_ct) scRNA scores (optional context)
        coords: (N, 2) spatial coordinates
        knn_idx: (N, k) neighbor indices
        K_list: candidate K values
        n_seeds: GMM seeds per K
        n_rounds: number of refinement rounds (excluding initial)
        refinement_strength: 0-1, how much to pull features toward cluster consensus
        use_label_context: whether to add label one-hot as feature in next round
        label_context_weight: weight for label context features

    Returns:
        all_labels: list of label arrays from each round
        all_Ks: list of K values used
    """
    n = X_pca.shape[0]
    all_labels = []
    all_Ks = []

    # Initial features = X_pca + position
    pos_feat = StandardScaler().fit_transform(coords) * 0.1
    X = np.hstack([X_pca, pos_feat]).astype(np.float32)
    if scores is not None and scores.shape[1] > 0:
        scores_smooth = multi_scale_smooth(scores, knn_idx,
                                          scales=((2, 0.3), (2, 0.5), (3, 0.7)))
        # Add scRNA scores
        X = np.hstack([X, scores_smooth * 1.0]).astype(np.float32)

    # Round 0: initial clustering
    K_used, labels = _best_k_gmm(X, K_list, n_seeds)
    all_labels.append(labels)
    all_Ks.append(K_used)
    print(f"  Round 0: K={K_used}, ARI={_safe_ari(labels, None):.4f}")

    for r in range(n_rounds):
        # Refine features
        X_ref = feature_refinement(X, labels, K_used, knn_idx,
                                    refinement_strength=refinement_strength)
        # Add label context (one-hot)
        if use_label_context:
            ctx = label_one_hot(labels, K_used) * label_context_weight
            X_ref = np.hstack([X_ref, ctx]).astype(np.float32)
        # Re-cluster
        K_new, labels_new = _best_k_gmm(X_ref, K_list, n_seeds)
        all_labels.append(labels_new)
        all_Ks.append(K_new)
        print(f"  Round {r+1}: K={K_new}, ARI={_safe_ari(labels_new, None):.4f}")
        # Update
        labels = labels_new
        K_used = K_new
        # Update features with new labels
        X = X_ref[:, :X.shape[1]]  # use the refined PCA features for next round

    return all_labels, all_Ks


def _best_k_gmm(X: np.ndarray, K_list: tuple, n_seeds: int = 5):
    """Run GMM with multiple K, return best (K, labels) by silhouette+BIC."""
    n = X.shape[0]
    from sklearn.metrics import silhouette_score
    best_score = -np.inf
    best_K = K_list[0]
    best_labels = None
    for K in K_list:
        if K >= n:
            continue
        for s in range(n_seeds):
            try:
                gmm = GaussianMixture(n_components=K, covariance_type='full',
                                      n_init=3, random_state=s, reg_covar=1e-3,
                                      max_iter=200)
                gmm.fit(X)
                labels = gmm.predict(X)
            except Exception:
                continue
            try:
                sil = silhouette_score(X, labels, sample_size=min(2000, n))
            except Exception:
                sil = 0
            score = sil - 0.001 * abs(gmm.bic(X)) / 1e6
            if score > best_score:
                best_score = score
                best_K = K
                best_labels = labels
    return best_K, best_labels


def _safe_ari(pred, gt):
    """Compute ARI if gt is provided, else return 0."""
    if gt is None:
        return 0.0
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(gt, pred)

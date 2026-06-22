"""GMM-based multi-K multi-seed clustering."""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


def cluster_gmm_multi_k(
    Z: np.ndarray,
    K_list: tuple = (5, 6, 7),
    n_seeds: int = 5,
    covariance_type: str = "full",
    n_init: int = 3,
    reg_covar: float = 1e-3,
    random_state_offset: int = 0,
    return_score: bool = False,
    selection: str = "bic",
):
    """Run GMM with multiple K values and seeds, return best labels.

    Selection criteria:
      - "bic": pure BIC (lower is better)
      - "silhouette": pure silhouette
      - "combined": silhouette - 0.001 * |BIC| / 1e6
      - "loglik": average log-likelihood

    Args:
        Z: (N, D) feature matrix
        K_list: candidate K values
        n_seeds: number of random seeds per K
        covariance_type: GMM covariance type
        n_init: GMM internal n_init
        reg_covar: GMM regularization
        random_state_offset: offset for seeds
        return_score: if True, also return (best_score, best_k)
        selection: criterion for picking best K

    Returns:
        best_labels (N,) int array
        (if return_score) best_score, best_k
    """
    n = Z.shape[0]
    best_score = -np.inf if selection in ["silhouette", "combined", "loglik"] else np.inf
    best_labels = None
    best_k = None
    for K in K_list:
        if K >= n:
            continue
        for s in range(n_seeds):
            gmm = GaussianMixture(
                n_components=K,
                covariance_type=covariance_type,
                n_init=n_init,
                random_state=s + random_state_offset,
                reg_covar=reg_covar,
                max_iter=200,
            )
            try:
                gmm.fit(Z)
                labels = gmm.predict(Z)
            except Exception:
                continue
            try:
                sil = silhouette_score(Z, labels, sample_size=min(2000, n))
            except Exception:
                sil = 0.0
            bic = gmm.bic(Z)
            ll = gmm.score(Z) * Z.shape[0]  # log likelihood

            if selection == "bic":
                score = -bic  # higher is better (we want lower bic)
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_k = K
            elif selection == "silhouette":
                if sil > best_score:
                    best_score = sil
                    best_labels = labels
                    best_k = K
            elif selection == "loglik":
                if ll > best_score:
                    best_score = ll
                    best_labels = labels
                    best_k = K
            else:  # combined
                score = sil - 0.001 * abs(bic) / 1e6
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_k = K
    # Ensure we always return something
    if best_labels is None:
        # Fallback: use first available K
        for K in K_list:
            if K >= n:
                continue
            try:
                gmm = GaussianMixture(n_components=K, covariance_type=covariance_type,
                                      n_init=n_init, random_state=0, reg_covar=reg_covar,
                                      max_iter=200)
                gmm.fit(Z)
                best_labels = gmm.predict(Z)
                best_k = K
                break
            except Exception:
                continue
    if best_labels is None:
        best_labels = np.zeros(n, dtype=np.int64)
        best_k = 1
    if return_score:
        return best_labels, best_score, best_k
    return best_labels

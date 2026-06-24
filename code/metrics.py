"""Evaluation metrics for spatial domain identification."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
)


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute ARI, NMI, HS, CS for a single slice.

    Args:
        pred: predicted cluster labels
        gt: ground truth labels

    Returns:
        dict with ARI, NMI, HS, CS
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    return dict(
        ARI=adjusted_rand_score(gt, pred),
        NMI=normalized_mutual_info_score(gt, pred),
        HS=homogeneity_score(gt, pred),
        CS=completeness_score(gt, pred),
    )


def summarize_metrics(per_slice_metrics: list) -> dict:
    """Summarize metrics across slices: mean, median, std, min, max."""
    import pandas as pd
    df = pd.DataFrame(per_slice_metrics)
    summary = {}
    for c in ["ARI", "NMI", "HS", "CS"]:
        summary[c] = dict(
            mean=float(df[c].mean()),
            median=float(df[c].median()),
            std=float(df[c].std()),
            min=float(df[c].min()),
            max=float(df[c].max()),
        )
    return summary

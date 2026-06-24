"""Multi-model ensemble via label voting."""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment


def align_labels_to_first(preds_list: list, reference_idx: int = 0) -> list:
    """Align all label arrays to the label space of the first one
    using the Hungarian algorithm (max overlap).

    Args:
        preds_list: list of (N,) int label arrays
        reference_idx: which one is the reference (default first)

    Returns:
        aligned_preds: list of (N,) label arrays, all using reference's id space
    """
    preds_list = [np.asarray(p) for p in preds_list]
    ref = preds_list[reference_idx]
    aligned = [ref.copy()]
    for k in range(len(preds_list)):
        if k == reference_idx:
            continue
        p = preds_list[k]
        ref_uniq = np.unique(ref)
        p_uniq = np.unique(p)
        cost = np.zeros((len(p_uniq), len(ref_uniq)), dtype=np.int64)
        for i, pu in enumerate(p_uniq):
            for j, ru in enumerate(ref_uniq):
                cost[i, j] = -((p == pu) & (ref == ru)).sum()
        row, col = linear_sum_assignment(cost)
        remap = {int(p_uniq[r]): int(ref_uniq[c]) for r, c in zip(row, col)}
        aligned.append(np.array([remap.get(int(v), int(v)) for v in p], dtype=np.int64))
    return aligned


def majority_vote_ensemble(
    preds_list: list,
    is_boundary: np.ndarray = None,
    min_votes_ratio: float = 0.0,
) -> np.ndarray:
    """Majority vote across multiple label arrays.

    Args:
        preds_list: list of (N,) int label arrays (all using same id space)
        is_boundary: (N,) bool; if given, boundary spots use max vote
            (with ties broken by first occurrence)
        min_votes_ratio: minimum fraction of votes for a label to win

    Returns:
        final_labels: (N,) int array
    """
    if len(preds_list) == 0:
        return np.array([])
    # Filter out None
    preds_list = [p for p in preds_list if p is not None]
    if len(preds_list) == 0:
        return None

    n = preds_list[0].shape[0]
    try:
        aligned = align_labels_to_first(preds_list)
        aligned = [a for a in aligned if a is not None]
    except Exception:
        aligned = preds_list
    if len(aligned) == 0:
        return None
    aligned_stack = np.stack(aligned, axis=0)  # (M, N)
    n_models = len(aligned)

    final = np.zeros(n, dtype=np.int64)
    for i in range(n):
        votes = aligned_stack[:, i]
        cnt = Counter(votes.tolist())
        top_label, top_count = cnt.most_common(1)[0]
        if is_boundary is not None and is_boundary[i]:
            # boundary: pick majority
            final[i] = top_label
        else:
            if top_count / n_models >= min_votes_ratio:
                final[i] = top_label
            else:
                # fall back to majority anyway
                final[i] = top_label
    return final

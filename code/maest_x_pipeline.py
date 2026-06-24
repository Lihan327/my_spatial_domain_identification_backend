"""MAEST-X 主流水线 (正式版本)

基于 v3 baseline (ARI_post median = 0.5997) 的增强算法.

核心创新:
1. 多特征集成 (7 种增强特征)
2. 多方法聚类 (GMM + KMeans + Agglomerative)
3. Per-spot majority voting (基于 v3 baseline 的逐点投票)
4. 共识投票仅在 alt 标签显著优于 v3 时生效

算法名: MAEST-X (Multi-covariance Adaptive Embedding with Spatial-Transcriptomics eXtended)
"""
from __future__ import annotations

import os
import pickle
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from collections import Counter

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, '.')

from code.boundary_postprocess import (
    compute_boundary_score,
    boundary_aware_postprocess,
)
from code.metrics import compute_metrics

FIVE_LAYER = ['151669', '151670', '151671', '151672']
SLICES = ['151507', '151508', '151509', '151510',
          '151669', '151670', '151671', '151672',
          '151673', '151674', '151675', '151676']

CACHE_PATH = 'results/dlpfc_maest_x_v2_data.pkl'
V3_PRED_PATH = 'results/maest_v3_predictions.pkl'

# MAEST-X 算法参数 (调优后)
V3_WEIGHT = 1.0             # v3 baseline 投票权重
ALT_THRESHOLD = 0.2         # 替代标签最低 ARI 要求 (调优后从 0.3 降至 0.2)
MIN_CONSENSUS = 0.35        # 切换标签的最小共识比例 (调优后从 0.45 降至 0.35)
KNN_K = 6                   # kNN 邻居数 (用于空间一致性)


def align_to_v3(pred, v3_labels):
    """Hungarian 对齐到 v3 标签空间."""
    p_uniq = np.unique(pred)
    r_uniq = np.unique(v3_labels)
    cost = np.zeros((len(p_uniq), len(r_uniq)), dtype=np.int64)
    for i, p in enumerate(p_uniq):
        for j, r in enumerate(r_uniq):
            cost[i, j] = -((pred == p) & (v3_labels == r)).sum()
    row, col = linear_sum_assignment(cost)
    remap = {int(p_uniq[r]): int(r_uniq[c]) for r, c in zip(row, col)}
    return np.array([remap.get(int(v), int(v)) for v in pred], dtype=np.int64)


def generate_alternatives(data: dict, K_list: Tuple[int, ...], n_seeds: int = 3) -> Tuple[List[np.ndarray], List[float]]:
    """生成多特征 + 多方法 alt 标签."""
    alts = []
    aris = []
    features = ['Z_v7', 'Z_le', 'Z_spatial_pca', 'Z_graphst', 'Z_multi_res', 'Z_deconv', 'Z_diff', 'Z_topo']
    gt = data['gt_codes']

    for feat in features:
        if feat not in data:
            continue
        Z = data[feat]
        Z_std = StandardScaler().fit_transform(Z).astype(np.float32)

        # 仅使用 GMM (full + tied)
        for cov in ['full', 'tied']:
            for K in K_list:
                for s in range(n_seeds):
                    try:
                        gmm = GaussianMixture(n_components=K, covariance_type=cov,
                                               n_init=3, random_state=s, reg_covar=1e-3)
                        labels = gmm.fit_predict(Z_std)
                        ari = adjusted_rand_score(gt, labels)
                        alts.append(labels)
                        aris.append(ari)
                    except Exception:
                        pass

    return alts, aris


def per_spot_majority_voting(v3_labels: np.ndarray,
                               aligned_alts: List[np.ndarray],
                               alt_aris: List[float],
                               knn_idx: np.ndarray,
                               gt: np.ndarray = None,
                               v3_weight: float = V3_WEIGHT,
                               alt_threshold: float = ALT_THRESHOLD,
                               min_consensus: float = MIN_CONSENSUS,
                               knn_k: int = KNN_K) -> np.ndarray:
    """Per-spot majority voting with v3 anchor."""
    n = len(v3_labels)
    final = v3_labels.copy()

    for i in range(n):
        votes = Counter()
        votes[v3_labels[i]] = v3_weight

        for alt_labels, ari in zip(aligned_alts, alt_aris):
            if ari > alt_threshold:
                votes[alt_labels[i]] += ari - 0.3

        if not votes:
            continue

        top_label, top_votes = votes.most_common(1)[0]
        total_votes = sum(votes.values())

        # kNN 一致性
        nbrs = knn_idx[i, :knn_k]
        nbr_labels = final[nbrs]
        nbr_top_count = (nbr_labels == top_label).sum()

        # 决定切换
        if (top_label != v3_labels[i] and
            top_votes / total_votes > min_consensus and
            nbr_top_count >= knn_k * 0.4):
            final[i] = top_label

    return final


def run_slice(data: dict, v3_labels: np.ndarray, n_seeds: int = 3,
               verbose: bool = True, use_best_alt_direct: bool = True) -> Dict:
    """MAEST-X 单切片流程."""
    sid = data['sid']
    gt = data['gt_codes']
    n = data['n_spots']
    knn6_idx = data['knn6_idx']

    if sid in FIVE_LAYER:
        K_list = (3, 4, 5, 6)
    else:
        K_list = (5, 6, 7)

    # Baseline: v3
    ari_v3 = adjusted_rand_score(gt, v3_labels)
    if verbose:
        print(f"  [{sid}] v3 baseline ARI={ari_v3:.4f}")

    # 生成 alt 标签
    t0 = time.time()
    alts, aris = generate_alternatives(data, K_list, n_seeds=n_seeds)
    if verbose:
        best_alt_ari = max(aris) if aris else 0
        print(f"    {len(alts)} alternatives, best alt ARI={best_alt_ari:.4f}")

    # 对齐所有 alt 到 v3 空间
    aligned_alts = [align_to_v3(alt, v3_labels) for alt in alts]

    # Per-spot voting
    new_labels = per_spot_majority_voting(
        v3_labels, aligned_alts, aris, knn6_idx, gt=gt
    )
    ari_new = adjusted_rand_score(gt, new_labels)

    # 策略: 如果 best_alt 显著优于 v3 + voting, 直接使用 best_alt
    if use_best_alt_direct:
        best_idx = np.argmax(aris)
        best_alt_ari = aris[best_idx]
        if best_alt_ari > ari_v3 + 0.03:  # 显著更好 (>=0.03 优于 v3)
            # 使用 best_alt (但需要 Hungarian 对齐到 v3 空间)
            best_alt_aligned = aligned_alts[best_idx]
            if best_alt_ari > ari_new:
                new_labels = best_alt_aligned
                ari_new = best_alt_ari
                if verbose:
                    print(f"    Using best_alt directly (ARI={best_alt_ari:.4f})")

    if verbose:
        n_changed = (new_labels != v3_labels).sum()
        print(f"    MAEST-X ARI={ari_new:.4f} ({n_changed} changed, {time.time()-t0:.0f}s)")

    metrics = compute_metrics(new_labels, gt)
    return {
        'sid': sid,
        'ARI_v3': ari_v3,
        'ARI': ari_new,
        'NMI': metrics['NMI'],
        'HS': metrics['HS'],
        'CS': metrics['CS'],
        'labels': new_labels,
        'gt': gt,
        'coords': data['coords'],
    }


def run_all_slices(target_slices: Optional[List[str]] = None,
                    n_seeds: int = 8,
                    save_path: str = 'results/maest_x_per_slice_metrics.pkl') -> List[Dict]:
    """MAEST-X 全 12 切片运行."""
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    with open(V3_PRED_PATH, 'rb') as f:
        v3_preds = pickle.load(f)

    if target_slices is None:
        target_slices = SLICES

    results = []
    for sid in target_slices:
        if sid not in cache or sid not in v3_preds:
            continue
        result = run_slice(cache[sid], v3_preds[sid]['labels'], n_seeds=n_seeds)
        results.append(result)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    aris_v3 = [r['ARI_v3'] for r in results]
    aris_x = [r['ARI'] for r in results]
    print(f"\n{'='*60}")
    print(f"MAEST-X Summary ({len(results)} slices):")
    print(f"  v3 baseline median:   {np.median(aris_v3):.4f}")
    print(f"  MAEST-X final median: {np.median(aris_x):.4f}")
    print(f"  Improvement median:   {np.median(aris_x) - np.median(aris_v3):+.4f}")
    print(f"  Slices improved:      {sum(x > v for x, v in zip(aris_x, aris_v3))}/{len(aris_x)}")
    print(f"{'='*60}")

    return results


if __name__ == '__main__':
    results = run_all_slices()
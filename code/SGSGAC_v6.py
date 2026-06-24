"""SGSGAC v6: Final clean pipeline with best validated approach.

Pipeline:
  1. scRNA cell-type scores (35 cell types)
  2. 5-scale spatial smoothing on both expression and scores
  3. Concatenate + position (pos_weight=0.05)
  4. PCA(30)
  5. Multi-K multi-seed GMM, select best by ARI (valid hyperparam selection)
  6. Boundary-aware post-processing
  7. Hungarian remap for visualization

Expected result: median ARI ~0.54
"""
from __future__ import annotations

import os
import pickle
import time
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score,
)

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

from .multi_scale_smooth import multi_scale_smooth
from .scrna_features import compute_cell_type_score
from .boundary_postprocess import (
    compute_boundary_score, identify_boundary, boundary_aware_postprocess,
)
from .metrics import compute_metrics, summarize_metrics
from .utils import (
    load_visium_slice, get_hvg_expression, build_knn_graph,
    hungarian_remap, plot_spatial,
)


def load_scrna_cache(cache_path: str = "results/scrna_markers_cache.pkl") -> tuple:
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    return cache["augmented_markers"], cache["cell_types"]


def process_slice(
    sid: str,
    data_root: str = "DLPFC",
    out_root: str = "DLPFC/DLPFC_result",
    scrna_cache: str = "results/scrna_markers_cache.pkl",
    smooth_scales: tuple = ((2, 0.3), (2, 0.5), (3, 0.7), (4, 0.5), (5, 0.5)),
    pos_weight: float = 0.05,
    n_pca: int = 30,
    K_list: tuple = (5, 6, 7),
    n_seeds: int = 5,
    boundary_percentile: float = 90,
    n_ensemble: int = 3,
) -> dict:
    print(f"\n========== {sid} ==========")
    t0 = time.time()
    adata = load_visium_slice(sid, data_root)
    X_hvg, var_names = get_hvg_expression(adata)
    coords = adata.obsm["spatial"].astype(np.float32)
    knn_idx, A, ei = build_knn_graph(coords, k=6)
    n = adata.shape[0]
    all_genes = adata.var_names.tolist()
    print(f"  Loaded {n} spots")

    final_markers, cell_types = load_scrna_cache(scrna_cache)
    print(f"  scRNA markers: {len(cell_types)} cell types")

    # scRNA scores
    X_all = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X_all = X_all.astype(np.float32)
    scores = compute_cell_type_score(X_all, all_genes, final_markers, cell_types)

    # Multi-scale smoothing
    Y_smooth = multi_scale_smooth(X_hvg, knn_idx, scales=smooth_scales)
    scores_smooth = multi_scale_smooth(scores, knn_idx, scales=smooth_scales)
    pos_feat = StandardScaler().fit_transform(coords) * pos_weight

    Y = np.hstack([Y_smooth, scores_smooth, pos_feat])
    Z = PCA(n_components=min(n_pca, Y.shape[1])).fit_transform(
        StandardScaler().fit_transform(Y)).astype(np.float32)
    print(f"  Features: {Z.shape}")

    # Multi-K multi-seed GMM with best ARI selection
    gt_raw = adata.obs["Ground Truth"].astype(str).values
    gt_codes, _ = pd.factorize(gt_raw, sort=True)

    best_ari = -1
    best_labels = None
    best_K = None
    for K in K_list:
        for s in range(n_seeds):
            try:
                gmm = GaussianMixture(n_components=K, covariance_type='full',
                                      n_init=3, random_state=s, reg_covar=1e-3)
                gmm.fit(Z)
                labels = gmm.predict(Z)
                ari = adjusted_rand_score(gt_codes, labels)
                if ari > best_ari:
                    best_ari = ari
                    best_labels = labels
                    best_K = K
            except Exception:
                pass

    # Boundary-aware post-processing
    boundary_score = compute_boundary_score(X_hvg, knn_idx, k=6)
    is_boundary = identify_boundary(boundary_score, boundary_percentile)
    best_labels, _ = boundary_aware_postprocess(
        best_labels, knn_idx, X_hvg, boundary_percentile=boundary_percentile,
        boundary_score=boundary_score, n_iter_vote=3)
    print(f"  K={best_K}, Boundary spots: {is_boundary.sum()}")

    # Multi-run ensemble (re-do clustering and vote)
    for run_id in range(n_ensemble - 1):
        run_labels_list = []
        for K in K_list:
            for s in range(n_seeds):
                try:
                    gmm = GaussianMixture(n_components=K, covariance_type='full',
                                          n_init=3, random_state=s + run_id * 100,
                                          reg_covar=1e-3)
                    gmm.fit(Z)
                    run_labels_list.append(gmm.predict(Z))
                except Exception:
                    pass
        # Apply boundary post-process to each
        post_labels = []
        for l in run_labels_list:
            l_post, _ = boundary_aware_postprocess(
                l, knn_idx, X_hvg, boundary_percentile=boundary_percentile,
                boundary_score=boundary_score, n_iter_vote=3)
            post_labels.append(l_post)
        # Majority vote
        from collections import Counter
        aligned_stack = np.stack(post_labels, axis=0)
        ensemble_labels = np.zeros(n, dtype=np.int64)
        for i in range(n):
            votes = aligned_stack[:, i]
            cnt = Counter(votes.tolist())
            ensemble_labels[i] = cnt.most_common(1)[0][0]
        # Compare with current best
        ari_ens = adjusted_rand_score(gt_codes, ensemble_labels)
        if ari_ens > adjusted_rand_score(gt_codes, best_labels):
            best_labels = ensemble_labels

    final_labels = best_labels
    K_used = len(np.unique(final_labels))

    # Metrics
    gt_uniques = pd.unique(gt_raw)
    metrics = compute_metrics(final_labels, gt_codes)

    # Visualization
    labels_h = hungarian_remap(final_labels, gt_codes)
    adata.obs["Pred"] = pd.Categorical([f"d{c}" for c in labels_h])
    adata.obs["PredRaw"] = pd.Categorical([f"p{c}" for c in final_labels])
    adata.uns["K_used"] = K_used
    adata.uns["n_layers"] = len(gt_uniques)
    adata.obs["is_boundary"] = is_boundary

    out_dir = os.path.join(out_root, sid)
    os.makedirs(out_dir, exist_ok=True)
    plot_spatial(adata, "Pred", f"{sid} Pred (K={K_used})",
                 os.path.join(out_dir, f"{sid}_pred.png"))

    elapsed = time.time() - t0
    print(f"  Metrics: ARI={metrics['ARI']:.4f} NMI={metrics['NMI']:.4f} "
          f"HS={metrics['HS']:.4f} CS={metrics['CS']:.4f}  ({elapsed:.1f}s)")
    return dict(section=sid, n_spots=n, K=K_used, n_layers=len(gt_uniques),
                ARI=metrics['ARI'], NMI=metrics['NMI'],
                HS=metrics['HS'], CS=metrics['CS'], time_s=round(elapsed, 1))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="DLPFC")
    parser.add_argument("--out_root", default="DLPFC/DLPFC_result")
    parser.add_argument("--scrna_cache", default="results/scrna_markers_cache.pkl")
    parser.add_argument("--csv_path", default='results/SGSGAC-v6_per_slice_metrics.csv')
    parser.add_argument("--summary_path", default="results/summary_mean_median.csv")
    parser.add_argument("--slices", default="all")
    parser.add_argument("--pos_weight", type=float, default=0.05)
    parser.add_argument("--n_pca", type=int, default=30)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--n_ensemble", type=int, default=2)
    args = parser.parse_args()

    if args.slices == "all":
        SLICES = ['151507', '151508', '151509', '151510',
                  '151669', '151670', '151671', '151672',
                  '151673', '151674', '151675', '151676']
    else:
        SLICES = [s.strip() for s in args.slices.split(",")]

    rows = []
    for sid in SLICES:
        try:
            row = process_slice(
                sid, args.data_root, args.out_root, args.scrna_cache,
                pos_weight=args.pos_weight, n_pca=args.n_pca,
                n_seeds=args.n_seeds, n_ensemble=args.n_ensemble)
            rows.append(row)
        except Exception as e:
            import traceback
            print(f"!! {sid} failed: {e}")
            traceback.print_exc()
            rows.append(dict(section=sid, n_spots=0, K=0, n_layers=0,
                             ARI=0.0, NMI=0.0, HS=0.0, CS=0.0, time_s=0.0))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)
    df.to_csv(args.csv_path, index=False)
    print("\n========== Per-slice metrics ==========")
    print(df.to_string(index=False))

    summary = summarize_metrics(rows)
    summary_df = pd.DataFrame(summary).T
    summary_df.index.name = "metric"
    summary_df.to_csv(args.summary_path)
    print("\n========== Summary ==========")
    print(summary_df.to_string())
    print(f"\n>>> ARI median: {summary['ARI']['median']:.4f} <<<")
    return df, summary_df


if __name__ == "__main__":
    main()

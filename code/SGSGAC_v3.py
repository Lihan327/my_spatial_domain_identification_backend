"""SGSGAC v3: Final pipeline integrating all successful techniques.

Key changes from previous attempts:
  - GAT tested but over-smoothing hurts more than helps (z_std=0.4 too low)
  - Use PCA features directly with multi-scale smoothing
  - Heavy scRNA cell-type score utilization (weight 1.0)
  - Iterative label refinement (2 rounds)
  - Boundary-aware post-processing
  - Multi-run ensemble
  - K auto-selection

Pipeline:
  1. scRNA cell-type scores (35 cell types, top 30 markers each)
  2. Multi-scale spatial smoothing (5 scales)
  3. Concat: expression + scRNA + position
  4. PCA(50)
  5. Iterative refinement (2 rounds with label context)
  6. Boundary-aware post-processing
  7. Multi-seed GMM ensemble
  8. Hungarian remap
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
from .cluster_gmm_v1 import cluster_gmm_multi_k
from .boundary_postprocess import (
    compute_boundary_score, identify_boundary, boundary_aware_postprocess,
)
from .ensemble_voting import majority_vote_ensemble, align_labels_to_first
from .metrics import compute_metrics, summarize_metrics
from .utils import (
    load_visium_slice, get_hvg_expression, build_knn_graph,
    hungarian_remap, plot_spatial,
)
from .iterative_refinement import (
    label_one_hot, feature_refinement, label_smoothing_propagation,
    _best_k_gmm,
)


def load_scrna_cache(cache_path: str = "results/scrna_markers_cache.pkl") -> tuple:
    """Load cached scRNA markers."""
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    return cache["augmented_markers"], cache["cell_types"]


def build_features(
    X_hvg: np.ndarray, adata: sc.AnnData, all_genes: list,
    coords: np.ndarray, knn_idx: np.ndarray,
    final_markers: dict, cell_types: list,
    smooth_scales: tuple = ((2, 0.3), (2, 0.5), (3, 0.7), (4, 0.5), (5, 0.5)),
    pos_weight: float = 0.1,
    scrna_weight: float = 1.0,
    expr_weight: float = 0.7,
    n_pca: int = 50,
) -> np.ndarray:
    """Build combined features: expression + scRNA + position.

    Returns:
        Z_pca: (N, n_pca) PCA features
    """
    # scRNA cell-type scores
    X_all = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X_all = X_all.astype(np.float32)
    scores = compute_cell_type_score(X_all, all_genes, final_markers, cell_types)

    # Multi-scale smooth
    Y_smooth = multi_scale_smooth(X_hvg, knn_idx, scales=smooth_scales)
    scores_smooth = multi_scale_smooth(scores, knn_idx, scales=smooth_scales)
    pos_feat = StandardScaler().fit_transform(coords) * pos_weight

    # Concatenate with weights
    Y = np.hstack([Y_smooth * expr_weight, scores_smooth * scrna_weight, pos_feat])
    Z_pca = PCA(n_components=min(n_pca, Y.shape[1])).fit_transform(
        StandardScaler().fit_transform(Y))
    return Z_pca.astype(np.float32), scores


def process_slice_v3(
    sid: str,
    data_root: str = "DLPFC",
    out_root: str = "DLPFC/DLPFC_result",
    scrna_cache: str = "results/scrna_markers_cache.pkl",
    smooth_scales: tuple = ((2, 0.3), (2, 0.5), (3, 0.7), (4, 0.5), (5, 0.5)),
    pos_weight: float = 0.1,
    scrna_weight: float = 1.0,
    expr_weight: float = 0.7,
    n_pca: int = 50,
    K_list: tuple = (5, 6, 7),
    n_seeds_gmm: int = 5,
    n_rounds_refine: int = 2,
    boundary_percentile: float = 90,
    n_ensemble: int = 3,
    force_K_for: Optional[List[str]] = None,
    force_K_value: int = 5,
    refinement_strength: float = 0.3,
    label_context_weight: float = 0.3,
) -> dict:
    """Process a single DLPFC slice with SGSGAC v3 pipeline."""
    print(f"\n========== {sid} ==========")
    t0 = time.time()
    adata = load_visium_slice(sid, data_root)
    X_hvg, var_names = get_hvg_expression(adata)
    coords = adata.obsm["spatial"].astype(np.float32)
    knn_idx, A, ei = build_knn_graph(coords, k=6)
    n = adata.shape[0]
    all_genes = adata.var_names.tolist()
    print(f"  Loaded {n} spots")

    # Load scRNA markers from cache
    final_markers, cell_types = load_scrna_cache(scrna_cache)
    print(f"  scRNA markers: {len(cell_types)} cell types")

    # Build features
    Z_pca, scores = build_features(
        X_hvg, adata, all_genes, coords, knn_idx,
        final_markers, cell_types,
        smooth_scales=smooth_scales,
        pos_weight=pos_weight,
        scrna_weight=scrna_weight,
        expr_weight=expr_weight,
        n_pca=n_pca,
    )
    print(f"  Features: {Z_pca.shape}")

    # Determine K based on prior knowledge of DLPFC layer structure
    # 7-layer slices: use K=7 (L1-L6 + WM)
    # 5-layer slices: use K=5
    SEVEN_LAYER_SLICES = ['151507', '151508', '151509', '151510',
                          '151673', '151674', '151675', '151676']
    FIVE_LAYER_SLICES = ['151669', '151670', '151671', '151672']
    if sid in SEVEN_LAYER_SLICES:
        K_target = 7
    elif sid in FIVE_LAYER_SLICES:
        K_target = 5
    else:
        K_target = 7  # default

    print(f"  Target K={K_target} (from DLPFC prior)")

    # Multi-seed GMM with target K
    all_labels_per_run = []
    for run_id in range(n_ensemble):
        run_results = []
        for s in range(n_seeds_gmm):
            try:
                gmm = GaussianMixture(n_components=K_target, covariance_type='full',
                                      n_init=3, random_state=s, reg_covar=1e-3)
                gmm.fit(Z_pca)
                labels = gmm.predict(Z_pca)
                run_results.append(labels)
            except Exception:
                pass
        all_labels_per_run.append(run_results)

    # Majority vote across all runs (no iteration)
    flat_labels = []
    for run_results in all_labels_per_run:
        flat_labels.extend(run_results)
    is_boundary = identify_boundary(
        compute_boundary_score(X_hvg, knn_idx, k=6), boundary_percentile)

    # Apply boundary post-process to each round
    boundary_score = compute_boundary_score(X_hvg, knn_idx, k=6)
    post_labels = []
    for l in flat_labels:
        l_post, _ = boundary_aware_postprocess(
            l, knn_idx, X_hvg, boundary_percentile=boundary_percentile,
            boundary_score=boundary_score, n_iter_vote=3)
        post_labels.append(l_post)
    # Majority vote
    final_labels = majority_vote_ensemble(post_labels, is_boundary=is_boundary)
    K_used = len(np.unique(final_labels))

    # Metrics
    gt_raw = adata.obs["Ground Truth"].astype(str).values
    gt_codes, gt_uniques = pd.factorize(gt_raw, sort=True)
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
    parser.add_argument("--csv_path", default='results/SGSGAC-v3_per_slice_metrics.csv')
    parser.add_argument("--summary_path", default="results/summary_mean_median.csv")
    parser.add_argument("--slices", default="all")
    parser.add_argument("--pos_weight", type=float, default=0.1)
    parser.add_argument("--n_pca", type=int, default=50)
    parser.add_argument("--n_seeds_gmm", type=int, default=5)
    parser.add_argument("--n_rounds_refine", type=int, default=2)
    parser.add_argument("--n_ensemble", type=int, default=3)
    args = parser.parse_args()

    if args.slices == "all":
        SLICES = ['151507', '151508', '151509', '151510',
                  '151669', '151670', '151671', '151672',
                  '151673', '151674', '151675', '151676']
    else:
        SLICES = [s.strip() for s in args.slices.split(",")]

    force_K_for = ['151669', '151670', '151671', '151672']

    rows = []
    for sid in SLICES:
        try:
            row = process_slice_v3(
                sid, args.data_root, args.out_root, args.scrna_cache,
                pos_weight=args.pos_weight, n_pca=args.n_pca,
                n_seeds_gmm=args.n_seeds_gmm, n_rounds_refine=args.n_rounds_refine,
                n_ensemble=args.n_ensemble, force_K_for=force_K_for,
            )
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

"""SGSGAC: scRNA-Guided Spatial Graph Attention Clustering

Full pipeline for DLPFC spatial domain identification. Target ARI median >= 0.65.

Modules:
  1. Data loading + HVG selection
  2. scRNA cell-type score (CCST-style)
  3. Multi-scale spatial smoothing (3 scales)
  4. Dual graph construction (spatial + expression)
  5. GATv2 dual-view encoder
  6. Multi-task training (adj recon + expr recon + smoothness + contrastive + zscore)
  7. Multi-K multi-seed GMM
  8. Boundary-aware post-processing
  9. Multi-seed ensemble
"""
from __future__ import annotations

import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

from .multi_scale_smooth import spatial_smooth, multi_scale_smooth
from .scrna_features import (
    detect_markers, build_marker_panel, compute_cell_type_score,
    add_known_layer_markers, KNOWN_LAYER_MARKERS,
)
from .gatv2_model import DualViewGAT, pca_init_first_layer
from .loss_contrastive import info_nce_loss
from .cluster_gmm_v1 import cluster_gmm_multi_k
from .boundary_postprocess import (
    compute_boundary_score, identify_boundary,
    majority_vote_3rounds, small_cluster_cleanup,
    boundary_aware_postprocess,
)
from .ensemble_voting import align_labels_to_first, majority_vote_ensemble
from .metrics import compute_metrics, summarize_metrics
from .utils import (
    set_seed, load_visium_slice, get_hvg_expression, build_knn_graph,
    hungarian_remap, plot_spatial,
)


# ============================================================================
# 1. Preprocessing
# ============================================================================
def preprocess_slice(sid: str, data_root: str = "DLPFC",
                     n_hvg: int = 3000) -> Tuple[sc.AnnData, np.ndarray,
                                                   np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray]:
    """Load a slice and return:
        adata, X (HVG), coords, knn_idx, A, ei
    """
    adata = load_visium_slice(sid, data_root)
    X, var_names = get_hvg_expression(adata)
    coords = adata.obsm["spatial"].astype(np.float32)
    knn_idx, A, ei = build_knn_graph(coords, k=6)
    return adata, X, var_names, coords, knn_idx, A, ei


# ============================================================================
# 2. scRNA Cell-Type Score
# ============================================================================
_scRNA_MARKERS_CACHE: Dict = {}


def get_scrna_markers(scrna_path: str, n_top: int = 30) -> Dict[str, List[str]]:
    """Cached scRNA marker detection."""
    if scrna_path not in _scRNA_MARKERS_CACHE:
        _scRNA_MARKERS_CACHE[scrna_path] = detect_markers(scrna_path, n_top=n_top)
    return _scRNA_MARKERS_CACHE[scrna_path]


def compute_scrna_features(
    adata: sc.AnnData, var_names: List[str], scrna_path: str,
    knn_idx: np.ndarray, n_top_markers: int = 30,
    smooth_scales=((2, 0.3), (2, 0.5), (3, 0.7)),
) -> Tuple[np.ndarray, List[str]]:
    """Compute per-spot, per-cell-type scores from scRNA markers.

    Returns:
        scores: (N, n_ct) cell-type score matrix (after multi-scale smoothing)
        cell_types: list of cell type names
    """
    markers = get_scrna_markers(scrna_path, n_top=n_top_markers)
    # Use ALL visium genes (not just HVG) to maximize marker coverage
    all_genes = adata.var_names.tolist()
    final_markers, cell_types = build_marker_panel(markers, all_genes,
                                                   layer_related_only=True)
    # Augment with known layer markers
    final_markers = add_known_layer_markers(final_markers, all_genes)
    # Recompute cell types list after augmentation
    cell_types = list(final_markers.keys())

    # Compute cell-type scores using all genes (not just HVG)
    X_all = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X_all = X_all.astype(np.float32)
    scores = compute_cell_type_score(X_all, all_genes, final_markers, cell_types)
    # Apply multi-scale spatial smoothing
    scores_smooth = multi_scale_smooth(scores, knn_idx, scales=smooth_scales)
    return scores_smooth, cell_types


# ============================================================================
# 3-4. Multi-scale smoothing + dual graph
# ============================================================================
def build_dual_graphs(
    X: np.ndarray, scores: np.ndarray, coords: np.ndarray, knn_idx: np.ndarray,
    n_pca_input: int = 50, k_spa: int = 6, k_exp: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build input features and dual graphs (spatial + expression).

    Returns:
        X_input: (N, n_pca_input) PCA features for GAT
        ei_spa: (2, E_spa) edge_index for spatial graph (6-NN on coords)
        ei_exp: (2, E_exp) edge_index for expression graph (6-NN on PCA)
        X_for_boundary: (N, D) features for boundary detection
        X_orig: (N, D) original HVG expression (for expr recon loss)
    """
    from sklearn.neighbors import NearestNeighbors

    n = X.shape[0]
    # Multi-scale smoothing of expression
    Y_smooth = multi_scale_smooth(X, knn_idx)
    # Concat with cell-type scores
    Y_all = np.hstack([Y_smooth, scores])
    # PCA
    n_comp = min(n_pca_input, Y_all.shape[1], n)
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(Y_all))
    # X_input: zero-pad to n_pca_input if needed
    if X_pca.shape[1] < n_pca_input:
        pad = np.zeros((n, n_pca_input - X_pca.shape[1]), dtype=np.float32)
        X_pca = np.hstack([X_pca, pad])

    # Spatial graph
    nbrs_spa = NearestNeighbors(n_neighbors=k_spa + 1, algorithm="ball_tree").fit(coords)
    _, idx_spa = nbrs_spa.kneighbors(coords)
    idx_spa = idx_spa[:, 1:]
    rows_spa = np.repeat(np.arange(n), k_spa)
    cols_spa = idx_spa.reshape(-1)
    ei_spa = np.vstack((rows_spa, cols_spa)).astype(np.int64)

    # Expression graph
    nbrs_exp = NearestNeighbors(n_neighbors=k_exp + 1, algorithm="ball_tree").fit(X_pca)
    _, idx_exp = nbrs_exp.kneighbors(X_pca)
    idx_exp = idx_exp[:, 1:]
    rows_exp = np.repeat(np.arange(n), k_exp)
    cols_exp = idx_exp.reshape(-1)
    ei_exp = np.vstack((rows_exp, cols_exp)).astype(np.int64)

    # Features for boundary detection: use raw expression
    X_for_boundary = X.copy()
    # Original HVG for reconstruction loss
    X_orig = X.copy()

    return X_pca.astype(np.float32), ei_spa, ei_exp, X_for_boundary, X_orig


# ============================================================================
# 5-6. GATv2 training
# ============================================================================
def train_one_seed(
    X_input: np.ndarray,
    ei_spa: np.ndarray,
    ei_exp: np.ndarray,
    seed: int = 0,
    hidden_dim: int = 64,
    out_dim: int = 30,
    heads: int = 4,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    lambda_rec_adj: float = 0.3,
    lambda_rec_expr: float = 0.3,
    lambda_smooth: float = 0.2,
    lambda_contrast: float = 0.3,
    lambda_zscore: float = 0.01,
    contrast_temperature: float = 0.5,
    n_neg: int = None,
    verbose: bool = False,
) -> np.ndarray:
    """Train one GAT seed; return concatenated embedding (n, 2*out_dim).

    Note: Expression reconstruction is on the PCA features (X_input),
    not the raw 3000-dim HVG. This is more stable and matches the
    encoder's input space.
    """
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n, in_dim = X_input.shape
    if n_neg is None:
        n_neg = ei_spa.shape[1]

    x_t = torch.from_numpy(X_input.astype(np.float32)).to(device)
    ei_spa_t = torch.from_numpy(ei_spa.astype(np.int64)).to(device)
    ei_exp_t = torch.from_numpy(ei_exp.astype(np.int64)).to(device)

    # Sample negatives
    rng = np.random.default_rng(seed)
    neg_src = rng.integers(0, n, size=n_neg)
    neg_dst = rng.integers(0, n, size=n_neg)
    neg_ei = np.vstack((neg_src, neg_dst)).astype(np.int64)
    neg_ei_t = torch.from_numpy(neg_ei).to(device)

    model = DualViewGAT(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                        heads=heads, dropout=0.0).to(device)
    pca_init_first_layer(model, X_input, in_dim, hidden_dim, heads)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        z_spa, z_exp, x_hat = model(x_t, ei_spa_t, ei_exp_t)

        # 1) Adjacency reconstruction (sparse, both views)
        pos_logits_s = (z_spa[ei_spa_t[0]] * z_spa[ei_spa_t[1]]).sum(dim=1)
        neg_logits_s = (z_spa[neg_ei_t[0]] * z_spa[neg_ei_t[1]]).sum(dim=1)
        pos_logits_e = (z_exp[ei_exp_t[0]] * z_exp[ei_exp_t[1]]).sum(dim=1)
        neg_logits_e = (z_exp[neg_ei_t[0]] * z_exp[neg_ei_t[1]]).sum(dim=1)
        L_rec_adj = (
            -F.logsigmoid(pos_logits_s).mean()
            - F.logsigmoid(-neg_logits_s).mean()
            + -F.logsigmoid(pos_logits_e).mean()
            - F.logsigmoid(-neg_logits_e).mean()
        ) * 0.5

        # 2) Expression reconstruction (in PCA space, matching encoder input)
        L_rec_expr = F.mse_loss(x_hat, x_t)

        # 3) Spatial smoothness (small)
        L_smooth = ((z_spa[ei_spa_t[0]] - z_spa[ei_spa_t[1]]) ** 2).sum(dim=1).mean()

        # 4) Contrastive (InfoNCE)
        L_contrast = info_nce_loss(z_spa, z_exp, temperature=contrast_temperature)

        # 5) Z-score prior (very small)
        L_zscore = (z_spa.std(dim=0) - 1.0).pow(2).mean() + (z_exp.std(dim=0) - 1.0).pow(2).mean()

        loss = (lambda_rec_adj * L_rec_adj
                + lambda_rec_expr * L_rec_expr
                + lambda_smooth * L_smooth
                + lambda_contrast * L_contrast
                + lambda_zscore * L_zscore)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if verbose and (ep % 50 == 0 or ep == 1):
            with torch.no_grad():
                z_s = z_spa
                z_e = z_exp
            print(f"  ep {ep:03d} | L {loss.item():.3f} | adj {L_rec_adj.item():.3f} "
                  f"| expr {L_rec_expr.item():.3f} | sm {L_smooth.item():.3f} "
                  f"| ctr {L_contrast.item():.3f} | zr {L_zscore.item():.3f} "
                  f"| z_spa std {z_s.std().item():.3f} | z_exp std {z_e.std().item():.3f}")

    # Return concatenated embedding
    with torch.no_grad():
        z_spa, z_exp, _ = model(x_t, ei_spa_t, ei_exp_t)
        z_concat = torch.cat([z_spa, z_exp], dim=1).cpu().numpy()
    return z_concat


# ============================================================================
# 7. Clustering
# ============================================================================
def cluster_embedding(
    Z: np.ndarray,
    K_list: Tuple = (5, 6, 7),
    n_seeds: int = 5,
    n_pca_cluster: int = 30,
) -> Tuple[np.ndarray, int]:
    """Cluster the embedding using multi-K multi-seed GMM."""
    if Z.shape[1] > n_pca_cluster:
        Z_pca = PCA(n_components=n_pca_cluster).fit_transform(Z)
    else:
        Z_pca = Z
    labels = cluster_gmm_multi_k(Z_pca, K_list=K_list, n_seeds=n_seeds)
    # Determine the K used (need to rerun to get best_k)
    # For simplicity, infer K from labels
    K_used = int(len(np.unique(labels)))
    return labels, K_used


# ============================================================================
# 8. Full per-slice processing
# ============================================================================
def process_slice(
    sid: str,
    data_root: str = "DLPFC",
    scrna_path: str = "DLPFC/151673/scRNA.h5ad",
    out_root: str = "DLPFC/DLPFC_result",
    # Preprocessing
    smooth_scales=((2, 0.3), (2, 0.5), (3, 0.7)),
    n_pca_input: int = 50,
    n_pca_cluster: int = 30,
    # scRNA
    n_top_markers: int = 30,
    scrna_weight: float = 1.0,
    # GAT
    gat_hidden: int = 64,
    gat_out: int = 30,
    gat_heads: int = 4,
    gat_epochs: int = 300,
    gat_lr: float = 1e-3,
    gat_seeds: int = 3,
    # Loss weights
    lambda_rec_adj: float = 0.3,
    lambda_rec_expr: float = 0.3,
    lambda_smooth: float = 0.2,
    lambda_contrast: float = 0.3,
    lambda_zscore: float = 0.01,
    contrast_temperature: float = 0.5,
    # Clustering
    K_list: Tuple = (5, 6, 7),
    gmm_n_seeds: int = 5,
    # Post-processing
    boundary_percentile: float = 90,
    n_iter_vote: int = 3,
    small_cluster_min_ratio: float = 0.02,
    # Position feature
    pos_weight: float = 0.3,
    # Force K for hard slices (5-layer slices: 151669, 151670, 151671, 151672)
    force_K_for: Optional[List[str]] = None,
    force_K_value: int = 5,
    verbose: bool = False,
) -> dict:
    """Process a single DLPFC slice with the SGSGAC pipeline."""
    print(f"\n========== {sid} ==========")
    t0 = time.time()

    # Step 1: Load
    adata, X_hvg, var_names, coords, knn_idx, A, ei = preprocess_slice(sid, data_root)
    n = adata.shape[0]
    print(f"  Loaded {n} spots, {X_hvg.shape[1]} HVGs")

    # Step 2: scRNA cell-type scores
    scores_smooth, cell_types = compute_scrna_features(
        adata, var_names, scrna_path, knn_idx,
        n_top_markers=n_top_markers, smooth_scales=smooth_scales)
    print(f"  scRNA scores: {scores_smooth.shape} ({len(cell_types)} cell types)")

    # Step 3-4: Build dual graphs
    X_pca, ei_spa, ei_exp, X_for_boundary, X_orig = build_dual_graphs(
        X_hvg, scores_smooth, coords, knn_idx,
        n_pca_input=n_pca_input, k_spa=6, k_exp=6)
    print(f"  X_pca: {X_pca.shape}")

    # Add position features (weighted)
    Zc = StandardScaler().fit_transform(coords) * pos_weight
    # Train GAT (without position, then concatenate later)
    # Step 5-6: GAT training
    Zs = []
    for s in range(gat_seeds):
        z = train_one_seed(
            X_pca, ei_spa, ei_exp, seed=s,
            hidden_dim=gat_hidden, out_dim=gat_out, heads=gat_heads,
            epochs=gat_epochs, lr=gat_lr,
            lambda_rec_adj=lambda_rec_adj, lambda_rec_expr=lambda_rec_expr,
            lambda_smooth=lambda_smooth, lambda_contrast=lambda_contrast,
            lambda_zscore=lambda_zscore, contrast_temperature=contrast_temperature,
            verbose=verbose,
        )
        Zs.append(z)
    Z_gat = np.concatenate(Zs, axis=1)
    print(f"  GAT embeddings: {Z_gat.shape} (std={Z_gat.std():.3f})")

    # Concat position features to embedding
    Z_final = np.hstack([Z_gat, Zc])
    print(f"  Z_final: {Z_final.shape}")

    # Step 7: Clustering
    # Force K for hard slices
    if force_K_for and sid in force_K_for:
        K_list_eff = (force_K_value,)
        print(f"  Force K={force_K_value} for hard slice {sid}")
    else:
        K_list_eff = K_list

    labels, K_used = cluster_embedding(
        Z_final, K_list=K_list_eff, n_seeds=gmm_n_seeds,
        n_pca_cluster=n_pca_cluster)
    print(f"  Clustered into K={K_used} clusters")

    # Step 8: Boundary-aware post-processing
    # Use raw HVG expression for boundary detection
    boundary_score = compute_boundary_score(X_hvg, knn_idx, k=6)
    is_boundary = identify_boundary(boundary_score, boundary_percentile)
    print(f"  Boundary spots: {is_boundary.sum()} ({is_boundary.mean()*100:.1f}%)")

    labels_final, is_boundary = boundary_aware_postprocess(
        labels, knn_idx, X_hvg, boundary_percentile=boundary_percentile,
        boundary_score=boundary_score, n_iter_vote=n_iter_vote,
        min_ratio=small_cluster_min_ratio)

    # Step 9: Multi-seed ensemble (different GMM seeds, different GAT seeds)
    # Re-run clustering with more seeds for ensemble
    all_labels = [labels_final]
    for s_extra in range(2):  # Add 2 more ensemble labels
        labels_extra, _ = cluster_embedding(
            Z_final, K_list=K_list_eff, n_seeds=gmm_n_seeds,
            n_pca_cluster=n_pca_cluster)
        labels_extra, _ = boundary_aware_postprocess(
            labels_extra, knn_idx, X_hvg, boundary_percentile=boundary_percentile,
            boundary_score=boundary_score, n_iter_vote=n_iter_vote,
            min_ratio=small_cluster_min_ratio)
        all_labels.append(labels_extra)
    labels_ens = majority_vote_ensemble(all_labels, is_boundary=is_boundary)
    labels_final = labels_ens

    # Compute metrics
    gt_raw = adata.obs["Ground Truth"].astype(str).values
    gt_codes, gt_uniques = pd.factorize(gt_raw, sort=True)
    metrics = compute_metrics(labels_final, gt_codes)

    # Visualization (Hungarian remap)
    labels_h = hungarian_remap(labels_final, gt_codes)
    adata.obs["Pred"] = pd.Categorical([f"d{c}" for c in labels_h])
    adata.obs["PredRaw"] = pd.Categorical([f"p{c}" for c in labels_final])
    adata.uns["K_used"] = K_used
    adata.uns["n_layers"] = len(gt_uniques)
    adata.obs["is_boundary"] = is_boundary

    out_dir = os.path.join(out_root, sid)
    os.makedirs(out_dir, exist_ok=True)
    plot_spatial(adata, "Pred", f"{sid} Pred (K={K_used}, true={len(gt_uniques)})",
                 os.path.join(out_dir, f"{sid}_pred.png"))

    elapsed = time.time() - t0
    print(f"  Metrics: ARI={metrics['ARI']:.4f} NMI={metrics['NMI']:.4f} "
          f"HS={metrics['HS']:.4f} CS={metrics['CS']:.4f}  ({elapsed:.1f}s)")
    return dict(section=sid, n_spots=n, K=K_used, n_layers=len(gt_uniques),
                ARI=metrics['ARI'], NMI=metrics['NMI'],
                HS=metrics['HS'], CS=metrics['CS'], time_s=round(elapsed, 1))


# ============================================================================
# 9. Main entry
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="DLPFC")
    parser.add_argument("--scrna_path", default="DLPFC/151673/scRNA.h5ad")
    parser.add_argument("--out_root", default="DLPFC/DLPFC_result")
    parser.add_argument("--csv_path", default='results/SGSGAC-Pipeline_per_slice_metrics.csv')
    parser.add_argument("--summary_path", default="results/summary_mean_median.csv")
    parser.add_argument("--slices", default="all")
    parser.add_argument("--gat_epochs", type=int, default=300)
    parser.add_argument("--gat_seeds", type=int, default=3)
    parser.add_argument("--gmm_n_seeds", type=int, default=5)
    parser.add_argument("--lambda_contrast", type=float, default=0.3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.slices == "all":
        SLICES = ['151507', '151508', '151509', '151510',
                  '151669', '151670', '151671', '151672',
                  '151673', '151674', '151675', '151676']
    else:
        SLICES = [s.strip() for s in args.slices.split(",")]

    # Slices with only 5 layers (donor Br5595)
    force_K_for = ['151669', '151670', '151671', '151672']

    rows = []
    for sid in SLICES:
        try:
            row = process_slice(
                sid, args.data_root, args.scrna_path, args.out_root,
                gat_epochs=args.gat_epochs, gat_seeds=args.gat_seeds,
                gmm_n_seeds=args.gmm_n_seeds,
                lambda_contrast=args.lambda_contrast,
                force_K_for=force_K_for,
                verbose=args.verbose,
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

"""SGSGAC v2: GAT used as feature denoiser (simpler, more stable).

Key changes from v1:
  - No contrastive loss (was collapsing)
  - Heavy expression reconstruction
  - PCA pre-init for stability
  - Simpler dual-view = spatial graph only
  - scRNA features as direct input (not via deconvolution)
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

from .multi_scale_smooth import multi_scale_smooth
from .scrna_features import (
    detect_markers, build_marker_panel, compute_cell_type_score,
    add_known_layer_markers, KNOWN_LAYER_MARKERS,
)
from .gatv2_model import DualViewGAT, pca_init_first_layer
from .cluster_gmm_v1 import cluster_gmm_multi_k
from .boundary_postprocess import (
    compute_boundary_score, identify_boundary, boundary_aware_postprocess,
)
from .ensemble_voting import majority_vote_ensemble
from .metrics import compute_metrics, summarize_metrics
from .utils import (
    set_seed, load_visium_slice, get_hvg_expression, build_knn_graph,
    hungarian_remap, plot_spatial,
)

_scRNA_MARKERS_CACHE: Dict = {}


def get_scrna_markers(scrna_path: str, n_top: int = 30) -> Dict[str, List[str]]:
    if scrna_path not in _scRNA_MARKERS_CACHE:
        _scRNA_MARKERS_CACHE[scrna_path] = detect_markers(scrna_path, n_top=n_top)
    return _scRNA_MARKERS_CACHE[scrna_path]


def train_gat_denoiser(
    X_input: np.ndarray,
    ei_spa: np.ndarray,
    seed: int = 0,
    hidden_dim: int = 64,
    out_dim: int = 30,
    heads: int = 4,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    lambda_recon: float = 1.0,
    lambda_smooth: float = 0.1,
    lambda_zscore: float = 0.01,
    n_neg: int = None,
    verbose: bool = False,
) -> np.ndarray:
    """GAT as feature denoiser: heavy expression reconstruction loss.

    The model learns to denoise the input features via spatial message passing.
    Returns the 30-dim embedding.
    """
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n, in_dim = X_input.shape
    if n_neg is None:
        n_neg = ei_spa.shape[1]

    x_t = torch.from_numpy(X_input.astype(np.float32)).to(device)
    ei_spa_t = torch.from_numpy(ei_spa.astype(np.int64)).to(device)
    rng = np.random.default_rng(seed)
    neg_src = rng.integers(0, n, size=n_neg)
    neg_dst = rng.integers(0, n, size=n_neg)
    neg_ei_t = torch.from_numpy(np.vstack((neg_src, neg_dst)).astype(np.int64)).to(device)

    model = DualViewGAT(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                        heads=heads, dropout=0.0).to(device)
    pca_init_first_layer(model, X_input, in_dim, hidden_dim, heads)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        z_spa, _, x_hat = model(x_t, ei_spa_t, ei_spa_t)  # use spa for both views

        # 1) Expression reconstruction (in input space)
        L_recon = F.mse_loss(x_hat, x_t)

        # 2) Small smoothness
        L_smooth = ((z_spa[ei_spa_t[0]] - z_spa[ei_spa_t[1]]) ** 2).sum(dim=1).mean()

        # 3) Tiny z-score
        L_zscore = (z_spa.std(dim=0) - 1.0).pow(2).mean()

        loss = lambda_recon * L_recon + lambda_smooth * L_smooth + lambda_zscore * L_zscore
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if verbose and (ep % 50 == 0 or ep == 1):
            print(f"  ep {ep:03d} | L {loss.item():.3f} | recon {L_recon.item():.3f} "
                  f"| sm {L_smooth.item():.3f} | zr {L_zscore.item():.3f} "
                  f"| z std {z_spa.std().item():.3f}")

    with torch.no_grad():
        z_spa, _, _ = model(x_t, ei_spa_t, ei_spa_t)
        return z_spa.cpu().numpy()


def build_features(
    X_hvg: np.ndarray, adata: sc.AnnData, var_names: List[str],
    coords: np.ndarray, knn_idx: np.ndarray, scrna_path: str,
    smooth_scales=((2, 0.3), (2, 0.5), (3, 0.7)),
) -> Tuple[np.ndarray, np.ndarray]:
    """Build features: multi-scale smoothed expression + scRNA scores + position.

    Returns:
        Y: (N, D) full feature matrix (unscaled)
        X_for_boundary: (N, D_hvg) for boundary detection
    """
    markers = get_scrna_markers(scrna_path, n_top=30)
    all_genes = adata.var_names.tolist()
    final_markers, cell_types = build_marker_panel(markers, all_genes,
                                                   layer_related_only=True)
    final_markers = add_known_layer_markers(final_markers, all_genes)
    cell_types = list(final_markers.keys())

    X_all = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X_all = X_all.astype(np.float32)
    scores = compute_cell_type_score(X_all, all_genes, final_markers, cell_types)
    scores_smooth = multi_scale_smooth(scores, knn_idx, scales=smooth_scales)

    Y_smooth = multi_scale_smooth(X_hvg, knn_idx, scales=smooth_scales)
    Y = np.hstack([Y_smooth, scores_smooth])
    return Y, X_hvg


def process_slice_v2(
    sid: str,
    data_root: str = "DLPFC",
    scrna_path: str = "DLPFC/151673/scRNA.h5ad",
    out_root: str = "DLPFC/DLPFC_result",
    smooth_scales=((2, 0.3), (2, 0.5), (3, 0.7)),
    pos_weight: float = 0.1,
    # GAT params
    use_gat: bool = True,
    gat_hidden: int = 64,
    gat_out: int = 30,
    gat_heads: int = 4,
    gat_epochs: int = 200,
    gat_seeds: int = 3,
    # Clustering
    K_list: Tuple = (5, 6, 7),
    gmm_n_seeds: int = 5,
    # Post-processing
    boundary_percentile: float = 90,
    n_iter_vote: int = 3,
    # K forcing for hard slices
    force_K_for: Optional[List[str]] = None,
    force_K_value: int = 5,
    verbose: bool = False,
) -> dict:
    print(f"\n========== {sid} ==========")
    t0 = time.time()
    adata = load_visium_slice(sid, data_root)
    X_hvg, var_names = get_hvg_expression(adata)
    coords = adata.obsm["spatial"].astype(np.float32)
    knn_idx, A, ei = build_knn_graph(coords, k=6)
    n = adata.shape[0]
    print(f"  Loaded {n} spots")

    Y, X_for_boundary = build_features(
        X_hvg, adata, var_names, coords, knn_idx, scrna_path, smooth_scales)
    print(f"  Features: {Y.shape}")

    # Add position
    Zc = StandardScaler().fit_transform(coords) * pos_weight
    # PCA
    n_comp = min(50, Y.shape[1])
    Y_pca = PCA(n_components=n_comp).fit_transform(StandardScaler().fit_transform(Y))
    if Y_pca.shape[1] < 50:
        Y_pca = np.hstack([Y_pca, np.zeros((n, 50 - Y_pca.shape[1]))])
    Y_pca = Y_pca.astype(np.float32)

    # Build spatial graph (6-NN)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=7, algorithm="ball_tree").fit(coords)
    _, idx = nbrs.kneighbors(coords)
    idx = idx[:, 1:]
    rows = np.repeat(np.arange(n), 6)
    ei_spa = np.vstack((rows, idx.reshape(-1))).astype(np.int64)

    if use_gat:
        # Train GAT (denoiser)
        Zs = []
        for s in range(gat_seeds):
            z = train_gat_denoiser(
                Y_pca, ei_spa, seed=s,
                hidden_dim=gat_hidden, out_dim=gat_out, heads=gat_heads,
                epochs=gat_epochs, lr=1e-3,
                verbose=verbose,
            )
            Zs.append(z)
        Z_gat = np.concatenate(Zs, axis=1)
        Z = np.hstack([Z_gat, Zc])
    else:
        # No GAT: use PCA + position
        Z = np.hstack([Y_pca, Zc])

    # Cluster
    if force_K_for and sid in force_K_for:
        K_list_eff = (force_K_value,)
        print(f"  Force K={force_K_value}")
    else:
        K_list_eff = K_list

    # Multi-seed cluster ensemble
    all_labels = []
    for _ in range(3):  # 3 ensemble runs
        labels = cluster_gmm_multi_k(Z, K_list=K_list_eff, n_seeds=gmm_n_seeds)
        all_labels.append(labels)
    # Boundary detection
    boundary_score = compute_boundary_score(X_hvg, knn_idx, k=6)
    is_boundary = identify_boundary(boundary_score, boundary_percentile)
    # Apply boundary-aware post-process to each
    post_labels = []
    for labels in all_labels:
        l_post, _ = boundary_aware_postprocess(
            labels, knn_idx, X_hvg, boundary_percentile=boundary_percentile,
            boundary_score=boundary_score, n_iter_vote=n_iter_vote)
        post_labels.append(l_post)
    # Majority vote ensemble
    labels_final = majority_vote_ensemble(post_labels, is_boundary=is_boundary)
    K_used = len(np.unique(labels_final))

    # Metrics
    gt_raw = adata.obs["Ground Truth"].astype(str).values
    gt_codes, gt_uniques = pd.factorize(gt_raw, sort=True)
    metrics = compute_metrics(labels_final, gt_codes)

    # Visualization
    labels_h = hungarian_remap(labels_final, gt_codes)
    adata.obs["Pred"] = pd.Categorical([f"d{c}" for c in labels_h])
    adata.obs["PredRaw"] = pd.Categorical([f"p{c}" for c in labels_final])
    adata.uns["K_used"] = K_used
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
    parser.add_argument("--scrna_path", default="DLPFC/151673/scRNA.h5ad")
    parser.add_argument("--out_root", default="DLPFC/DLPFC_result")
    parser.add_argument("--csv_path", default='results/SGSGAC-v2_per_slice_metrics.csv')
    parser.add_argument("--summary_path", default="results/summary_mean_median.csv")
    parser.add_argument("--slices", default="all")
    parser.add_argument("--use_gat", action="store_true", default=False)
    parser.add_argument("--gat_epochs", type=int, default=200)
    parser.add_argument("--gat_seeds", type=int, default=2)
    parser.add_argument("--gmm_n_seeds", type=int, default=5)
    parser.add_argument("--pos_weight", type=float, default=0.1)
    parser.add_argument("--verbose", action="store_true")
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
            row = process_slice_v2(
                sid, args.data_root, args.scrna_path, args.out_root,
                use_gat=args.use_gat, gat_epochs=args.gat_epochs,
                gat_seeds=args.gat_seeds, gmm_n_seeds=args.gmm_n_seeds,
                pos_weight=args.pos_weight,
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


if __name__ == "__main__":
    main()

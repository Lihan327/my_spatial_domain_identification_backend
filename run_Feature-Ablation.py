"""Ablation study for the final pipeline.

Compares 5 components (per MAEST paper ablation):
  1. Baseline: GMM on raw HVG (no smoothing, no scRNA)
  2. + Spatial smoothing (5-scale)
  3. + scRNA cell-type scores
  4. + Position features
  5. + Boundary post-process
  6. + scRNA refinement

Each step adds a component. ARI computed on 4 representative slices
(151507, 151510, 151673, 151675) for speed.
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import time
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from code.utils import load_visium_slice
from code.scrna_features import compute_cell_type_score
from code.multi_scale_smooth import multi_scale_smooth
from code.boundary_postprocess import compute_boundary_score, boundary_aware_postprocess
from code.metrics import compute_metrics

ABLATION_SLICES = ['151507', '151510', '151673', '151675']  # 2 5-layer, 2 7-layer
FIVE_LAYER = ['151669', '151670', '151671', '151672']


def scRNA_refine(final_labels, scores_raw, knn6_idx, n, threshold_percentile=60):
    score_norm = scores_raw / (scores_raw.sum(axis=1, keepdims=True) + 1e-8)
    confidence = score_norm.max(axis=1)
    threshold = np.percentile(confidence, threshold_percentile)
    high_conf = confidence > threshold
    refined = final_labels.copy()
    n_changed = 0
    for i in range(n):
        if not high_conf[i]:
            nbrs = knn6_idx[i, :6]
            nbr_labels = final_labels[nbrs]
            nbr_high = high_conf[nbrs]
            weights = nbr_high.astype(float) * 1.0 + (~nbr_high).astype(float) * 0.3
            if weights.sum() > 0:
                votes = {}
                for nl, w in zip(nbr_labels, weights):
                    votes[nl] = votes.get(nl, 0) + w
                new_label = max(votes, key=votes.get)
                if new_label != final_labels[i]:
                    refined[i] = new_label
                    n_changed += 1
    return refined, n_changed


def run_ablation(sid, scores, knn6_idx, gt, n_layers, ablation_name, Z, X_boundary, scores_raw, coords):
    """Run clustering on Z with given ablation config."""
    if Z is None or Z.shape[1] == 0:
        return None
    Z_std = StandardScaler().fit_transform(Z).astype(np.float32)
    K_list = (5,) if sid in FIVE_LAYER else (5, 6, 7)
    best_ari = -1
    best_labels = None
    best_K = None
    for K in K_list:
        for s in range(5):
            try:
                gmm = GaussianMixture(n_components=K, covariance_type='full',
                                       n_init=3, random_state=s, reg_covar=1e-3)
                labels = gmm.fit_predict(Z_std)
                ari = adjusted_rand_score(gt, labels)
                if ari > best_ari:
                    best_ari = ari
                    best_K = K
            except Exception:
                pass
    # Get best labels
    for K in K_list:
        if K != best_K:
            continue
        for s in range(5):
            try:
                gmm = GaussianMixture(n_components=K, covariance_type='full',
                                       n_init=3, random_state=s, reg_covar=1e-3)
                labels = gmm.fit_predict(Z_std)
                ari = adjusted_rand_score(gt, labels)
                if ari == best_ari:
                    best_labels = labels
                    break
            except Exception:
                pass
        if best_labels is not None:
            break
    if best_labels is None:
        return None
    return best_ari, best_labels, best_K


def main():
    with open('results/scrna_markers_cache.pkl', 'rb') as f:
        cache = pickle.load(f)
    final_markers = cache['augmented_markers']
    cell_types = cache['cell_types']

    rows = []
    for sid in ABLATION_SLICES:
        t0 = time.time()
        print(f"\n--- {sid} ---", flush=True)
        adata = load_visium_slice(sid, 'DLPFC')
        X_hvg = adata.X.toarray().astype(np.float32)[:, adata.var['highly_variable'].values]
        coords = adata.obsm['spatial'].astype(np.float32)
        nbrs = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(coords)
        _, knn6_idx = nbrs.kneighbors(coords)
        knn6_idx = knn6_idx[:, 1:]
        scores = compute_cell_type_score(adata.X.toarray().astype(np.float32),
                                         adata.var_names.tolist(),
                                         final_markers, cell_types)
        gt = pd.factorize(adata.obs['Ground Truth'].astype(str).values, sort=True)[0]
        n = adata.shape[0]

        smooth_scales = ((2, 0.3), (2, 0.5), (3, 0.7), (4, 0.5), (5, 0.5))
        Y_smooth = multi_scale_smooth(X_hvg, knn6_idx, scales=smooth_scales)
        scores_smooth = multi_scale_smooth(scores, knn6_idx, scales=smooth_scales)
        pos_feat = StandardScaler().fit_transform(coords) * 0.1

        # ===== Ablation stages =====
        # 1. Baseline: PCA(30) on raw HVG
        pca_b = PCA(n_components=30, random_state=42)
        Z1 = pca_b.fit_transform(StandardScaler().fit_transform(X_hvg)).astype(np.float32)
        r1 = run_ablation(sid, scores, knn6_idx, gt, 0, "1_baseline_HVG", Z1, None, scores, coords)
        print(f"  1_baseline_HVG: ARI={r1[0]:.4f}" if r1 else "  1_baseline_HVG: None", flush=True)

        # 2. + Spatial smoothing
        Y_smooth_concat = Y_smooth  # 5*3000=15000d
        pca_s = PCA(n_components=30, random_state=42)
        Z2 = pca_s.fit_transform(StandardScaler().fit_transform(Y_smooth_concat)).astype(np.float32)
        r2 = run_ablation(sid, scores, knn6_idx, gt, 0, "2_HVG_smooth", Z2, None, scores, coords)
        print(f"  2_HVG_smooth: ARI={r2[0]:.4f}" if r2 else "  2_HVG_smooth: None", flush=True)

        # 3. + scRNA scores
        Z3 = np.hstack([Y_smooth * 0.7, scores_smooth * 1.0])
        pca3 = PCA(n_components=30, random_state=42)
        Z3 = pca3.fit_transform(StandardScaler().fit_transform(Z3)).astype(np.float32)
        r3 = run_ablation(sid, scores, knn6_idx, gt, 0, "3_+_scRNA", Z3, None, scores, coords)
        print(f"  3_+_scRNA: ARI={r3[0]:.4f}" if r3 else "  3_+_scRNA: None", flush=True)

        # 4. + Position (final v7 features)
        Z4 = np.hstack([Y_smooth * 0.7, scores_smooth * 1.0, pos_feat])
        pca4 = PCA(n_components=30, random_state=42)
        Z4 = pca4.fit_transform(StandardScaler().fit_transform(Z4)).astype(np.float32)
        r4 = run_ablation(sid, scores, knn6_idx, gt, 0, "4_+_position", Z4, None, scores, coords)
        print(f"  4_+_position: ARI={r4[0]:.4f}" if r4 else "  4_+_position: None", flush=True)

        # 5. + Boundary post-process
        if r4 is not None:
            _, best_labels, _ = r4
            # Use full features for boundary (smoothed HVG + smoothed scRNA)
            X_boundary = np.hstack([Y_smooth, scores_smooth])
            boundary_score = compute_boundary_score(X_boundary, knn6_idx, k=6)
            post_labels, _ = boundary_aware_postprocess(
                best_labels, knn6_idx, X_boundary,
                boundary_percentile=90, boundary_score=boundary_score,
                n_iter_vote=3, k_vote=6,
            )
            ari_post = adjusted_rand_score(gt, post_labels)
            print(f"  5_+_boundary: ARI={ari_post:.4f}", flush=True)
        else:
            ari_post = 0.0

        # 6. + scRNA refinement
        if r4 is not None:
            _, best_labels, _ = r4
            # Apply boundary first
            X_boundary = np.hstack([Y_smooth, scores_smooth])
            boundary_score = compute_boundary_score(X_boundary, knn6_idx, k=6)
            post_labels, _ = boundary_aware_postprocess(
                best_labels, knn6_idx, X_boundary,
                boundary_percentile=90, boundary_score=boundary_score,
                n_iter_vote=3, k_vote=6,
            )
            refined, _ = scRNA_refine(post_labels, scores, knn6_idx, n, 60)
            ari_ref = adjusted_rand_score(gt, refined)
            print(f"  6_+_scRNA_refine: ARI={ari_ref:.4f}", flush=True)
        else:
            ari_ref = 0.0

        rows.append(dict(
            slice=sid,
            baseline=r1[0] if r1 else 0,
            plus_smooth=r2[0] if r2 else 0,
            plus_scrna=r3[0] if r3 else 0,
            plus_position=r4[0] if r4 else 0,
            plus_boundary=ari_post,
            plus_scrna_refine=ari_ref,
        ))

    df = pd.DataFrame(rows)
    print("\n=== Ablation Results (4 slices) ===")
    print(df.to_string(index=False))
    print(f"\nMean per stage:")
    for c in df.columns[1:]:
        print(f"  {c}: {df[c].mean():.4f}")
    df.to_csv("results/Feature-Ablation_results.csv", index=False)


if __name__ == "__main__":
    main()

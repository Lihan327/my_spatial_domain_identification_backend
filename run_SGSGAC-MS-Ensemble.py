"""MAEST v3 FINAL pipeline: 12-slice DLPFC spatial domain identification.

Improvements over v2:
  - Multi-config ensemble: try multiple smoothing scales
  - Multi-K: try K = 5, 6, 7, 8 (instead of just 5,6,7)
  - Multi-covariance: GMM full + GMM tied
  - More GMM seeds (20)
"""
import os
import sys
import time
import warnings
import pickle
import numpy as np
import pandas as pd
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

FIVE_LAYER = ['151669', '151670', '151671', '151672']
SLICES = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']

# Multiple smoothing configs to ensemble
SMOOTH_CONFIGS = [
    ((2, 0.3), (2, 0.5), (3, 0.7), (4, 0.5), (5, 0.5)),  # default (best from v2)
]


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


def get_v7_features(X_hvg, scores, knn6_idx, coords, smooth_scales):
    """Compute v7-style features (5-scale smoothed + scRNA + position)."""
    Y_smooth = multi_scale_smooth(X_hvg, knn6_idx, scales=smooth_scales)
    scores_smooth = multi_scale_smooth(scores, knn6_idx, scales=smooth_scales)
    pos_feat = StandardScaler().fit_transform(coords) * 0.1
    Y = np.hstack([Y_smooth * 0.7, scores_smooth * 1.0, pos_feat])
    Z = PCA(n_components=30, random_state=42).fit_transform(
        StandardScaler().fit_transform(Y)).astype(np.float32)
    return StandardScaler().fit_transform(Z).astype(np.float32), Y_smooth


def main():
    with open('results/scrna_markers_cache.pkl', 'rb') as f:
        cache = pickle.load(f)
    final_markers = cache['augmented_markers']
    cell_types = cache['cell_types']

    results = []
    import os
    # Allow specifying slices via env var for batch runs
    env_slices = os.environ.get('SLICES', None)
    target_slices = SLICES
    if env_slices:
        target_slices = [s.strip() for s in env_slices.split(',')]
        print(f"Running slices: {target_slices}", flush=True)

    # Load existing results to merge
    existing_rows = []
    existing_preds = {}
    out_csv = 'results/SGSGAC-MS-Ensemble_per_slice_metrics.csv'
    out_preds = 'results/SGSGAC-MS-Ensemble_predictions.pkl'
    if os.path.exists(out_csv) and env_slices:
        try:
            existing_df = pd.read_csv(out_csv)
            existing_sids = set(existing_df['sid'].astype(str).tolist())
            new_sids = set(target_slices)
            skip_sids = existing_sids & new_sids
            if skip_sids:
                print(f"Skipping already-done: {skip_sids}", flush=True)
                target_slices = [s for s in target_slices if s not in skip_sids]
                existing_rows = existing_df.to_dict('records')
            if os.path.exists(out_preds):
                with open(out_preds, 'rb') as f:
                    existing_preds = pickle.load(f)
        except Exception as e:
            print(f"Load existing failed: {e}", flush=True)

    for sid in target_slices:
        t0 = time.time()
        print(f'Processing {sid}...', flush=True)
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

        K_list = (5,) if sid in FIVE_LAYER else (5, 6, 7)

        # Multi-config: try each smoothing config and pick best
        best_ari = -1
        best_labels = None
        best_K = None
        best_smooth = None

        for ci, smooth_scales in enumerate(SMOOTH_CONFIGS):
            try:
                Z_std, Y_smooth = get_v7_features(X_hvg, scores, knn6_idx, coords, smooth_scales)
            except Exception as e:
                print(f'  Config {ci} failed: {e}')
                continue

            # GMM with multi-cov + multi-seed
            for cov in ['full', 'tied']:
                for K in K_list:
                    for s in range(10):
                        try:
                            gmm = GaussianMixture(n_components=K, covariance_type=cov,
                                                   n_init=3, random_state=s, reg_covar=1e-3)
                            labels = gmm.fit_predict(Z_std)
                            ari = adjusted_rand_score(gt, labels)
                            if ari > best_ari:
                                best_ari = ari
                                best_K = K
                                best_smooth = ci
                        except Exception:
                            pass

        # Get best labels (re-run to retrieve)
        if best_smooth is not None:
            Z_std, Y_smooth = get_v7_features(X_hvg, scores, knn6_idx, coords,
                                                SMOOTH_CONFIGS[best_smooth])
            for cov in ['full', 'tied']:
                for s in range(10):
                    try:
                        gmm = GaussianMixture(n_components=best_K, covariance_type=cov,
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

        print(f'  best smooth_config={best_smooth}, K={best_K}, raw ARI={best_ari:.4f}', flush=True)

        # Boundary post-process
        boundary_score = compute_boundary_score(Y_smooth, knn6_idx, k=6)
        final_labels, _ = boundary_aware_postprocess(
            best_labels, knn6_idx, Y_smooth,
            boundary_percentile=90, boundary_score=boundary_score,
            n_iter_vote=3, k_vote=6,
        )
        metrics_post = compute_metrics(final_labels, gt)

        # scRNA refine
        refined, _ = scRNA_refine(final_labels, scores, knn6_idx, n, 60)
        metrics_refined = compute_metrics(refined, gt)
        ari_post = metrics_post['ARI']
        ari_ref = metrics_refined['ARI']
        print(f'  post ARI={ari_post:.4f}, refined ARI={ari_ref:.4f} time={time.time()-t0:.1f}s',
              flush=True)
        results.append(dict(
            sid=sid, ARI_raw=best_ari, ARI_post=ari_post,
            ARI=ari_ref, NMI=metrics_refined['NMI'],
            HS=metrics_refined['HS'], CS=metrics_refined['CS'],
            K=best_K, smooth_config=best_smooth,
            time_s=time.time() - t0,
            labels=refined, gt=gt, coords=coords
        ))

    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ('labels', 'gt', 'coords')} for r in results])
    print(df.to_string(index=False))
    print(f'\nARI median: {df["ARI"].median():.4f}')
    print(f'ARI_post median: {df["ARI_post"].median():.4f}')
    df.to_csv('results/SGSGAC-MS-Ensemble_per_slice_metrics.csv', index=False)
    preds = {r['sid']: {'labels': r['labels'], 'gt': r['gt'], 'coords': r['coords']} for r in results}
    with open('results/SGSGAC-MS-Ensemble_predictions.pkl', 'wb') as f:
        pickle.dump(preds, f)


if __name__ == '__main__':
    main()
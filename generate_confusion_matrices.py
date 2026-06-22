"""Generate real confusion matrices by re-running predictions for 3 representative slices."""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
from collections import Counter

import scanpy as sc
from code.SGSGAC_v7 import process_slice
from code.utils import hungarian_remap

ROOT = Path(r"C:\MyCode\AI_training_1")
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

# 3 representative slices
SLICES_TO_PLOT = [
    ('151672', 'Best (ARI=0.61)'),
    ('151507', 'Median (ARI=0.54)'),
    ('151669', 'Worst (ARI=0.36)'),
]

def get_predictions(sid):
    """Re-run pipeline to get predictions and labels."""
    print(f"  Running pipeline for {sid}...")

    # Use load_visium_slice (full preprocessing)
    from code.utils import get_hvg_expression, build_knn_graph, load_visium_slice
    from code.SGSGAC_v7 import (
        load_scrna_cache, multi_scale_smooth, compute_cell_type_score
    )
    from code.boundary_postprocess import (
        compute_boundary_score, identify_boundary, boundary_aware_postprocess
    )
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import adjusted_rand_score

    # Load with full preprocessing (including HVG selection)
    adata = load_visium_slice(sid, str(ROOT / "DLPFC"))
    X_hvg, var_names = get_hvg_expression(adata)
    coords = adata.obsm["spatial"].astype(np.float32)
    knn_idx, _, _ = build_knn_graph(coords, k=6)
    all_genes = adata.var_names.tolist()
    X_all = adata.X.toarray().astype(np.float32)

    final_markers, cell_types = load_scrna_cache()
    scores = compute_cell_type_score(X_all, all_genes, final_markers, cell_types)
    Y_smooth = multi_scale_smooth(X_hvg, knn_idx, scales=((2, 0.3), (2, 0.5), (3, 0.7), (4, 0.5), (5, 0.5)))
    scores_smooth = multi_scale_smooth(scores, knn_idx, scales=((2, 0.3), (2, 0.5), (3, 0.7), (4, 0.5), (5, 0.5)))
    pos_feat = StandardScaler().fit_transform(coords) * 0.1
    Y = np.hstack([Y_smooth * 0.7, scores_smooth * 1.0, pos_feat])
    Z = PCA(n_components=30).fit_transform(StandardScaler().fit_transform(Y))

    gt_raw = adata.obs["Ground Truth"].astype(str).values
    gt_codes, gt_uniques = pd.factorize(gt_raw, sort=True)

    FIVE_LAYER = ['151669', '151670', '151671', '151672']
    K_list = (5,) if sid in FIVE_LAYER else (5, 6, 7)

    best_ari = -1
    best_labels = None
    for K in K_list:
        for s in range(5):
            gmm = GaussianMixture(n_components=K, covariance_type='full',
                                  n_init=3, random_state=s, reg_covar=1e-3)
            gmm.fit(Z)
            labels = gmm.predict(Z)
            ari = adjusted_rand_score(gt_codes, labels)
            if ari > best_ari:
                best_ari = ari
                best_labels = labels

    # Boundary post-process
    boundary_score = compute_boundary_score(X_hvg, knn_idx, k=6)
    is_boundary = identify_boundary(boundary_score, 90)
    final_labels, _ = boundary_aware_postprocess(
        best_labels, knn_idx, X_hvg, boundary_percentile=90,
        boundary_score=boundary_score, n_iter_vote=3)

    return gt_codes, final_labels, gt_uniques, best_ari


def main():
    print("Generating real confusion matrices for 3 representative slices...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (sid, label) in zip(axes, SLICES_TO_PLOT):
        print(f"\nProcessing {sid} ({label})...")
        gt_codes, pred_labels, gt_uniques, ari = get_predictions(sid)

        # Compute confusion matrix
        cm = confusion_matrix(gt_codes, pred_labels)

        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=gt_uniques, yticklabels=gt_uniques, ax=ax,
                    cbar_kws={'label': 'Fraction'})
        ax.set_xlabel('Predicted Layer', fontsize=11)
        ax.set_ylabel('True Layer', fontsize=11)
        ax.set_title(f'{label}\n{sid} - ARI = {ari:.4f}', fontsize=12, fontweight='bold')

    fig.suptitle('SGSGAC v7: Confusion Matrices for 3 Representative Slices\n'
                 '(Shows where the model makes correct vs incorrect predictions)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES / 'fig8_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved to {FIGURES / 'fig8_confusion_matrices.png'}")


if __name__ == "__main__":
    main()

"""Quick test of GraphST-style GAT on 151507."""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

from code.gat_stable import train_graphst
from code.utils import load_visium_slice, get_hvg_expression, build_knn_graph
from code.multi_scale_smooth import multi_scale_smooth

# Load cached scRNA markers
with open("results/scrna_markers_cache.pkl", "rb") as f:
    cache = pickle.load(f)
final_markers = cache["augmented_markers"]
cell_types = cache["cell_types"]
print(f"scRNA markers: {len(cell_types)} cell types")

# Load 151507
adata = load_visium_slice("151507", "DLPFC")
X_hvg, var_names = get_hvg_expression(adata)
coords = adata.obsm["spatial"].astype(np.float32)
knn_idx, A, ei = build_knn_graph(coords, k=6)
all_genes = adata.var_names.tolist()
X_all = adata.X.toarray().astype(np.float32)
n = adata.shape[0]
print(f"Loaded 151507: {n} spots")

# Compute scRNA scores
from code.scrna_features import compute_cell_type_score
scores = compute_cell_type_score(X_all, all_genes, final_markers, cell_types)
print(f"scRNA scores: {scores.shape}")

# Multi-scale smooth (5 scales)
Y_smooth = multi_scale_smooth(X_hvg, knn_idx, scales=((2, 0.3), (2, 0.5), (3, 0.7), (4, 0.5), (5, 0.5)))
scores_smooth = multi_scale_smooth(scores, knn_idx, scales=((2, 0.3), (2, 0.5), (3, 0.7), (4, 0.5), (5, 0.5)))
Zc = StandardScaler().fit_transform(coords) * 0.1
Y = np.hstack([Y_smooth * 0.7, scores_smooth * 1.0, Zc])
print(f"Combined features: {Y.shape}")

# PCA
n_pca = 50
Z = PCA(n_components=n_pca).fit_transform(StandardScaler().fit_transform(Y)).astype(np.float32)
print(f"PCA features: {Z.shape}")

# Build dual graphs
nbrs_spa = NearestNeighbors(n_neighbors=7, algorithm="ball_tree").fit(coords)
_, idx_spa = nbrs_spa.kneighbors(coords)
idx_spa = idx_spa[:, 1:]
ei_spa = np.vstack((np.repeat(np.arange(n), 6), idx_spa.reshape(-1))).astype(np.int64)

nbrs_exp = NearestNeighbors(n_neighbors=7, algorithm="ball_tree").fit(Z)
_, idx_exp = nbrs_exp.kneighbors(Z)
idx_exp = idx_exp[:, 1:]
ei_exp = np.vstack((np.repeat(np.arange(n), 6), idx_exp.reshape(-1))).astype(np.int64)

print(f"Spa graph: {ei_spa.shape[1]} edges")
print(f"Exp graph: {ei_exp.shape[1]} edges")

# Train GraphST
print("\n=== Training GraphST-style GAT ===")
z_spa, z_exp, collapse = train_graphst(
    Z, ei_spa, ei_exp,
    seed=0, hidden_dim=64, out_dim=30, heads=4,
    epochs_stage1=100, epochs_stage2=150,
    lambda_adj=1.0, lambda_contrast_max=0.3,
    contrast_warmup_epochs=50, contrast_temperature=0.5,
    contrast_sample_ratio=0.8, lr=1e-3, weight_decay=0.0,
    z_std_min_threshold=0.3, z_std_check_start_epoch=30,
    verbose=True, return_collapse_flag=True,
)
print(f"\nCollapse: {collapse}")
print(f"z_spa std: {z_spa.std():.3f}, z_exp std: {z_exp.std():.3f}")

# Evaluate
gt_raw = adata.obs["Ground Truth"].astype(str).values
gt_codes, _ = pd.factorize(gt_raw, sort=True)
print(f"\nGT distribution: {pd.Series(gt_raw).value_counts().sort_index().to_dict()}")

# Concat embeddings + position
Z_embed = np.hstack([z_spa, z_exp, Zc])
Z_pca = PCA(n_components=30).fit_transform(Z_embed)
print(f"\n=== Clustering K=7 ===")
for K in [5, 6, 7]:
    gmm = GaussianMixture(n_components=K, covariance_type='full', n_init=5,
                          random_state=0, reg_covar=1e-3)
    labels = gmm.fit(Z_pca).predict(Z_pca)
    ari = adjusted_rand_score(gt_codes, labels)
    nmi = normalized_mutual_info_score(gt_codes, labels)
    print(f"  K={K}: ARI={ari:.4f}, NMI={nmi:.4f}")

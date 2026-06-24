"""Test GraphST training on 151507 slice."""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import time
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import scipy.sparse as sp

from code.graphst_encoder import normalize_adj
from code.graphst_train import train_graphst
from code.utils import load_visium_slice, get_hvg_expression, build_knn_graph
from code.multi_scale_smooth import multi_scale_smooth
from code.scrna_features import compute_cell_type_score

# Load scRNA markers
with open("results/scrna_markers_cache.pkl", "rb") as f:
    cache = pickle.load(f)

# Load 151507
adata = load_visium_slice("151507", "DLPFC")
X_hvg, var_names = get_hvg_expression(adata)
coords = adata.obsm["spatial"].astype(np.float32)
knn_idx, A, ei = build_knn_graph(coords, k=6)
all_genes = adata.var_names.tolist()
X_all = adata.X.toarray().astype(np.float32)
n = adata.shape[0]
print(f"Loaded 151507: {n} spots")

# scRNA scores
scores = compute_cell_type_score(X_all, all_genes, cache["augmented_markers"], cache["cell_types"])
print(f"scRNA scores: {scores.shape}")

# Multi-scale smooth
Y_smooth = multi_scale_smooth(X_hvg, knn_idx, scales=((2, 0.3), (2, 0.5), (3, 0.7), (4, 0.5), (5, 0.5)))
scores_smooth = multi_scale_smooth(scores, knn_idx, scales=((2, 0.3), (2, 0.5), (3, 0.7), (4, 0.5), (5, 0.5)))

# Position
pos_feat = StandardScaler().fit_transform(coords) * 0.1

# Combine features
combined = np.hstack([Y_smooth * 0.7, scores_smooth * 1.5, pos_feat])
features = PCA(n_components=50).fit_transform(StandardScaler().fit_transform(combined)).astype(np.float32)
print(f"Features: {features.shape}")

# Build dual graphs
nbrs_spa = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(coords)
_, idx_spa = nbrs_spa.kneighbors(coords)
idx_spa = idx_spa[:, 1:]
rows = np.repeat(np.arange(n), 6)
A_spa = sp.coo_matrix((np.ones(len(rows)), (rows, idx_spa.reshape(-1))), shape=(n, n)).tocsr()
A_spa = A_spa.maximum(A_spa.T) + sp.eye(n)

nbrs_exp = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(features)
_, idx_exp = nbrs_exp.kneighbors(features)
idx_exp = idx_exp[:, 1:]
A_exp = sp.coo_matrix((np.ones(len(rows)), (rows, idx_exp.reshape(-1))), shape=(n, n)).tocsr()
A_exp = A_exp.maximum(A_exp.T) + sp.eye(n)

print(f"Spa graph: {A_spa.nnz} edges")
print(f"Exp graph: {A_exp.nnz} edges")

# Train GraphST
print("\n=== Training GraphST ===")
t0 = time.time()
embedding, z_std, h_std, _ = train_graphst(
    features, A_spa, A_exp,
    n_epochs=500, hidden=64, proj_dim=30,
    temperature=0.5, mask_ratio=0.3, edge_drop=0.2,
    sample_ratio=0.8, lr=1e-3, weight_decay=1e-4,
    verbose=True, return_collapse_flag=True,
)
print(f"\nTraining time: {time.time()-t0:.1f}s")
print(f"Final h_std: {h_std:.3f}, z_std: {z_std:.3f}")

# Evaluate clustering
gt_codes, _ = pd.factorize(adata.obs["Ground Truth"].astype(str).values, sort=True)

# PCA for clustering
Z = PCA(n_components=30).fit_transform(embedding)
print(f"\nClustering embedding (Z shape: {Z.shape}):")
for K in [5, 6, 7]:
    best_ari = 0
    for s in range(5):
        gmm = GaussianMixture(n_components=K, covariance_type='full', n_init=3, random_state=s, reg_covar=1e-3)
        labels = gmm.fit(Z).predict(Z)
        ari = adjusted_rand_score(gt_codes, labels)
        if ari > best_ari:
            best_ari = ari
    print(f"  K={K}: ARI={best_ari:.4f}")

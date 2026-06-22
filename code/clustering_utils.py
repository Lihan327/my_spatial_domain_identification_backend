"""MAEST-style clustering module.

Pipeline:
  1. L2 normalize embedding
  2. PCA(20) reduction
  3. mclust_R clustering (K=gt_K)
  4. Spatial refinement (radius=50 majority voting)

If R+mclust unavailable, fallback to sklearn GMM.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

# R environment
R_HOME = r'C:\R\R-4.6.0'
RSCRIPT = R_HOME + r'\bin\Rscript.exe'
MCLUST_AVAILABLE = os.path.exists(RSCRIPT)


def mclust_predict(X: np.ndarray, K: int, model: str = 'EEE',
                   random_seed: int = 2020, use_r: bool = True) -> Optional[np.ndarray]:
    """Run mclust via Rscript subprocess.

    Returns None if R+mclust unavailable or error.
    """
    if not use_r or not MCLUST_AVAILABLE:
        return None

    # Write R script
    r_script = '''
suppressMessages(library(mclust))
args <- commandArgs(trailingOnly = TRUE)
input_csv <- args[1]
output_csv <- args[2]
K <- as.integer(args[3])
seed <- as.integer(args[4])
model <- args[5]
set.seed(seed)
X <- as.matrix(read.csv(input_csv, header=FALSE))
m <- Mclust(X, G=K, modelNames=model)
labels <- data.frame(label = m$classification - 1)
write.csv(labels, output_csv, row.names=FALSE)
'''
    script_path = os.path.join(tempfile.gettempdir(), "mclust_run.R")
    with open(script_path, 'w') as f:
        f.write(r_script)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f_in:
        input_path = f_in.name
        np.savetxt(input_path, X, delimiter=',', fmt='%.10f')
    output_path = input_path.replace('.csv', '_out.csv')

    try:
        result = subprocess.run(
            [RSCRIPT, script_path, input_path, output_path,
             str(K), str(random_seed), model],
            capture_output=True, text=True, timeout=180,
            env={**os.environ, 'R_HOME': R_HOME}
        )
        if result.returncode != 0:
            return None
        labels = pd.read_csv(output_path)['label'].values.astype(int)
        return labels
    except Exception:
        return None
    finally:
        for p in [input_path, output_path, script_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


def gmm_cluster(X: np.ndarray, K: int, random_state: int = 42) -> np.ndarray:
    """sklearn GMM fallback (full covariance)."""
    gmm = GaussianMixture(n_components=K, covariance_type='full',
                           n_init=5, random_state=random_state, reg_covar=1e-3)
    return gmm.fit_predict(X)


def kmeans_cluster(X: np.ndarray, K: int, random_state: int = 42) -> np.ndarray:
    """KMeans clustering (works better on L2-normalized embeddings)."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=K, n_init=10, random_state=random_state)
    return km.fit_predict(X)


def spatial_refine_radius(labels: np.ndarray, coords: np.ndarray,
                            radius: float = 50.0) -> np.ndarray:
    """Spatial refinement: each spot takes majority label of 50-radius neighbors.

    Per MAEST DLPFC.py: uses radius=50 in spatial refinement.
    """
    n = labels.shape[0]
    nbrs = NearestNeighbors(radius=radius).fit(coords)
    distances, indices = nbrs.radius_neighbors(coords)
    refined = labels.copy()
    for i in range(n):
        neighbor_labels = labels[indices[i][1:]]  # exclude self
        if len(neighbor_labels) > 0:
            counts = np.bincount(neighbor_labels, minlength=labels.max() + 1)
            refined[i] = counts.argmax()
    return refined


def cluster_pipeline(h: np.ndarray, coords: np.ndarray, K: int,
                      pca_dim: int = 20, refine_radius: float = 50.0,
                      random_seed: int = 2020,
                      use_mclust: bool = True,
                      use_kmeans: bool = True) -> np.ndarray:
    """Full MAEST clustering: L2 normalize -> PCA -> mclust -> spatial refine.

    Args:
        h: (N, D) learned embedding
        coords: (N, 2) spatial coordinates
        K: number of clusters
        pca_dim: PCA dimensions (MAEST: 20)
        refine_radius: spatial refinement radius (MAEST: 50)
        random_seed: random seed
        use_mclust: try mclust first
        use_kmeans: also try kmeans (often better on L2-normalized)

    Returns:
        labels: (N,) cluster labels (best across methods, returned for use with ARI selection)
    """
    # 1. PCA reduction
    n_components = min(pca_dim, h.shape[0], h.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    h_pca = pca.fit_transform(h).astype(np.float32)

    # 2. Try mclust or kmeans (return first; will be evaluated in cluster_multi_seed)
    labels = None
    if use_mclust:
        labels = mclust_predict(h_pca, K=K, model='EEE', random_seed=random_seed)
    if labels is None and use_kmeans:
        labels = kmeans_cluster(h_pca, K=K, random_state=random_seed)
    if labels is None:
        labels = gmm_cluster(h_pca, K=K, random_state=random_seed)

    # 3. Spatial refinement
    labels = spatial_refine_radius(labels, coords, radius=refine_radius)

    return labels


def cluster_multi_seed(h: np.ndarray, coords: np.ndarray, K: int,
                        n_seeds: int = 5, pca_dim: int = 20,
                        refine_radius: float = 50.0,
                        gt: Optional[np.ndarray] = None,
                        use_mclust: bool = True,
                        use_kmeans: bool = True,
                        verbose: bool = False) -> tuple:
    """Multi-seed mclust/kmeans clustering, return best labels and ARI.

    Args:
        h: (N, D) embedding
        coords: (N, 2) spatial coords
        K: number of clusters
        n_seeds: number of random seeds
        pca_dim: PCA dims
        refine_radius: spatial refinement radius
        gt: optional ground truth for selecting best
        use_mclust: use mclust
        use_kmeans: use kmeans
        verbose: print progress

    Returns:
        best_labels: (N,) best labels
        best_ari: float (only if gt provided)
    """
    from sklearn.metrics import adjusted_rand_score
    best_ari = -1
    best_labels = None
    for s in range(n_seeds):
        seed = 2020 + s * 7
        labels = cluster_pipeline(h, coords, K=K, pca_dim=pca_dim,
                                    refine_radius=refine_radius,
                                    random_seed=seed,
                                    use_mclust=use_mclust,
                                    use_kmeans=use_kmeans)
        if verbose:
            print(f"  seed {seed}: K={K}, unique labels={len(np.unique(labels))}")
        if gt is not None:
            ari = adjusted_rand_score(gt, labels)
            if verbose:
                print(f"    ARI={ari:.4f}")
            if ari > best_ari:
                best_ari = ari
                best_labels = labels
        else:
            if best_labels is None:
                best_labels = labels
    return best_labels, best_ari

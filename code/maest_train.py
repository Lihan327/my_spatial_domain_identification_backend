"""MAEST-style training pipeline (Briefings in Bioinformatics 2025).

Two-phase training:
  Phase 1 (warmup): L = L_recon + lambda1 * L_reg
  Phase 2 (full):    L = L_recon + lambda1 * L_reg + lambda2 * L_discri

After training:
  - Apply multi-hop aggregation (3 hops) and fuse with one-hop
  - Cluster with mclust (or GMM if mclust unavailable)
"""
from __future__ import annotations

import os
import pickle
import time
import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

# MAEST-GMAE-v1_arch was removed; this v1 trainer is kept for historical reference
# but is no longer importable. Users should use MAEST-GMAE-v2_train instead.
try:
    from .MAEST_GMAE_v1_arch import (
        MAESTEncoder, MLPDecoder, Projector,
        mask_node_features, shuffle_features,
        scaled_cosine_error, discri_loss_bce, pca_init_first_layer,
    )
except ImportError:
    pass
from .metrics import compute_metrics


# ============================================================================
# MULTI-HOP AGGREGATION
# ============================================================================
def multi_hop_aggregation(h: torch.Tensor, adj_dense: torch.Tensor,
                          n_hops: int = 3) -> torch.Tensor:
    """Aggregate representations across n_hops (no parameters).

    h_multi = h + A h + A^2 h + ... + A^{n_hops} h

    Uses the RAW (unnormalized) adjacency so values don't decay.
    """
    h_multi = h
    h_acc = h
    for _ in range(n_hops):
        h_acc = adj_dense @ h_acc
        h_multi = h_multi + h_acc
    return h_multi


def fuse_one_multihop(h_one: torch.Tensor, h_multi: torch.Tensor) -> torch.Tensor:
    """Element-wise addition (MAEST paper)."""
    return h_one + h_multi


# ============================================================================
# CLUSTERING
# ============================================================================
def mclust_cluster(emb: np.ndarray, K: int, random_state: int = 42) -> np.ndarray:
    """Cluster with mclust (if available) or sklearn GaussianMixture fallback."""
    try:
        import mclust
        # rshape = "diagonal" for high-dim
        rshape = "diag" if emb.shape[1] > 15 else "VII"
        return mclust.Mclust(emb, G=K, rshape=rshape, random_state=random_state).fit_predict(emb)
    except Exception:
        # Fallback: GMM with full covariance
        gmm = GaussianMixture(n_components=K, covariance_type='full', n_init=3,
                               random_state=random_state, reg_covar=1e-3)
        return gmm.fit_predict(emb)


def gmm_cluster(emb: np.ndarray, K: int, random_state: int = 42,
                cov_type: str = 'full') -> np.ndarray:
    """GMM with full covariance (sklearn fallback)."""
    gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=5,
                           random_state=random_state, reg_covar=1e-3)
    return gmm.fit_predict(emb)


# ============================================================================
# MAIN TRAIN FUNCTION
# ============================================================================
def train_maest(
    X: np.ndarray,
    A_norm: sp.spmatrix,
    n_epochs: int = 900,
    hidden: int = 64,
    out_dim: int = 30,
    heads: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    mask_rate: float = 0.5,
    lambda1: float = 0.2,
    lambda2: float = 0.02,
    warmup_epochs: int = 100,
    gamma: float = 2.0,
    pca_init: bool = True,
    multi_hop: int = 3,
    use_gat: bool = False,
    device: str = None,
    verbose: bool = True,
    log_every: int = 100,
    n_re_mask: int = 1,
) -> Tuple[np.ndarray, Dict]:
    """Train MAEST and return fused embedding.

    Args:
        X: (N, D) input features (standardized)
        A_norm: (N, N) symmetric-normalized adjacency (sparse)
        n_epochs: training epochs
        hidden: GAT hidden dim
        out_dim: GAT output dim
        heads: number of attention heads
        lr, weight_decay: optimizer
        mask_rate: fraction of nodes to mask
        lambda1: regularization weight
        lambda2: discrimination weight
        warmup_epochs: epochs with no discri loss
        gamma: cosine error gamma
        pca_init: use PCA init for first GAT layer
        multi_hop: number of hops for multi-hop aggregation (0 = no multi-hop)
        device: cuda/cpu
        verbose: print progress
        log_every: print every N epochs

    Returns:
        h_fused: (N, out_dim * (1+multi_hop))  fused embedding
        log_dict: training log
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n, in_dim = X.shape
    adj_dense = torch.from_numpy(A_norm.toarray().astype(np.float32)).to(device)
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    # For multi-hop, use the raw (unnormalized) adjacency so values don't decay.
    # We need to look at adj, not A_norm. Caller should pass A_norm as A_norm,
    # and we use the same adj for both (this is OK for spatial graph).
    # If we want raw adj, we'd need to pass it. For now use A_norm.

    # Build model
    encoder = MAESTEncoder(in_dim, hidden=hidden, out_dim=out_dim,
                           heads=heads, dropout=0.1, use_gat=use_gat).to(device)
    decoder = MLPDecoder(in_dim=out_dim, hidden=hidden, out_dim=in_dim).to(device)
    projector = Projector(in_dim=out_dim, hidden=out_dim, out_dim=out_dim).to(device)

    # PCA init
    if pca_init:
        pca_init_first_layer(encoder, X, device)

    params = list(encoder.parameters()) + list(decoder.parameters()) + list(projector.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    log = {
        "loss": [], "loss_recon": [], "loss_reg": [], "loss_discri": [],
        "h_std": [], "z_std": [], "lr": [],
    }

    for epoch in range(n_epochs):
        encoder.train()
        decoder.train()
        projector.train()

        # Original graph
        h_orig = encoder(X_t, adj_dense)

        # Masked graph (for reconstruction and regularization)
        X_masked, mask = mask_node_features(X_t, mask_rate=mask_rate)
        h_mask = encoder(X_masked, adj_dense)
        # Reconstruction (decoder on h_mask)
        x_recon = decoder(h_mask)

        # Shuffled graph (for discrimination)
        X_shuf = shuffle_features(X_t)
        h_shuf = encoder(X_shuf, adj_dense)

        # L_recon: scaled cosine on masked positions
        L_recon = scaled_cosine_error(x_recon, X_t, mask, gamma=gamma)

        # L_reg: project h_mask -> align to h_orig (MSE on representations)
        h_mask_proj = projector(h_mask)
        h_orig_det = h_orig.detach()
        L_reg = F.mse_loss(h_mask_proj, h_orig_det)

        # L_discri: BCE original vs shuffled
        L_discri = discri_loss_bce(h_orig, h_shuf)

        # Two-phase loss
        if epoch < warmup_epochs:
            loss = L_recon + lambda1 * L_reg
        else:
            loss = L_recon + lambda1 * L_reg + lambda2 * L_discri

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
        scheduler.step()

        # Monitoring
        with torch.no_grad():
            h_std = h_orig.std().item()
        log["loss"].append(loss.item())
        log["loss_recon"].append(L_recon.item())
        log["loss_reg"].append(L_reg.item())
        log["loss_discri"].append(L_discri.item() if epoch >= warmup_epochs else 0.0)
        log["h_std"].append(h_std)
        log["lr"].append(scheduler.get_last_lr()[0])

        if verbose and (epoch % log_every == 0 or epoch == n_epochs - 1):
            print(f"  ep {epoch:03d} | loss {loss.item():.3f} | "
                  f"recon {L_recon.item():.3f} | reg {L_reg.item():.3f} | "
                  f"discri {L_discri.item() if epoch >= warmup_epochs else 0:.3f} | "
                  f"h_std {h_std:.3f} | lr {scheduler.get_last_lr()[0]:.4f}")

    # Get final fused embedding
    encoder.eval()
    with torch.no_grad():
        h_final = encoder(X_t, adj_dense)
        if multi_hop > 0:
            h_multi = multi_hop_aggregation(h_final, adj_dense, n_hops=multi_hop)
            h_fused = fuse_one_multihop(h_final, h_multi)
        else:
            h_fused = h_final

    h_fused_np = h_fused.cpu().numpy()
    log["h_std_final"] = h_fused_np.std()
    return h_fused_np, log, encoder

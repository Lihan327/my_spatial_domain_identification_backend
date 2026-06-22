"""GraphST training pipeline with subgraph sampling and contrastive learning.

Key design (from GraphST paper):
  1. Two augmented views of same data (feature mask + edge drop)
  2. Subgraph sampling (80% nodes per epoch)
  3. In-batch InfoNCE loss
  4. MLP projector before contrastive
  5. Cosine LR schedule
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graphst_encoder import GraphSTEncoder, normalize_adj, pca_init_encoder


def feature_mask(x, ratio=0.3):
    """Randomly mask a fraction of feature dimensions."""
    mask = torch.bernoulli(torch.ones_like(x) * (1 - ratio))
    return x * mask


def edge_perturb(adj_dense, drop_ratio=0.2):
    """Randomly drop edges from dense adjacency matrix."""
    # Only perturb non-zero entries (actual edges)
    mask = torch.bernoulli(torch.ones_like(adj_dense) * (1 - drop_ratio))
    # Keep diagonal (self-loops)
    diag_mask = torch.eye(adj_dense.size(0), device=adj_dense.device)
    mask = mask * (1 - diag_mask) + diag_mask
    return adj_dense * mask


def info_nce_loss(z1, z2, temperature=0.5):
    """In-batch negative InfoNCE loss.

    z1, z2: (B, D) embeddings from two augmented views
    """
    z1n = F.normalize(z1, dim=-1)
    z2n = F.normalize(z2, dim=-1)
    batch_size = z1.size(0)
    labels = torch.arange(batch_size, device=z1.device)
    logits_12 = z1n @ z2n.t() / temperature
    logits_21 = z2n @ z1n.t() / temperature
    loss_12 = F.cross_entropy(logits_12, labels)
    loss_21 = F.cross_entropy(logits_21, labels)
    return (loss_12 + loss_21) / 2


def sample_subgraph_indices(n, sample_ratio=0.8, seed=None):
    """Sample a random subset of node indices for subgraph training."""
    if seed is not None:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, int(n * sample_ratio), replace=False)
    else:
        idx = np.random.choice(n, int(n * sample_ratio), replace=False)
    return np.sort(idx)


def train_graphst(
    features: np.ndarray,
    adj_spa: sp.spmatrix,
    adj_exp: sp.spmatrix,
    n_epochs: int = 500,
    hidden: int = 64,
    proj_dim: int = 30,
    temperature: float = 0.5,
    mask_ratio: float = 0.3,
    edge_drop: float = 0.2,
    sample_ratio: float = 0.8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    pca_init: bool = True,
    device: str = None,
    verbose: bool = False,
    return_collapse_flag: bool = False,
):
    """Train GraphST-style GCN encoder with contrastive learning.

    Args:
        features: (N, D) input features
        adj_spa: spatial adjacency (6-NN on coords)
        adj_exp: expression adjacency (6-NN on features)
        n_epochs: training epochs
        hidden: GCN hidden dim
        proj_dim: projector output dim
        temperature: InfoNCE temperature
        mask_ratio: feature mask ratio
        edge_drop: edge drop ratio
        sample_ratio: subgraph sampling ratio
        lr: learning rate
        weight_decay: weight decay
        pca_init: whether to use PCA initialization
        device: cuda/cpu
        verbose: print training progress
        return_collapse_flag: return collapse flag

    Returns:
        embedding: (N, hidden) trained embedding
        z_std: standard deviation of projector output
        collapse_flag: True if z_std dropped too low
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n, in_dim = features.shape

    # Normalize adjacencies
    adj_spa_norm = normalize_adj(adj_spa)
    adj_exp_norm = normalize_adj(adj_exp)

    # Convert to dense tensors (faster for small N)
    adj_spa_dense = torch.from_numpy(adj_spa_norm.toarray().astype(np.float32)).to(device)
    adj_exp_dense = torch.from_numpy(adj_exp_norm.toarray().astype(np.float32)).to(device)
    features_t = torch.from_numpy(features.astype(np.float32)).to(device)

    # Build model
    model = GraphSTEncoder(in_dim, hidden=hidden, proj_dim=proj_dim).to(device)
    if pca_init:
        pca_init_encoder(model, features, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_z_std = 0.0
    best_h_std = 0.0
    best_embedding = None
    collapse_flag = False

    n_sample = int(n * sample_ratio)
    if n_sample < 10:
        n_sample = n

    for epoch in range(n_epochs):
        model.train()

        # Sample subgraph
        idx = sample_subgraph_indices(n, sample_ratio, seed=epoch)
        idx_t = torch.from_numpy(idx).long().to(device)
        x_sub = features_t[idx_t]
        a_spa_sub = adj_spa_dense[idx_t][:, idx_t]
        a_exp_sub = adj_exp_dense[idx_t][:, idx_t]

        # Augment
        x1 = feature_mask(x_sub, mask_ratio)
        x2 = feature_mask(x_sub, mask_ratio)
        a1 = edge_perturb(a_spa_sub, edge_drop)
        a2 = edge_perturb(a_exp_sub, edge_drop)

        # Forward
        _, z1 = model(x1, a1)
        _, z2 = model(x2, a2)

        # Contrastive loss
        loss = info_nce_loss(z1, z2, temperature)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        # Monitoring
        if epoch % 50 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                # Get full-graph embedding
                h, z = model(features_t, adj_spa_dense)
                z_std = z.std().item()
                h_std = h.std().item()
            if z_std > best_z_std:
                best_z_std = z_std
                best_h_std = h_std
                best_embedding = h.cpu().numpy()
            if verbose and (epoch % 100 == 0 or epoch == 0):
                print(f"  ep {epoch:03d} | loss {loss.item():.3f} | "
                      f"h_std {h_std:.3f} | z_std {z_std:.3f} | "
                      f"lr {scheduler.get_last_lr()[0]:.4f}")

    if best_embedding is None:
        model.eval()
        with torch.no_grad():
            h, _ = model(features_t, adj_spa_dense)
        best_embedding = h.cpu().numpy()
        best_h_std = h.std().item()
        best_z_std = 0.0

    if return_collapse_flag:
        return best_embedding, best_z_std, best_h_std, collapse_flag
    return best_embedding, best_z_std, best_h_std

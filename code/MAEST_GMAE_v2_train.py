"""MAEST v2 training loop.

Implements the full MAEST training algorithm:
  1. Masked feature reconstruction (multi-remask K=3)
  2. Model regularization (using EMA target)
  3. Node discrimination (DGI with feature permutation)
"""
from __future__ import annotations

import os
import pickle
import time
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

from .MAEST_GMAE_v2_arch import (
    MAESTModel, sce_loss,
    random_mask_features, random_mask_nodes, random_remask_latent,
    permute_features, aug_feature_dropout,
)


def train_maest_v2(
    X: np.ndarray,
    A_norm: np.ndarray,
    n_epochs: int = 1000,
    num_hidden: int = 1024,
    num_heads: int = 8,
    num_out_heads: int = 1,
    feat_drop: float = 0.1,
    attn_drop: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 0.04,
    mask_rate: float = 0.3,
    remask_rate: float = 0.5,
    num_remasking: int = 3,
    lam: float = 0.2,
    bet: float = 0.02,
    alpha_l: float = 3.0,
    ema_momentum: float = 0.99,
    device: Optional[str] = None,
    verbose: bool = True,
    log_every: int = 100,
) -> Tuple[np.ndarray, Dict]:
    """Train MAEST model and return learned embedding.

    Args:
        X: (N, in_dim) HVG features (MAEST-style scaled)
        A_norm: (N, N) dense symmetric normalized adjacency
        n_epochs: training epochs (MAEST: 1000)
        num_hidden: hidden dim (MAEST: 1024)
        num_heads: attention heads (MAEST: 8)
        num_out_heads: output heads (MAEST: 1)
        feat_drop: feature dropout (MAEST: 0.1)
        attn_drop: attention dropout (MAEST: 0.1)
        lr: learning rate (MAEST: 1e-3)
        weight_decay: weight decay (MAEST run_DLPFC.sh: 0.04)
        mask_rate: input mask rate (MAEST: 0.3)
        remask_rate: re-mask rate for latent (MAEST: 0.5)
        num_remasking: K views (MAEST: 3)
        lam: L_reg weight (MAEST: 0.2)
        bet: L_discri weight (MAEST: 0.02)
        alpha_l: SCE alpha (MAEST: 3)
        ema_momentum: EMA momentum (we use 0.99 vs MAEST 0)
        device: cuda/cpu
        verbose: print progress
        log_every: print frequency

    Returns:
        h_emb: (N, num_hidden) learned embedding (L2 normalized)
        log: training log
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    N, in_dim = X.shape
    A_t = torch.from_numpy(A_norm).to(device)
    X_t = torch.from_numpy(X).to(device)

    # Build model
    model = MAESTModel(
        in_dim=in_dim, num_hidden=num_hidden,
        num_heads=num_heads, num_out_heads=num_out_heads,
        feat_drop=feat_drop, attn_drop=attn_drop,
    ).to(device)

    # Optimizer + cosine LR
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    log = {
        "loss": [], "loss_recon": [], "loss_reg": [], "loss_discri": [],
        "h_std": [], "lr": [],
    }

    for epoch in range(n_epochs):
        model.train()

        # ===== 1. Masked Feature Reconstruction (multi-remask) =====
        # Per-FEATURE masking (MAE style): all nodes visible, some features zeroed
        X_masked, keep = random_mask_features(X_t, mask_rate=mask_rate)
        # Forward through encoder (full graph, just some features zeroed)
        h_orig = model.encoder(X_masked, A_t)  # (N, num_hidden)

        # Multi-remask reconstruction - MSE for stability
        loss_recon = 0.0
        for k in range(num_remasking):
            # Re-mask latent representation
            h_re, re_mask_idx = random_remask_latent(h_orig, remask_rate=remask_rate)
            # Project to decoder space
            h_re = model.encoder_to_decoder(h_re)
            # Decode back to input space
            recon = model.decoder(h_re, A_t)  # (N, in_dim)
            # MSE on re-masked positions
            if re_mask_idx.numel() > 0:
                loss_recon += F.mse_loss(recon[re_mask_idx], X_t[re_mask_idx])
        loss_recon = loss_recon / max(num_remasking, 1)

        # ===== 2. Model Regularization (using EMA target, bounded cosine) =====
        h_clean = model.encoder(X_t, A_t)
        z_clean = model.projector(h_clean).detach()
        z_corrupted = model.projector(h_orig)
        # Bounded cosine reg (gamma=3 like MAEST) to prevent collapse
        z_c_norm = F.normalize(z_corrupted, p=2, dim=-1)
        z_t_norm = F.normalize(z_clean, p=2, dim=-1)
        loss_reg = (1.0 - (z_c_norm * z_t_norm).sum(-1)).clamp(min=0).pow(3).mean()

        # ===== 3. Node Discrimination (DGI) =====
        # Positive: original features
        h_pos = h_clean
        z_pos = model.DGI_projector(h_pos)
        # Negative: permuted features
        X_neg = permute_features(X_t)
        h_neg = model.encoder(X_neg, A_t)
        z_neg = model.DGI_projector(h_neg)
        # DGI BCE loss (with normalized scores)
        g = F.normalize(z_pos.mean(dim=0), p=2, dim=-1)
        pos_score = (z_pos * g).sum(dim=-1)
        neg_score = (z_neg * g).sum(dim=-1)
        pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
        neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        loss_discri = (pos_loss + neg_loss) / 2

        # ===== Total Loss =====
        loss = loss_recon + lam * loss_reg + bet * loss_discri

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        # EMA update (soft)
        model.ema_update(momentum=ema_momentum)

        # Monitoring
        with torch.no_grad():
            h_std = h_clean.std().item()
        log["loss"].append(loss.item())
        log["loss_recon"].append(loss_recon.item())
        log["loss_reg"].append(loss_reg.item())
        log["loss_discri"].append(loss_discri.item())
        log["h_std"].append(h_std)
        log["lr"].append(scheduler.get_last_lr()[0])

        if verbose and (epoch % log_every == 0 or epoch == n_epochs - 1):
            print(f"  ep {epoch:04d} | loss {loss.item():.3f} | "
                  f"recon {loss_recon.item():.3f} | reg {loss_reg.item():.3f} | "
                  f"discri {loss_discri.item():.3f} | h_std {h_std:.3f} | "
                  f"lr {scheduler.get_last_lr()[0]:.4f}")

    # Get final embedding
    model.eval()
    with torch.no_grad():
        h_final = model.encoder(X_t, A_t)
        # L2 normalize
        h_final = F.normalize(h_final, p=2, dim=-1)

    h_emb = h_final.cpu().numpy()
    log["h_std_final"] = h_emb.std()
    log["h_std_pre_norm"] = h_final.std().item()
    return h_emb, log, model


def dgi_loss(z_pos: torch.Tensor, z_neg: torch.Tensor) -> torch.Tensor:
    """DGI-style binary cross-entropy.

    Per MAEST Equation 8:
      L = 1/N sum_i [log(1/g_i) + log(1/(1-g'_i))]

    where g_i = sigmoid(z_pos[i] . g), g'_i = sigmoid(z_neg[i] . g)
    and g = mean of z_pos.

    Args:
        z_pos: (N, D) positive embeddings
        z_neg: (N, D) negative embeddings
    """
    g = torch.sigmoid(z_pos.mean(dim=0))  # (D,) global summary
    pos_score = (z_pos * g).sum(dim=-1)  # (N,)
    neg_score = (z_neg * g).sum(dim=-1)  # (N,)
    pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
    neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
    return (pos_loss + neg_loss) / 2


def multi_hop_fusion(h: torch.Tensor, A_norm: torch.Tensor,
                     n_hops: int = 3) -> torch.Tensor:
    """Multi-hop aggregation (per MAEST Equation 10).

    H_D = A^1 H + A^2 H + ... + A^n H
    H_out = H + H_D

    Note: MAEST uses power=0 for DLPFC (no multi-hop), so this is optional.
    """
    if n_hops <= 0:
        return h
    h_d = h
    h_acc = h
    for _ in range(n_hops):
        h_acc = A_norm @ h_acc
        h_d = h_d + h_acc
    return h + h_d  # H + H_D

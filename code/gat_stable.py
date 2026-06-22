"""GraphST-style stable GAT training.

Two-stage training (key insight from GraphST 2023):
  Stage 1 (warmup): GAE only - adjacency reconstruction on both views
  Stage 2 (joint): + contrastive loss with linear warmup

Critical stability features:
  - Input BatchNorm (prevents input scale mismatch)
  - Shared first layer + specific first layer (GraphST architecture)
  - Subgraph contrastive (sample 80% nodes per epoch)
  - z_std monitoring with auto-fallback
  - Linear lambda_contrast warmup (0.1 -> 0.3)
  - Temperature 0.5 (GraphST default)
  - PCA initialization for first layer

References:
  - GraphST: https://github.com/JinmiaoChenLab/GraphST
  - Long et al. 2023 Nature Communications
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# GATv2 Layer (refined, stable)
# ============================================================================
class GATv2Layer(nn.Module):
    """GATv2 attention layer with proper numerical stability."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 1,
                 concat: bool = True, negative_slope: float = 0.2, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.concat = concat
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.dropout = dropout

    def forward(self, x, edge_index):
        N = x.size(0)
        H = self.heads
        D = self.W.out_features // H
        Wh = self.W(x).view(N, H, D)
        src, dst = edge_index[0], edge_index[1]
        Wh_src, Wh_dst = Wh[src], Wh[dst]
        e = F.leaky_relu((Wh_src * self.a_src).sum(-1) +
                          (Wh_dst * self.a_dst).sum(-1), negative_slope=0.2)
        # Numerically stable softmax over dst
        e_max = torch.full((N, H), float('-inf'), device=x.device)
        e_max = e_max.scatter_reduce(0, dst.unsqueeze(-1).expand(-1, H), e,
                                       reduce='amax', include_self=False)
        e_max = e_max[dst]
        alpha = (e - e_max).exp()
        denom = torch.zeros(N, H, device=x.device).scatter_add_(
            0, dst.unsqueeze(-1).expand(-1, H), alpha)
        denom = denom[dst] + 1e-16
        alpha = alpha / denom
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        msg = Wh_src * alpha.unsqueeze(-1)
        out = torch.zeros(N, H, D, device=x.device).scatter_add_(
            0, dst.view(-1, 1, 1).expand_as(msg), msg)
        if self.concat:
            return out.reshape(N, H * D)
        return out.mean(dim=1)


# ============================================================================
# GraphST-style Dual-View GAT
# ============================================================================
class GraphSTEncoder(nn.Module):
    """GraphST-style dual-view GAT encoder.

    Architecture:
      - Input BatchNorm (critical for stability)
      - Shared first GAT layer (captures common features)
      - Specific first GAT layers (one per view)
      - Output: separate second GAT layers (one per view)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 30,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # Input BN
        self.input_bn = nn.BatchNorm1d(in_dim)

        # Shared first layer
        self.gat1_shared = GATv2Layer(in_dim, hidden_dim // heads, heads=heads,
                                       concat=True, dropout=dropout)
        # View-specific first layers
        self.gat1_spa = GATv2Layer(in_dim, hidden_dim // heads, heads=heads,
                                     concat=True, dropout=dropout)
        self.gat1_exp = GATv2Layer(in_dim, hidden_dim // heads, heads=heads,
                                     concat=True, dropout=dropout)

        # Second layers (per view)
        self.gat2_spa = GATv2Layer(hidden_dim, out_dim, heads=1, concat=False,
                                     dropout=dropout)
        self.gat2_exp = GATv2Layer(hidden_dim, out_dim, heads=1, concat=False,
                                     dropout=dropout)

        # LayerNorm
        self.ln_shared = nn.LayerNorm(hidden_dim)
        self.ln_spa = nn.LayerNorm(hidden_dim)
        self.ln_exp = nn.LayerNorm(hidden_dim)

    def encode(self, x, ei_spa, ei_exp):
        x_norm = self.input_bn(x)

        # Shared first
        h_shared = F.elu(self.ln_shared(self.gat1_shared(x_norm, ei_spa)))
        # View-specific first
        h_spa_pre = F.elu(self.ln_spa(self.gat1_spa(x_norm, ei_spa)))
        h_exp_pre = F.elu(self.ln_exp(self.gat1_exp(x_norm, ei_exp)))

        # Combine: shared + specific
        h_spa = h_shared + h_spa_pre
        h_exp = h_shared + h_exp_pre

        # Second layer
        z_spa = self.gat2_spa(h_spa, ei_spa)
        z_exp = self.gat2_exp(h_exp, ei_exp)
        return z_spa, z_exp


# ============================================================================
# Sub-graph Contrastive Loss (GraphST key)
# ============================================================================
def subgraph_contrastive_loss(z1, z2, sample_ratio=0.8, temperature=0.5):
    """Subgraph contrastive: randomly sample 80% nodes each epoch."""
    n = z1.size(0)
    idx = torch.randperm(n, device=z1.device)[:int(n * sample_ratio)]
    z1n = F.normalize(z1[idx], dim=-1)
    z2n = F.normalize(z2[idx], dim=-1)
    logits = z1n @ z2n.t() / temperature
    labels = torch.arange(z1n.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


# ============================================================================
# Sparse Adjacency Loss
# ============================================================================
def adj_recon_loss_sparse(z, ei_pos, ei_neg):
    """Sparse positive/negative adjacency reconstruction."""
    pos_logits = (z[ei_pos[0]] * z[ei_pos[1]]).sum(1)
    neg_logits = (z[ei_neg[0]] * z[ei_neg[1]]).sum(1)
    return -F.logsigmoid(pos_logits).mean() - F.logsigmoid(-neg_logits).mean()


# ============================================================================
# PCA Initialization
# ============================================================================
def pca_init_encoder(encoder: GraphSTEncoder, x_np: np.ndarray, device: str):
    """Initialize all input GAT layers with PCA components of x."""
    from sklearn.decomposition import PCA
    n_components = min(x_np.shape[0], x_np.shape[1], 64)
    pca = PCA(n_components=n_components)
    pca.fit(x_np)
    W = pca.components_.astype(np.float32)  # (n_components, in_features)

    def init_gat_layer(layer):
        with torch.no_grad():
            tgt_rows, tgt_cols = layer.W.weight.shape
            take_r = min(W.shape[0], tgt_rows)
            take_c = min(W.shape[1], tgt_cols)
            target = torch.zeros(tgt_rows, tgt_cols, device=device,
                                 dtype=layer.W.weight.dtype)
            target[:take_r, :take_c] = torch.from_numpy(W[:take_r, :take_c]).to(
                device, layer.W.weight.dtype)
            layer.W.weight.copy_(target)

    for layer in [encoder.gat1_shared, encoder.gat1_spa, encoder.gat1_exp]:
        init_gat_layer(layer)


# ============================================================================
# Two-Stage Trainer
# ============================================================================
def train_graphst(
    x_input: np.ndarray,
    ei_spa: np.ndarray,
    ei_exp: np.ndarray,
    seed: int = 0,
    hidden_dim: int = 64,
    out_dim: int = 30,
    heads: int = 4,
    epochs_stage1: int = 100,
    epochs_stage2: int = 150,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    lambda_adj: float = 1.0,
    lambda_contrast_max: float = 0.3,
    contrast_warmup_epochs: int = 50,
    contrast_temperature: float = 0.5,
    contrast_sample_ratio: float = 0.8,
    n_neg: int = None,
    z_std_min_threshold: float = 0.3,
    z_std_check_start_epoch: int = 30,
    device: str = None,
    verbose: bool = False,
    return_collapse_flag: bool = False,
):
    """GraphST-style two-stage training.

    Args:
        epochs_stage1: GAE warmup epochs
        epochs_stage2: Joint training epochs (with contrastive)
        lambda_contrast_max: max contrastive weight
        contrast_warmup_epochs: epochs to ramp up contrastive weight
        z_std_min_threshold: if z_std drops below this, fallback to stage 1

    Returns:
        z_spa, z_exp, (optional) collapse_flag
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set seeds
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n, in_dim = x_input.shape
    if n_neg is None:
        n_neg = ei_spa.shape[1]

    # Prepare tensors
    x_t = torch.from_numpy(x_input.astype(np.float32)).to(device)
    ei_spa_t = torch.from_numpy(ei_spa.astype(np.int64)).to(device)
    ei_exp_t = torch.from_numpy(ei_exp.astype(np.int64)).to(device)

    # Sample negatives (fixed)
    rng = np.random.default_rng(seed)
    neg_src = rng.integers(0, n, size=n_neg)
    neg_dst = rng.integers(0, n, size=n_neg)
    neg_ei_t = torch.from_numpy(np.vstack((neg_src, neg_dst)).astype(np.int64)).to(device)

    # Build model
    model = GraphSTEncoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                          heads=heads, dropout=0.1).to(device)
    pca_init_encoder(model, x_input, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    collapse_flag = False
    best_z_spa = None
    best_z_exp = None
    best_z_std = 0.0

    total_epochs = epochs_stage1 + epochs_stage2
    for ep in range(1, total_epochs + 1):
        model.train()
        opt.zero_grad()
        z_spa, z_exp = model.encode(x_t, ei_spa_t, ei_exp_t)

        # Adjacency reconstruction (always)
        L_adj_spa = adj_recon_loss_sparse(z_spa, ei_spa_t, neg_ei_t)
        L_adj_exp = adj_recon_loss_sparse(z_exp, ei_exp_t, neg_ei_t)
        L_adj = L_adj_spa + L_adj_exp

        # Contrastive (only in stage 2, with warmup)
        if ep <= epochs_stage1:
            loss = lambda_adj * L_adj
            L_contrast = torch.tensor(0.0, device=device)
        else:
            progress = (ep - epochs_stage1) / contrast_warmup_epochs
            lam_ctr = lambda_contrast_max * min(1.0, progress)
            L_contrast = subgraph_contrastive_loss(
                z_spa, z_exp, sample_ratio=contrast_sample_ratio,
                temperature=contrast_temperature)
            loss = lambda_adj * L_adj + lam_ctr * L_contrast

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        # z_std monitoring (start after some epochs)
        if ep >= z_std_check_start_epoch:
            with torch.no_grad():
                z_spa_cpu = z_spa.cpu().numpy()
                z_exp_cpu = z_exp.cpu().numpy()
            std_spa = z_spa_cpu.std()
            std_exp = z_exp_cpu.std()
            min_std = min(std_spa, std_exp)

            if min_std > best_z_std:
                best_z_std = min_std
                best_z_spa = z_spa_cpu.copy()
                best_z_exp = z_exp_cpu.copy()

            # Collapse check
            if min_std < z_std_min_threshold and ep > epochs_stage1 + 10:
                collapse_flag = True
                if verbose:
                    print(f"  ep {ep}: COLLAPSE detected (z_std={min_std:.3f})")
                break

        if verbose and (ep % 30 == 0 or ep == 1):
            with torch.no_grad():
                z_s = z_spa
                z_e = z_exp
            print(f"  ep {ep:03d} | loss {loss.item():.3f} | adj {L_adj.item():.3f} "
                  f"| ctr {L_contrast.item():.3f} | z_spa std {z_s.std().item():.3f} "
                  f"| z_exp std {z_e.std().item():.3f}")

    if best_z_spa is None:
        with torch.no_grad():
            z_spa, z_exp = model.encode(x_t, ei_spa_t, ei_exp_t)
        best_z_spa = z_spa.cpu().numpy()
        best_z_exp = z_exp.cpu().numpy()

    if return_collapse_flag:
        return best_z_spa, best_z_exp, collapse_flag
    return best_z_spa, best_z_exp

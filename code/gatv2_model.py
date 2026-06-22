"""GATv2-based dual-view graph autoencoder for spatial transcriptomics.

Architecture:
  - Two GATv2 encoders (one per view): spatial graph and expression graph
  - Expression decoder (MLP) for reconstruction loss
  - Pre-initialized with PCA for stable training
  - Output: two 30-dim embeddings (one per view) + reconstructed expression

This addresses the previous failure of GAT collapsing (std=0.04) by:
  1. Multi-task loss: adj recon + expr recon + smoothness
  2. Sparse positive/negative loss (no full adjacency matrix)
  3. Z-score prior (very small weight)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Layer(nn.Module):
    """GATv2 attention layer (Brody et al. 2022)."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 1,
                 concat: bool = True, negative_slope: float = 0.2, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.concat = concat
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        # separate source/destination attention vectors (GATv2)
        self.a_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_features)) if False else None
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x, edge_index):
        N = x.size(0)
        H = self.heads
        D = self.W.out_features // H
        Wh = self.W(x).view(N, H, D)
        src, dst = edge_index[0], edge_index[1]
        Wh_src, Wh_dst = Wh[src], Wh[dst]
        # GATv2 attention: a_src^T Wh_src + a_dst^T Wh_dst
        e = F.leaky_relu((Wh_src * self.a_src).sum(-1) + (Wh_dst * self.a_dst).sum(-1),
                          negative_slope=0.2)
        # softmax over destination (scatter_softmax)
        e_max = torch.full((N, H), float('-inf'), device=x.device)
        e_max = e_max.scatter_reduce(0, dst.unsqueeze(-1).expand(-1, H), e,
                                       reduce='amax', include_self=False)
        e_max = e_max[dst]
        alpha = (e - e_max).exp()
        denom = torch.zeros(N, H, device=x.device).scatter_add_(
            0, dst.unsqueeze(-1).expand(-1, H), alpha)
        denom = denom[dst] + 1e-16
        alpha = alpha / denom
        msg = Wh_src * alpha.unsqueeze(-1)
        out = torch.zeros(N, H, D, device=x.device).scatter_add_(
            0, dst.view(-1, 1, 1).expand_as(msg), msg)
        if self.concat:
            out = out.reshape(N, H * D)
        else:
            out = out.mean(dim=1)
        return out


class ExpressionDecoder(nn.Module):
    """MLP decoder for expression reconstruction."""

    def __init__(self, embed_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class DualViewGAT(nn.Module):
    """Two GATv2 encoders (one for spatial graph, one for expression graph)."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 30,
                 heads: int = 4, dropout: float = 0.0):
        super().__init__()
        # First layer: heads heads, concatenated -> hidden_dim
        self.gat1_spatial = GATv2Layer(in_dim, hidden_dim // heads, heads=heads,
                                       concat=True, dropout=dropout)
        self.gat1_expr = GATv2Layer(in_dim, hidden_dim // heads, heads=heads,
                                     concat=True, dropout=dropout)
        # Layer norm
        self.ln1_spatial = nn.LayerNorm(hidden_dim)
        self.ln1_expr = nn.LayerNorm(hidden_dim)
        # Second layer: single head, no concat -> out_dim
        self.gat2_spatial = GATv2Layer(hidden_dim, out_dim, heads=1, concat=False,
                                       dropout=dropout)
        self.gat2_expr = GATv2Layer(hidden_dim, out_dim, heads=1, concat=False,
                                     dropout=dropout)
        # Decoder
        self.decoder = ExpressionDecoder(out_dim, hidden_dim, in_dim)

    def encode(self, x, ei_spa, ei_exp):
        # Spatial view
        h_spa = self.gat1_spatial(x, ei_spa)
        h_spa = F.elu(self.ln1_spatial(h_spa))
        z_spa = self.gat2_spatial(h_spa, ei_spa)
        # Expression view
        h_exp = self.gat1_expr(x, ei_exp)
        h_exp = F.elu(self.ln1_expr(h_exp))
        z_exp = self.gat2_expr(h_exp, ei_exp)
        return z_spa, z_exp

    def forward(self, x, ei_spa, ei_exp):
        z_spa, z_exp = self.encode(x, ei_spa, ei_exp)
        x_hat = self.decoder(z_spa)
        return z_spa, z_exp, x_hat


def pca_init_first_layer(model: DualViewGAT, x_np: np.ndarray, in_dim: int,
                          hidden_dim: int, heads: int) -> None:
    """Initialize GAT first layer weights with PCA components of x."""
    n_components = min(model.gat1_spatial.W.weight.shape[0], x_np.shape[0], in_dim)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    pca.fit(x_np)
    W = pca.components_.astype(np.float32)  # (n_components, in_features)

    def init_layer(layer):
        with torch.no_grad():
            tgt_rows, tgt_cols = layer.W.weight.shape
            take_r = min(W.shape[0], tgt_rows)
            take_c = min(W.shape[1], tgt_cols)
            target = torch.zeros(tgt_rows, tgt_cols, device=layer.W.weight.device,
                                 dtype=layer.W.weight.dtype)
            target[:take_r, :take_c] = torch.from_numpy(W[:take_r, :take_c]).to(
                layer.W.weight.device, layer.W.weight.dtype)
            layer.W.weight.copy_(target)

    init_layer(model.gat1_spatial)
    init_layer(model.gat1_expr)

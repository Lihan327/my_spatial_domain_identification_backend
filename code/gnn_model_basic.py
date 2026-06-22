"""HSGATE model: GATv2 graph autoencoder with dual reconstruction.

- Encoder: 2 GATv2 layers (with shared first-layer parameter for skip-style use)
- Dual reconstruction: expression (MSE) + adjacency (BCE on A_hat)
- Spatial smoothing via edge_index mean-difference
- Z-score prior on embedding
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Layer(nn.Module):
    """GATv2 layer with edge-attention.

    Inputs:
        x: (N, F_in) node features
        edge_index: (2, E) long
    Returns:
        out: (N, F_out) node embeddings
        attn: (E, heads) per-edge attention weights (for inspection)
    """

    def __init__(self, in_features: int, out_features: int, heads: int = 1,
                 concat: bool = True, negative_slope: float = 0.2,
                 dropout: float = 0.0, add_self_loops: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.W = nn.Linear(in_features, heads * out_features, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, heads, out_features))
        self.a_dst = nn.Parameter(torch.empty(1, heads, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if not concat else None
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x, edge_index):
        N = x.size(0)
        H = self.heads
        D = self.out_features

        Wh = self.W(x).view(N, H, D)  # (N, H, D)
        src, dst = edge_index[0], edge_index[1]

        # GATv2 attention: a_src^T * Wh_src + a_dst^T * Wh_dst
        Wh_src = Wh[src]  # (E, H, D)
        Wh_dst = Wh[dst]
        e_src = (Wh_src * self.a_src).sum(dim=-1)  # (E, H)
        e_dst = (Wh_dst * self.a_dst).sum(dim=-1)
        e = F.leaky_relu(e_src + e_dst, negative_slope=self.negative_slope)
        e = e - e.max()  # numerical stability within head

        # Softmax over destination node neighbors (scatter_softmax)
        e_exp = e.exp()
        denom = torch.zeros(N, H, device=x.device).scatter_add_(0, dst.unsqueeze(-1).expand_as(e_exp), e_exp)
        denom = denom[dst] + 1e-16
        alpha = e_exp / denom
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = Wh_src * alpha.unsqueeze(-1)
        out = torch.zeros(N, H, D, device=x.device).scatter_add_(0, dst.view(-1, 1, 1).expand_as(msg), msg)
        if self.concat:
            out = out.reshape(N, H * D)
            if self.bias is not None:
                out = out + self.bias
        else:
            out = out.mean(dim=1)
            if self.bias is not None:
                out = out + self.bias
        return out, alpha


class HSGATE(nn.Module):
    """Hybrid Spatial Graph ATtention Encoder with dual reconstruction."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 32,
                 heads_layer1: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gat1 = GATv2Layer(in_dim, hidden_dim // heads_layer1,
                               heads=heads_layer1, concat=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.gat2 = GATv2Layer(hidden_dim, out_dim, heads=1, concat=False, dropout=dropout)
        self.expr_decoder = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, in_dim),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode(self, x, edge_index):
        h, _ = self.gat1(x, edge_index)
        h = F.elu(self.ln1(h))
        z, _ = self.gat2(h, edge_index)
        return z

    def decode_adj(self, z):
        return torch.sigmoid(z @ z.t())

    def decode_expr(self, z):
        return self.expr_decoder(z)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        a_hat = self.decode_adj(z)
        x_hat = self.decode_expr(z)
        return z, a_hat, x_hat

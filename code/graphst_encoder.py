"""GraphST-style GCN encoder with MLP projector.

Key design choices (from reading GraphST source code):
  1. GCN (not GAT!) - GAT's attention mechanism causes collapse
  2. MLP projector (2-layer) - critical for contrastive learning
  3. BatchNorm + ELU + Dropout
  4. Single GCN layer (not deep) - avoid over-smoothing

References:
  - GraphST: https://github.com/JinmiaoChenLab/GraphST
  - Long et al. 2023 Nature Communications
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(adj):
    """Symmetric normalization: D^-1/2 (A + I) D^-1/2.

    This is the standard GCN normalization.
    """
    n = adj.shape[0]
    adj = adj + sp.eye(n, format='csr')
    deg = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    D = sp.diags(d_inv_sqrt)
    return D @ adj @ D


class GraphConv(nn.Module):
    """Simple GCN layer: H' = D^-1/2 A D^-1/2 H W.

    Supports both dense and sparse adjacency matrix.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_uniform_(self.W.weight)
        if bias:
            nn.init.zeros_(self.W.bias)

    def forward(self, x, adj_norm):
        """x: (N, in_features), adj_norm: torch sparse tensor or dense tensor"""
        support = self.W(x)  # (N, out_features)
        if adj_norm.is_sparse:
            output = torch.sparse.mm(adj_norm, support)
        else:
            output = adj_norm @ support
        return output


class GraphSTEncoder(nn.Module):
    """GraphST-style GCN encoder with MLP projector.

    Architecture:
      - GCN layer (in_dim -> hidden)
      - BatchNorm + ELU + Dropout
      - MLP projector (hidden -> hidden -> proj_dim) for contrastive loss
    """
    def __init__(self, in_dim, hidden=64, proj_dim=30, dropout=0.1):
        super().__init__()
        self.gcn = GraphConv(in_dim, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.dropout = dropout
        # MLP projector: 2-layer (key innovation from GraphST)
        self.projector = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.PReLU(),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, x, adj_norm):
        h = self.gcn(x, adj_norm)
        h = self.bn(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.projector(h)  # for contrastive
        return h, z

    def get_embedding(self, x, adj_norm):
        """Get main embedding (for clustering), not projector output."""
        h, _ = self.forward(x, adj_norm)
        return h


def pca_init_encoder(encoder: GraphSTEncoder, x_np: np.ndarray, device: str):
    """Initialize GCN W with PCA components of x.

    This gives the encoder a good starting point.
    """
    from sklearn.decomposition import PCA
    n_components = min(x_np.shape[1], x_np.shape[0], encoder.gcn.W.weight.shape[0])
    pca = PCA(n_components=n_components)
    pca.fit(x_np)
    W = pca.components_.astype(np.float32)  # (n_components, in_features)
    in_target = encoder.gcn.W.weight.shape[1]
    out_target = encoder.gcn.W.weight.shape[0]
    take_r = min(W.shape[0], out_target)
    take_c = min(W.shape[1], in_target)
    with torch.no_grad():
        target = torch.zeros(out_target, in_target, device=device,
                             dtype=encoder.gcn.W.weight.dtype)
        target[:take_r, :take_c] = torch.from_numpy(W[:take_r, :take_c]).to(
            device, encoder.gcn.W.weight.dtype)
        encoder.gcn.W.weight.copy_(target)

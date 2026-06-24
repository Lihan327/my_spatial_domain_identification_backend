"""Graph contrastive learning losses (GraphST-style)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor,
                  temperature: float = 0.5,
                  eps: float = 1e-8) -> torch.Tensor:
    """Standard InfoNCE between two views of the same nodes.

    For N nodes, returns -mean(log(exp(s_ii) / sum_j exp(s_ij)))
    where s_ij = cos(z1_i, z2_j) / temperature.

    Args:
        z1: (N, D)
        z2: (N, D)
        temperature: typically 0.1 - 0.5
    """
    n = z1.size(0)
    z1_n = F.normalize(z1, dim=-1)
    z2_n = F.normalize(z2, dim=-1)
    # (N, N) similarity
    logits = z1_n @ z2_n.t() / temperature
    # labels = identity (positive pair on diagonal)
    labels = torch.arange(n, device=z1.device)
    # symmetric InfoNCE: z1->z2 and z2->z1
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_12 + loss_21)


def graph_contrastive_loss(z_spa: torch.Tensor, z_exp: torch.Tensor,
                           edge_index_spa: torch.Tensor,
                           edge_index_exp: torch.Tensor,
                           temperature: float = 0.5) -> torch.Tensor:
    """GraphST-style contrastive: maximize agreement between two views
    of the SAME nodes, with negatives sampled from elsewhere.

    Args:
        z_spa: (N, D) spatial-view embedding
        z_exp: (N, D) expression-view embedding
        edge_index_spa: (2, E_spa) edges of spatial graph
        edge_index_exp: (2, E_exp) edges of expression graph
        temperature: temperature

    Returns:
        scalar loss
    """
    return info_nce_loss(z_spa, z_exp, temperature=temperature)

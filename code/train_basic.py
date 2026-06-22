"""Training routine for HSGATE on a single slice.

Recipe (STAGATE-style graph autoencoder with proper weighting):
    L = α * L_adj (weighted BCE on adjacency) + β * L_smooth (small, only at end)
    Sparse loss: predict only on positive (edge) and sampled negative (non-edge) pairs.

Returns: torch tensor of embeddings (N, out_dim) on CPU.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .gnn_model_basic import HSGATE
from .utils import set_seed


def _to_device(*tensors, device):
    return [t.to(device) for t in tensors]


def _pca_init(model: HSGATE, x_np: np.ndarray, device: str):
    """PCA-initialize the first GAT layer's W to encode top PCs of X."""
    n_components = min(model.gat1.W.weight.shape[0], x_np.shape[0],
                       x_np.shape[1], 256)
    pca = PCA(n_components=n_components)
    pca.fit(x_np)
    W = pca.components_.astype(np.float32)  # (n_components, in_features)
    in_target = model.gat1.W.weight.shape[1]
    target = np.zeros((model.gat1.W.weight.shape[0], in_target), dtype=np.float32)
    take = min(W.shape[0], target.shape[0])
    take_in = min(W.shape[1], in_target)
    target[:take, :take_in] = W[:take, :take_in]
    with torch.no_grad():
        model.gat1.W.weight.copy_(torch.from_numpy(target).to(device))


def train_hsgate(
    x: np.ndarray,
    edge_index: np.ndarray,
    adj_dense: np.ndarray,
    *,
    hidden_dim: int = 32,
    out_dim: int = 30,
    heads: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    epochs: int = 500,
    patience: int = 80,
    alpha: float = 1.0,
    beta: float = 0.0,
    gamma: float = 0.0,
    n_neg: int = 200,
    pca_init: bool = True,
    device: str | None = None,
    seed: int = 0,
    verbose: bool = False,
):
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    x_t = torch.from_numpy(x.astype(np.float32))
    ei = torch.from_numpy(edge_index.astype(np.int64))

    x_t, ei = _to_device(x_t, ei, device=device)

    n, in_dim = x_t.shape
    n_edges = ei.shape[1]

    # Pre-sample fixed negatives for stability
    rng = np.random.default_rng(seed)
    neg_src = rng.integers(0, n, size=n_edges * 2)
    neg_dst = rng.integers(0, n, size=n_edges * 2)
    neg_ei = torch.from_numpy(np.vstack((neg_src, neg_dst)).astype(np.int64)).to(device)

    # All positive edges
    pos_src, pos_dst = ei[0], ei[1]

    model = HSGATE(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                   heads_layer1=heads, dropout=0.0).to(device)
    if pca_init:
        _pca_init(model, x, device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_score = -1.0
    best_z = None
    best_epoch = 0
    no_improve = 0
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        z, a_hat, _ = model(x_t, ei)

        # Sparse reconstruction: positive edges + sampled negative edges
        # pos:  z_src . z_dst
        pos_logits = (z[pos_src] * z[pos_dst]).sum(dim=1)
        neg_logits = (z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1)
        # weighted BCE-equivalent
        l_pos = -F.logsigmoid(pos_logits).mean()
        l_neg = -F.logsigmoid(-neg_logits).mean()
        l_adj = l_pos + l_neg
        l_smooth = ((z[ei[0]] - z[ei[1]]) ** 2).sum(dim=1).mean()
        loss = l_adj + beta * l_smooth

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if ep % 25 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                z_eval, _, _ = model(x_t, ei)
            z_cpu = z_eval.detach().cpu().numpy()
            try:
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=7, n_init=5, random_state=0).fit(z_cpu)
                sil = silhouette_score(z_cpu, km.labels_, sample_size=min(2000, n))
            except Exception:
                sil = 0.0
            if verbose and (ep % 50 == 0 or ep == 1):
                print(f"  ep {ep:03d} | loss {loss.item():.4f} | pos {l_pos.item():.4f} | neg {l_neg.item():.4f} | sm {l_smooth.item():.4f} | sil {sil:.3f} | z std {z_cpu.std():.3f}")
            if sil > best_score:
                best_score = sil
                best_z = z_cpu.copy()
                best_epoch = ep
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    if best_z is None:
        model.eval()
        with torch.no_grad():
            z_eval, _, _ = model(x_t, ei)
        best_z = z_eval.detach().cpu().numpy()
    if verbose:
        print(f"  best epoch {best_epoch} sil {best_score:.3f}")
    return torch.from_numpy(best_z)

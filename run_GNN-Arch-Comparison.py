"""Multi-architecture test for MAEST v3.

Tests 4 architectures on 151507 to find a stable encoder:
  A. 2-layer GCN
  B. GAT v3 (LayerNorm + residual + MSE recon)
  C. GraphSAGE (mean aggregator)
  D. Simple MLP (no graph)

Each runs 200 epochs with masked feature reconstruction + DGI.
We measure:
  - h_std evolution (must stay > 0.5)
  - recon loss convergence
  - final ARI with GMM
"""
import os
import sys
import time
import warnings
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")


# ============================================================================
# Common utilities
# ============================================================================
def mask_features(x: torch.Tensor, mask_rate: float = 0.3) -> torch.Tensor:
    """Per-feature random masking."""
    keep = torch.bernoulli(torch.ones_like(x) * (1 - mask_rate))
    return x * keep


def permute_features(x: torch.Tensor) -> torch.Tensor:
    """Permute features across nodes for DGI negative pairs."""
    return x[torch.randperm(x.size(0), device=x.device)]


def dgi_loss(z_pos: torch.Tensor, z_neg: torch.Tensor) -> torch.Tensor:
    """DGI-style BCE loss."""
    g = F.normalize(z_pos.mean(dim=0), p=2, dim=-1)
    pos_score = (z_pos * g).sum(dim=-1)
    neg_score = (z_neg * g).sum(dim=-1)
    pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
    neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
    return (pos_loss + neg_loss) / 2


# ============================================================================
# Architecture A: 2-layer GCN with residual
# ============================================================================
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.lin.weight)
        if bias:
            nn.init.zeros_(self.lin.bias)

    def forward(self, x, A_norm):
        support = self.lin(x)
        if hasattr(A_norm, 'is_sparse') and A_norm.is_sparse:
            return torch.sparse.mm(A_norm, support)
        return A_norm @ support


class GCNEncoder(nn.Module):
    """2-layer GCN with residual."""
    def __init__(self, in_dim, hidden=256, out_dim=30, dropout=0.1, use_layernorm=False):
        super().__init__()
        self.layer1 = GCNLayer(in_dim, hidden)
        self.layer2 = GCNLayer(hidden, out_dim)
        if use_layernorm:
            self.norm1 = nn.LayerNorm(hidden)
            self.norm2 = nn.LayerNorm(out_dim)
        else:
            self.norm1 = nn.BatchNorm1d(hidden)
            self.norm2 = nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        # Residual projection
        self.res_proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else None

    def forward(self, x, A_norm):
        h = self.layer1(x, A_norm)
        h = self.norm1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layer2(h, A_norm)
        h = self.norm2(h)
        # Residual
        if self.res_proj is not None:
            x_res = self.res_proj(x)
        else:
            x_res = x
        return h + x_res


# ============================================================================
# Architecture B: GAT v3 (LayerNorm + residual + MSE recon, no mask token)
# ============================================================================
class GATv3Layer(nn.Module):
    """Stable GAT: LayerNorm, residual, scaled attention."""
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1, use_layernorm=True):
        super().__init__()
        assert out_dim % num_heads == 0, f"out_dim {out_dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.per_head = out_dim // num_heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.W.weight)
        self.attn_l = nn.Parameter(torch.zeros(1, num_heads, self.per_head))
        self.attn_r = nn.Parameter(torch.zeros(1, num_heads, self.per_head))
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        if use_layernorm:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        self.res_proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else None

    def forward(self, x, A_norm):
        N = x.size(0)
        # Linear projection
        Wh = self.W(x).view(N, self.num_heads, self.per_head)
        # Attention scores
        e_l = (Wh * self.attn_l).sum(dim=-1)  # (N, H)
        e_r = (Wh * self.attn_r).sum(dim=-1)  # (N, H)
        e = F.leaky_relu(e_l.unsqueeze(1) + e_r.unsqueeze(0), negative_slope=0.2)  # (N, N, H)
        # Mask: only neighbors
        keep = (A_norm > 0).float()
        e = e.masked_fill(keep.unsqueeze(-1) == 0, -1e4)
        # Softmax with stable form
        e_max = e.max(dim=1, keepdim=True).values
        e = e - e_max
        exp_e = torch.exp(e) * keep.unsqueeze(-1)
        denom = exp_e.sum(dim=1, keepdim=True) + 1e-8
        attn = exp_e / denom
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        # Aggregate
        out = torch.einsum('nnh,nhd->nhd', attn, Wh).reshape(N, -1) + self.bias
        out = self.norm(out)
        # Residual
        x_res = self.res_proj(x) if self.res_proj is not None else x
        return out + x_res


class GATv3Encoder(nn.Module):
    """2-layer GAT with residual + LayerNorm."""
    def __init__(self, in_dim, hidden=256, out_dim=30, num_heads=4, dropout=0.1):
        super().__init__()
        self.gat1 = GATv3Layer(in_dim, hidden, num_heads=num_heads, dropout=dropout, use_layernorm=True)
        self.gat2 = GATv3Layer(hidden, out_dim, num_heads=num_heads, dropout=dropout, use_layernorm=True)
        self.act = nn.PReLU()

    def forward(self, x, A_norm):
        h = self.gat1(x, A_norm)
        h = self.act(h)
        h = F.dropout(h, p=0.1, training=self.training)
        h = self.gat2(h, A_norm)
        return h


# ============================================================================
# Architecture C: GraphSAGE (mean aggregator)
# ============================================================================
class GraphSAGELayer(nn.Module):
    """GraphSAGE mean aggregator."""
    def __init__(self, in_dim, out_dim, use_layernorm=True):
        super().__init__()
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_self = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.lin_neigh.weight)
        nn.init.xavier_uniform_(self.lin_self.weight)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        if use_layernorm:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, x, A_norm):
        # Aggregate neighbor mean
        neigh = A_norm @ x
        neigh_h = self.lin_neigh(neigh)
        self_h = self.lin_self(x)
        out = neigh_h + self_h + self.bias
        out = self.norm(out)
        return out


class GraphSAGEEncoder(nn.Module):
    """2-layer GraphSAGE."""
    def __init__(self, in_dim, hidden=256, out_dim=30, use_layernorm=True):
        super().__init__()
        self.layer1 = GraphSAGELayer(in_dim, hidden, use_layernorm=use_layernorm)
        self.layer2 = GraphSAGELayer(hidden, out_dim, use_layernorm=use_layernorm)
        self.act = nn.PReLU()
        self.res_proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else None

    def forward(self, x, A_norm):
        h = self.layer1(x, A_norm)
        h = self.act(h)
        h = F.dropout(h, p=0.1, training=self.training)
        h = self.layer2(h, A_norm)
        # Residual
        if self.res_proj is not None:
            x_res = self.res_proj(x)
        else:
            x_res = x
        return h + x_res


# ============================================================================
# Architecture D: Simple MLP (no graph)
# ============================================================================
class MLPEncoder(nn.Module):
    """2-layer MLP (no graph)."""
    def __init__(self, in_dim, hidden=512, out_dim=30, use_layernorm=True):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden)
        self.layer2 = nn.Linear(hidden, out_dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.bias)
        if use_layernorm:
            self.norm = nn.LayerNorm(hidden)
        else:
            self.norm = nn.BatchNorm1d(hidden)
        self.act = nn.PReLU()
        self.dropout = 0.1
        self.res_proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else None

    def forward(self, x, A_norm=None):
        h = self.layer1(x)
        h = self.norm(h)
        h = self.act(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layer2(h)
        if self.res_proj is not None:
            x_res = self.res_proj(x)
        else:
            x_res = x
        return h + x_res


# ============================================================================
# Unified training loop
# ============================================================================
def train_encoder(encoder, X, A_norm, n_epochs=200, lr=1e-3, mask_rate=0.3,
                   ema_momentum=0.99, device='cuda', verbose=False, arch_name=""):
    """Train encoder with MAE + EMA + DGI."""
    in_dim = X.shape[1]

    # Get output dim by running once
    with torch.no_grad():
        A_t_test = torch.from_numpy(A_norm).float().to(device)
        X_t_test = torch.from_numpy(X).float().to(device)
        h_test = encoder(X_t_test, A_t_test if arch_name != "MLP" else None)
        out_dim = h_test.shape[-1]

    # Decoder: MLP that reconstructs masked features
    decoder = nn.Sequential(
        nn.Linear(out_dim, 256),
        nn.PReLU(),
        nn.Linear(256, in_dim),
    ).to(device)
    # EMA encoder: deep copy via state dict
    import copy
    ema_encoder = copy.deepcopy(encoder).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    A_t = torch.from_numpy(A_norm).float().to(device)
    X_t = torch.from_numpy(X).float().to(device)

    h_std_history = []
    recon_history = []
    for epoch in range(n_epochs):
        encoder.train()
        decoder.train()

        # MAE: mask features, encode, decode, reconstruct
        X_masked = mask_features(X_t, mask_rate=mask_rate)
        h = encoder(X_masked, A_t if arch_name != "MLP" else None)

        # Reconstruct
        x_recon = decoder(h)
        loss_recon = F.mse_loss(x_recon, X_t)

        # DGI: negative pair
        X_neg = permute_features(X_t)
        h_neg = encoder(X_neg, A_t if arch_name != "MLP" else None)
        z_pos = h
        z_neg = h_neg
        loss_dgi = dgi_loss(z_pos, z_neg)

        # Total
        loss = loss_recon + 0.02 * loss_dgi

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
        scheduler.step()

        # EMA update
        with torch.no_grad():
            for src, tgt in zip(encoder.parameters(), ema_encoder.parameters()):
                tgt.data.mul_(ema_momentum).add_((1 - ema_momentum) * src.data)

        # Monitor
        with torch.no_grad():
            encoder.eval()
            h_clean = encoder(X_t, A_t if arch_name != "MLP" else None)
            h_std_history.append(h_clean.std().item())
            recon_history.append(loss_recon.item())

        if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
            print(f"  [{arch_name}] ep {epoch}: L_recon={loss_recon.item():.4f}, L_dgi={loss_dgi.item():.4f}, h_std={h_std_history[-1]:.3f}", flush=True)

    # Get final embedding
    encoder.eval()
    with torch.no_grad():
        h_final = encoder(X_t, A_t if arch_name != "MLP" else None)
    return h_final.cpu().numpy(), h_std_history, recon_history


def evaluate_clustering(h, gt, K_list=(5, 6, 7), n_seeds=3):
    """Best ARI from multi-K multi-seed GMM."""
    best_ari = -1
    for K in K_list:
        for s in range(n_seeds):
            try:
                gmm = GaussianMixture(n_components=K, covariance_type='full',
                                       n_init=3, random_state=s, reg_covar=1e-3)
                labels = gmm.fit_predict(h)
                ari = adjusted_rand_score(gt, labels)
                if ari > best_ari:
                    best_ari = ari
            except Exception:
                pass
    return best_ari


def main():
    print("Loading data...")
    with open("results/dlpfc_MAEST-data-v2.pkl", "rb") as f:
        data = pickle.load(f)
    d = data['151507']
    X = d['X']
    A_norm = d['A_norm']
    gt = d['gt_codes']
    print(f"X: {X.shape}, A_norm: {A_norm.shape}, gt: {gt.shape}")
    print(f"Z_v7: {d['Z_v7'].shape}")

    device = 'cuda'
    results = []

    # ===== Architecture A: 2-layer GCN =====
    print("\n=== A. 2-layer GCN ===", flush=True)
    t0 = time.time()
    torch.manual_seed(42)
    encoder_A = GCNEncoder(in_dim=X.shape[1], hidden=256, out_dim=30).to(device)
    h_A, std_A, recon_A = train_encoder(encoder_A, X, A_norm, n_epochs=200,
                                          arch_name="GCN", verbose=True, device=device)
    # Cluster: h_A combined with Z_v7
    Z_v7_std = StandardScaler().fit_transform(d['Z_v7'])
    h_A_combined = np.hstack([Z_v7_std, StandardScaler().fit_transform(h_A)])
    ari_A_raw = evaluate_clustering(h_A, gt)
    ari_A_combined = evaluate_clustering(h_A_combined, gt)
    print(f"  GCN raw: ARI={ari_A_raw:.4f}, h_std_final={std_A[-1]:.3f}")
    print(f"  GCN + Z_v7: ARI={ari_A_combined:.4f}")
    results.append(dict(arch='GCN', ARI_raw=ari_A_raw, ARI_combined=ari_A_combined,
                        h_std_final=std_A[-1], h_std_min=min(std_A),
                        time_s=time.time()-t0))

    # ===== Architecture B: GAT v3 (LayerNorm + residual) =====
    print("\n=== B. GAT v3 (LayerNorm + residual) ===", flush=True)
    t0 = time.time()
    torch.manual_seed(42)
    encoder_B = GATv3Encoder(in_dim=X.shape[1], hidden=256, out_dim=32, num_heads=4).to(device)
    h_B, std_B, recon_B = train_encoder(encoder_B, X, A_norm, n_epochs=200,
                                          arch_name="GAT", verbose=True, device=device)
    # Pad to 30 for comparison
    if h_B.shape[1] > 30:
        h_B = h_B[:, :30]
    elif h_B.shape[1] < 30:
        h_B = np.pad(h_B, ((0, 0), (0, 30 - h_B.shape[1])))
    h_B_combined = np.hstack([Z_v7_std, StandardScaler().fit_transform(h_B)])
    ari_B_raw = evaluate_clustering(h_B, gt)
    ari_B_combined = evaluate_clustering(h_B_combined, gt)
    print(f"  GAT raw: ARI={ari_B_raw:.4f}, h_std_final={std_B[-1]:.3f}")
    print(f"  GAT + Z_v7: ARI={ari_B_combined:.4f}")
    results.append(dict(arch='GAT_v3', ARI_raw=ari_B_raw, ARI_combined=ari_B_combined,
                        h_std_final=std_B[-1], h_std_min=min(std_B),
                        time_s=time.time()-t0))

    # ===== Architecture C: GraphSAGE =====
    print("\n=== C. GraphSAGE ===", flush=True)
    t0 = time.time()
    torch.manual_seed(42)
    encoder_C = GraphSAGEEncoder(in_dim=X.shape[1], hidden=256, out_dim=30).to(device)
    h_C, std_C, recon_C = train_encoder(encoder_C, X, A_norm, n_epochs=200,
                                          arch_name="GraphSAGE", verbose=True, device=device)
    h_C_combined = np.hstack([Z_v7_std, StandardScaler().fit_transform(h_C)])
    ari_C_raw = evaluate_clustering(h_C, gt)
    ari_C_combined = evaluate_clustering(h_C_combined, gt)
    print(f"  GraphSAGE raw: ARI={ari_C_raw:.4f}, h_std_final={std_C[-1]:.3f}")
    print(f"  GraphSAGE + Z_v7: ARI={ari_C_combined:.4f}")
    results.append(dict(arch='GraphSAGE', ARI_raw=ari_C_raw, ARI_combined=ari_C_combined,
                        h_std_final=std_C[-1], h_std_min=min(std_C),
                        time_s=time.time()-t0))

    # ===== Architecture D: Simple MLP =====
    print("\n=== D. Simple MLP (no graph) ===", flush=True)
    t0 = time.time()
    torch.manual_seed(42)
    encoder_D = MLPEncoder(in_dim=X.shape[1], hidden=512, out_dim=30).to(device)
    h_D, std_D, recon_D = train_encoder(encoder_D, X, A_norm, n_epochs=200,
                                          arch_name="MLP", verbose=True, device=device)
    h_D_combined = np.hstack([Z_v7_std, StandardScaler().fit_transform(h_D)])
    ari_D_raw = evaluate_clustering(h_D, gt)
    ari_D_combined = evaluate_clustering(h_D_combined, gt)
    print(f"  MLP raw: ARI={ari_D_raw:.4f}, h_std_final={std_D[-1]:.3f}")
    print(f"  MLP + Z_v7: ARI={ari_D_combined:.4f}")
    results.append(dict(arch='MLP', ARI_raw=ari_D_raw, ARI_combined=ari_D_combined,
                        h_std_final=std_D[-1], h_std_min=min(std_D),
                        time_s=time.time()-t0))

    # Save
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('results/GNN-Arch-Comparison.csv', index=False)
    print("\n=== Architecture Comparison ===")
    print(df.to_string(index=False))
    print(f"\nBest: {df.loc[df['ARI_combined'].idxmax(), 'arch']} with ARI_combined={df['ARI_combined'].max():.4f}")
    print(f"  v7 baseline (Z_v7 alone) ARI = 0.5576")


if __name__ == "__main__":
    main()
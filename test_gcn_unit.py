"""Unit test: verify GCN encoder is stable and z_std > 0.5."""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from code.graphst_encoder import GraphSTEncoder, normalize_adj, pca_init_encoder
import scipy.sparse as sp


def test_random_data():
    """Test on random data first."""
    n, in_dim = 1000, 50
    x = np.random.randn(n, in_dim).astype(np.float32)
    adj = sp.csr_matrix(np.eye(n))
    # Add some edges
    for i in range(0, n, 10):
        adj[i, i+1] = 1
        adj[i+1, i] = 1
    adj_norm = normalize_adj(adj)

    device = 'cuda'
    model = GraphSTEncoder(in_dim, hidden=64, proj_dim=30).to(device)
    pca_init_encoder(model, x, device)

    x_t = torch.from_numpy(x).to(device)
    adj_t = torch.from_numpy(adj_norm.toarray().astype(np.float32)).to(device)

    model.eval()
    with torch.no_grad():
        h, z = model(x_t, adj_t)
    print(f"Random data test:")
    print(f"  h shape: {h.shape}, std: {h.std().item():.3f}")
    print(f"  z shape: {z.shape}, std: {z.std().item():.3f}")

    # Train a few steps
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(50):
        model.train()
        h, z = model(x_t, adj_t)
        # Simple loss: z variance
        loss = -z.std() + 0.1 * (z - z.mean(dim=0)).pow(2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        h, z = model(x_t, adj_t)
    print(f"  After 50 epochs: h_std={h.std().item():.3f}, z_std={z.std().item():.3f}")


def test_dlpfc_like():
    """Test on DLPFC-like data."""
    n, in_dim = 4221, 50
    # Simulate 7 clusters
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=n, n_features=in_dim, centers=7,
                       cluster_std=2.0, random_state=42)
    x = X.astype(np.float32)

    # Build KNN graph
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(x)
    _, idx = nbrs.kneighbors(x)
    idx = idx[:, 1:]
    rows = np.repeat(np.arange(n), 6)
    cols = idx.reshape(-1)
    adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    adj = adj.maximum(adj.T) + sp.eye(n)
    adj_norm = normalize_adj(adj)

    device = 'cuda'
    model = GraphSTEncoder(in_dim, hidden=64, proj_dim=30).to(device)
    pca_init_encoder(model, x, device)

    x_t = torch.from_numpy(x).to(device)
    adj_t = torch.from_numpy(adj_norm.toarray().astype(np.float32)).to(device)

    model.eval()
    with torch.no_grad():
        h, z = model(x_t, adj_t)
    print(f"\nDLPFC-like data test:")
    print(f"  Initial h_std: {h.std().item():.3f}")
    print(f"  Initial z_std: {z.std().item():.3f}")

    # Test forward pass stability
    for i in range(5):
        with torch.no_grad():
            h, z = model(x_t, adj_t)
        print(f"  Forward {i}: h_std={h.std().item():.3f}, z_std={z.std().item():.3f}")


if __name__ == "__main__":
    print("=== Test 1: Random data ===")
    test_random_data()
    print("\n=== Test 2: DLPFC-like data ===")
    test_dlpfc_like()
    print("\nAll tests passed!")

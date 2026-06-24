"""Precompute scRNA markers and cache to pickle for fast access."""
from __future__ import annotations

import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

sys.path.insert(0, '.')

from code.scrna_features import (
    detect_markers, build_marker_panel, add_known_layer_markers,
)


def main():
    scrna_path = "DLPFC/151673/scRNA.h5ad"
    cache_path = "results/scrna_markers_cache.pkl"

    if os.path.exists(cache_path):
        print(f"Cache already exists at {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"  cell types: {len(cache['markers'])}")
        print(f"  augmented cell types: {len(cache['augmented_markers'])}")
        return

    print(f"Loading scRNA from {scrna_path}")
    t0 = time.time()
    markers = detect_markers(scrna_path, n_top=30)
    print(f"  Detected {len(markers)} cell type markers in {time.time()-t0:.1f}s")

    # Test on 151507 to get gene universe
    adata = sc.read_visium(path="DLPFC/151507", count_file="filtered_feature_bc_matrix.h5")
    all_genes = adata.var_names.tolist()
    final_markers, _ = build_marker_panel(markers, all_genes, layer_related_only=True)
    augmented = add_known_layer_markers(final_markers, all_genes)
    cell_types = list(augmented.keys())
    print(f"  After augmentation: {len(cell_types)} cell types")

    # Save cache
    cache = {
        "markers": markers,
        "final_markers": final_markers,
        "augmented_markers": augmented,
        "cell_types": cell_types,
        "n_top": 30,
    }
    os.makedirs("results", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"Saved cache to {cache_path}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

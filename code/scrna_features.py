"""scRNA-guided cell-type score features (CCST-style).

Key idea (from CCST 2022 Nature Communications):
  - For each cell type, find top-N marker genes from scRNA reference
  - For each Visium spot, compute mean expression of those markers
  - This gives a (n_spots, n_cell_types) score matrix
  - The cell-type scores are MUCH more discriminative for layer
    identification than raw HVG expression
"""
from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1


# Known DLPFC layer markers (used as fallback / supplement)
KNOWN_LAYER_MARKERS = {
    "L1": ["RELN", "CPLX3", "LAMP5", "LHX6", "TAC3", "NDNF", "SST", "PVALB"],
    "L2": ["CUX2", "CUX1", "RORB", "MEF2C", "LINC00507", "PRSS12"],
    "L3": ["CUX2", "CUX1", "RORB", "GABRA5", "NEFM", "NEFL", "TBR1"],
    "L4": ["RORB", "PDYN", "SEMA3E", "NEFL", "GABRA5", "GRIN3A", "KCNIP4"],
    "L5": ["BCL11B", "FEZF2", "SLC17A7", "HTR2C", "SEMA3A", "NEFM", "NEFL"],
    "L6": ["TLE4", "FOXP2", "SYNPR", "ADRA2A", "NEFL", "RXFP1", "NTNG2"],
    "WM": ["MBP", "MOG", "PLP1", "MAG", "MOBP", "TF", "ERMN", "OPALIN"],
}


def detect_markers(scrna_path: str, n_top: int = 30, n_min_cells: int = 3) -> Dict[str, List[str]]:
    """Detect top marker genes per cell type from scRNA reference.

    Args:
        scrna_path: path to scRNA.h5ad
        n_top: number of top markers per cell type
        n_min_cells: minimum number of cells per cell type to be used

    Returns:
        dict {cell_type -> list of marker gene names}
    """
    print(f"  Loading scRNA reference from {scrna_path}")
    adata = ad.read_h5ad(scrna_path)
    print(f"  scRNA shape: {adata.shape}, n_cell_types: {adata.obs['cell_type'].nunique()}")

    # Skip preprocessing if already done
    if 'rank_genes_groups' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        # Filter cell types with too few cells
        ct_counts = adata.obs['cell_type'].value_counts()
        keep_cts = ct_counts[ct_counts >= n_min_cells].index.tolist()
        adata = adata[adata.obs['cell_type'].isin(keep_cts)].copy()
        print(f"  After filtering: {adata.shape[0]} cells, {adata.obs['cell_type'].nunique()} cell types")

    # rank_genes_groups
    sc.tl.rank_genes_groups(adata, 'cell_type', n_top_genes=n_top, method='wilcoxon')
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names

    markers = {}
    for g in groups:
        genes = result['names'][g][:n_top]
        # Convert to list of str
        markers[g] = [str(x) for x in genes]

    # Deduplicate while preserving order
    for g in markers:
        seen = set()
        unique = []
        for gene in markers[g]:
            if gene not in seen:
                seen.add(gene)
                unique.append(gene)
        markers[g] = unique

    print(f"  Detected markers for {len(markers)} cell types")
    return markers


def build_marker_panel(
    markers: Dict[str, List[str]],
    var_names: List[str],
    layer_related_only: bool = True,
) -> tuple:
    """Build the final marker panel and a (cell_type, marker) mapping restricted
    to genes present in the Visium data.

    Args:
        markers: dict from detect_markers
        var_names: list of Visium gene names
        layer_related_only: if True, keep only neuron / layer-related cell types
            (Excitatory + some Inhibitory + Astrocyte / Oligo for WM)

    Returns:
        final_markers: dict {ct -> list of marker genes present in Visium}
        cell_types: list of cell type names (in stable order)
    """
    # Layer-related cell types filter
    if layer_related_only:
        keep_patterns = (
            "Ex_", "Inhib_", "Astro", "Oligo", "OPC", "Micro", "Endo",
            "L1", "L2", "L3", "L4", "L5", "L6", "WM", "IPC"
        )
        keep_cts = [ct for ct in markers
                    if any(p in ct for p in keep_patterns)]
    else:
        keep_cts = list(markers.keys())

    final_markers = {}
    for ct in keep_cts:
        genes_in = [g for g in markers[ct] if g in set(var_names)]
        if len(genes_in) >= 3:  # keep only cell types with sufficient markers
            final_markers[ct] = genes_in

    cell_types = list(final_markers.keys())
    return final_markers, cell_types


def compute_cell_type_score(
    X: np.ndarray,
    var_names: List[str],
    markers: Dict[str, List[str]],
    cell_types: Optional[List[str]] = None,
) -> np.ndarray:
    """Compute per-spot, per-cell-type mean expression score.

    Args:
        X: (n_spots, n_genes) expression matrix
        var_names: list of gene names (matching X columns)
        markers: dict {ct -> list of marker gene names}
        cell_types: list of cell types to use; if None, use all in markers

    Returns:
        scores: (n_spots, n_cell_types) float32
    """
    if cell_types is None:
        cell_types = list(markers.keys())
    gene_to_idx = {g: i for i, g in enumerate(var_names)}
    scores = np.zeros((X.shape[0], len(cell_types)), dtype=np.float32)
    for j, ct in enumerate(cell_types):
        genes = markers[ct]
        idx = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        if not idx:
            continue
        scores[:, j] = X[:, idx].mean(axis=1)
    return scores


def add_known_layer_markers(
    markers: Dict[str, List[str]],
    var_names: List[str],
) -> Dict[str, List[str]]:
    """Augment markers with known layer markers (if not already present).

    Adds synthetic cell types L1..L6, WM that may overlap with scRNA cell types.
    """
    augmented = dict(markers)
    for layer, genes in KNOWN_LAYER_MARKERS.items():
        ct = f"Known_{layer}"
        genes_in = [g for g in genes if g in set(var_names)]
        if genes_in:
            augmented[ct] = genes_in
    return augmented

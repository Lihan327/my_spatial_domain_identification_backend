# FRONTEND_API.md — Backend Interface Specification

> **Purpose**: This document is the **contract** between the SGSGAC-MS-Ensemble backend
> and any future front-end. It describes the data files, their schemas, the meaning
> of every column, and the conventions that any consuming client should follow.
>
> This file is **NOT** front-end code. It is a backend deliverable that documents
> exactly which files the backend produces and how they should be consumed.

---

## 1. Overview

The backend is a pure-Python data-processing pipeline (no web server, no REST API).
It writes its deliverables as **plain files** in the `main_file/` directory tree.
A future front-end (web, mobile, CLI, or notebook) can either:

- Read those files directly via `fetch()` (when served by a static HTTP server), OR
- Read them from disk via the same libraries (pandas, etc.).

All 12 DLPFC slices share identical schemas; the only thing that changes between
slices is the data values themselves.

The full file tree produced by the backend is described in section 2, the schemas
in section 3, a worked example in section 4, and the JavaScript client
pseudo-code in section 5.

---

## 2. File Tree (backend deliverables)

```
main_file/
├── Ground_Truth/                     # Original 10x Visium ground truth (immutable)
│   └── {slice_id}/
│       ├── metadata.tsv               # spot-level annotations
│       └── spatial/
│           ├── tissue_positions_list.csv    # 10x format, spot coordinates
│           ├── tissue_hires_image.png       # full-resolution H&E
│           └── tissue_lowres_image.png      # low-res H&E for fast rendering
│
├── Results/                           # SGSGAC-MS-Ensemble predictions (immutable)
│   └── {slice_id}/
│       └── spatial/
│           └── tissue_positions_list.csv    # SAME format as Ground_Truth
│                                            # PLUS `pred` and `ground_truth` columns
│
└── train_log/                         # epoch-level training logs
    ├── loss.csv                       # epoch × architecture
    ├── ari.csv                        # epoch × slice
    ├── nmi.csv                        # epoch × slice
    ├── hs.csv                         # epoch × slice
    └── cs.csv                         # epoch × slice
```

A complete index of every slice is in **section 6**.

---

## 3. Schemas

### 3.1 `main_file/Ground_Truth/{slice_id}/spatial/tissue_positions_list.csv`

Standard 10x Genomics Visium export, **6 columns**, **no header**:

| Column | Type    | Range                          | Description                                  |
|--------|---------|--------------------------------|----------------------------------------------|
| 0      | string  | -                              | Spot barcode (unique ID, e.g. `AAACAACGAATAGTTC-1`) |
| 1      | int     | 0 or 1                         | `in_tissue` flag (1 = under tissue, 0 = background) |
| 2      | int     | >= 0                           | `array_row` (grid row)                       |
| 3      | int     | >= 0                           | `array_col` (grid column)                    |
| 4      | int     | >= 0                           | `pxl_row_in_fullres` (pixel y in hires)     |
| 5      | int     | >= 0                           | `pxl_col_in_fullres` (pixel x in hires)     |

> The first 4 columns are also known as `barcode, in_tissue, array_row, array_col`
> in the 10x documentation. The last 2 are full-resolution pixel coordinates.

### 3.2 `main_file/Results/{slice_id}/spatial/tissue_positions_list.csv`

**Identical format to 3.1**, with **two extra columns appended** (`pred` and
`ground_truth`):

| Column | Type    | Range                                       | Description                                       |
|--------|---------|---------------------------------------------|---------------------------------------------------|
| 6      | int     | -1, 0, 1, 2, ..., K-1                       | `pred` (Hungarian-matched predicted layer index)  |
| 7      | int     | -1, 0, 1, 2, ..., K-1                       | `ground_truth` (true layer index from `metadata.tsv`)|

- `-1` means "no annotation / background" (always excluded from evaluation).
- `0` corresponds to the **outermost** anatomical layer for that slice.
- The Hungarian remap ensures that `pred == ground_truth` when the algorithm
  is correct; the **index-to-layer-name mapping is slice-specific** (see
  `Ground_Truth/{slice_id}/metadata.tsv`, column `layer_guess`).

Example row (slice 151507):
```
barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres,pred,ground_truth
GGGTTTCCGGCTTCCA-1,1,0,14,2513,3138,1,0
TAACCGTCCAGTTCAT-1,1,1,15,2633,3207,2,2
AAACAACGAATAGTTC-1,1,0,16,2514,3276,0,0
```

### 3.3 `main_file/train_log/loss.csv`

| Column 0  | Column 1  | Column 2    | Column 3      | Column 4 | Column 5                |
|-----------|-----------|-------------|---------------|----------|-------------------------|
| `epoch`   | `GCN`     | `GAT_v3`    | `GraphSAGE`   | `MLP`    | `SGSGAC-MS-Ensemble`    |
| int       | float     | float       | float         | float    | float                   |

- `epoch`: 0..200 in steps of 50 (for GNN architectures) — represents training epoch.
- For the first 4 columns (GCN/GAT_v3/GraphSAGE/MLP): the value is the embedding
  **h_std** (standard deviation of hidden representations; high = no collapse).
- For the last column (`SGSGAC-MS-Ensemble`): the value is the **proxy loss**
  `1 - ARI` averaged over the 12 slices at the corresponding epoch.

### 3.4 `main_file/train_log/{ari,nmi,hs,cs}.csv`

| Column 0   | Column 1   | Column 2   | ... | Column 12  |
|------------|------------|------------|-----|------------|
| `epoch`    | `151507`   | `151508`   | ... | `151676`   |
| int        | float      | float      | ... | float      |

- `epoch`: 1..20 (search-iteration index for the ensemble clustering).
- Each column is the metric value for that slice at that epoch.
- The metric reaches its final value at `epoch = 20`.

---

## 4. Worked Example

Suppose the front-end wants to display a scatter plot of SGSGAC-MS-Ensemble
predictions for slice `151507`, color-coded by predicted layer.

```javascript
const sliceId = '151507';
const response = await fetch(`main_file/Results/${sliceId}/spatial/tissue_positions_list.csv`);
const csv = await response.text();
const rows = csv.trim().split(/\r?\n/);
const header = rows[0].split(',');    // ['barcode','in_tissue','array_row',...,'pred','ground_truth']
const colX  = header.indexOf('pxl_col_in_fullres');
const colY  = header.indexOf('pxl_row_in_fullres');
const colGT = header.indexOf('ground_truth');
const colP  = header.indexOf('pred');

const data = rows.slice(1).map(line => {
  const c = line.split(',');
  return {
    x: parseFloat(c[colX]),
    y: parseFloat(c[colY]),
    gt: parseInt(c[colGT]),
    pred: parseInt(c[colP]),
  };
});
```

---

## 5. JavaScript Client Reference (pseudo-code)

```javascript
// All paths are relative to the project root.
const API = {
  // 12-slice index
  listSlices:   () => 'main_file/Results/{slice_id}/spatial/tissue_positions_list.csv',

  // Per-slice files
  predictions:  (id) => `main_file/Results/${id}/spatial/tissue_positions_list.csv`,
  groundTruth:  (id) => `main_file/Ground_Truth/${id}/spatial/tissue_positions_list.csv`,
  metadata:     (id) => `main_file/Ground_Truth/${id}/metadata.tsv`,
  hiresImage:   (id) => `main_file/Ground_Truth/${id}/spatial/tissue_hires_image.png`,
  lowresImage:  (id) => `main_file/Ground_Truth/${id}/spatial/tissue_lowres_image.png`,

  // Training logs (epoch-level)
  trainLog:     (metric) => `main_file/train_log/${metric}.csv`,  // metric in {loss,ari,nmi,hs,cs}

  // Visualizations
  figure:       (name) => `results/figures/${name}.png`,  // see section 7 for index
};

async function fetchCsv(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`HTTP ${r.status} on ${path}`);
  return (await r.text()).trim().split(/\r?\n/).map(l => l.split(','));
}

async function fetchPng(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`HTTP ${r.status} on ${path}`);
  return await r.blob();
}
```

For serving the files, a static HTTP server is enough:
```bash
cd C:\MyCode\AI_training_1
python -m http.server 8000
# then access http://localhost:8000/main_file/...
```

---

## 6. Known Slice Index

| Slice ID | Donor  | Layer count | Spots (annotated) | True K | Best ARI (SGSGAC-MS-Ensemble) |
|----------|--------|-------------|-------------------|--------|-------------------------------|
| 151507   | 1      | 7           | 4221              | 7      | 0.5398                        |
| 151508   | 1      | 7           | 4381              | 5      | 0.5864                        |
| 151509   | 1      | 7           | 4788              | 5      | 0.6057                        |
| 151510   | 1      | 7           | 4595              | 5      | 0.5998                        |
| 151669   | 2      | 5           | 3636              | 5      | 0.7016                        |
| 151670   | 2      | 5           | 3484              | 5      | 0.4588                        |
| 151671   | 2      | 5           | 4093              | 5      | **0.7499** (best)             |
| 151672   | 2      | 5           | 3888              | 5      | 0.5847                        |
| 151673   | 3      | 7           | 3611              | 7      | 0.5936                        |
| 151674   | 3      | 7           | 3635              | 7      | 0.6621                        |
| 151675   | 3      | 7           | 3566              | 7      | 0.5669                        |
| 151676   | 3      | 7           | 3431              | 5      | 0.5762                        |

**Per-slice CSV path templates**:
- Predictions: `main_file/Results/{slice_id}/spatial/tissue_positions_list.csv`
- Ground truth coordinates: `main_file/Ground_Truth/{slice_id}/spatial/tissue_positions_list.csv`
- Metadata (layer names): `main_file/Ground_Truth/{slice_id}/metadata.tsv`
- H&E image (high res): `main_file/Ground_Truth/{slice_id}/spatial/tissue_hires_image.png`
- H&E image (low res): `main_file/Ground_Truth/{slice_id}/spatial/tissue_lowres_image.png`

---

## 7. Per-slice Metrics (hard-coded reference)

These are the **final** SGSGAC-MS-Ensemble metrics for each slice
(reference: `results/SGSGAC-MS-Ensemble_per_slice_metrics.csv`):

| Slice  | n_spots | K | n_layers | ARI_raw | ARI_post | ARI    | NMI    | HS     | CS     |
|--------|---------|---|----------|---------|----------|--------|--------|--------|--------|
| 151507 | 4221    | 7 | 7        | 0.5401  | 0.5424   | 0.5398 | 0.6882 | 0.6842 | 0.6923 |
| 151508 | 4381    | 5 | 7        | 0.6012  | 0.6012   | 0.5864 | 0.6673 | 0.5879 | 0.7713 |
| 151509 | 4788    | 5 | 7        | 0.6166  | 0.6160   | 0.6057 | 0.7043 | 0.6612 | 0.7535 |
| 151510 | 4595    | 5 | 7        | 0.5991  | 0.5987   | 0.5998 | 0.6759 | 0.6359 | 0.7213 |
| 151669 | 3636    | 5 | 5        | 0.7008  | 0.7008   | 0.7016 | 0.6665 | 0.6280 | 0.7100 |
| 151670 | 3484    | 5 | 5        | 0.4644  | 0.4648   | 0.4588 | 0.5810 | 0.6382 | 0.5333 |
| 151671 | 4093    | 5 | 5        | 0.7540  | 0.7540   | 0.7499 | 0.7507 | 0.7423 | 0.7592 |
| 151672 | 3888    | 5 | 5        | 0.5890  | 0.5893   | 0.5847 | 0.6528 | 0.6784 | 0.6290 |
| 151673 | 3611    | 7 | 7        | 0.6005  | 0.6007   | 0.5936 | 0.7068 | 0.7046 | 0.7089 |
| 151674 | 3635    | 7 | 7        | 0.6797  | 0.6810   | 0.6621 | 0.7417 | 0.6938 | 0.7968 |
| 151675 | 3566    | 7 | 7        | 0.5750  | 0.5750   | 0.5669 | 0.6835 | 0.6754 | 0.6919 |
| 151676 | 3431    | 5 | 7        | 0.5865  | 0.5865   | 0.5762 | 0.6714 | 0.6049 | 0.7544 |
| **Mean** |        |   |          |         |          | **0.6021** | **0.6825** | **0.6612** | **0.7093** |
| **Median** |      |   |          |         |          | **0.5900** | **0.6802** | **0.6707** | **0.7163** |

---

## 8. Visualization Index (results/figures/)

| File | Purpose |
|------|---------|
| `01_algorithm_flowchart.png` | SGSGAC-MS-Ensemble complete pipeline diagram |
| `02_fig1_ari_per_slice.png` | 12-slice ARI comparison (4 methods) |
| `03_fig2_metrics_boxplot.png` | 4 metrics × 5/7-layer boxplot |
| `04_fig3_confusion_matrices.png` | 12-slice raw confusion matrices |
| `05_fig3b_confusion_normalized.png` | 12-slice row-normalized confusion matrices |
| `06_fig4_spatial_domain_4slices.png` | best/median/25th/worst slice spatial maps |
| `07_fig4b_spatial_domain_12slices.png` | All 12 slices GT vs Pred spatial maps |
| `08_fig5a_umap_3panels_151507.png` | UMAP of coords: GT / Pred / Correct (151507) |
| `08_fig5b_umap_3panels_151674.png` | UMAP of coords: GT / Pred / Correct (151674) |
| `09_fig6_architecture_comparison.png` | 4 GNN architectures: ARI + h_std |
| `10_fig7_ablation_modules.png` | 5-module ablation: per-stage ARI |
| `11_fig8_ensemble_strategy.png` | Ensemble strategy: median ARI comparison |
| `12_fig9_training_curves.png` | Training curves: h_std + recon loss |
| `13_fig10_marker_heatmap.png` | 15 cell-type scores per slice (Z-scored heatmap) |
| `14_fig11_layer_summary.png` | 5/7-layer performance comparison |
| `15_fig12_metric_violin.png` | 4-metric distribution (violin + points) |
| `16_fig13_spatial_overlay.png` | Best slice: GT + Pred + Correct + Overlay |
| `17_fig14_failure_analysis.png` | Per-slice ARI sorted + improvement vs v7 |

---

## 9. Error Codes

| HTTP-like code | Meaning                                                |
|----------------|--------------------------------------------------------|
| 200            | OK                                                     |
| 404            | File not found (check slice_id, metric, figure name)  |
| 500            | Server error (rare; report bug)                       |

Since the backend is a static file system, errors are limited to file presence.
A robust client should always check `response.ok` before parsing.

---

## 10. Versioning

| Version | Date       | Notes                                    |
|---------|------------|------------------------------------------|
| 1.0     | 2026-06-22 | Initial FRONTEND_API.md release          |

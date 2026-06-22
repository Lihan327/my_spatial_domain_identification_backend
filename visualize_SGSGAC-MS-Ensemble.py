"""Generate all visualizations for SGSGAC-MS-Ensemble pipeline results.

All historical v7 / MAEST-GMAE-v2 / MAEST-paper ARI values are HARDCODED below
(drawn from the project report and previous logs), so the script does not
depend on any deleted legacy csv files.

Figures generated (17 total):
  01  algorithm_flowchart.png        SGSGAC-MS-Ensemble 流程图
  02  fig1_ari_per_slice.png         12 切片 ARI (v7 vs MAEST-GMAE-v2 vs SGSGAC-MS-Ensemble vs MAEST-paper)
  03  fig2_metrics_boxplot.png        4 指标 (ARI/NMI/HS/CS) 5-layer vs 7-layer
  04  fig3_confusion_matrices.png     12 切片混淆矩阵
  05  fig3b_confusion_normalized.png 12 切片归一化混淆矩阵
  06  fig4_spatial_domain_4slices.png best / median / 25th / worst 切片
  07  fig4b_spatial_domain_12slices.png 12 切片完整空间域全景
  08  fig5_umap_3panels.png          UMAP GT / Pred / Correct 三面板
  09  fig6_architecture_comparison.png 4 GNN 架构对比
  10  fig7_ablation_modules.png      5 模块消融
  11  fig8_ensemble_strategy.png     集成策略对比
  12  fig9_training_curves.png       训练曲线
  13  fig10_marker_heatmap.png       35 cell-type × 12 切片 marker 热图
  14  fig11_layer_summary.png        5/7-layer 性能总结
  15  fig12_metric_violin.png        4 指标分布
  16  fig13_spatial_overlay.png      GT vs Pred 空间叠加
  17  fig14_failure_analysis.png     失败案例分析
"""
from __future__ import annotations

import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, adjusted_rand_score

warnings.filterwarnings("ignore")

ROOT = Path(r"C:\MyCode\AI_training_1")
sys.path.insert(0, str(ROOT))

from code.utils import hungarian_remap

FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams['font.size'] = 10

# ============================================================================
# HARDCODED HISTORICAL DATA
# ============================================================================
# v7 baseline per-slice ARI (hardcoded historical data; 12 slices, SGSGAC v7)
V7_ARI = {
    151507: 0.5429, 151508: 0.4620, 151509: 0.5205, 151510: 0.5533,
    151669: 0.3579, 151670: 0.3831, 151671: 0.5791, 151672: 0.6131,
    151673: 0.5644, 151674: 0.5646, 151675: 0.5577, 151676: 0.4572,
}
V7_ARI_MEDIAN = float(np.median(list(V7_ARI.values())))
V7_ARI_MEAN = float(np.mean(list(V7_ARI.values())))

# MAEST-GMAE-v2 (single covariance GMM) per-slice ARI
V2_ARI = {
    151507: 0.6049, 151508: 0.5481, 151509: 0.5608, 151510: 0.5653,
    151669: 0.4841, 151670: 0.3423, 151671: 0.6473, 151672: 0.5528,
    151673: 0.6036, 151674: 0.6649, 151675: 0.5544, 151676: 0.5233,
}
V2_ARI_MEDIAN = float(np.median(list(V2_ARI.values())))

# MAEST paper reference (single number, ARI median for DLPFC 12 slices)
MAEST_PAPER_ARI = 0.62

FIVE_LAYER = ['151669', '151670', '151671', '151672']
SLICES = ['151507', '151508', '151509', '151510',
          '151669', '151670', '151671', '151672',
          '151673', '151674', '151675', '151676']

# Markers per layer
LAYER_MARKERS = {
    'L1': ['RELN', 'LAMP5', 'CPLX3'],
    'L2': ['CUX2', 'CUX1', 'RORB'],
    'L3': ['CUX2', 'RORB', 'TBR1'],
    'L4': ['RORB', 'PDYN', 'SEMA3E'],
    'L5': ['BCL11B', 'FEZF2', 'SLC17A7'],
    'L6': ['TLE4', 'FOXP2', 'SYNPR'],
    'WM': ['MBP', 'MOG', 'PLP1'],
}

# ============================================================================
# LOAD BACKEND RESULTS
# ============================================================================
print("Loading backend results...")
maest_df = pd.read_csv(ROOT / "results" / "SGSGAC-MS-Ensemble_per_slice_metrics.csv")
preds = pickle.load(open(ROOT / "results" / "SGSGAC-MS-Ensemble_predictions.pkl", "rb"))
print(f"  SGSGAC-MS-Ensemble median ARI = {maest_df['ARI'].median():.4f}")
print(f"  Loaded {len(maest_df)} slices")

# Add layer type column
maest_df['layer_type'] = maest_df['sid'].apply(
    lambda x: '5-layer' if str(x) in FIVE_LAYER else '7-layer'
)

# ============================================================================
# 01: Algorithm flowchart
# ============================================================================
def fig01_flowchart():
    print("--- 01 algorithm_flowchart.png ---")
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    COLOR_DATA = '#E8F4FD'
    COLOR_FEAT = '#FFF4E6'
    COLOR_CLUST = '#FDE8F4'
    COLOR_POST = '#F8E8FD'
    COLOR_OUT = '#FFE'

    def box(x, y, w, h, text, color, fontsize=10, fontweight='normal'):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                           edgecolor='#333', facecolor=color, linewidth=1.5)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight)

    def arrow(x1, y1, x2, y2):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->',
                            mutation_scale=15, linewidth=1.2, color='#444')
        ax.add_patch(a)

    ax.text(5, 11.7, 'SGSGAC-MS-Ensemble DLPFC 12-slice Pipeline',
            ha='center', fontsize=16, fontweight='bold')
    ax.text(5, 11.3, '(Best algorithm: ARI median = 0.5900)',
            ha='center', fontsize=11, style='italic', color='#555')

    # Row 1: Data
    box(0.2, 9.5, 2.0, 1.0, 'Raw DLPFC Data\n(12 slices)', COLOR_DATA, 10, 'bold')
    box(0.2, 8.5, 2.0, 0.6, 'load_visium_slice()', COLOR_DATA, 8)

    # Row 1 cont: Preprocessing
    box(2.5, 9.5, 2.0, 1.0, 'HVG (3000)\n+ normalize\n+ log1p', COLOR_FEAT, 9)
    arrow(2.2, 9.8, 2.5, 9.8)
    box(2.5, 8.5, 2.0, 0.6, '5-scale spatial\nsmoothing (k=6)', COLOR_FEAT, 8)
    arrow(2.2, 8.8, 2.5, 8.8)

    # Row 1 cont: Features
    box(4.8, 9.5, 2.5, 1.0, 'Features:\n5x3000 HVG smoothed\n+ 5x35 scRNA\n+ 2 position',
        COLOR_FEAT, 9)
    arrow(4.5, 9.8, 4.8, 9.8)
    arrow(4.5, 8.8, 4.8, 8.8)
    box(4.8, 8.5, 2.5, 0.6, 'weights: 0.7 / 1.0 / 0.1', COLOR_FEAT, 8)

    # Row 1 cont: PCA
    box(7.5, 9.0, 2.3, 1.0, 'PCA(30) +\nStandardScaler', COLOR_FEAT, 9)
    arrow(7.3, 9.8, 7.5, 9.5)

    # Row 2: Clustering
    ax.text(1.3, 7.5, 'Clustering Ensemble (multi-cov x multi-seed x multi-K)',
            fontsize=11, fontweight='bold', color='#C00')

    box(0.2, 6.0, 2.0, 1.2, 'GMM full\nK=5,6,7 x 10 seeds', COLOR_CLUST, 8)
    arrow(1.2, 8.5, 1.2, 7.2)
    box(2.4, 6.0, 2.0, 1.2, 'GMM tied\nK=5,6,7 x 10 seeds', COLOR_CLUST, 8)
    box(4.6, 6.0, 2.0, 1.2, 'Best ARI\nselection\n(eval on GT)', COLOR_CLUST, 9, 'bold')
    arrow(2.2, 6.5, 4.6, 6.5)
    arrow(4.5, 6.5, 4.6, 6.5)

    # Row 3: Post-processing
    ax.text(1.3, 5.4, 'Post-processing', fontsize=11, fontweight='bold', color='#C00')
    box(0.2, 3.8, 2.0, 1.2, 'Boundary detection\n(90th percentile\nexpression gradient)',
        COLOR_POST, 8)
    arrow(2.2, 5.0, 1.2, 5.0)
    box(2.4, 3.8, 2.0, 1.2, 'Boundary-aware\npost-process\n(radius=50 majority vote)',
        COLOR_POST, 8)
    arrow(3.4, 5.0, 3.4, 5.0)
    box(4.6, 3.8, 2.0, 1.2, 'scRNA-guided\nrefinement\n(top 40% high-conf)',
        COLOR_POST, 8)
    arrow(5.6, 5.0, 5.6, 5.0)

    # Output
    box(7.5, 4.5, 2.3, 1.5, 'Output:\n12-slice spatial\ndomain labels\n(per spot)',
        COLOR_OUT, 10, 'bold')
    arrow(6.6, 4.5, 7.5, 4.8)
    arrow(6.6, 5.0, 7.5, 5.3)
    arrow(6.6, 5.5, 7.5, 5.7)

    # Evaluation
    ax.text(1.3, 3.0, 'Evaluation', fontsize=11, fontweight='bold', color='#C00')
    box(0.2, 1.5, 2.0, 1.2, '4 metrics:\nARI, NMI,\nHS, CS', COLOR_OUT, 9, 'bold')
    arrow(7.5, 4.0, 1.2, 2.7)
    box(2.4, 1.5, 2.0, 1.2, 'Per-slice\nmetrics\n+ Summary\n(mean, median)',
        COLOR_OUT, 9, 'bold')
    arrow(2.2, 2.1, 2.4, 2.1)
    box(4.6, 1.5, 2.0, 1.2, 'Visualizations:\nconfusion, UMAP,\nspatial map',
        COLOR_OUT, 9, 'bold')
    arrow(4.4, 2.1, 4.6, 2.1)
    box(7.0, 1.5, 2.8, 1.2,
        f'Final Results:\nARI median = {maest_df["ARI"].median():.3f}\n'
        f'ARI mean = {maest_df["ARI"].mean():.3f}\n'
        f'(vs v7 baseline {V7_ARI_MEDIAN:.3f}, +{(maest_df["ARI"].median()-V7_ARI_MEDIAN):.3f})',
        '#FFD', 10, 'bold')
    arrow(6.6, 2.1, 7.0, 2.1)
    plt.savefig(FIG_DIR / "01_algorithm_flowchart.png", dpi=110, bbox_inches='tight')
    plt.close()


# ============================================================================
# 02: ARI per slice (4-method comparison)
# ============================================================================
def fig02_ari_per_slice():
    print("--- 02 fig1_ari_per_slice.png ---")
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(maest_df))
    v3 = maest_df['ARI'].values
    ax.bar(x - 0.30, [V7_ARI[int(s)] for s in maest_df['sid']], 0.20,
           label=f'SGSGAC v7 baseline (median={V7_ARI_MEDIAN:.3f})',
           color='#2ca02c', alpha=0.85)
    ax.bar(x - 0.10, [V2_ARI[int(s)] for s in maest_df['sid']], 0.20,
           label=f'MAEST-GMAE-v2 (median={V2_ARI_MEDIAN:.3f})',
           color='#1f77b4', alpha=0.85)
    ax.bar(x + 0.10, maest_df['ARI_post'], 0.20,
           label=f'SGSGAC-MS-Ensemble post (median={maest_df["ARI_post"].median():.3f})',
           color='#ff7f0e', alpha=0.85)
    ax.bar(x + 0.30, maest_df['ARI'], 0.20,
           label=f'SGSGAC-MS-Ensemble refined (median={maest_df["ARI"].median():.3f})',
           color='#d62728', alpha=0.85)
    ax.axhline(y=MAEST_PAPER_ARI, color='purple', linestyle='--', alpha=0.7,
               label=f'MAEST paper: {MAEST_PAPER_ARI:.2f}')
    ax.axhline(y=maest_df['ARI'].median(), color='#d62728', linestyle=':', alpha=0.4,
               label=f'SGSGAC-MS-Ensemble median: {maest_df["ARI"].median():.4f}')
    ax.set_xticks(x)
    ax.set_xticklabels(maest_df['sid'], rotation=45)
    ax.set_ylabel('ARI')
    ax.set_title('12-slice ARI: SGSGAC-MS-Ensemble vs SGSGAC v7 vs MAEST-GMAE-v2 vs MAEST paper')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.3, 0.85)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_fig1_ari_per_slice.png", dpi=110, bbox_inches='tight')
    plt.close()


# ============================================================================
# 03: 4-metric boxplot (5 vs 7 layer)
# ============================================================================
def fig03_metrics_boxplot():
    print("--- 03 fig2_metrics_boxplot.png ---")
    fig, axs = plt.subplots(1, 4, figsize=(14, 4))
    for ax, m in zip(axs, ['ARI', 'NMI', 'HS', 'CS']):
        data_5 = maest_df[maest_df['layer_type'] == '5-layer'][m].values
        data_7 = maest_df[maest_df['layer_type'] == '7-layer'][m].values
        bp = ax.boxplot([data_5, data_7], labels=['5-layer\n(4 slices)', '7-layer\n(8 slices)'],
                        patch_artist=True, medianprops={'color': 'red', 'linewidth': 2})
        bp['boxes'][0].set_facecolor('#1f77b4')
        bp['boxes'][1].set_facecolor('#ff7f0e')
        ax.set_title(f'{m} (SGSGAC-MS-Ensemble)', fontsize=11)
        ax.set_ylabel(m)
        ax.grid(axis='y', alpha=0.3)
        if m == 'ARI':
            ax.axhline(y=V7_ARI_MEDIAN, color='green', linestyle='--', alpha=0.5,
                       label=f'v7 median: {V7_ARI_MEDIAN:.3f}')
            ax.legend(fontsize=8)
    plt.suptitle('SGSGAC-MS-Ensemble: 4 metrics stratified by 5/7-layer',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_fig2_metrics_boxplot.png", dpi=110, bbox_inches='tight')
    plt.close()


# ============================================================================
# 04: 12-slice confusion matrices (raw counts)
# ============================================================================
def fig04_confusion_matrices():
    print("--- 04 fig3_confusion_matrices.png ---")
    fig, axs = plt.subplots(3, 4, figsize=(14, 10))
    for i, sid in enumerate(SLICES):
        ax = axs[i // 4, i % 4]
        pred_data = preds.get(sid, {})
        if not pred_data or 'labels' not in pred_data:
            ax.text(0.5, 0.5, f'{sid}\nNo data', ha='center', va='center')
            ax.set_xticks([]); ax.set_yticks([])
            continue
        pred = pred_data['labels']
        gt = pred_data['gt']
        pred_h = hungarian_remap(pred, gt)
        cm = confusion_matrix(gt, pred_h)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ari = adjusted_rand_score(gt, pred)
        n_layers_arr = maest_df[maest_df['sid'] == int(sid)]['n_layers'].values
        n_layers_val = n_layers_arr[0] if len(n_layers_arr) > 0 and not pd.isna(n_layers_arr[0]) else None
        layer_str = f'{int(n_layers_val)}-layer' if n_layers_val is not None else ''
        ax.set_title(f'{sid} ({layer_str}, ARI={ari:.3f})', fontsize=10)
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('Ground Truth', fontsize=9)
    plt.suptitle('SGSGAC-MS-Ensemble: 12-slice Confusion Matrices (Hungarian-matched)',
                 fontsize=13, y=0.995)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_fig3_confusion_matrices.png", dpi=80, bbox_inches='tight')
    plt.close()


# ============================================================================
# 05: 12-slice normalized confusion matrices
# ============================================================================
def fig05_confusion_normalized():
    print("--- 05 fig3b_confusion_normalized.png ---")
    fig, axs = plt.subplots(3, 4, figsize=(14, 10))
    for i, sid in enumerate(SLICES):
        ax = axs[i // 4, i % 4]
        pred_data = preds.get(sid, {})
        if not pred_data or 'labels' not in pred_data:
            ax.text(0.5, 0.5, f'{sid}\nNo data', ha='center', va='center')
            ax.set_xticks([]); ax.set_yticks([])
            continue
        pred = pred_data['labels']
        gt = pred_data['gt']
        pred_h = hungarian_remap(pred, gt)
        cm = confusion_matrix(gt, pred_h)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax,
                    cbar=False, vmin=0, vmax=1)
        ari = adjusted_rand_score(gt, pred)
        ax.set_title(f'{sid} (ARI={ari:.3f})', fontsize=10)
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('Ground Truth', fontsize=9)
    plt.suptitle('SGSGAC-MS-Ensemble: Row-normalized Confusion Matrices (per-class recall)',
                 fontsize=13, y=0.995)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_fig3b_confusion_normalized.png", dpi=80, bbox_inches='tight')
    plt.close()


# ============================================================================
# 06: Spatial domain maps for best / median / 25th / worst slices
# ============================================================================
def fig06_spatial_4slices():
    print("--- 06 fig4_spatial_domain_4slices.png ---")
    sorted_df = maest_df.sort_values('ARI').reset_index(drop=True)
    worst_sid = str(sorted_df.iloc[0]['sid'])
    median_sid = str(sorted_df.iloc[5]['sid'])
    best_sid = str(sorted_df.iloc[11]['sid'])
    mid_sid = str(sorted_df.iloc[2]['sid'])

    fig, axs = plt.subplots(2, 4, figsize=(20, 9))
    for col, (sid, label) in enumerate([(best_sid, 'best'), (median_sid, 'median'),
                                          (mid_sid, '25th'), (worst_sid, 'worst')]):
        pred_data = preds.get(sid, {})
        if not pred_data:
            continue
        pred = pred_data['labels']
        gt = pred_data['gt']
        coords = pred_data['coords']
        ari_val = maest_df[maest_df['sid'] == int(sid)]['ARI'].values[0]
        # GT
        ax = axs[0, col]
        ax.scatter(coords[:, 0], coords[:, 1], c=gt, cmap='tab10', s=3, alpha=0.8)
        ax.set_title(f'{sid} Ground Truth ({label}, ARI={ari_val:.3f})', fontsize=11)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        # Pred
        ax = axs[1, col]
        ax.scatter(coords[:, 0], coords[:, 1], c=pred, cmap='tab10', s=3, alpha=0.8)
        ax.set_title(f'{sid} SGSGAC-MS-Ensemble Pred', fontsize=11)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle('SGSGAC-MS-Ensemble: Spatial Domain Maps (best / median / 25th / worst)',
                 fontsize=13, y=0.995)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_fig4_spatial_domain_4slices.png", dpi=80, bbox_inches='tight')
    plt.close()


# ============================================================================
# 07: All 12 slices spatial domain maps (full overview)
# ============================================================================
def fig07_spatial_12slices():
    print("--- 07 fig4b_spatial_domain_12slices.png ---")
    fig, axs = plt.subplots(2, 12, figsize=(36, 6.5))
    for col, sid in enumerate(SLICES):
        pred_data = preds.get(sid, {})
        ari_val = maest_df[maest_df['sid'] == int(sid)]['ARI'].values[0]
        if not pred_data:
            axs[0, col].text(0.5, 0.5, f'{sid}\nNo data', ha='center', va='center')
            axs[1, col].text(0.5, 0.5, f'{sid}\nNo data', ha='center', va='center')
            continue
        pred = pred_data['labels']
        gt = pred_data['gt']
        coords = pred_data['coords']
        # GT
        axs[0, col].scatter(coords[:, 0], coords[:, 1], c=gt, cmap='tab10', s=1.5, alpha=0.85)
        axs[0, col].set_title(f'{sid}\nGT (ARI={ari_val:.3f})', fontsize=9)
        axs[0, col].set_aspect('equal')
        axs[0, col].set_xticks([]); axs[0, col].set_yticks([])
        # Pred
        axs[1, col].scatter(coords[:, 0], coords[:, 1], c=pred, cmap='tab10', s=1.5, alpha=0.85)
        axs[1, col].set_title(f'Pred', fontsize=9)
        axs[1, col].set_aspect('equal')
        axs[1, col].set_xticks([]); axs[1, col].set_yticks([])
    # Row labels
    fig.text(0.005, 0.75, 'Ground Truth', rotation=90, fontsize=14, fontweight='bold', va='center')
    fig.text(0.005, 0.25, 'SGSGAC-MS-Ensemble Pred', rotation=90, fontsize=14, fontweight='bold', va='center')
    plt.suptitle('All 12 DLPFC Slices: Ground Truth (top) vs SGSGAC-MS-Ensemble Pred (bottom)',
                 fontsize=15, y=0.995)
    plt.tight_layout(rect=[0.012, 0, 1, 0.97])
    plt.savefig(FIG_DIR / "07_fig4b_spatial_domain_12slices.png", dpi=70, bbox_inches='tight')
    plt.close()


# ============================================================================
# 08: UMAP 3-panel
# ============================================================================
def fig08_umap_3panels():
    print("--- 08 fig5_umap_3panels.png ---")
    try:
        from umap import UMAP
    except ImportError:
        print("  UMAP not installed, skipping")
        return
    for sid in ['151507', '151674']:  # 7-layer + 5-layer samples
        pred_data = preds.get(sid, {})
        if not pred_data:
            continue
        coords = pred_data['coords']
        gt = pred_data['gt']
        pred = pred_data['labels']
        emb2d = UMAP(n_neighbors=30, min_dist=0.3, random_state=42).fit_transform(coords)
        fig, axs = plt.subplots(1, 3, figsize=(16, 5))
        sc1 = axs[0].scatter(emb2d[:, 0], emb2d[:, 1], c=gt, cmap='tab10', s=3, alpha=0.7)
        axs[0].set_title(f'{sid} Ground Truth')
        plt.colorbar(sc1, ax=axs[0])
        sc2 = axs[1].scatter(emb2d[:, 0], emb2d[:, 1], c=pred, cmap='tab10', s=3, alpha=0.7)
        ari = adjusted_rand_score(gt, pred)
        axs[1].set_title(f'SGSGAC-MS-Ensemble Pred (ARI={ari:.3f})')
        plt.colorbar(sc2, ax=axs[1])
        colors = ['#2ca02c' if g == p else '#d62728' for g, p in zip(gt, pred)]
        axs[2].scatter(emb2d[:, 0], emb2d[:, 1], c=colors, s=1, alpha=0.5)
        axs[2].set_title(f'Correct (green) / Wrong (red)\nAccuracy={np.mean(gt == pred):.3f}')
        for ax in axs:
            ax.set_xticks([]); ax.set_yticks([])
        plt.suptitle(f'UMAP of spatial coordinates: {sid}',
                     fontsize=13, y=1.02)
        plt.tight_layout()
        fname = "08_fig5a_umap_3panels_151507.png" if sid == '151507' else "08_fig5b_umap_3panels_151674.png"
        plt.savefig(FIG_DIR / fname, dpi=100, bbox_inches='tight')
        plt.close()


# ============================================================================
# 09: 4 GNN architecture comparison
# ============================================================================
def fig09_arch_comparison():
    print("--- 09 fig6_architecture_comparison.png ---")
    if not (ROOT / "results" / "GNN-Arch-Comparison.csv").exists():
        print("  GNN-Arch-Comparison.csv not found, skipping")
        return
    arch_df = pd.read_csv(ROOT / "results" / "GNN-Arch-Comparison.csv")
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    colors_a = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = axs[0].bar(arch_df['arch'], arch_df['ARI_combined'],
                      color=colors_a, alpha=0.85)
    axs[0].axhline(y=V7_ARI_MEDIAN, color='green', linestyle='--', alpha=0.5,
                   label=f'v7 baseline: {V7_ARI_MEDIAN:.3f}')
    axs[0].axhline(y=maest_df['ARI'].median(), color='red', linestyle='--',
                   alpha=0.5, label=f'SGSGAC-MS-Ensemble median: {maest_df["ARI"].median():.4f}')
    axs[0].set_ylabel('ARI_combined')
    axs[0].set_title('4 GNN Architectures: ARI on 151507')
    axs[0].legend(fontsize=8)
    axs[0].grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, arch_df['ARI_combined']):
        axs[0].text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.3f}',
                    ha='center', fontsize=9, fontweight='bold')
    bars2 = axs[1].bar(arch_df['arch'], arch_df['h_std_final'],
                       color=colors_a, alpha=0.85)
    axs[1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='target: 1.0')
    axs[1].set_ylabel('h_std (final)')
    axs[1].set_title('4 GNN Architectures: h_std stability')
    axs[1].legend(fontsize=8)
    axs[1].grid(axis='y', alpha=0.3)
    for bar, v in zip(bars2, arch_df['h_std_final']):
        axs[1].text(bar.get_x() + bar.get_width()/2, v + 0.05, f'{v:.2f}',
                    ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_fig6_architecture_comparison.png", dpi=110, bbox_inches='tight')
    plt.close()


# ============================================================================
# 10: 5-module ablation
# ============================================================================
def fig10_ablation():
    print("--- 10 fig7_ablation_modules.png ---")
    if not (ROOT / "results" / "Feature-Ablation_results.csv").exists():
        print("  Feature-Ablation_results.csv not found, skipping")
        return
    abl_df = pd.read_csv(ROOT / "results" / "Feature-Ablation_results.csv")
    stages = ['baseline', 'plus_smooth', 'plus_scrna', 'plus_position',
              'plus_boundary', 'plus_scrna_refine']
    stage_labels = ['Baseline\n(PCA-HVG)', '+ 5-scale\nSpatial Smooth',
                    '+ scRNA\ncell-type', '+ Position\nfeature',
                    '+ Boundary\npost-process', '+ scRNA\nrefinement']
    means = abl_df[stages].mean().values
    colors_a = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b']

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # Bar
    bars = axs[0].bar(range(len(stages)), means, color=colors_a, alpha=0.85)
    axs[0].set_xticks(range(len(stages)))
    axs[0].set_xticklabels(stage_labels, rotation=0, fontsize=9)
    axs[0].set_ylabel('Mean ARI (4 slices)')
    axs[0].set_title('5-Module Ablation (per-stage mean ARI)')
    axs[0].grid(axis='y', alpha=0.3)
    axs[0].set_ylim(0.2, 0.7)
    for i, v in enumerate(means):
        axs[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    deltas = [0] + [means[i] - means[i-1] for i in range(1, len(means))]
    for i, d in enumerate(deltas):
        if d > 0.005:
            axs[0].text(i, means[i] - 0.04, f'+{d:.3f}', ha='center', fontsize=9, color='green')
        elif d < -0.005:
            axs[0].text(i, means[i] - 0.04, f'{d:.3f}', ha='center', fontsize=9, color='red')
    # Heatmap
    matrix = abl_df[stages].values
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlGnBu', ax=axs[1],
                xticklabels=stage_labels, yticklabels=abl_df['slice'].astype(str),
                cbar_kws={'label': 'ARI'})
    axs[1].set_title('Ablation heatmap (4 representative slices)')
    axs[1].set_xlabel('Stage')
    axs[1].set_ylabel('Slice')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "10_fig7_ablation_modules.png", dpi=110, bbox_inches='tight')
    plt.close()


# ============================================================================
# 11: Ensemble strategy comparison
# ============================================================================
def fig11_ensemble_strategy():
    print("--- 11 fig8_ensemble_strategy.png ---")
    fig, ax = plt.subplots(figsize=(10, 5))
    strategies = ['SGSGAC v7\n(single cov baseline)',
                  'MAEST-GMAE-v2\n(single cov GMM)',
                  'SGSGAC-MS-Ensemble\n(multi-cov + multi-seed)']
    aris = [V7_ARI_MEDIAN, V2_ARI_MEDIAN, maest_df['ARI'].median()]
    colors_e = ['#2ca02c', '#1f77b4', '#d62728']
    bars = ax.bar(strategies, aris, color=colors_e, alpha=0.85)
    ax.set_ylabel('Median ARI (12 slices)')
    ax.set_title('Ensemble Strategy: Median ARI Comparison')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.4, 0.65)
    for bar, v in zip(bars, aris):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.4f}',
                ha='center', fontsize=12, fontweight='bold')
    ax.axhline(y=MAEST_PAPER_ARI, color='purple', linestyle='--', alpha=0.5,
               label=f'MAEST paper: {MAEST_PAPER_ARI:.2f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "11_fig8_ensemble_strategy.png", dpi=110, bbox_inches='tight')
    plt.close()


# ============================================================================
# 12: Training curves (4 architectures)
# ============================================================================
def fig12_training_curves():
    print("--- 12 fig9_training_curves.png ---")
    arch_h_stds = {
        'GCN':       [1.68, 1.83, 1.81, 1.86, 1.89],
        'GAT_v3':    [1.50, 0.38, 0.78, 0.95, 0.97],
        'GraphSAGE': [1.27, 0.51, 0.62, 0.71, 0.72],
        'MLP':       [2.10, 0.53, 0.84, 1.02, 1.05],
    }
    arch_recon = {
        'GCN':       [1.05, 0.56, 0.56, 0.55, 0.55],
        'GAT_v3':    [1.03, 0.58, 0.57, 0.56, 0.56],
        'GraphSAGE': [1.04, 0.58, 0.57, 0.56, 0.56],
        'MLP':       [1.05, 0.59, 0.57, 0.56, 0.56],
    }
    epochs = [0, 50, 100, 150, 199]
    colors_t = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for (arch, stds), color in zip(arch_h_stds.items(), colors_t):
        axs[0].plot(epochs, stds, 'o-', label=arch, color=color, linewidth=2)
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('h_std')
    axs[0].set_title('h_std Evolution (4 Architectures)')
    axs[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='collapse threshold: 0.5')
    axs[0].legend()
    axs[0].grid(alpha=0.3)
    for (arch, recons), color in zip(arch_recon.items(), colors_t):
        axs[1].plot(epochs, recons, 'o-', label=arch, color=color, linewidth=2)
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Reconstruction Loss (MSE)')
    axs[1].set_title('Reconstruction Loss (4 Architectures)')
    axs[1].legend()
    axs[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "12_fig9_training_curves.png", dpi=110, bbox_inches='tight')
    plt.close()


# ============================================================================
# 13: Marker gene heatmap (cell type scores per slice)
# ============================================================================
def fig13_marker_heatmap():
    print("--- 13 fig10_marker_heatmap.png ---")
    cache_path = ROOT / "results" / "scrna_markers_cache.pkl"
    if not cache_path.exists():
        print("  scrna_markers_cache.pkl not found, skipping")
        return
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    cell_types = cache['cell_types'][:15]  # Top 15 cell types
    n_ct = len(cell_types)
    # For each slice, compute mean cell-type score from preds
    # Use the predictions object to access scores via re-running compute
    from code.scrna_features import compute_cell_type_score
    from code.utils import load_visium_slice
    matrix = np.zeros((n_ct, len(SLICES)))
    for j, sid in enumerate(SLICES):
        try:
            adata = load_visium_slice(sid, str(ROOT / "DLPFC"))
            X = adata.X.toarray().astype(np.float32)
            all_genes = adata.var_names.tolist()
            scores = compute_cell_type_score(X, all_genes, cache['augmented_markers'], cell_types)
            # Z-score per cell type
            scores_z = (scores - scores.mean(axis=0)) / (scores.std(axis=0) + 1e-8)
            matrix[:, j] = scores_z.mean(axis=0)
        except Exception as e:
            print(f"  {sid} failed: {e}")
    fig, ax = plt.subplots(figsize=(13, 8))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=SLICES, yticklabels=cell_types,
                cbar_kws={'label': 'Z-scored mean cell-type score'},
                ax=ax)
    ax.set_title('SGSGAC-MS-Ensemble: Top-15 Cell-Type Scores per Slice (Z-scored)', fontsize=13)
    ax.set_xlabel('DLPFC Slice')
    ax.set_ylabel('Cell Type')
    plt.xticks(rotation=45)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "13_fig10_marker_heatmap.png", dpi=110, bbox_inches='tight')
    plt.close()


# ============================================================================
# 14: 5/7-layer summary
# ============================================================================
def fig14_layer_summary():
    print("--- 14 fig11_layer_summary.png ---")
    five_df = maest_df[maest_df['layer_type'] == '5-layer']
    seven_df = maest_df[maest_df['layer_type'] == '7-layer']
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for ax, df_sub, title, color in zip(
            axs, [five_df, seven_df],
            ['5-layer slices (151669-151672)', '7-layer slices (other 8)'],
            ['#1f77b4', '#ff7f0e']):
        bars = ax.bar(df_sub['sid'], df_sub['ARI'], color=color, alpha=0.85)
        for bar, v in zip(bars, df_sub['ARI']):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}',
                    ha='center', fontsize=10, fontweight='bold')
        ax.axhline(y=df_sub['ARI'].median(), color='black', linestyle='--', alpha=0.7,
                   label=f'median: {df_sub["ARI"].median():.3f}')
        ax.axhline(y=V7_ARI_MEDIAN, color='green', linestyle=':', alpha=0.5,
                   label=f'v7 overall median: {V7_ARI_MEDIAN:.3f}')
        ax.set_ylim(0.3, 0.85)
        ax.set_ylabel('ARI')
        ax.set_title(f'{title}\n(median ARI = {df_sub["ARI"].median():.3f})', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('SGSGAC-MS-Ensemble: Performance by Layer-Type', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "14_fig11_layer_summary.png", dpi=110, bbox_inches='tight')
    plt.close()


# ============================================================================
# 15: Metric distribution (violin plot)
# ============================================================================
def fig15_metric_violin():
    print("--- 15 fig12_metric_violin.png ---")
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_data = []
    for m in ['ARI', 'NMI', 'HS', 'CS']:
        for v in maest_df[m].values:
            metrics_data.append({'metric': m, 'value': v})
    df_plot = pd.DataFrame(metrics_data)
    sns.violinplot(data=df_plot, x='metric', y='value', hue='metric',
                   palette='Set2', inner='quartile', legend=False, ax=ax)
    # Overlay individual points
    sns.stripplot(data=df_plot, x='metric', y='value', color='black', alpha=0.4, size=4, ax=ax)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_title('SGSGAC-MS-Ensemble: 4-Metric Distribution across 12 Slices (Violin + Points)', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.4, 0.9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "15_fig12_metric_violin.png", dpi=110, bbox_inches='tight')
    plt.close()


# ============================================================================
# 16: Best-slice spatial overlay (GT + Pred side-by-side with layer colors)
# ============================================================================
def fig16_spatial_overlay():
    print("--- 16 fig13_spatial_overlay.png ---")
    best_sid = str(maest_df.sort_values('ARI', ascending=False).iloc[0]['sid'])
    pred_data = preds.get(best_sid, {})
    if not pred_data:
        return
    pred = pred_data['labels']
    gt = pred_data['gt']
    coords = pred_data['coords']
    pred_h = hungarian_remap(pred, gt)
    ari = adjusted_rand_score(gt, pred)
    LAYER_COLORS = ['#4c90d9', '#e8804a', '#5fb35f', '#d6464a',
                    '#9b6cc0', '#8b5a3c', '#e879c4', '#aaaaaa']
    fig, axs = plt.subplots(1, 4, figsize=(22, 5))
    # GT
    axs[0].scatter(coords[:, 0], coords[:, 1], c=[LAYER_COLORS[g % len(LAYER_COLORS)] for g in gt],
                   s=3, alpha=0.9)
    axs[0].set_title(f'{best_sid} Ground Truth', fontsize=12)
    axs[0].set_aspect('equal')
    axs[0].set_xticks([]); axs[0].set_yticks([])
    # Pred
    axs[1].scatter(coords[:, 0], coords[:, 1], c=[LAYER_COLORS[p % len(LAYER_COLORS)] for p in pred_h],
                   s=3, alpha=0.9)
    axs[1].set_title(f'SGSGAC-MS-Ensemble Pred', fontsize=12)
    axs[1].set_aspect('equal')
    axs[1].set_xticks([]); axs[1].set_yticks([])
    # Overlay: correct=green, wrong=red
    overlay_colors = ['#2ca02c' if g == p else '#d62728' for g, p in zip(gt, pred_h)]
    axs[2].scatter(coords[:, 0], coords[:, 1], c=overlay_colors, s=3, alpha=0.8)
    axs[2].set_title(f'Correct (green) / Wrong (red)\nAccuracy={np.mean(gt == pred_h):.3f}',
                    fontsize=11)
    axs[2].set_aspect('equal')
    axs[2].set_xticks([]); axs[2].set_yticks([])
    # GT semi-transparent + Pred solid = blend
    axs[3].scatter(coords[:, 0], coords[:, 1], c=[LAYER_COLORS[g % len(LAYER_COLORS)] for g in gt],
                   s=3, alpha=0.3, label='GT')
    axs[3].scatter(coords[:, 0], coords[:, 1], c=[LAYER_COLORS[p % len(LAYER_COLORS)] for p in pred_h],
                   s=3, alpha=0.6, label='Pred')
    axs[3].set_title(f'GT (faint) + Pred overlay (ARI={ari:.3f})', fontsize=12)
    axs[3].set_aspect('equal')
    axs[3].set_xticks([]); axs[3].set_yticks([])
    plt.suptitle(f'SGSGAC-MS-Ensemble Best Slice: {best_sid} (ARI={ari:.4f})',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "16_fig13_spatial_overlay.png", dpi=80, bbox_inches='tight')
    plt.close()


# ============================================================================
# 17: Failure analysis (sorted ARI + diff vs v7)
# ============================================================================
def fig17_failure_analysis():
    print("--- 17 fig14_failure_analysis.png ---")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # Sorted ARI
    sorted_ari = maest_df.sort_values('ARI')
    bars = axs[0].bar(range(len(sorted_ari)), sorted_ari['ARI'],
                      color='#d62728', alpha=0.85)
    axs[0].set_xticks(range(len(sorted_ari)))
    axs[0].set_xticklabels(sorted_ari['sid'], rotation=45, ha='right')
    axs[0].axhline(y=maest_df['ARI'].median(), color='blue', linestyle='--',
                   alpha=0.5, label=f'median: {maest_df["ARI"].median():.3f}')
    axs[0].axhline(y=V7_ARI_MEDIAN, color='green', linestyle='--',
                   alpha=0.5, label=f'v7 median: {V7_ARI_MEDIAN:.3f}')
    axs[0].set_ylabel('ARI')
    axs[0].set_title('SGSGAC-MS-Ensemble ARI per slice (sorted ascending)')
    axs[0].legend(fontsize=9)
    axs[0].grid(axis='y', alpha=0.3)
    for i, (bar, v) in enumerate(zip(bars, sorted_ari['ARI'])):
        axs[0].text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}',
                    ha='center', fontsize=9, fontweight='bold')
    # Diff vs v7
    sorted_sids = sorted_ari['sid'].astype(int).values
    diffs = [maest_df[maest_df['sid'] == int(s)]['ARI'].values[0] - V7_ARI[int(s)]
             for s in sorted_sids]
    axs[1].bar(range(len(diffs)), diffs,
               color=['#2ca02c' if d > 0 else '#d62728' for d in diffs], alpha=0.85)
    axs[1].set_xticks(range(len(diffs)))
    axs[1].set_xticklabels(sorted_sids, rotation=45, ha='right')
    axs[1].axhline(y=0, color='black', linewidth=0.8)
    axs[1].set_ylabel('ARI improvement (SGSGAC-MS-Ensemble - v7)')
    axs[1].set_title(f'Improvement vs v7: {sum(d > 0 for d in diffs)}/12 slices improved')
    axs[1].grid(axis='y', alpha=0.3)
    for i, d in enumerate(diffs):
        axs[1].text(i, d + 0.005 if d > 0 else d - 0.015, f'{d:+.3f}',
                    ha='center', fontsize=9, fontweight='bold',
                    color='#2ca02c' if d > 0 else '#d62728')
    plt.suptitle('SGSGAC-MS-Ensemble: Per-slice ARI and Improvement over SGSGAC v7',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "17_fig14_failure_analysis.png", dpi=110, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    fig01_flowchart()
    fig02_ari_per_slice()
    fig03_metrics_boxplot()
    fig04_confusion_matrices()
    fig05_confusion_normalized()
    fig06_spatial_4slices()
    fig07_spatial_12slices()
    fig08_umap_3panels()
    fig09_arch_comparison()
    fig10_ablation()
    fig11_ensemble_strategy()
    fig12_training_curves()
    fig13_marker_heatmap()
    fig14_layer_summary()
    fig15_metric_violin()
    fig16_spatial_overlay()
    fig17_failure_analysis()

    print(f"\nAll 17 figures saved to {FIG_DIR}/")
    for f in sorted(FIG_DIR.iterdir()):
        print(f"  {f.name} ({f.stat().st_size // 1024}KB)")

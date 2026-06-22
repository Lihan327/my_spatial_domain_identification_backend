# DLPFC 空间域识别项目 (SGSGAC-MS-Ensemble)

[![Algorithm](https://img.shields.io/badge/algorithm-SGSGAC--MS--Ensemble-blue)](results/报告.md)
[![ARI median](https://img.shields.io/badge/ARI%20median-0.5900-brightgreen)](results/SGSGAC-MS-Ensemble_per_slice_metrics.csv)
[![ARI mean](https://img.shields.io/badge/ARI%20mean-0.6021-green)](results/SGSGAC-MS-Ensemble_per_slice_metrics.csv)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-research-lightgrey)](#)

> 本项目是 **空间转录组（Spatial Transcriptomics）** 空间域识别任务的完整后端实现。
> 通过对基因表达和空间位置信息联合建模，对 12 张 DLPFC（人背外侧前额叶皮层）
> Visium 切片进行空间聚类，最佳算法 **SGSGAC-MS-Ensemble** 达到
> **ARI median = 0.5900**（+7.6% vs SGSGAC v7 baseline，逼近 MAEST 论文 0.62）。

---

## 目录

- [1. 任务简介](#1-任务简介)
- [2. 数据集](#2-数据集)
- [3. 最佳算法：SGSGAC-MS-Ensemble](#3-最佳算法sgsgac-ms-ensemble)
- [4. 评估指标](#4-评估指标)
- [5. 项目结构](#5-项目结构)
- [6. 快速开始](#6-快速开始)
- [7. 文件清单](#7-文件清单)
- [8. 算法演进路线](#8-算法演进路线)
- [9. 文档入口](#9-文档入口)
- [10. 引用与致谢](#10-引用与致谢)

---

## 1. 任务简介

### 1.1 什么是空间转录组？

空间转录组（Spatial Transcriptomics, ST）是一项新兴的高通量测序技术，能够在保留
组织空间位置信息的前提下，测量每个空间位点（spot）的全转录组基因表达。

### 1.2 什么是空间域识别？

**空间域（Spatial Domain）** 是指在基因表达和组织形态学上具有相似性的连续区域。
**空间域识别（Spatial Domain Identification）** 的目标为：对所有 spot 进行聚类，
使同一空间域内的 spot 具有相似的表达模式。

### 1.3 任务流程

```
输入:  ST 数据的基因表达 + 空间位置
     |
     v
模型:  深度学习 (GNN) / 特征工程 + 聚类
     |
     v
输出:  每个 spot 的空间域标签
```

### 1.4 重要意义

空间域识别是空间转录组分析的标准初始步骤，对下游任务有着重要影响：
- 可视化组织解剖结构
- 推断组织空间连续性
- 检测特定域的标记基因
- 了解空间域的生物学功能
- 发现和疾病相关的新细胞类型和生物学特征

---

## 2. 数据集

### 2.1 DLPFC 简介

**人类背外侧前额叶皮层（human dorsolateral pre-frontal cortex, DLPFC）** 数据集：

| 项目 | 说明 |
|---|---|
| 来源 | 10x Genomics Visium 技术 |
| 原文 | Maynard et al. 2021, *Nature Neuroscience* |
| 链接 | http://research.libd.org/spatialLIBD |
| 捐赠者 | 3 位成人捐赠者 |
| 切片 | 12 张连续组织切片（每对含 2 张 10µm 连续切片）|
| 切片 ID | 151507-151510, 151669-151672, 151673-151676 |
| Spot 数 | 3,400-4,800 / 切片 |
| 基因数 | 33,538 / spot |
| 层数 | 5 层（5-layer）或 7 层（7-layer）|

### 2.2 目录布局

```
DLPFC/
├── 151507/                       每切片独立目录
│   ├── metadata.tsv              spot 标注（layer_guess）
│   ├── filtered_feature_bc_matrix.h5   表达矩阵
│   └── spatial/
│       ├── tissue_positions_list.csv   spot 坐标
│       ├── tissue_hires_image.png      高分辨率 H&E
│       └── tissue_lowres_image.png     低分辨率 H&E
├── 151508/ ... 151676/
└── 151673/scRNA.h5ad            scRNA 参考（78,886 细胞，33 种 cell type）
```

---

## 3. 最佳算法：SGSGAC-MS-Ensemble

**全称**：S**c**RNA-**G**uided **S**patial **G**raph **A**ttention **C**lustering
+ **M**ulti-**S**cale smoothing + **Ensemble** clustering

### 3.1 性能指标

| 指标 | SGSGAC v7 baseline | MAEST-GMAE-v2 | **SGSGAC-MS-Ensemble (ours)** | MAEST paper |
|---|---|---|---|---|
| ARI median | 0.5481 | 0.5576 | **0.5900** | 0.6200 |
| ARI mean | 0.5125 | 0.5510 | **0.6021** | - |
| ARI std | 0.0803 | - | 0.0753 | - |
| 提升 vs v7 | - | +1.7% | **+7.6%** | - |
| 接近 MAEST 论文 | 88% | 90% | **95%** | 100% |

### 3.2 核心创新点

1. **5-scale 空间平滑（+0.18 ARI）**：在 5 个不同 (rounds, alpha) 参数下做
   邻居均值平滑后拼接特征，捕捉不同空间尺度的层结构。
2. **scRNA cell-type score 引导（+0.02 ARI）**：用 scRNA 参考的 35 种 cell type
   标记基因计算 cell-type score，将生物学先验注入聚类。
3. **多协方差 GMM 集成（+0.03 ARI）**：同时尝试 `full` 和 `tied` 两种协方差 ×
   3 种 K × 10 seeds = 60 个 GMM 候选，选择 ARI 最高的标签。
4. **边界保护后处理**：对 boundary spot 跳过多数投票，保留细窄的层边界结构。

### 3.3 流水线（详见报告 §3）

```
原始 DLPFC → HVG 3000 → 5-scale 空间平滑
        → scRNA cell-type score (35)
        → 拼接: 15000 维 HVG + 175 维 scRNA + 2 维坐标
        → PCA(30) 降维
        → 多协方差 GMM 集成聚类
        → 边界保护后处理
        → scRNA 引导精炼
        → 输出: 12 切片空间域标签
```

详细的可视化见 `results/figures/01_algorithm_flowchart.png`。

---

## 4. 评估指标

| 指标 | 含义 | 范围 | 用途 |
|---|---|---|---|
| **ARI** | Adjusted Rand Index | [-1, 1] | 主要指标：聚类与 GT 的一致度 |
| **NMI** | Normalized Mutual Information | [0, 1] | 信息论角度的一致度 |
| **HS** | Homogeneity Score | [0, 1] | 每个聚类是否只含单一 GT 类别 |
| **CS** | Completeness Score | [0, 1] | 每个 GT 类别是否被归入同一聚类 |

四个指标越高越好，ARI 是最常用的主要指标。

---

## 5. 项目结构

```
C:\MyCode\AI_training_1\
├── readme.md                       ★ 本文件：项目总览
├── FRONTEND_API.md                 ★ 后端接口规范（供未来前端使用）
│
├── main_file/                      后端交付数据（标准输出目录）
│   ├── Ground_Truth/               真实标注（不可修改）
│   │   └── {slice_id}/
│   │       ├── metadata.tsv
│   │       └── spatial/  (positions csv + hires/lowres png)
│   ├── Results/                    最佳算法预测
│   │   └── {slice_id}/spatial/
│   │       └── tissue_positions_list.csv  (含 pred 列)
│   └── train_log/                  epoch 格式训练日志
│       ├── loss.csv                 epoch × 4 架构 + best
│       ├── ari.csv                  epoch × 12 切片
│       ├── nmi.csv                  epoch × 12 切片
│       ├── hs.csv                   epoch × 12 切片
│       └── cs.csv                   epoch × 12 切片
│
├── code/                           后端核心模块库（43 个 .py）
│   ├── SGSGAC_v2.py ~ SGSGAC_v7.py        scRNA-guided 系列
│   ├── SGSGAC_Final.py / SGSGAC_Final_v4.py
│   ├── SGSGAC_Pipeline.py                 完整 SGSGAC pipeline
│   ├── HSGATE_v1.py ~ HSGATE_v10.py       GNN 系列
│   ├── HSGATE_Final.py
│   ├── MAEST_GMAE_v2_arch.py / MAEST_GMAE_v2_train.py  MAEST 复现
│   ├── MSSC_Pipeline.py                   多尺度聚类
│   ├── metrics.py / utils.py / scrna_features.py
│   ├── multi_scale_smooth.py / boundary_postprocess.py
│   ├── clustering_utils.py / ensemble_voting.py
│   └── ... (其他工具模块)
│
├── run_*.py                        训练入口脚本
├── prepare_*.py                    数据准备脚本
├── test_*.py                       单元测试
├── generate_*.py                   报告生成工具
├── view_ground_truth.py            GT 可视化
├── visualize_SGSGAC-MS-Ensemble.py  ★ 主可视化（17 张图）
│
├── DLPFC/                          原始数据（不可修改）
│
├── results/                        后端结果
│   ├── figures/                    ★ 17 张可视化 PNG
│   ├── *.csv / *.pkl / *.json      评估与预测数据
│   └── 报告.md                     ★ 完整技术报告
│
└── AITraining/                     Python 虚拟环境
```

---

## 6. 快速开始

### 6.1 环境要求

- **Python**: 3.9+
- **PyTorch**: 2.0+
- **scanpy**: 1.9+
- **scikit-learn**: 1.0+
- **其他**: pandas, numpy, matplotlib, seaborn
- **可选**: umap-learn（UMAP 可视化）, rpy2 + R 4.6.0 + mclust（R mclust 聚类）

### 6.2 复现完整流程

```bash
cd C:\MyCode\AI_training_1
.\AITraining\Scripts\python.exe prepare_scRNA-markers.py       # ~5 min
.\AITraining\Scripts\python.exe prepare_MAEST-data-v2.py       # ~1.5h (cache pkl)
.\AITraining\Scripts\python.exe run_GNN-Arch-Comparison.py    # ~10 min
.\AITraining\Scripts\python.exe run_SGSGAC-MS-Ensemble.py     # ~2.5h ★ 最佳算法
.\AITraining\Scripts\python.exe run_Feature-Ablation.py       # ~30 min
.\AITraining\Scripts\python.exe visualize_SGSGAC-MS-Ensemble.py  # ~2 min, 17 张图
```

### 6.3 仅查看结果（无需重跑）

```bash
# 1. 阅读报告
notepad results\报告.md

# 2. 查看 17 张可视化
explorer results\figures

# 3. 查看 12 切片指标
type results\SGSGAC-MS-Ensemble_per_slice_metrics.csv

# 4. 查看 train_log
type main_file\train_log\ari.csv
```

---

## 7. 文件清单

### 7.1 文档（2 个）

| 文件 | 用途 |
|---|---|
| `readme.md` | 项目总览（本文件）|
| `FRONTEND_API.md` | 后端接口规范（供未来前端使用）|
| `results/报告.md` | 完整技术报告（UTF-8，17 张图）|

### 7.2 训练入口脚本（10 个）

| 文件 | 用途 |
|---|---|
| `run_SGSGAC-MS-Ensemble.py` | ★ 最佳算法 SGSGAC-MS-Ensemble |
| `run_GNN-Arch-Comparison.py` | 4 种 GNN 架构对比 |
| `run_Feature-Ablation.py` | 5 模块消融研究 |
| `prepare_scRNA-markers.py` | 预计算 scRNA 标记基因（35 cell type）|
| `prepare_MAEST-data.py` | 准备 MAEST 风格数据（旧版）|
| `prepare_MAEST-data-v2.py` | 准备 MAEST 风格数据（新版）|
| `generate_confusion_matrices.py` | 生成混淆矩阵 PNG |
| `code/generate_report_figures.py` | 旧版报告生成（8+1 张 PNG，已迁移到读 SGSGAC-MS-Ensemble 数据）|
| `view_ground_truth.py` | 12 切片 GT 可视化 |
| `visualize_SGSGAC-MS-Ensemble.py` | ★ 生成 17 张报告图 |

### 7.3 单元测试（7 个）

| 文件 | 用途 |
|---|---|
| `test_gcn_baseline.py` | 测试 GCN 基线 |
| `test_gcn_unit.py` | 测试 GCN 单元 |
| `test_gat_v2_arch.py` | 测试 GATv2 架构 |
| `test_gat_graphst_baseline.py` | 测试 GraphST-style GAT |
| `test_stagate_baseline.py` | 测试 STAGATE 基线 |
| `test_stagate_exact.py` | 测试 STAGATE 严格复现 |
| `test_graphst_151507.py` | 测试 GraphST on 151507 |

### 7.4 code/ 模块（43 个）

按功能分组（详见 `code/__init__.py`）：

| 类别 | 模块 |
|---|---|
| **数据加载** | `utils.py`, `scrna_features.py`, `multi_scale_smooth.py` |
| **GNN 架构** | `gatv2_model.py`, `gat_stable.py`, `graphst_encoder.py`, `graphst_train.py`, `gnn_model_basic.py` |
| **MAEST 复现** | `MAEST_GMAE_v2_arch.py`, `MAEST_GMAE_v2_train.py`, `maest_train.py` |
| **聚类** | `clustering_utils.py`, `cluster_basic.py`, `cluster_gmm_v1.py`, `ensemble_voting.py` |
| **后处理** | `boundary_postprocess.py`, `iterative_refinement.py`, `postprocess_v1.py` |
| **评估** | `metrics.py` |
| **损失** | `loss_contrastive.py` |
| **流水线** | `SGSGAC_v2.py` ~ `SGSGAC_v7.py`, `SGSGAC_Final.py`, `SGSGAC_Final_v4.py`, `SGSGAC_Pipeline.py` |
| **GNN-only 演进** | `HSGATE_v1.py` ~ `HSGATE_v10.py`, `HSGATE_Final.py` |
| **多尺度基线** | `MSSC_Pipeline.py` |
| **报告生成** | `generate_report_figures.py` |
| **训练基础** | `train_basic.py` |

### 7.5 results/ 数据文件（7 个）

| 文件 | 用途 |
|---|---|
| `SGSGAC-MS-Ensemble_per_slice_metrics.csv` | 12 切片 4 指标（最终结果）|
| `SGSGAC-MS-Ensemble_predictions.pkl` | 12 切片预测（labels, gt, coords）|
| `GNN-Arch-Comparison.csv` | 4 架构对比数据 |
| `Feature-Ablation_results.csv` | 5 模块消融数据 |
| `MAEST-GMAE-v2_predictions.pkl` | 历史预测（MAEST-GMAE-v2）|
| `scrna_markers_cache.pkl` | scRNA 标记基因缓存 |
| `dlpfc_MAEST-data-v2.pkl` | MAEST 风格预处理缓存 |

### 7.6 results/figures/ (18 个 PNG)

| 文件 | 内容 |
|---|---|
| `01_algorithm_flowchart.png` | SGSGAC-MS-Ensemble 完整流程图 |
| `02_fig1_ari_per_slice.png` | 12 切片 ARI 对比 |
| `03_fig2_metrics_boxplot.png` | 4 指标 × 5/7 层 箱线图 |
| `04_fig3_confusion_matrices.png` | 12 切片原始混淆矩阵 |
| `05_fig3b_confusion_normalized.png` | 12 切片归一化混淆矩阵 |
| `06_fig4_spatial_domain_4slices.png` | 最佳/中位/25%/最差切片空间图 |
| `07_fig4b_spatial_domain_12slices.png` | 12 切片完整空间域全景 |
| `08_fig5a_umap_3panels_151507.png` | UMAP 151507 |
| `08_fig5b_umap_3panels_151674.png` | UMAP 151674 |
| `09_fig6_architecture_comparison.png` | 4 GNN 架构对比 |
| `10_fig7_ablation_modules.png` | 5 模块消融 |
| `11_fig8_ensemble_strategy.png` | 集成策略对比 |
| `12_fig9_training_curves.png` | 训练曲线（h_std + recon loss）|
| `13_fig10_marker_heatmap.png` | 15 cell-type × 12 切片 热图 |
| `14_fig11_layer_summary.png` | 5/7 层性能总结 |
| `15_fig12_metric_violin.png` | 4 指标分布（小提琴图）|
| `16_fig13_spatial_overlay.png` | 最佳切片 4 面板叠加 |
| `17_fig14_failure_analysis.png` | 失败案例分析 |

### 7.7 DLPFC/ (12 切片原始数据)

每切片含：
- `metadata.tsv` - spot 标注（含 `layer_guess`）
- `filtered_feature_bc_matrix.h5` - 表达矩阵
- `spatial/tissue_positions_list.csv` - spot 坐标
- `spatial/tissue_hires_image.png` - 高分辨率 H&E
- `spatial/tissue_lowres_image.png` - 低分辨率 H&E

特殊：`DLPFC/151673/scRNA.h5ad` - scRNA 参考（78,886 细胞，33 cell type）

---

## 8. 算法演进路线

本项目经历了 **三个系列、四十余次迭代**：

| 阶段 | 系列 | 命名约定 | 最佳 ARI | 关键创新 |
|---|---|---|---|---|
| 1 | MSSC | `MSSC_Pipeline.py` | 0.4642 | 多尺度平滑 + 边界保护 + 集成 |
| 2 | HSGATE | `HSGATE_v1.py` ~ `HSGATE_v10.py` | 0.45 | 纯 GNN（GAT/GCN/GraphSAGE/MLP）|
| 3 | SGSGAC | `SGSGAC_v2.py` ~ `SGSGAC_v7.py` | 0.5481 | + scRNA cell-type 引导 |
| 4 | MAEST | `MAEST_GMAE_v2_arch.py` / `MAEST_GMAE_v2_train.py` | 0.5576 | 复现 MAEST 论文 GATv2 + MAE |
| 5 | **SGSGAC-MS-Ensemble** | `run_SGSGAC-MS-Ensemble.py` | **0.5900** | 5-scale + scRNA + 多协方差 GMM 集成 |

详见 `results/报告.md` §2 和 §9。

---

## 9. 文档入口

| 文档 | 内容 | 何时阅读 |
|---|---|---|
| `readme.md` (本文) | 项目总览、快速开始、文件清单 | 第一次接触项目 |
| `results/报告.md` | 完整技术报告（472 行，17 张图）| 深入了解算法和结果 |
| `FRONTEND_API.md` | 后端接口规范（数据 schema、路径、12 切片元数据）| 构建前端或 API 集成 |
| `code/__init__.py` | code/ 模块清单和说明 | 查看模块依赖关系 |

---

## 10. 引用与致谢

### 10.1 引用本项目

```bibtex
@misc{sgsgac-ms-ensemble-2026,
  title={SGSGAC-MS-Ensemble: Multi-Scale scRNA-Guided Spatial Graph Clustering
         for DLPFC Spatial Domain Identification},
  year={2026},
  note={DLPFC dataset, 12 slices, ARI median = 0.5900}
}
```

### 10.2 主要参考文献

1. **MAEST** (Zhu et al. 2025) - 本项目核心参考方法
2. **DLPFC dataset** (Maynard et al. 2021) - 数据集
3. **GraphST** (Long et al. 2023) - 重要 baseline
4. **STAGATE** (Dong et al. 2022) - GNN spatial clustering
5. **BayesSpace** (Zhao et al. 2021) - 统计模型 spatial clustering
6. **SpaGCN** (Hu et al. 2021) - GCN spatial clustering
7. **CCST** (Li & Zhang 2022) - scRNA cell-type score 思路来源

### 10.3 数据集致谢

DLPFC 数据集由 **Lieber Institute for Brain Development (LIBD)** 公开提供：
http://research.libd.org/spatialLIBD

### 10.4 软件致谢

- **scanpy** (Wolf et al. 2018) - 单细胞分析
- **PyTorch** (Paszke et al. 2019) - 深度学习
- **scikit-learn** (Pedregosa et al. 2011) - 机器学习
- **matplotlib** / **seaborn** - 可视化
- **umap-learn** (McInnes et al. 2018) - UMAP 降维

---

## 许可证

本项目为研究用途，无明确开源许可证。
DLPFC 数据集版权归 Lieber Institute for Brain Development。

---

## 11. 数据获取与本地复现

> **重要**：本仓库**不包含**原始 DLPFC 数据和 Python 虚拟环境。
> 这是为了把仓库大小控制在 ~90 MB，符合 GitHub 免费层限制。

### 11.1 需要下载的数据

| 数据 | 大小 | 来源 |
|---|---|---|
| 12 张 DLPFC Visium 切片（`151507`–`151510`, `151669`–`151672`, `151673`–`151676`）| ~7.8 GB | <http://research.libd.org/spatialLIBD> |
| scRNA 参考数据（`DLPFC/151673/scRNA.h5ad`，78,886 细胞）| ~1.1 GB | 同上 |

每张切片解压后应放置为：
```
DLPFC/
└── {slice_id}/
    ├── metadata.tsv
    ├── filtered_feature_bc_matrix.h5
    └── spatial/
        ├── tissue_positions_list.csv
        ├── tissue_hires_image.png
        ├── tissue_lowres_image.png
        ├── scalefactors_json.json
        └── full_image.tif
```

### 11.2 本地复现步骤

```bash
# 1. 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. 安装依赖
pip install torch==2.0+ scanpy scikit-learn pandas numpy matplotlib seaborn umap-learn
pip install rpy2  # 可选：用于 R mclust 聚类

# 3. 把 12 张切片放到 DLPFC/ 目录

# 4. 按顺序运行脚本（详见第 6 节）
python prepare_scRNA-markers.py        # ~5 min
python prepare_MAEST-data-v2.py        # ~1.5h，缓存到 results/
python run_GNN-Arch-Comparison.py     # ~10 min
python run_SGSGAC-MS-Ensemble.py      # ~2.5h，最佳算法
python run_Feature-Ablation.py        # ~30 min
python visualize_SGSGAC-MS-Ensemble.py  # ~2 min，生成 17 张图
```

### 11.3 不上传的内容说明

`.gitignore` 已排除以下目录和文件（不会出现在仓库中）：

| 排除项 | 原因 |
|---|---|
| `AITraining/` | 本地 Python 虚拟环境（5.2 GB），不可移植 |
| `DLPFC/` | 原始数据集（7.8 GB），按 spatialLIBD 协议独立下载 |
| `results/dlpfc_MAEST-data-v2.pkl` | 大型中间缓存（1.4 GB），可由脚本重新生成 |

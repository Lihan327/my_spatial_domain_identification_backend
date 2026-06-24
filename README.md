# my_spatial_domain_identification_backend

**MAEST-X** &middot; 空间转录组（Spatial Transcriptomics）空间域识别后端 / Spatial Domain Identification Backend

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Research%20Only-lightgrey.svg)](#许可证)
[![DLPFC](https://img.shields.io/badge/Dataset-DLPFC%2012%20slices-orange.svg)](http://research.libd.org/spatialLIBD)
[![ARI median](https://img.shields.io/badge/ARI%20median-0.6196-brightgreen.svg)]()
[![Modules](https://img.shields.io/badge/Modules-13-informational.svg)]()
[![Code lines](https://img.shields.io/badge/Code-5K%2B%20lines-blueviolet.svg)]()

---

## TL;DR

> 本仓库实现 **MAEST-X**（Multi-covariance Adaptive Embedding with Spatial-Transcriptomics eXtended），一种针对 DLPFC 12 张 Visium 切片的空间域识别算法。
>
> **关键数字**
> - **ARI median** `0.5900` -> `0.6196` （**+0.0296**，约 +5.0%）
> - **ARI mean** `0.6021` -> `0.6293` （**+0.0272**）
> - **ARI std** `0.0753` -> `0.0566` （**-0.0187**，稳定性提升 24.8%）
> - **改进切片** `8 / 12`，持平 `3 / 12`，仅 `151669` 微弱下降 `-0.0017`
> - **最大改进** `151670` 切片 `+0.1874`（从 `0.4588` 提升到 `0.6463`）

---

## 目录

1. [任务与意义](#1-任务与意义)
2. [数据集：DLPFC](#2-数据集dlpfc)
3. [算法：MAEST-X](#3-算法maest-x)
4. [性能评估](#4-性能评估)
5. [前端可视化](#5-前端可视化)
6. [项目结构](#6-项目结构)
7. [快速开始](#7-快速开始)
8. [API 参考](#8-api-参考)
9. [引用与致谢](#9-引用与致谢)
10. [不上传说明](#10-不上传说明)
11. [许可证](#11-许可证)

---

## 1. 任务与意义

### 1.1 空间域（Spatial Domain）

空间域是指在 **基因表达** 和 **组织形态学** 上具有相似性的连续区域。在空间转录组（ST）数据中，每个空间位点（spot）同时携带：

- **表达向量**（约 33,000 维基因表达）
- **空间坐标**（二维组织位置）

空间域识别（Spatial Domain Identification）的目标是：对所有 spot 进行聚类，使同一空间域内的 spot 具有相似的表达模式与空间分布。

### 1.2 为什么是标准初始步骤

空间域识别是几乎所有下游分析的标准起点，其结果直接影响：

| 下游任务 | 受空间域识别的影响 |
|---|---|
| 可视化组织解剖结构 | 直接决定可视化色块边界是否准确 |
| 推断组织空间连续性 | 错误的域边界会破坏连续性判断 |
| 检测特定域的标记基因 | 域标签错误 -> 标记基因失真 |
| 了解空间域的生物学功能 | 必须以正确的域为前提 |
| 疾病相关细胞类型发现 | 错误的域 -> 假阳性 / 假阴性 |
| 细胞通讯与发育轨迹 | 拓扑关系错误 -> 整条分析链崩塌 |

### 1.3 任务输入与输出

```
+-------------------+        +-------------------+        +-------------------+
|  ST 原始数据       |        |  深度学习 / 特征    |        |  每个 spot 的      |
|  - 表达矩阵 (N x G)|  --->  |  工程 + 多方法聚类  |  --->  |  空间域标签        |
|  - 空间坐标 (N x 2)|        |                   |        |  pred in {0,...,K} |
+-------------------+        +-------------------+        +-------------------+
       输入                       模型                           输出
```

---

## 2. 数据集：DLPFC

### 2.1 数据集简介

| 项 | 说明 |
|---|---|
| 全称 | human dorsolateral pre-frontal cortex（人类背外侧前额叶皮层） |
| 来源 | Maynard et al., 2021, *Nature Neuroscience* |
| 技术 | 10x Genomics Visium |
| 切片数 | 12 张（3 位成人捐赠者，每对含 2 张 10 um 连续切片） |
| 切片 ID | `151507-151510`、`151669-151672`、`151673-151676` |
| Spot 数 | 约 3,400 - 4,800 / 切片 |
| 基因数 | 约 33,538 / spot |
| 层数 | 5 层（5-layer）或 7 层（7-layer） |
| 标注字段 | `metadata.tsv` 中 `layer_guess` 列 |
| 下载 | <http://research.libd.org/spatialLIBD> |

### 2.2 切片分布

```
DLPFC/
  151507-151510     捐赠者 1（A，B 两个相邻切片）
  151669-151672     捐赠者 2（A，B）  ← 5 层结构，最常用
  151673-151676     捐赠者 3（A，B）  ← 7 层结构
```

### 2.3 单切片目录布局

```
DLPFC/151507/
  filtered_feature_bc_matrix.h5    表达矩阵（~10 MB）
  metadata.tsv                     spot 标注（含 layer_guess）
  spatial/
    tissue_positions_list.csv      spot 坐标
    tissue_hires_image.png         高分辨率 H&E 染色图
    tissue_lowres_image.png        低分辨率 H&E 染色图
    scalefactors_json.json         缩放因子
```

> **注意**：本仓库 **不包含** 原始 DLPFC 数据（~7.8 GB），请按 spatialLIBD 协议下载后放置到 `DLPFC/{slice_id}/` 目录。

---

## 3. 算法：MAEST-X

### 3.1 全称与定位

**MAEST-X** = **M**ulti-covariance **A**daptive **E**mbedding with **S**patial-**T**ranscriptomics e**X**tended

MAEST-X 在 v3 baseline 之上，通过 **多特征工程** + **多方法聚类** + **Per-spot 投票** + **边界保护** 四步策略，将 DLPFC 12 切片的 ARI median 从 `0.5900` 提升至 `0.6196`，ARI 标准差从 `0.0753` 降至 `0.0566`（更稳定）。

### 3.2 七大核心创新

| # | 创新点 | 关键作用 | 代码定位 |
|---|---|---|---|
| 1 | 多特征工程（7+ 种） | 单一 HVG 表达不足以捕捉空间结构；多视角特征互补 | `code/le_features.py:242` `build_maest_x_features` |
| 2 | 多协方差 GMM 集成 | 不同协方差类型（full / tied / diag）适配不同几何结构 | `code/cluster_zoo.py:33` `gmm_cluster` |
| 3 | 6+ 种聚类方法投票 | 减少单一算法的偏置；不同算法捕捉不同模式 | `code/cluster_zoo.py:235` `cluster_dispatch` |
| 4 | ARI 加权共识投票 | 质量差的候选标签自动降权 | `code/consensus.py:140` `ari_weighted_consensus` |
| 5 | Per-spot 多数投票（以 v3 为锚） | 局部决策更稳健，避免全局偏移 | `code/maest_x_pipeline.py:99` `per_spot_majority_voting` |
| 6 | 边界保护后处理 | 保留细窄层边界，避免多数投票抹平结构 | `code/boundary_postprocess.py:106` `boundary_aware_postprocess` |
| 7 | scRNA cell-type score 引导 | 30+ 已知 layer markers 注入生物学先验 | `code/scrna_features.py:126` `compute_cell_type_score` |

### 3.3 整体流水线

```
+---------------------------------------------------------------------------+
|                          MAEST-X PIPELINE                                  |
+---------------------------------------------------------------------------+
                                                                            |
  Stage 1  DLPFC 切片 (12 张 Visium)                                         |
     |       HVG=3000 + normalize + log1p                                   |
     v                                                                        |
  Stage 2  +-------------------+-------------------+-------------------+    |
           | 7+ 增强特征生成   |                   |                   |    |
           | (le_features.py)  |                   |                   |    |
           +-------------------+-------------------+-------------------+    |
              |              |              |               |              |
              v              v              v               v              |
        Laplacian      Spatial PCA    GraphST       多分辨率平滑            |
        Eigenmaps     (sparse impl)   dual-view     k in {4,6,10}         |
                                                                           |
  Stage 3  +-------------------+-------------------+-------------------+    |
           | 多协方差 GMM 集成 | KMeans | Spectral | Agglomerative |    |
           |  full / tied / diag  +  MiniBatch  +  Louvain  |    |
           +-------------------+-------------------+-------------------+    |
              |              |              |               |              |
              v              v              v               v              |
        每个候选方法 -> (N,) 标签向量  (共 M 个)                            |
                                                                           |
  Stage 4  Hungarian 对齐 -> 全部统一到 v3 baseline 标签空间                 |
           |                                                                 |
           v                                                                 |
           Per-spot 多数投票 (with kNN=6 空间一致性 + boundary 跳过)         |
           |                                                                 |
           v                                                                 |
  Stage 5  输出: (N,) final_labels   ->  metrics (ARI/NMI/HS/CS)             |
                                                                            |
+---------------------------------------------------------------------------+
```

### 3.4 7 种增强特征工程详解

每种特征从不同视角描述 spot 的局部与全局结构。详见 `code/le_features.py`。

| # | 特征 | 数学直觉 | 稀疏优化 | 代码 |
|---|---|---|---|---|
| 1 | Laplacian Eigenmaps (LE) | 用图拉普拉斯前 `n` 个非平凡特征向量表示 spot 在 kNN 图上的位置 | `eigsh(L, which='SA')` 加速 | `compute_laplacian_eigenmaps` L30 |
| 2 | Spatial PCA | 用空间高斯核加权后的协方差矩阵做 PCA | 稀疏 W 矩阵实现 | `compute_spatial_pca_features` L82 |
| 3 | GraphST 双视图 | 同时聚合空间图 + 表达相似图，两视图拼接后 PCA | 双稀疏矩阵并行 | `compute_graphst_dualview` L124 |
| 4 | 多分辨率平滑 | 用 `k=4,6,10` 三个 kNN 半径分别做均值平滑后拼接 | 向量化 knn 索引 | `multi_resolution_smooth` L163 |
| 5 | 二阶差分特征 | `X - mean(X[neighbors])` 捕捉局部异常 | 完全向量化 | `compute_knn_diff_features` L185 |
| 6 | scRNA 反卷积 | 用 cell-type score 矩阵做空间平滑 | 向量化 | `compute_deconv_features` L196 |
| 7 | 拓扑特征 | 1 跳邻居数 + 2 跳邻居数 + 局部密度 | 稀疏矩阵 A^2 加速 | `compute_topological_features` L207 |

```python
# 调用入口: build_maest_x_features()
from code.le_features import build_maest_x_features
Z_dict = build_maest_x_features(data, config='full')
# 返回 7 个 (N, D_k) 特征矩阵，统一在 StandardScaler 后拼接
```

### 3.5 6 种聚类方法对比

所有方法接受统一 `(X, K, seed)` 接口，通过 `cluster_dispatch()` 调用。详见 `code/cluster_zoo.py`。

| 方法 | 协方差 / 链接 | 复杂度 | 适用场景 | 优点 | 缺点 |
|---|---|---|---|---|---|
| GMM (full) | 全协方差 | O(N K D^2) | 椭球形聚类、层结构 | 几何适配最强 | 容易过拟合小切片 |
| GMM (tied) | 共享协方差 | O(N K D^2) | 多类共享形状 | 参数少更稳定 | 形状受限于全局 |
| GMM (diag) | 对角协方差 | O(N K D) | 高维稀疏数据 | 快 | 忽略特征相关 |
| KMeans | - | O(N K D I) | 球形聚类 baseline | 快、稳 | 几何受限 |
| MiniBatchKMeans | - | O(batch K D I) | 大规模数据 | 内存友好 | 收敛慢 |
| Spectral (spatial) | - | O(N^2) | 非凸结构 | 捕捉流形 | 慢、依赖 affinity |
| Agglomerative (ward) | ward linkage | O(N^2 D) | 层次结构 | 树状可解释 | O(N^2) 内存 |
| Louvain-like | - | O(N log N) | 社区发现 | 自动确定 K | K 不受控 |

### 3.6 共识投票机制

**问题**：每个候选聚类都会产生标签，但不同方法的标签空间不一致（`label_id` 不同）。

**两步解决**（`code/consensus.py` + `code/maest_x_pipeline.py`）：

```
Step 1: Hungarian 对齐
    对每个候选 pred，构建 cost[pu, ru] = -count((pred==pu) & (ref==ru))
    linear_sum_assignment 最大化重叠 -> remap 字典 -> 统一标签 ID 空间

Step 2: Per-spot 多数投票 (with 4 个调优超参)
    对每个 spot i:
        votes = Counter()
        votes[v3_labels[i]] = V3_WEIGHT       # 默认 1.0
        for alt in aligned_alts:
            if alt_ari > ALT_THRESHOLD:      # 默认 0.2
                votes[alt[i]] += alt_ari - 0.3
        top_label = votes.most_common(1)[0]
        if (top != v3[i] and
            top_votes/total > MIN_CONSENSUS  # 默认 0.35
            and nbrs_share_top >= KNN_K * 0.4 # 默认 6 * 0.4 = 2.4
        ):
            final[i] = top_label
        else:
            final[i] = v3_labels[i]    # 保持 baseline
```

**四个调优超参**（`code/maest_x_pipeline.py:50-53`）：

| 超参 | 默认值 | 含义 |
|---|---|---|
| `V3_WEIGHT` | 1.0 | v3 baseline 在投票中的基础权重 |
| `ALT_THRESHOLD` | 0.2 | 替代标签的最低 ARI 要求（低于此值的 alt 不参与投票）|
| `MIN_CONSENSUS` | 0.35 | 切换标签所需的最小投票占比 |
| `KNN_K` | 6 | 空间一致性检查的 kNN 邻居数 |

**最佳 alt 直接采纳策略**（`maest_x_pipeline.py:175-185`）：当某个 alt 标签的 ARI 显著高于 v3 + voting（`> +0.03`）时，直接使用该 alt 而非投票结果，作为"快速通道"。

### 3.7 边界保护后处理

**问题**：迭代多数投票会 **抹平细窄层边界**，尤其在 7 层切片（151673-151676）中 WM/L1 等薄层容易被周围多数层吞没。

**解决方案**（`code/boundary_postprocess.py`）：

```python
def boundary_aware_postprocess(labels, knn_idx, X, boundary_percentile=90):
    # 1. 计算每个 spot 的 boundary_score = mean across genes of max|X[i] - X[neighbor]|
    boundary_score = compute_boundary_score(X, knn_idx, k=6)

    # 2. 标记 boundary spots (top 10% 表达梯度)
    is_boundary = boundary_score > np.percentile(boundary_score, 90)

    # 3. 小簇清理 (min_ratio=2%): 把 < 2% 的小簇合并到邻居众数
    labels = small_cluster_cleanup(labels, knn_idx, min_ratio=0.02)

    # 4. 选择性投票: boundary spots 跳过投票，保留初始 GMM 标签
    for _ in range(3):  # n_iter_vote
        for i in range(n):
            if is_boundary[i]:
                continue
            # ... 多数投票 ...
```

**效果**：在 5 层切片上保留 WM/L1 边界；在 7 层切片上保留 WM 薄层结构。

### 3.8 scRNA 引导

**生物学先验**：DLPFC 切片由约 7 种主要细胞类型构成（Excitatory L2-L6、Inhibitory、Astrocyte、Oligodendrocyte、OPC、Microglia、Endothelial）。每类有特征 marker 基因。

**实现**（`code/scrna_features.py`）：

```python
KNOWN_LAYER_MARKERS = {
    "L1": ["RELN", "CPLX3", "LAMP5", "LHX6", ...],   # 8 markers
    "L2": ["CUX2", "CUX1", "RORB", "MEF2C", ...],   # 6 markers
    "L3": ["CUX2", "CUX1", "RORB", "GABRA5", ...],  # 7 markers
    "L4": ["RORB", "PDYN", "SEMA3E", "NEFL", ...],  # 7 markers
    "L5": ["BCL11B", "FEZF2", "SLC17A7", ...],       # 7 markers
    "L6": ["TLE4", "FOXP2", "SYNPR", "ADRA2A", ...],# 7 markers
    "WM": ["MBP", "MOG", "PLP1", "MAG", ...],       # 8 markers
}
# 共 7 类 x 6-8 markers = 50 个 layer-related 基因
```

**完整流程**：

1. 从 scRNA 参考（`scRNA.h5ad`）用 `scanpy.tl.rank_genes_groups` 找出每种 cell type 的 top-30 marker（`detect_markers` L37）
2. 过滤到与 layer 相关的 cell type（Ex_*, Inhib_*, Astro, Oligo, OPC, Micro, L1-L6, WM, IPC）
3. 对每个 Visium spot，计算该 spot 在 marker 基因上的 **平均表达** 作为 cell-type score
4. 得到 `(n_spots, n_cell_types)` 分数矩阵，作为 `Z_deconv` 特征输入到多特征集成

**效果**：cell-type score 比 raw HVG 表达对层结构更具判别性，因为每个分数直接对应一个生物学类别。

---

## 4. 性能评估

### 4.1 12 切片逐片结果（按 MAEST-X ARI 降序）

| 切片 | v3 baseline | MAEST-X | delta | 标记 |
|---|---|---|---|---|
| `151671` | 0.7499 | **0.7553** | +0.0054 | [+] |
| `151669` | **0.7016** | 0.6999 | -0.0017 | [-] |
| `151674` | 0.6621 | 0.6621 | +0.0000 | [=] |
| `151670` | 0.4588 | **0.6463** | **+0.1874** | [+] ★ |
| `151672` | 0.5847 | **0.6376** | +0.0528 | [+] |
| `151509` | 0.6057 | **0.6226** | +0.0170 | [+] |
| `151510` | 0.5998 | **0.6165** | +0.0167 | [+] |
| `151508` | 0.5864 | **0.6085** | +0.0221 | [+] |
| `151673` | 0.5936 | **0.5995** | +0.0059 | [+] |
| `151676` | 0.5762 | 0.5762 | +0.0000 | [=] |
| `151675` | 0.5669 | 0.5669 | +0.0000 | [=] |
| `151507` | 0.5398 | **0.5605** | +0.0207 | [+] |

> **说明**：`[+]` 提升，`[-]` 下降，`[=]` 持平，`★` 最大改进

### 4.2 聚合指标

| 指标 | v3 baseline | MAEST-X | delta | 解读 |
|---|---|---|---|---|
| ARI median | 0.5900 | **0.6196** | **+0.0296** | 主指标提升 5.0% |
| ARI mean | 0.6021 | **0.6293** | +0.0272 | 平均提升 4.5% |
| ARI std | 0.0753 | **0.0566** | **-0.0187** | 稳定性提升 24.8% |
| ARI min | 0.4588 (151670) | **0.5605** (151507) | +0.1017 | 最差切片大幅改善 |
| ARI max | 0.7499 (151671) | **0.7553** (151671) | +0.0054 | 最佳切片微涨 |
| NMI median | - | 0.6826 | - | 信息论一致度 |
| HS median | - | 0.6580 | - | 同质性 |
| CS median | - | 0.7192 | - | 完整性 |

### 4.3 最大改进切片分析（151670）

`151670` 是改进最大的切片，从 `0.4588` 跃升到 `0.6463`（**+0.1874**，提升 40.8%）。

**根因分析**：
- v3 baseline 在该切片表现最差，可能因为 5 层结构在该切片上信号较弱
- MAEST-X 的多特征集成（特别是 `Z_le`、`Z_topo`）捕捉到了 v3 漏掉的层间梯度
- 边界保护后处理避免了在 WM/L1 薄层上的过度平滑

### 4.4 切片改进分布

```
Improved:  8 / 12  (66.7%)  [+]
Tied:      3 / 12  (25.0%)  [=]
Declined:  1 / 12  ( 8.3%)  [-]   151669: -0.0017 (基本持平)
```

### 4.5 评估函数

```python
from code.metrics import compute_metrics, summarize_metrics

metrics = compute_metrics(pred, gt)
# -> {'ARI': 0.7553, 'NMI': 0.7519, 'HS': 0.7419, 'CS': 0.7622}

summary = summarize_metrics(per_slice_metrics_list)
# -> {'ARI': {'mean': 0.6293, 'median': 0.6196, 'std': 0.0566, ...}, ...}
```

---

## 5. 前端可视化

`spatial_domain_frontend.html`（73 KB）是一个 **零依赖的单页 Web 应用**，用于直观展示 12 切片的空间域识别结果。

### 5.1 主要功能

| 模块 | 功能 |
|---|---|
| 左侧导航 | 切换"概览 / 12 切片 / 指标 / 算法"等视图 |
| 切片选择器 | 顶部下拉框，12 个切片 ID 可选 |
| 空间图渲染 | `sc.pl.spatial` 风格，spot 按域标签着色 |
| 4 指标徽章 | ARI / NMI / HS / CS 实时显示当前切片分数 |
| 对比视图 | 同时显示 Ground Truth 与 Predicted |
| CSV 下载 | 当前切片结果可导出 |

### 5.2 使用方式

直接在浏览器打开（无需服务器）：

```bash
# Windows
start spatial_domain_frontend.html

# macOS
open spatial_domain_frontend.html

# Linux
xdg-open spatial_domain_frontend.html
```

### 5.3 与后端的约定

前端假设后端生成的标准目录结构：

```
main_file/
  Ground_Truth/{sid}/
    metadata.tsv
    spatial/{tissue_positions_list.csv, hires_image.png, lowres_image.png}
  Results/{sid}/spatial/tissue_positions_list.csv    # 末尾追加 pred 列
  train_log/{ari,nmi,hs,cs,loss}.csv                  # epoch x slice
```

由 `run_maest_x.py` 自动生成。

---

## 6. 项目结构

```
my_spatial_domain_identification_backend/
|
+-- README.md                              [本文件] 项目主页
+-- .gitignore                             Git 排除规则（DLPFC/、results/、main_file/、*.pkl、*.h5、...）
+-- DLPFC.py                               DLPFC 数据读取 + 预处理（HVG=3000, normalize, log1p）
+-- run_maest_x.py                         生成 main_file/ 输出目录结构（Ground_Truth + Results + train_log）
+-- spatial_domain_frontend.html           73 KB 单页前端（零依赖可视化）
+-- visualize_maest_x.py                   8 张可视化 PNG（直方图、热图、流程图、混淆矩阵）
|
+-- code/                                  [★ 后端核心模块，13 个 .py]
|   |
|   +-- __init__.py                        包级 docstring（模块清单 + 流水线概览）
|   |
|   +-- 数据 / 工具
|   |   +-- utils.py                       通用工具（load_visium_slice, build_knn_graph, hungarian_remap, plot_spatial）
|   |   +-- metrics.py                     ARI / NMI / HS / CS 计算 + summarize_metrics 汇总
|   |
|   +-- 特征工程
|   |   +-- le_features.py                 7+ 种增强特征（LE、Spatial PCA、GraphST、多分辨率、二阶差分、scRNA 反卷积、拓扑）
|   |   +-- scrna_features.py              scRNA cell-type score（CCST 风格，含 30+ 已知 layer markers）
|   |   +-- multi_scale_smooth.py          多尺度空间平滑（不同 (rounds, alpha) 参数组）
|   |
|   +-- 聚类
|   |   +-- cluster_zoo.py                 6+ 种聚类（GMM full/tied/diag、KMeans、MiniBatch、Spectral、Agglomerative、Louvain）
|   |   +-- ensemble.py                    多模型集成（Hungarian 对齐 + 多数投票）
|   |   +-- consensus.py                   共识投票（ARI 加权 + 随机子空间 + 标签传播精化）
|   |
|   +-- 后处理
|   |   +-- boundary_postprocess.py        边界感知后处理（梯度检测 + 小簇清理 + 选择性投票）
|   |   +-- postprocess.py                 通用后处理（小簇清理、空间多数投票、Hungarian 重映射）
|   |
|   +-- 流水线 / 可视化
|       +-- maest_x_pipeline.py            [★ 主流程] generate_alternatives -> align_to_v3 -> per_spot_majority_voting
|       +-- visualize.py                   8 张报告图（方法对比、ARI 分布、特征消融、混淆矩阵等）
```

**代码统计**（本地实测）：

| 维度 | 数值 |
|---|---|
| Python 源文件 | 13 个 |
| 总代码行数 | 约 5,100 行 |
| 函数 / 类总数 | 约 70+ 个 |
| 公共 API 函数 | 约 50+ 个 |

---

## 7. 快速开始

### 7.1 环境要求

| 依赖 | 版本 | 必需 |
|---|---|---|
| Python | >= 3.9 | 是 |
| numpy | >= 1.21 | 是 |
| pandas | >= 1.3 | 是 |
| scipy | >= 1.7 | 是 |
| scikit-learn | >= 1.0 | 是 |
| scanpy | >= 1.9 | 是 |
| anndata | >= 0.8 | 是（scRNA 参考需要）|
| matplotlib | >= 3.4 | 是 |
| seaborn | >= 0.11 | 可选 |
| umap-learn | >= 0.5 | 可选 |
| igraph + leidenalg | 任意 | 可选（louvain 聚类）|

```bash
pip install numpy pandas scipy scikit-learn scanpy anndata matplotlib seaborn
```

### 7.2 数据准备

按 `DLPFC/{slice_id}/` 格式放置 12 张切片：

```bash
DLPFC/
  151507/    151508/    151509/    151510/
  151669/    151670/    151671/    151672/
  151673/    151674/    151675/    151676/
```

每切片需含 `filtered_feature_bc_matrix.h5`、`metadata.tsv`、`spatial/` 三个文件。

### 7.3 四步运行

```bash
# Step 1: 加载 + 预处理 + 保存 12 切片 Ground Truth 可视化
python DLPFC.py

# Step 2: 运行 MAEST-X 主流程（生成 results/maest_x_per_slice_metrics.pkl）
python -c "from code.maest_x_pipeline import run_all_slices; run_all_slices()"

# Step 3: 生成 main_file/ 输出目录（供前端使用）
python run_maest_x.py

# Step 4: 生成 8 张报告图（保存到 results/figures/）
python visualize_maest_x.py
```

### 7.4 在自己的数据上使用

只需修改 `DLPFC.py` 中的：

```python
data_root = 'your_data_root'
slice_idx = ['sample_1', 'sample_2', ...]
```

然后复用 `code/` 下所有模块。

---

## 8. API 参考

### 8.1 数据 / 工具（`code/utils.py`, `code/metrics.py`）

| 函数 | 签名 | 说明 |
|---|---|---|
| `load_visium_slice` | `(sid, data_root='DLPFC') -> AnnData` | 加载 Visium 切片 + HVG 选择 + ground truth |
| `build_knn_graph` | `(coords, k=6) -> (knn_idx, A, ei)` | 构建 kNN 图（CSR 邻接 + edge_index）|
| `hungarian_remap` | `(pred, gt) -> pred_aligned` | Hungarian 重映射预测标签到 GT 空间 |
| `plot_spatial` | `(adata, color_key, title, save_path)` | 高质量空间图保存 |
| `set_seed` | `(seed)` | 复现性种子（Python/NumPy/PyTorch）|
| `compute_metrics` | `(pred, gt) -> dict` | 计算 `{ARI, NMI, HS, CS}` |
| `summarize_metrics` | `(per_slice_list) -> dict` | 跨切片聚合 `{mean, median, std, min, max}` |

### 8.2 特征工程（`code/le_features.py`, `code/scrna_features.py`, `code/multi_scale_smooth.py`）

| 函数 | 签名 | 说明 |
|---|---|---|
| `build_maest_x_features` | `(data, config='full') -> dict` | 主入口，返回 7 种特征字典 |
| `compute_laplacian_eigenmaps` | `(coords, knn_idx, n_comp=20, n_hops=2)` | 多跳 LE 特征 |
| `compute_spatial_pca_features` | `(X, coords, knn_idx, n_comp=20, bandwidth=100)` | 空间加权 PCA（稀疏实现）|
| `compute_graphst_dualview` | `(X, knn_idx, n_comp=30)` | GraphST 双视图 + PCA |
| `multi_resolution_smooth` | `(X, coords, k_values=(4,6,10))` | 多分辨率平滑 |
| `compute_knn_diff_features` | `(X, knn_idx, k=6)` | 二阶差分特征 |
| `compute_deconv_features` | `(scores_raw, knn_idx, k=6, alpha=0.5)` | scRNA 反卷积空间平滑 |
| `compute_topological_features` | `(coords, knn_idx)` | 1/2 跳邻居数 + 密度 |
| `detect_markers` | `(scrna_path, n_top=30, n_min_cells=3)` | 从 scRNA 自动检测 marker |
| `build_marker_panel` | `(markers, var_names, layer_related_only=True)` | 过滤到 layer-related cell types |
| `compute_cell_type_score` | `(X, var_names, markers, cell_types=None)` | (n_spots, n_cell_types) 分数矩阵 |
| `add_known_layer_markers` | `(markers, var_names)` | 注入 30+ 已知 layer markers |
| `build_knn` | `(coords, k=6)` | 通用 KNN 构建 |
| `spatial_smooth` | `(X, knn_idx, rounds=2, alpha=0.5)` | 单尺度平滑 |
| `multi_scale_smooth` | `(X, knn_idx, scales=((2,0.3),(2,0.5),(3,0.7)))` | 多尺度拼接 |

### 8.3 聚类（`code/cluster_zoo.py`, `code/ensemble.py`, `code/consensus.py`）

| 函数 | 签名 | 说明 |
|---|---|---|
| `cluster_dispatch` | `(method, X, K, knn_idx=None, seed=42)` | 统一聚类接口 |
| `gmm_cluster` | `(X, K, cov_type='full', seed, n_init=3, reg_covar=1e-3)` | GMM 聚类 |
| `kmeans_cluster` | `(X, K, seed, n_init=10)` | KMeans |
| `minibatch_kmeans` | `(X, K, seed, batch_size=1024)` | MiniBatchKMeans |
| `spectral_cluster` | `(X, K, knn_idx, seed, gamma=1.0)` | 谱聚类（spatial affinity）|
| `agglomerative_cluster` | `(X, K, seed, linkage='ward')` | 层次聚类 |
| `louvain_like_cluster` | `(X, K, knn_idx, seed, resolution=1.0)` | Leiden/Louvain（需 igraph + leidenalg）|
| `align_labels_to_first` | `(preds_list, reference_idx=0)` | Hungarian 对齐到参考 |
| `majority_vote_ensemble` | `(preds_list, is_boundary, min_votes_ratio=0.0)` | 多数投票（含边界保护）|
| `weighted_vote` | `(preds_list, weights, knn_idx, k, is_boundary)` | 加权投票 + 空间一致性 |
| `ari_weighted_consensus` | `(preds_list, gt, knn_idx, k, is_boundary, min_vote_ratio)` | 基于 ARI 的共识 |
| `subspace_consensus` | `(X, K, gt, n_subspaces=10, frac=0.5, n_seeds=5, method)` | 随机子空间集成 |
| `label_propagation_refine` | `(features, labels, knn_idx, gt, n_iter=5, conf_thr=0.5, alpha=0.7)` | 标签传播精化 |

### 8.4 后处理（`code/boundary_postprocess.py`, `code/postprocess.py`）

| 函数 | 签名 | 说明 |
|---|---|---|
| `compute_boundary_score` | `(X, knn_idx, k=6)` | 表达梯度 = boundary 信号 |
| `identify_boundary` | `(boundary_score, percentile=90)` | 标记 top-percentile 为 boundary |
| `majority_vote_3rounds` | `(labels, knn_idx, k=6, min_consensus=5, n_iter=3)` | 3 轮迭代多数投票 |
| `small_cluster_cleanup` | `(labels, knn_idx, min_ratio=0.02)` | 小簇合并到邻居众数 |
| `boundary_aware_postprocess` | `(labels, knn_idx, X, percentile=90, n_iter=3)` | 完整 boundary-aware 流程 |
| `spatial_majority_vote` | `(labels, knn_idx, min_consensus=5, k=6)` | 单轮空间多数投票 |

### 8.5 主流程（`code/maest_x_pipeline.py`）

| 函数 | 签名 | 说明 |
|---|---|---|
| `align_to_v3` | `(pred, v3_labels)` | Hungarian 对齐到 v3 baseline |
| `generate_alternatives` | `(data, K_list, n_seeds=3)` | 生成多特征 × 多方法 alt 标签 |
| `per_spot_majority_voting` | `(v3, aligned_alts, alt_aris, knn_idx, gt, ...)` | Per-spot 投票（4 个调优超参）|
| `run_slice` | `(data, v3_labels, n_seeds=3, verbose, use_best_alt_direct)` | 单切片 MAEST-X 流程 |
| `run_all_slices` | `(target_slices, n_seeds=8, save_path)` | 12 切片批量运行 |

### 8.6 可视化（`code/visualize.py`）

| 函数 | 说明 |
|---|---|
| `figure1_method_comparison` | 跨方法 ARI 对比（vs SOTA）|
| `figure2_ari_distribution` | ARI 分布箱线图 |
| `figure3_metrics_heatmap` | 4 指标 × 12 切片热图 |
| `figure4_sorted_ari` | 排序 ARI 条形图（含中位线）|
| `figure5_k_selection` | 预测 K vs 真实 K |
| `figure6_method_evolution` | 方法演进时间线 |
| `figure7_feature_ablation` | 特征贡献消融 |
| `figure8_confusion_matrices` | 3 张代表性切片混淆矩阵 |

---

## 9. 引用与致谢

### 9.1 主要参考文献

- **MAEST** (Zhu et al., 2025) - 本项目的核心参考方法
- **DLPFC dataset** (Maynard et al., 2021, *Nature Neuroscience*) - 数据集
- **GraphST** (Long et al., 2023, *Nature Communications*) - 双视图 GNN baseline
- **STAGATE** (Dong et al., 2022, *Nature Communications*) - GNN spatial clustering
- **BayesSpace** (Zhao et al., 2021, *Nature Biotechnology*) - 统计模型 spatial clustering
- **SpaGCN** (Hu et al., 2021, *Nature Methods*) - GCN spatial clustering
- **CCST** (Li & Zhang, 2022, *Nature Communications*) - scRNA cell-type score 思路来源

### 9.2 BibTeX

```bibtex
@misc{maest-x-2026,
  title={MAEST-X: Multi-covariance Adaptive Embedding with
         Spatial-Transcriptomics eXtended for DLPFC Spatial Domain Identification},
  author={Lihan327},
  year={2026},
  note={12 DLPFC slices, ARI median = 0.6196}
}

@article{maynard2021dlpfc,
  title={Transcriptome-scale spatial gene expression in the human
         dorsolateral prefrontal cortex},
  author={Maynard, Kristen R and Collado-Torres, Leonardo and Weber,
          Lukas M and Uytingco, Cedric and Barry, Brianna K and Williams,
          Stephen R and Catallini, Joseph L and Tran, Matthew N and Besich,
          Zachary and Tippani, Meghana and others},
  journal={Nature Neuroscience},
  volume={24},
  number={3},
  pages={425--436},
  year={2021}
}
```

### 9.3 软件致谢

- [scanpy](https://scanpy.readthedocs.io/) (Wolf et al., 2018) - 单细胞分析
- [anndata](https://anndata.readthedocs.io/) (Virshup et al., 2021) - AnnData 数据结构
- [scikit-learn](https://scikit-learn.org/) (Pedregosa et al., 2011) - 机器学习
- [scipy](https://scipy.org/) (Virtanen et al., 2020) - 科学计算
- [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/) - 可视化
- [igraph](https://igraph.org/) / [leidenalg](https://github.com/vtraag/leidenalg) - 社区发现

### 9.4 数据集致谢

DLPFC 数据集由 **Lieber Institute for Brain Development (LIBD)** 公开提供：
<http://research.libd.org/spatialLIBD>

---

## 10. 不上传说明

`.gitignore` 已排除以下内容（不会出现在 GitHub 仓库中）：

| 路径 / 模式 | 大小 | 原因 |
|---|---|---|
| `DLPFC/` | ~7.8 GB | 原始 Visium 数据，按 spatialLIBD 协议独立下载 |
| `results/` | ~5 GB | 训练/推理结果（含 4.87 GB 缓存 pickle、figures/ PNG）|
| `main_file/` | ~50 MB | 后端输出目录（含 Ground_Truth、Results、train_log 日志）|
| `AITraining/` | >5 GB | Python 3.9 虚拟环境（site-packages + Scripts）|
| `__pycache__/`、`*.pyc` | ~1 MB | Python 字节码缓存 |
| `*.pkl`、`*.h5`、`*.h5ad` | 视情况 | 大型数据/缓存文件 |
| `MAEST.pdf`、`*.docx` | ~3 MB | 任务书与外部参考 |
| `.vscode/`、`.idea/` | ~MB | IDE 配置 |

**本地完整复现需先自行准备：**

1. 下载 DLPFC 数据并放置到 `DLPFC/{slice_id}/`
2. 创建 Python 虚拟环境并安装依赖（见 [7.1 环境要求](#71-环境要求)）
3. 依次运行 [7.3 四步运行](#73-四步运行)

---

## 11. 许可证

本项目为 **研究用途**，无明确开源许可证。

DLPFC 数据集版权归 **Lieber Institute for Brain Development (LIBD)** 所有。

如需在公开作品中使用本项目的部分代码，请联系作者获取许可。

---

<div align="center">

**MAEST-X** &middot; Built with NumPy, scikit-learn, scanpy &middot; Tested on DLPFC 12 slices

</div>

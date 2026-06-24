# my_spatial_domain_identification_backend

空间转录组（Spatial Transcriptomics）**空间域识别（Spatial Domain Identification）**后端项目。

本仓库实现了 **MAEST-X** 算法，针对人类背外侧前额叶皮层（DLPFC）12 张 Visium 切片进行空间聚类，并将基因表达与空间位置信息联合建模以识别连续的空间结构域。

---

## 1. 项目简介

空间域识别是空间转录组分析的标准初始步骤，对下游任务（标记基因检测、组织结构推断、疾病相关细胞类型发现等）具有重要影响。

**任务流程：**
```
输入:  ST 数据的基因表达 + 空间位置
       │
       ▼
模型:  特征工程 (HVG + scRNA + 拓扑 + 空间 PCA + ...) + 多方法聚类
       │
       ▼
输出:  每个 spot 的空间域标签
```

---

## 2. 数据集：DLPFC

| 项 | 说明 |
|---|---|
| 全称 | human dorsolateral pre-frontal cortex |
| 来源 | Maynard et al. 2021, *Nature Neuroscience* |
| 技术 | 10x Genomics Visium |
| 切片数 | 12 张（3 位捐赠者，每对 2 张 10 µm 连续切片） |
| 切片 ID | 151507–151510, 151669–151672, 151673–151676 |
| Spot 数 | ~3,400–4,800 / 切片 |
| 层数 | 5 层（5-layer）或 7 层（7-layer） |
| 下载 | http://research.libd.org/spatialLIBD |

> 数据未上传至本仓库，请按 spatialLIBD 协议下载后放置到 `DLPFC/{slice_id}/` 目录。

---

## 3. 算法：MAEST-X

**全称：** Multi-covariance Adaptive Embedding with Spatial-Transcriptomics eXtended

### 3.1 核心创新点

1. **多特征集成（7+ 种增强特征）**
   - HVG 表达
   - Laplacian Eigenmaps（LE）
   - Spatial PCA
   - GraphST 风格嵌入
   - 多分辨率平滑
   - scRNA cell-type score
   - 差异表达 / 拓扑特征

2. **多方法聚类**
   - Gaussian Mixture Model（full / tied 协方差）
   - KMeans
   - Agglomerative
   - Per-K × Per-seed 候选标签

3. **Per-spot 多数投票**
   - 以 v3 baseline 为锚点
   - 当 alternative 标签显著优于 v3 且满足空间一致性时切换

4. **边界保护后处理**
   - 基于表达梯度识别 boundary spot
   - 跳过 boundary 的多数投票，保留细窄层边界

### 3.2 流水线

```
DLPFC 切片 → HVG 3000
          → 7+ 种增强特征
          → StandardScaler
          → 多协方差 GMM × 多 K × 多 seed
          → Hungarian 对齐到 v3 标签空间
          → Per-spot voting (with kNN consistency)
          → 边界保护后处理
          → 输出 12 切片空间域标签
```

---

## 4. 评估指标

| 指标 | 全称 | 范围 | 含义 |
|---|---|---|---|
| ARI | Adjusted Rand Index | [-1, 1] | 聚类与 GT 一致度（主指标） |
| NMI | Normalized Mutual Information | [0, 1] | 信息论一致度 |
| HS | Homogeneity Score | [0, 1] | 每个聚类是否只含单一 GT 类别 |
| CS | Completeness Score | [0, 1] | 每个 GT 类别是否被归入同一聚类 |

四个指标均越高越好。

---

## 5. 目录结构

```
my_spatial_domain_identification_backend/
├── README.md                ★ 本文件
├── .gitignore               Git 排除规则
├── DLPFC.py                 DLPFC 数据读取脚本
├── run_maest_x.py           生成 main_file/ 输出目录结构
├── spatial_domain_frontend.html   前端可视化页面
├── visualize_maest_x.py     可视化脚本
└── code/                    ★ 后端核心模块
    ├── __init__.py
    ├── utils.py                      通用工具（加载、knn、Hungarian）
    ├── metrics.py                    ARI/NMI/HS/CS 计算
    ├── boundary_postprocess.py       边界感知后处理
    ├── multi_scale_smooth.py         多尺度空间平滑
    ├── le_features.py                Laplacian Eigenmaps 特征
    ├── scrna_features.py            scRNA cell-type score
    ├── cluster_zoo.py                聚类方法集合
    ├── consensus.py                  共识投票
    ├── ensemble.py                   集成
    ├── postprocess.py                通用后处理
    ├── maest_x_pipeline.py           ★ MAEST-X 主流程
    └── visualize.py                  内部可视化
```

---

## 6. 快速开始

### 6.1 环境要求

- Python 3.9+
- scanpy ≥ 1.9
- scikit-learn ≥ 1.0
- numpy, pandas, scipy, matplotlib
- 可选：umap-learn（UMAP 可视化）

### 6.2 数据准备

按 `DLPFC/{slice_id}/` 格式放置 12 张切片，每切片含：
```
DLPFC/151507/
├── filtered_feature_bc_matrix.h5
├── metadata.tsv
└── spatial/
    ├── tissue_positions_list.csv
    ├── tissue_hires_image.png
    ├── tissue_lowres_image.png
    └── scalefactors_json.json
```

### 6.3 运行

```bash
# 1. 加载并预处理 DLPFC 数据
python DLPFC.py

# 2. 运行 MAEST-X 主流程（需要 GPU 推荐）
python -c "from code.maest_x_pipeline import run_all_slices; run_all_slices()"

# 3. 生成 main_file/ 标准输出
python run_maest_x.py

# 4. 可视化
python visualize_maest_x.py
```

---

## 7. 性能结果

12 张 DLPFC 切片（5 层 + 7 层混合）：

| 切片 | ARI_v3 baseline | ARI MAEST-X | NMI | HS | CS |
|---|---|---|---|---|---|
| 151507 | 0.5398 | 0.5605 | 0.6999 | 0.6959 | 0.7041 |
| 151508 | 0.5864 | 0.6085 | 0.6776 | 0.5994 | 0.7793 |
| 151509 | 0.6057 | 0.6226 | 0.7183 | 0.6744 | 0.7683 |
| 151510 | 0.5998 | 0.6165 | 0.6818 | 0.6417 | 0.7272 |
| 151669 | 0.7016 | 0.6999 | 0.6632 | 0.6222 | 0.7101 |
| 151670 | 0.4588 | 0.6463 | 0.5282 | 0.4215 | 0.7072 |
| 151671 | 0.7499 | 0.7553 | 0.7519 | 0.7419 | 0.7622 |
| 151672 | 0.5847 | 0.6376 | 0.6089 | 0.5666 | 0.6579 |
| 151673 | 0.5936 | 0.5995 | 0.7096 | 0.7080 | 0.7112 |
| 151674 | 0.6621 | 0.6621 | 0.7417 | 0.6938 | 0.7968 |
| 151675 | 0.5669 | 0.5669 | 0.6835 | 0.6754 | 0.6919 |
| 151676 | 0.5762 | 0.5762 | 0.6714 | 0.6049 | 0.7544 |

**聚合指标（中位数）：**

| 指标 | v3 baseline | MAEST-X | 提升 |
|---|---|---|---|
| ARI | 0.5914 | 0.6177 | +0.0263 |
| NMI | 0.6804 | 0.6808 | +0.0004 |
| HS | 0.6503 | 0.6481 | -0.0022 |
| CS | 0.7151 | 0.7159 | +0.0008 |

---

## 8. 不上传到仓库的内容

`.gitignore` 已排除：

| 路径 / 模式 | 原因 |
|---|---|
| `DLPFC/` | 原始 Visium 数据（~7.8 GB），按 spatialLIBD 协议独立获取 |
| `results/` | 训练/推理结果（含 4.87 GB 缓存 pickle、figures/） |
| `main_file/` | 后端输出目录（含 Ground_Truth、Results、train_log 日志） |
| `AITraining/` | Python 3.9 虚拟环境（>5 GB） |
| `__pycache__/`、`*.pyc` | Python 字节码缓存 |
| `*.pkl`、`*.h5`、`*.h5ad` | 大型数据/缓存文件 |
| `MAEST.pdf`、`任务描述.docx` | 任务书与外部参考 |

---

## 9. 引用与致谢

### 9.1 主要参考文献

- **MAEST** (Zhu et al., 2025) — 本项目的核心参考方法
- **DLPFC dataset** (Maynard et al., 2021) — 数据集
- **GraphST** (Long et al., 2023) — GNN baseline
- **STAGATE** (Dong et al., 2022) — GNN spatial clustering
- **BayesSpace** (Zhao et al., 2021) — 统计模型 spatial clustering
- **SpaGCN** (Hu et al., 2021) — GCN spatial clustering

### 9.2 软件致谢

- [scanpy](https://scanpy.readthedocs.io/) (Wolf et al., 2018) — 单细胞分析
- [scikit-learn](https://scikit-learn.org/) (Pedregosa et al., 2011) — 机器学习
- [PyTorch](https://pytorch.org/) — 深度学习
- [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/) — 可视化

### 9.3 数据集致谢

DLPFC 数据集由 Lieber Institute for Brain Development (LIBD) 公开提供：http://research.libd.org/spatialLIBD

---

## 10. 许可证

本项目为研究用途，无明确开源许可证。
DLPFC 数据集版权归 Lieber Institute for Brain Development。

"""Code package for the SGSGAC-MS-Ensemble spatial domain identification project.

Best algorithm: SGSGAC-MS-Ensemble (Median ARI = 0.5900, Mean ARI = 0.6021)

The 43 modules in this package are organised into 11 functional groups:

============================================================================
1. Data loading and preprocessing
============================================================================
  utils.py                     Common utilities (set_seed, load_visium_slice, ...)
  scrna_features.py            scRNA cell-type score computation (CCST-style)
  multi_scale_smooth.py        5-scale spatial smoothing
  metrics.py                   ARI / NMI / HS / CS evaluation

============================================================================
2. GNN architecture modules
============================================================================
  gatv2_model.py               GATv2 dual-view model
  gat_stable.py                Stable GAT + subgraph contrastive
  graphst_encoder.py           GraphST encoder
  graphst_train.py             GraphST trainer
  gnn_model_basic.py           Basic GNN model (older)

============================================================================
3. MAEST paper reimplementation
============================================================================
  MAEST_GMAE_v2_arch.py        MAEST Graph Masked AutoEncoder architecture
  MAEST_GMAE_v2_train.py       MAEST training loop (with EMA + DGI)
  maest_train.py               MAEST v1 trainer (legacy)

============================================================================
4. Clustering
============================================================================
  clustering_utils.py          mclust_R + GMM + KMeans clustering
  cluster_basic.py             Basic GMM with BIC selection
  cluster_gmm_v1.py            Cluster GMM multi-k
  ensemble_voting.py           Multi-run ensemble voting

============================================================================
5. Post-processing
============================================================================
  boundary_postprocess.py      Boundary-aware post-processing
  iterative_refinement.py      Iterative refinement
  postprocess_v1.py            Basic post-processing (legacy)

============================================================================
6. Losses
============================================================================
  loss_contrastive.py          InfoNCE contrastive loss

============================================================================
7. SGSGAC pipeline iterations (best series)
============================================================================
  SGSGAC_v2.py                 SGSGAC v2  (scRNA cell-type score introduced)
  SGSGAC_v3.py                 SGSGAC v3
  SGSGAC_v5.py                 SGSGAC v5
  SGSGAC_v6.py                 SGSGAC v6  (5-scale smoothing + boundary + ensemble)
  SGSGAC_v7.py                 SGSGAC v7  (final: scRNA-guided supervised refinement)
  SGSGAC_Final.py              SGSGAC Final
  SGSGAC_Final_v4.py           SGSGAC Final v4
  SGSGAC_Pipeline.py            Full SGSGAC pipeline (GATv2 + scRNA + multi-task)

============================================================================
8. HSGATE pipeline iterations (GNN-only series)
============================================================================
  HSGATE_v1.py                 HSGATE v1  (basic GAT)
  HSGATE_v2.py                 HSGATE v2
  HSGATE_v3.py                 HSGATE v3
  HSGATE_v4.py                 HSGATE v4
  HSGATE_v5.py                 HSGATE v5
  HSGATE_v6.py                 HSGATE v6
  HSGATE_v7.py                 HSGATE v7
  HSGATE_v8.py                 HSGATE v8
  HSGATE_v9.py                 HSGATE v9
  HSGATE_v10.py                HSGATE v10
  HSGATE_Final.py              HSGATE Final

============================================================================
9. MSSC baseline
============================================================================
  MSSC_Pipeline.py             Multi-Scale Spatial Clustering baseline

============================================================================
10. Report figure generation (legacy)
============================================================================
  generate_report_figures.py   8-figure legacy SGSGAC v7 report generator

============================================================================
11. Training infrastructure
============================================================================
  train_basic.py               Basic training loop

============================================================================
BEST ALGORITHM WORKFLOW (SGSGAC-MS-Ensemble)
============================================================================
  utils.load_visium_slice      -> load 12 DLPFC slices, HVG=3000
  scrna_features.compute_cell_type_score  -> (n_spots, 35) cell-type scores
  multi_scale_smooth           -> 5-scale smoothing of HVG and cell-type scores
  utils.build_knn_graph        -> kNN6 graph
  multi_scale_smooth (concat)  -> (n_spots, 15177) weighted concatenation
  PCA(30)                      -> (n_spots, 30) cluster input
  clustering_utils.gmm_cluster -> multi-cov + multi-seed GMM ensemble
  boundary_postprocess.boundary_aware_postprocess  -> protect boundary spots
  scrna-guided refinement      -> use scRNA confidence to refine low-conf spots
  metrics.compute_metrics      -> ARI / NMI / HS / CS
"""

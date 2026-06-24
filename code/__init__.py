"""Code package for MAEST-X spatial domain identification.

Algorithm: MAEST-X (Multi-covariance Adaptive Embedding with
            Spatial-Transcriptomics eXtended)

Modules (13):
  Data / utils
    utils.py                     Common utilities (set_seed, load, knn, Hungarian)
    metrics.py                   ARI / NMI / HS / CS evaluation

  Features
    le_features.py               Laplacian Eigenmaps feature extraction
    scrna_features.py            scRNA cell-type score (CCST-style)
    multi_scale_smooth.py        Multi-scale spatial smoothing

  Clustering
    cluster_zoo.py               Cluster zoo (GMM / KMeans / Agglomerative)
    ensemble.py                  Ensemble voting
    consensus.py                 Consensus merging

  Post-processing
    boundary_postprocess.py      Boundary-aware post-processing
    postprocess.py               Generic post-processing

  Pipeline / visualisation
    maest_x_pipeline.py          ★ MAEST-X main pipeline
    visualize.py                 Visualisation helpers

MAEST-X pipeline (see code/maest_x_pipeline.py):
    load_visium_slice            -> load 12 DLPFC slices, HVG=3000
    multi-feature generation     -> 7+ enhanced features (HVG, LE, Spatial PCA,
                                     GraphST, multi-res, scRNA, topo, ...)
    StandardScaler               -> (N, D) standardised features
    multi-cov GMM ensemble       -> per-spot alternative labels
    Hungarian align to v3        -> unified label space
    per-spot majority voting     -> with kNN spatial consistency
    boundary-aware postprocess   -> protect boundary spots
    metrics.compute_metrics      -> ARI / NMI / HS / CS
"""

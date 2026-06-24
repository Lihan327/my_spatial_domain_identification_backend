import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
# pip install --user scikit-misc
from sklearn.metrics.cluster import adjusted_rand_score

data_root = 'DLPFC'  #根据自己的数据路径自行调整
result_path = "DLPFC/DLPFC_result/" #自行调整

slice_idx = ['151507', '151508', '151509', '151510',
             '151669', '151670', '151671', '151672',
             '151673', '151674', '151675', '151676']
# slice_idx = ['151673']

for section_id in slice_idx:
    input_dir = os.path.join(data_root, section_id)
    adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()
    print("=============原始{}切片信息===============".format(section_id))
    print(adata)

    #Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    ## add ground truth
    Ann_df = pd.read_csv(os.path.join(data_root, section_id,'metadata.tsv') , sep='\t')
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'layer_guess']
    # print(adata)

    # filter out NA nodes
    adata = adata[~pd.isnull(adata.obs['Ground Truth'])]
    print("=============过滤未注释节点后，{}切片信息===============".format(section_id))
    print(adata)

    out_path = os.path.join(result_path, section_id)
    # 检查文件夹是否存在，如果不存在则创建文件夹
    if not os.path.exists(out_path):
        os.makedirs(out_path)


    #可视化注释信息
    fig, ax = plt.subplots(figsize=(9,6))
    sc.pl.spatial(adata, img_key="hires", color="Ground Truth", show=False, ax=ax, legend_fontsize=15)
    plt.subplots_adjust(right=0.66)  # 可根据需要调整此值
    plt.title("{}".format(section_id), fontsize=25)
    # plt.show()
    plt.savefig(os.path.join(out_path,"{}_ground truth.png").format(section_id))


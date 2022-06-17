### Coupled autoencoders for M, E, and T analysis

Objectives:
 - Joint analysis of Morphology, Electrophysiology, and Transcriptomic data from Patch-seq experiments.
 - Extending results from Patch-seq dataset to EM reconstructions

### Data
 - Patch-seq dataset for V1 cortical interneurons ([Gala et al. 2021](https://www.nature.com/articles/s43588-021-00030-1): 3411 cells in T and E)
 - Patch-seq dataset for V1 cortical interneurons ([Gouwens et al. 2020](https://www.sciencedirect.com/science/article/pii/S009286742031254X): 3819 cells in T and E)
 - Density representations for morphology (721 cells)

### Environment

1. Navigate to the `cplAE_MET` folder with the `setup.py` file.
2. Create the conda environment with the main dependencies.
```bash
conda create -n cplAE_MET
conda activate cplAE_MET
conda install python=3.8
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch #see system specific instructions
pip install scikit-learn jupyterlab seaborn pandas rich tqdm timebudget statsmodels umap-learn
pip install tensorboard
```
3. Install the development version of this repository
```bash
pip install -e .
```
4. Install the `cplAE_TE` repository after cloning it.
```bash
# can do this within any directory on local machine
git clone https://github.com/AllenInstitute/coupledAE-patchseq
cd coupledAE-patchseq
pip install -e .
```

### Experiments
"T_ME_aT_5-0_aM_5-0_asd_1-0_aE_5-0_aME_5-0_lambda_ME_T_1-0_lambda_tune_ME_T_0-75_lambda_ME_M_1-0_lambda_ME_E_1-0_aug_dec_1_Enoise_0-05_Mnoise_0-0_scale_0-3_ld_5_ne_50000_ri_0_fold_2.pkl"

### Additional repositories
 - [celltype_hierarchy](https://github.com/AllenInstitute/celltype_hierarchy) - Helpers for dendrogram manipulation
 - [cplAE_TE](https://github.com/AllenInstitute/coupledAE-patchseq) - Coupled autoencoders for T and E

### Config
```toml
# config.toml contents
package_dir = '/Local/code/cplAE_MET/'
MET_inh_data = '/Local/data/inh_MET_model_input_mat.mat'
```

```
# config_preproc.toml contents
package_dir = '/Users/fahimehb/Documents/git-workspace/cplAE_MET/'

#For T
specimen_ids_file = "exc_inh_specimen_ids_30Mar22.txt"
gene_file = "good_genes_beta_score.csv"
t_data_output_file = "T_data_30Mar22.csv"
t_anno_output_file = "T_anno_30Mar22.csv"
gene_id_output_file = "gene_ids_30Mar22.csv"

#For M
m_data_folder = 'm_data'
m_anno = 'm_anno.csv'
hist2d_120x4_folder = 'hist2d_120x4'
m_output_file = 'M_data_30Mar22.mat'

#For E
E_timeseries_file = "fv_Ephys_timeseries_30Mar22.h5"
ipfx_features_file = "ipfx_features_30Mar22.csv"
e_output_file = "E_data_30Mar22.csv"

#For MET
met_output_file = "MET_data_30Mar22.mat"
```

### Contributors
Fahimeh Baftizadeh, Rohan Gala

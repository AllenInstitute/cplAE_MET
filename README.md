### Coupled autoencoders for M, E, and T analysis

 - Joint analysis of Morphology, Electrophysiology, and Transcriptomic data from Patch-seq experiments.
 - Extending results from Patch-seq dataset to EM reconstructions

### Data
 - Patch-seq dataset for V1 cortical interneurons (Gala et al. 2021: 3,411 cells in T and E)
 - Patch-seq dataset for V1 cortical interneurons (Gouwens et al. 2020: 3,819 cells in T and E)
 - Density representations for morphology ()

### Environment

1. Navigate to the cplAE_MET folder with the `setup.py` file.
2. Create the conda environment with the main dependencies.
```bash
conda create -n cplAE_MET
conda activate cplAE_MET
conda install python=3.8
pip install tensorflow scikit-learn jupyterlab
pip install seaborn pandas 
pip install rich tqdm timebudget
```
3. Install the development version of this repository
```bash
pip install -e .
```

### Additional repositories
 - [celltype_hierarchy](https://github.com/AllenInstitute/celltype_hierarchy) - Helpers for dendrogram manipulation
 - [cplAE_TE](https://github.com/AllenInstitute/coupledAE-patchseq) - Coupled autoencoders for T and E


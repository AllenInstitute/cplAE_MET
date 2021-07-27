# Prepare ET model input 
To prepare the input data for ET model, we need the following files:
       
    data.feather (the cpm values per cell)
    anno.feather (the metadata file)
    Ephys_timeseries.h5 (the ephys timeseries)
    ipfx_features.csv (the ipfx features)
    good_genes_beta_score.csv (genes with their beta scores)
    specimen_ids.txt
    
All the input info will be saved into a input_run_data_proc_ET.json file:

    {
    "input_path": "/Users/fahimehb/Documents/git-workspace/cplAE_MET/data/proc/",
    "output_path": "/Users/fahimehb/Documents/git-workspace/cplAE_MET/data/proc/",
    "T_data_file": "data.feather",
    "T_annotation_file": "anno.feather",
    "gene_file": "good_genes_beta_score.csv",
    "specimen_ids_file": "specimen_ids.txt",
    "E_timeseries_file": "Ephys_timeseris.h5",
    "ipfx_features_file": "inh_ipfx_features_20July2021.csv", 
    "beta_threshold": 0.4, 
    "pca_comp_threshold": 0.97, 
    "output_file_prefix": "inh"
    }

## 1. Prepare T data
To get the transcriptomic data ready:

    python data_proc_T.py --input input_run_data_proc_ET.json

## 2. Prepare E data
To get the ephys data ready:

    python data_proc_E.py --input input_run_data_proc_ET.json

## 3. Prepare ET data
To get the final .mat file with the E and T data:

    python data_proc_T.py --input input_run_data_proc_ET.json

The output will be save in the output directory as a .mat file which is ready to be used for ET model training.
1. Clone and install ipfx repository into a new conda env.:
```
conda create -n ipfx python==3.7
conda activate ipfx
git clone https://github.com/AllenInstitute/ipfx.git
cd ipfx
pip install -e .
```

2. Extract time series and ipfx features:
```
run_feature_collection.py           #extracts ephys features
run_feature_vector_extraction.py    #extracts time series
specimen_ids.txt                    #specimen id file
input_ephys_feature_vector.json     #input for run_feature_vector_extraction.py
input_ephys_feature_collection.json #input for run_feature_collection.py
```

`input_ephys_feature_vector.json` contents:
```json
{"output_dir": "/<your working directory path>",
"input": "/<your working directory path>/specimen_ids.txt"
"data_source":"lims",
"output_code":"fv_Ephys_timeseries",
"output_file_type":"h5"}
```

`input_ephys_feature_collection.json` contents:
```json
{"output_file": "/<your working directory path>/ipfx_features.csv",
"input": "/<your working directory path>/specimen_ids.txt",
"include_failed_cells":"True",
"data_source":"lims",
"run_parallel":"False",
"log_level":"DEBUG"}
```

Run it on hpc with `sbatch run_ipfx.sh` whose contents are:
```bash
#!/bin/bash
#SBATCH --partition=celltypes
#SBATCH --job-name=1seg_run2
#SBATCH --cpus-per-task=32
#SBATCH --mem=100g
#SBATCH --time=48:00:00
#SBATCH --output=log.out

export LIMS_HOST="limsdb2"
export LIMS_DBNAME="lims2"
export LIMS_USER="limsreader"
export LIMS_PASSWORD="limsro"

cd /<path to the working directory>/ 
source activate ipfx 

#Use only one python script at a time
python -m run_feature_vector_extraction --input_json input_ephys_feature_vector.json
python -m run_feature_collection --input_json input_ephys_feature_collection.json
```
Outputs are in `fv_Ephys_timeseries.h5` and `ipfx_features.csv`

**Note**: See `ET_model_input_preparation.md` for assembling the TE dataset.
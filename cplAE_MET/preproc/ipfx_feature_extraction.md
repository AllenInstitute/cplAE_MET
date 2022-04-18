# Ipfx features extraction

1. Clone the ipfx repository from this link:

       https://github.com/AllenInstitute/ipfx.git
   
2. Navigate to the ipfx folder with the setup.py file. 
3. Create a conda environment and name it ipfx:
   
       conda create -n ipfx
       conda activate ipfx
       conda install python=3.7
     

4. Install the development version of this repository 
   
       pip install -e .
    
5. Extract time seris and ipfx features. To do so, in your working directory, 
   prepare the following files (the python codes are already inside ipfx repo,
   you need to prepare the input json files and the specimen_id file):
   
       run_feature_collection.py  #the python code to exctract ephys features
       run_feature_vector_extraction.py  #the python code to exctract time series
       specimen_ids.txt #specimen id file
       input_ephys_feature_vector.json  #input file for run_feature_vector_extraction.py
       input_ephys_feature_collection.json #input file for run_feature_collection.py
   

input_ephys_feature_vector.json has the following info:

    {
     "output_dir": "/<your working directory path>",
     "input": "/<your working directory path>/specimen_ids.txt"
     "data_source":"lims",
     "output_code":"fv_Ephys_timeseries",
     "output_file_type":"h5"
     }

input_ephys_feature_collection.json has the following info:
          
    {
      "output_file": "/<your working directory path>/ipfx_features.csv",
      "input": "/<your working directory path>/specimen_ids.txt",
      "include_failed_cells":"True",
      "data_source":"lims",
      "run_parallel":"False",
      "log_level":"DEBUG"
    }

To run the these two python codes on slurm, the following run_ipfx.sh can be used:

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

    python -m run_feature_vector_extraction --input_json input_ephys_feature_vector.json
    
in order to submit this job on slurm, the following command can be used:

    sbatch run_ipfx.sh

The last line of the run_ipfx.sh can be modified to run the other python code as:

    python -m run_feature_collection --input_json input_ephys_feature_collection.json

The output of these two runs will be two files:

    fv_Ephys_timeseries.h5
    ipfx_features.csv
    
To prepare the transcriptomic and ephys data ready for training, read ET_model_input_preparation.md
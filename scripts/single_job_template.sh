#!/bin/bash
#SBATCH --partition=celltypes
#SBATCH --job-name=optuna_nopt0
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=20g
#SBATCH --time=100:00:00
#SBATCH --output=cpu_TEM_6k_5d_M120x4_2conv_10_10_0opset.out

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/miniconda3/envs/cplmet/lib
cd /allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/git_workspace/cplAE_MET/cplAE_MET/models
source /allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/miniconda3/etc/profile.d/conda.sh

python -m bayesian_optimization_conv --latent_dim 5 --exp_name TEM_6k_5d_M120x4_2conv_10_10 --n_epochs 2500 --opt_storage_db TEM_6k_5d_M120x4_2conv_10_10.db --config_file config.toml --optimization True --batch_size 1000

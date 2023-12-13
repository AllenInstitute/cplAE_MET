#!/bin/bash
#SBATCH --partition=celltypes
#SBATCH --job-name=cpl_simple
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=20g
#SBATCH --time=100:00:00
#SBATCH --output=cpl_simple.out

cd /allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Ian/code/cplAE_MET
source activate cplae

python cplAE_MET/models/bayesian_optimization_conv.py --latent_dim 5 --exp_name test_run --n_epochs 15 --config_file config.toml --optimization False --use_defined_params True --batch_size 128

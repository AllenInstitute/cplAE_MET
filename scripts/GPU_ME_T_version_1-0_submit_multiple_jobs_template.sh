model_dir="/allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/git_workspace/cplAE_MET/cplAE_MET/models"

config_file="config.toml"
exp_name="TEM_6k_5d_M120x4_2conv_10_10"
optimization=True
n_epochs=2500
latent_dim=5
batch_size=1000


for opset in {0..100};do
    jobname="optuna_nopt"${opset}
    exp_name=$exp_name
    opt_storage_db=${exp_name}'.db'
    echo '#!/bin/bash'>>subjob1.sh 
    echo '#SBATCH --partition=celltypes'>>subjob1.sh
    echo '#SBATCH --job-name='$jobname''>>subjob1.sh
    echo '#SBATCH --cpus-per-task=8'>>subjob1.sh
    #echo '#SBATCH --gpus=v100:1'>>subjob1.sh
    echo '#SBATCH --gpus=1'>>subjob1.sh
    echo '#SBATCH --nodes=1'>>subjob1.sh
    echo '#SBATCH --mem=20g'>>subjob1.sh
    echo '#SBATCH --time=10:00:00'>>subjob1.sh
    echo '#SBATCH --output='$exp_name'_'$opset'opset.out'>>subjob1.sh
    echo "#SBATCH --exclude=n278,n279,n280,n281,n287,n289">>subjob1.sh
    echo 'module load cuda'>>subjob1.sh 
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/miniconda3/envs/cplmet/lib'>>subjob1.sh
    echo 'cd '$model_dir>>subjob1.sh
    echo 'source /allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/miniconda3/etc/profile.d/conda.sh'>>subjob1.sh
    echo 'python -m bayesian_optimization_conv --latent_dim '${latent_dim}' --exp_name '${exp_name}' --n_epochs '$n_epochs' --opt_storage_db '$opt_storage_db' --config_file '$config_file' --optimization '${optimization}' --batch_size '${batch_size}''>>subjob1.sh
    echo '...'

    sleep 15
 
    sbatch subjob1.sh
    rm subjob1.sh

    echo 'Job: '$jobname''
done


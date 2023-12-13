model_dir="/allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/git_workspace/cplAE_MET/cplAE_MET/models"

config_file="config.toml"
exp_name="TEM_6k_5d_M120x4_2conv_10_10"
opt_storage_db=${exp_name}'.db'
optimization=True
n_epochs=2500
latent_dim=5
batch_size=1000


for opset in {0..2};do
    jobname="optuna_nopt"${opset}
    exp_name=$exp_name
    echo '#!/bin/bash'>>subjob.sh 
    echo '#SBATCH --partition=celltypes'>>subjob.sh
    echo '#SBATCH --job-name='$jobname''>>subjob.sh
    echo '#SBATCH --cpus-per-task=8'>>subjob.sh
    #echo '#SBATCH --gpus=v100:1'>>subjob.sh
    echo '#SBATCH --nodes=1'>>subjob.sh
    echo '#SBATCH --mem=20g'>>subjob.sh
    echo '#SBATCH --time=100:00:00'>>subjob.sh
    echo '#SBATCH --output='cpu_$exp_name'_'$opset'opset.out'>>subjob.sh
    #echo "#SBATCH --exclude=n278,n279,n280,n281,n289">>subjob.sh
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/miniconda3/envs/cplmet/lib'>>subjob.sh
    echo 'cd '$model_dir>>subjob.sh
    echo 'source /allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/miniconda3/etc/profile.d/conda.sh'>>subjob.sh
    echo 'python -m bayesian_optimization_conv --latent_dim '${latent_dim}' --exp_name '${exp_name}' --n_epochs '$n_epochs' --opt_storage_db '$opt_storage_db' --config_file '$config_file' --optimization '${optimization}' --batch_size '${batch_size}''>>subjob.sh
    echo '...'

    sleep 15
 
    sbatch subjob.sh
    rm subjob.sh

    echo 'Job: '$jobname''
done


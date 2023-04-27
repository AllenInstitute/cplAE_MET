model_dir="/allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/git_workspace/cplAE_MET/cplAE_MET/models"
exp_name="MET_10k_stratified_400nmfs_50met_removed_mass_norm_v0"
opt_storage_db=${exp_name}'.db'
n_epochs=2000
config_file="config_10k.toml"


for opset in {0..1};do
    jobname="optuna_nopt"${opset}
    exp_name=$exp_name
    echo '#!/bin/bash'>>subjob10.sh 
    echo '#SBATCH --partition=celltypes'>>subjob10.sh
    echo '#SBATCH --job-name='$jobname''>>subjob10.sh
    echo '#SBATCH --cpus-per-task=8'>>subjob10.sh
    echo '#SBATCH --nodes=1'>>subjob10.sh
    echo '#SBATCH --mem=20g'>>subjob10.sh
    echo '#SBATCH --time=20:00:00'>>subjob10.sh
    echo '#SBATCH --output='$exp_name'_'$opset'opset.out'>>subjob10.sh
    echo "#SBATCH --exclude=n266,n268,n94,n220,n74,n216,n251,n267,n175,n222,n269,n215,n257,n252,n79,n77,n255">>subjob10.sh
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/miniconda3/envs/cplmet/lib'>>subjob10.sh
    echo 'cd '$model_dir>>subjob10.sh
    echo 'source /allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/miniconda3/etc/profile.d/conda.sh'>>subjob10.sh
    echo 'python -m bayesian_optimization --exp_name '${exp_name}' --n_epochs '$n_epochs' --opt_storage_db '$opt_storage_db' --config_file '$config_file''>>subjob10.sh
    echo '...'

    sleep 15
 
    sbatch subjob10.sh
    rm subjob10.sh

    echo 'Job: '$jobname''
done


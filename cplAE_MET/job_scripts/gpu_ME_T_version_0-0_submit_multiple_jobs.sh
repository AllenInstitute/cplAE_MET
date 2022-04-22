model_dir="/allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/git_workspace/cplAE_MET/cplAE_MET/models/"

alpha_T=5.0
alpha_E=5.0
alpha_M=5.0
alpha_sd=5.0
alpha_ME=5.0
lambda_ME_T=1.0
lambda_ME_M=1.0
lambda_ME_E=1.0
lambda_tune_ME_T=0.75
scale_factor=0.3
latent_dim=10
model_id="corrected_val_cells_T_ME"
aug_dec=1
exp_name_pre="triple_mode_version_0.0"

c=0
for nf in {0..9};do
    #for lambda_tune_ME_T in {0.5,0.75,0.9,1.0}; do
    c=$((c+1))
    jobname="cpl_ME_T"$cpl_ME_T"_nfold"$nf
    exp_name=$exp_name_pre'/'$model_id'_aT_'${alpha_T}'_aE_'${alpha_E}'_aM_'${alpha_M}'_asd_'${alpha_sd}'_aME_'${alpha_ME}'_lmd_ME_T_'${lambda_ME_T}'_lmd_ME_M_'${lambda_ME_M}'_lmd_ME_E_'${lambda_ME_E}'_lmd_tune_ME_T_'${lambda_tune_ME_T}'_aug_dec_'${aug_dec}
    echo $exp_name
    echo '#!/bin/bash'>>subjob.sh 
    echo '#SBATCH --partition=celltypes'>>subjob.sh
    echo '#SBATCH --job-name='$jobname''>>subjob.sh
    echo '#SBATCH --cpus-per-task=8'>>subjob.sh
    echo '#SBATCH --gpus=v100:1'>>subjob.sh
    echo '#SBATCH --nodes=1'>>subjob.sh
    echo '#SBATCH --mem=60g'>>subjob.sh
    echo '#SBATCH --time=15:00:00'>>subjob.sh
    echo '#SBATCH --output=log.out'>>subjob.sh
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/miniconda3/envs/cplmet/lib'>>subjob.sh
    echo 'cd '$model_dir>>subjob.sh
    echo 'source /allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/miniconda3/etc/profile.d/conda.sh'>>subjob.sh
    echo 'python -m train_T_EM_AE --model_id '${model_id}'  --exp_name '${exp_name}' --alpha_T '${alpha_T}' --alpha_E '${alpha_E}' --alpha_M '${alpha_M}' --alpha_sd  '${alpha_sd}' --alpha_ME '${alpha_ME}' --lambda_ME_T '${lambda_ME_T}' --lambda_ME_M '${lambda_ME_M}' --lambda_ME_E '${lambda_ME_E}' --lambda_tune_ME_T '${lambda_tune_ME_T}'  --latent_dim '${latent_dim}' --n_epochs 50000 --n_fold '${nf} --augment_decoders ${aug_dec}''>>subjob.sh
    echo '...'

    wait 

#    sbatch subjob.sh

mv subjob.sh ${c}.sh
    echo 'Job: '$jobname''
#done
done


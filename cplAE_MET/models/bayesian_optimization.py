# From python
import os
import sys
import shutil
import logging
import argparse
import numpy as np
from pathlib import Path
from sklearn.mixture import GaussianMixture

# From torch
import torch
from torch.utils.data import DataLoader

# From optuna
import optuna

# From CplAE_MET
from cplAE_MET.utils.utils import savepkl
from cplAE_MET.models.torch_utils import tonumpy
from cplAE_MET.utils.dataset import MET_exc_inh_v2
from cplAE_MET.utils.load_config import load_config
from cplAE_MET.models.torch_utils import MET_dataset_v2
from cplAE_MET.models.model_classes import Model_ME_T_v2
from cplAE_MET.models.classification_functions import run_LDA
from cplAE_MET.models.train_tempcsfeatures_dev import set_paths, init_losses

from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name',              default='small_range_optuna_all_connected_gmm_10trial_2000epochs',   type=str,   help='Experiment set')
parser.add_argument('--alpha_T',               default=1.0,            type=float, help='T reconstruction loss weight')
parser.add_argument('--alpha_M',               default=1.0,            type=float, help='M reconstruction loss weight')
parser.add_argument('--alpha_E',               default=1.0,            type=float, help='E reconstruction loss weight')
parser.add_argument('--alpha_ME',              default=1.0,            type=float, help='ME reconstruction loss weight')
parser.add_argument('--lambda_TE',             default=1.0,            type=float, help='coupling loss weight between T and E')
parser.add_argument('--lambda_TM',             default=1.0,            type=float, help='coupling loss weight between T and M')
parser.add_argument('--lambda_ME',             default=1.0,            type=float, help='coupling loss weight between M and E')
parser.add_argument('--lambda_ME_T',           default=1.0,            type=float, help='coupling loss weight between ME and T')
parser.add_argument('--lambda_ME_M',           default=1.0,            type=float, help='coupling loss weight between ME and M')
parser.add_argument('--lambda_ME_E',           default=1.0,            type=float, help='coupling loss weight between ME and E')
parser.add_argument('--lambda_tune_T_E_range', default=(0,2),          type=float, help='Tune the directionality of coupling between T and E')
parser.add_argument('--lambda_tune_E_T_range', default=(0,2),          type=float, help='Tune the directionality of coupling between E and T')
parser.add_argument('--lambda_tune_T_M_range', default=(0,2),          type=float, help='Tune the directionality of coupling between T and M')
parser.add_argument('--lambda_tune_M_T_range', default=(0,2),          type=float, help='Tune the directionality of coupling between M and T')
parser.add_argument('--lambda_tune_E_M_range', default=(0,2),          type=float, help='Tune the directionality of coupling between E and M')
parser.add_argument('--lambda_tune_M_E_range', default=(0,2),          type=float, help='Tune the directionality of coupling between M and E')
parser.add_argument('--lambda_tune_T_ME_range',default=(0,2),          type=float, help='Tune the directionality of coupling between T and ME')
parser.add_argument('--lambda_tune_ME_T_range',default=(0,2),          type=float, help='Tune the directionality of coupling between ME and T')
parser.add_argument('--lambda_tune_ME_M_range',default=(0,2),          type=float, help='Tune the directionality of coupling between ME and M')
parser.add_argument('--lambda_tune_M_ME_range',default=(0,2),          type=float, help='Tune the directionality of coupling between M and ME')
parser.add_argument('--lambda_tune_ME_E_range',default=(0,2),          type=float, help='Tune the directionality of coupling between ME and E')
parser.add_argument('--lambda_tune_E_ME_range',default=(0,2),          type=float, help='Tune the directionality of coupling between E and ME')
parser.add_argument('--config_file',           default='config.toml',  type=str,   help='config file with data paths')
parser.add_argument('--n_epochs',              default=2000,           type=int,   help='Number of epochs to train')
parser.add_argument('--fold_n',                default=0,              type=int,   help='kth fold in 10-fold CV splits')
parser.add_argument('--latent_dim',            default=3,              type=int,   help='Number of latent dims')
parser.add_argument('--batch_size',            default=1000,           type=int,   help='Batch size')
parser.add_argument('--n_trials',              default=10,            type=int,   help='number trials for bayesian optimization, if it is larger than 1')
parser.add_argument('--opset',                 default=1,              type=int,   help='round of operation with n_trials')


def set_paths(config_file=None, exp_name='DEBUG', fold_n=0):
    paths = load_config(config_file=config_file, verbose=False)
    paths['result'] = f'{str(paths["package_dir"] / "data/results")}/{exp_name}/'
    Path(paths['result']).mkdir(parents=False, exist_ok=True)
    paths['tb_logs'] = f'{str(paths["package_dir"] / "data/results")}/tb_logs/{exp_name}/fold_{str(fold_n)}/'
    if os.path.exists(paths['tb_logs']):
        shutil.rmtree(paths['tb_logs'])
    Path(paths['tb_logs']).mkdir(parents=True, exist_ok=False)
    return paths

def save_ckp(state, checkpoint_dir, fname):
    filename = fname + '.pt'
    f_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, f_path)

def init_losses(loss_dict):
    t_loss = {}
    for k in loss_dict.keys():
        t_loss[k] = 0.
    return t_loss

def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val

def rm_emp_end_str(myarray):
    return np.array([mystr.rstrip() for mystr in myarray])

def save_results(model, dataloader, dat, fname):
    '''
    Takes the model, run it in the evaluation mode to calculate the embeddings and reconstructions for printing out.
    '''
    model.eval()

    for all_data in iter(dataloader):
        with torch.no_grad():
            loss_dict, z_dict, xr_dict = model(all_data)

    savedict = {'XT': tonumpy(all_data['xt']),
                'XM': tonumpy(all_data['xm']),
                'XE': tonumpy(all_data['xe']),
                'XrT': tonumpy(xr_dict['xrt']),
                'XrE': tonumpy(xr_dict['xre']),
                'XrM': tonumpy(xr_dict['xrm']),
                'XrM_me_paired': tonumpy(xr_dict['xrm_me_paired']),
                'XrE_me_paired': tonumpy(xr_dict['xre_me_paired']),
                'zm': tonumpy(z_dict['zm']),
                'ze': tonumpy(z_dict['ze']),
                'zt': tonumpy(z_dict['zt']),
                'zme_paired': tonumpy(z_dict['zme_paired']),
                'is_t_1d':tonumpy(all_data['is_t_1d']),
                'is_e_1d':tonumpy(all_data['is_e_1d']),
                'is_m_1d':tonumpy(all_data['is_m_1d']), 
                'cluster_id': dat.cluster_id,
                'gene_ids': dat.gene_ids,
                'e_features': dat.E_features,
                'specimen_id': rm_emp_end_str(dat.specimen_id),
                'cluster_label': rm_emp_end_str(dat.cluster_label),
                'cluster_color': rm_emp_end_str(dat.cluster_color)}

    savepkl(savedict, fname)
    model.train()
    return

def optimizer_to(optim, device):
    '''function to send the optimizer to device, this is required only when loading an optimizer 
    from a previous model'''
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

                            
def Criterion(model_config, loss_dict):
    ''' Loss function for the autoencoder'''

    criterion = model_config['T']['alpha_T'] * loss_dict['rec_t'] + \
                model_config['E']['alpha_E'] * loss_dict['rec_e'] + \
                model_config['M']['alpha_M'] * loss_dict['rec_m'] + \
                model_config['ME']['alpha_ME'] * (loss_dict['rec_m_me'] + loss_dict['rec_e_me']) + \
                model_config['TE']['lambda_TE'] * model_config['TE']['lambda_tune_T_E'] * loss_dict['cpl_t->e'] + \
                model_config['TE']['lambda_TE'] * model_config['TE']['lambda_tune_E_T'] * loss_dict['cpl_e->t'] + \
                model_config['TM']['lambda_TM'] * model_config['TM']['lambda_tune_T_M'] * loss_dict['cpl_t->m'] + \
                model_config['TM']['lambda_TM'] * model_config['TM']['lambda_tune_M_T'] * loss_dict['cpl_m->t'] + \
                model_config['ME']['lambda_ME'] * model_config['ME']['lambda_tune_E_M'] * loss_dict['cpl_e->m'] + \
                model_config['ME']['lambda_ME'] * model_config['ME']['lambda_tune_M_E'] * loss_dict['cpl_m->e'] + \
                model_config['ME_T']['lambda_ME_T'] * model_config['ME_T']['lambda_tune_T_ME'] * loss_dict['cpl_t->me'] + \
                model_config['ME_T']['lambda_ME_T'] * model_config['ME_T']['lambda_tune_ME_T'] * loss_dict['cpl_me->t'] + \
                model_config['ME_M']['lambda_ME_M'] * model_config['ME_M']['lambda_tune_ME_M'] * loss_dict['cpl_me->m'] + \
                model_config['ME_M']['lambda_ME_M'] * model_config['ME_M']['lambda_tune_M_ME'] * loss_dict['cpl_m->me'] + \
                model_config['ME_E']['lambda_ME_E'] * model_config['ME_E']['lambda_tune_ME_E'] * loss_dict['cpl_me->e'] + \
                model_config['ME_E']['lambda_ME_E'] * model_config['ME_E']['lambda_tune_E_ME'] * loss_dict['cpl_e->me'] 

    return criterion

def denovo_clustering_gmm(X, n_components_range=np.arange(1,93), covariance_type="full", random_state=0):

    models = [GaussianMixture(n, covariance_type=covariance_type, random_state=random_state).fit(X) 
                        for n in n_components_range]
    return np.argmin([m.bic(X) for m in models])


def run_gmm(model, dataloader):

    model.eval()
    for all_data in iter(dataloader):
        _, z_dict, _ = model(all_data) 

    is_t_1d = tonumpy(all_data['is_t_1d'])
    is_e_1d = tonumpy(all_data['is_e_1d'])
    is_m_1d = tonumpy(all_data['is_m_1d'])
    is_te_1d = np.logical_and(is_t_1d, is_e_1d)
    is_tm_1d = np.logical_and(is_t_1d, is_m_1d)
    is_me_1d = np.logical_and(is_m_1d, is_e_1d)
    is_met_1d = np.logical_and(is_t_1d, is_me_1d)

    zt = tonumpy(z_dict['zt'])
    ze = tonumpy(z_dict['ze'])
    zm = tonumpy(z_dict['zm'])
    zme_paired = tonumpy(z_dict['zme_paired'])
    
    n_t_gmm_types = denovo_clustering_gmm(zt[is_t_1d])
    n_e_gmm_types = denovo_clustering_gmm(ze[is_te_1d])
    n_m_gmm_types = denovo_clustering_gmm(zm[is_tm_1d])
    n_me_gmm_types = denovo_clustering_gmm(zme_paired[is_met_1d])
    
    print("n_t_gmm_types, n_e_gmm_types, n_m_gmm_types, n_me_gmm_types")
    print(n_t_gmm_types, n_e_gmm_types, n_m_gmm_types, n_me_gmm_types)
    model_score = np.min((n_t_gmm_types, n_e_gmm_types, n_m_gmm_types, n_me_gmm_types))

    print("model_score:", model_score)
    return model_score 


def main(exp_name="DEBUG",
         config_file="config.toml", 
         n_epochs=10, 
         fold_n=0, 
         alpha_T=0.0,
         alpha_M=0.0,
         alpha_E=0.0,
         alpha_ME=0.0,
         lambda_TE=0.0,
         lambda_TM=0.0,
         lambda_ME=0.0,
         lambda_ME_T=0.0,
         lambda_ME_M=0.0,
         lambda_ME_E=0.0,
         lambda_tune_T_E_range=(0,2),
         lambda_tune_E_T_range=(0,2),
         lambda_tune_T_M_range=(0,2),
         lambda_tune_M_T_range=(0,2),
         lambda_tune_T_ME_range=(0,2),
         lambda_tune_ME_T_range=(0,2),
         lambda_tune_ME_E_range=(0,2),
         lambda_tune_E_ME_range=(0,2),
         lambda_tune_ME_M_range=(0,2),
         lambda_tune_M_ME_range=(0,2),
         lambda_tune_M_E_range=(0,2),
         lambda_tune_E_M_range=(0,2),
         latent_dim=2,
         batch_size=1000, 
         n_trials=1,
         opset=0):

    
    def build_model(params):
        ''' Config and build the model'''

        model_config = dict(latent_dim=latent_dim, 
                            batch_size=batch_size,
                            T=dict(dropout_p=0.2, alpha_T=alpha_T),
                            E=dict(gnoise_std=train_dataset.gnoise_e_std, gnoise_std_frac=0.05, dropout_p=0.2, alpha_E=alpha_E),
                            M=dict(gnoise_std=train_dataset.gnoise_m_std, gnoise_std_frac=0.005, dropout_p=0.2, alpha_M=alpha_M),
                            TE=dict(lambda_TE=lambda_TE, lambda_tune_T_E=params['lambda_tune_T_E'], lambda_tune_E_T=params['lambda_tune_E_T']),
                            TM=dict(lambda_TM=lambda_TM, lambda_tune_T_M=params['lambda_tune_T_M'], lambda_tune_M_T=params['lambda_tune_M_T']),
                            ME=dict(alpha_ME=alpha_ME, lambda_ME=lambda_ME, lambda_tune_M_E=params['lambda_tune_M_E'], lambda_tune_E_M=params['lambda_tune_E_M']),
                            ME_T=dict(lambda_ME_T=lambda_ME_T, lambda_tune_ME_T=params['lambda_tune_ME_T'], lambda_tune_T_ME=params['lambda_tune_T_ME']),
                            ME_M=dict(lambda_ME_M=lambda_ME_M, lambda_tune_ME_M=params['lambda_tune_ME_M'], lambda_tune_M_ME=params['lambda_tune_M_ME']), 
                            ME_E=dict(lambda_ME_E=lambda_ME_E, lambda_tune_ME_E=params['lambda_tune_ME_E'], lambda_tune_E_ME=params['lambda_tune_E_ME'])
                            )  

        model = Model_ME_T_v2(model_config)
        return model, model_config


    class Objective:
        '''Objective class for optimization'''
        def __init__(self,
                     lambda_tune_T_E_range=None, 
                     lambda_tune_E_T_range=None, 
                     lambda_tune_T_M_range=None,
                     lambda_tune_M_T_range=None,
                     lambda_tune_T_ME_range=None,
                     lambda_tune_ME_T_range=None,
                     lambda_tune_M_E_range=None,
                     lambda_tune_E_M_range=None,
                     lambda_tune_ME_E_range=None,
                     lambda_tune_E_ME_range=None,
                     lambda_tune_ME_M_range=None,
                     lambda_tune_M_ME_range=None, 
                     previous_ML_model_weights_to_load=None
                     ):
                     
            self.lambda_tune_T_E_range = lambda_tune_T_E_range
            self.lambda_tune_E_T_range = lambda_tune_E_T_range
            self.lambda_tune_T_M_range = lambda_tune_T_M_range
            self.lambda_tune_M_T_range = lambda_tune_M_T_range
            self.lambda_tune_M_E_range = lambda_tune_M_E_range
            self.lambda_tune_E_M_range = lambda_tune_E_M_range
            self.lambda_tune_T_ME_range = lambda_tune_T_ME_range
            self.lambda_tune_ME_T_range = lambda_tune_ME_T_range
            self.lambda_tune_ME_E_range = lambda_tune_ME_E_range
            self.lambda_tune_E_ME_range = lambda_tune_E_ME_range
            self.lambda_tune_ME_M_range = lambda_tune_ME_M_range
            self.lambda_tune_M_ME_range = lambda_tune_M_ME_range
            self.previous_ML_model_weights_to_load = previous_ML_model_weights_to_load


        def __call__(self, trial):
            params = {'lambda_tune_T_E': trial.suggest_float('lambda_tune_T_E', self.lambda_tune_T_E_range[0], self.lambda_tune_T_E_range[1]),
                      'lambda_tune_E_T': trial.suggest_float('lambda_tune_E_T', self.lambda_tune_E_T_range[0], self.lambda_tune_E_T_range[1]),
                      'lambda_tune_T_M': trial.suggest_float('lambda_tune_T_M', self.lambda_tune_T_M_range[0], self.lambda_tune_T_M_range[1]),
                      'lambda_tune_M_T': trial.suggest_float('lambda_tune_M_T', self.lambda_tune_M_T_range[0], self.lambda_tune_M_T_range[1]),
                      'lambda_tune_E_M': trial.suggest_float('lambda_tune_E_M', self.lambda_tune_E_M_range[0], self.lambda_tune_E_M_range[1]),
                      'lambda_tune_M_E': trial.suggest_float('lambda_tune_M_E', self.lambda_tune_M_E_range[0], self.lambda_tune_M_E_range[1]),
                      'lambda_tune_T_ME': trial.suggest_float('lambda_tune_T_ME', self.lambda_tune_T_ME_range[0], self.lambda_tune_T_ME_range[1]),
                      'lambda_tune_ME_T': trial.suggest_float('lambda_tune_ME_T', self.lambda_tune_ME_T_range[0], self.lambda_tune_ME_T_range[1]),
                      'lambda_tune_ME_M': trial.suggest_float('lambda_tune_ME_M', self.lambda_tune_ME_M_range[0], self.lambda_tune_ME_M_range[1]),
                      'lambda_tune_M_ME': trial.suggest_float('lambda_tune_M_ME', self.lambda_tune_M_ME_range[0], self.lambda_tune_M_ME_range[1]),
                      'lambda_tune_ME_E': trial.suggest_float('lambda_tune_ME_E', self.lambda_tune_ME_E_range[0], self.lambda_tune_ME_E_range[1]),
                      'lambda_tune_E_ME': trial.suggest_float('lambda_tune_E_ME', self.lambda_tune_E_ME_range[0], self.lambda_tune_E_ME_range[1])}
           
            model, model_config = build_model(params)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            if self.previous_ML_model_weights_to_load is not None:
                loaded_model = torch.load(self.previous_ML_model_weights_to_load, map_location='cpu')
                model.load_state_dict(loaded_model['state_dict'])
                optimizer.load_state_dict(loaded_model['optimizer'])
                print("loaded previous best model weights and optimizer")
                print(self.previous_ML_model_weights_to_load)

            accuracy = train_and_evaluate(model_config, model, optimizer, trial)

            return accuracy

    def train_and_evaluate(model_config, model, optimizer, trial):
        '''Train and evaluation function, this will be called at each trial and epochs will start from zero'''

        model.to(device)
        optimizer_to(optimizer,device)

        # Training -----------
        for epoch in range(n_epochs):
            print(epoch)
            model.train()
            for step, batch in enumerate(iter(train_dataloader)):
                optimizer.zero_grad()
                # forward pass -----------
                loss_dict, _, _ = model(batch)
                loss = Criterion(model_config, loss_dict)

                loss.backward()
                optimizer.step()

                if step == 0:
                    train_loss = init_losses(loss_dict)
            
                # track loss over batches -----------
                for k, v in loss_dict.items():
                    train_loss[k] += v

            # Average losses over batches -----------
            for k, v in train_loss.items():
                train_loss[k] = train_loss[k] / len(train_dataloader)

            # Validation -----------
            with torch.no_grad():
                for val_batch in iter(val_dataloader):
                    model.eval()
                    val_loss, _, _ = model(val_batch)
                

                if ((epoch>0) and ((epoch+1) % int(n_epochs/2)) == 0):
                    print("time to check if the model can be pruned away at epoch:", epoch+1)
                    model_score = run_gmm(model, dataloader)
                    # model_score = 1
                    
                    # Prune if this trial is not good -----------
                    if trial is not None:
                        trial.report(model_score, epoch) 
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()   
    

        # save the model and the optuna study at the end of the trial -----------
        fname = dir_pth['result'] + f"model_trial_{trial.number}_epoch_{epoch+1}" 
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
            }
        save_ckp(checkpoint, dir_pth['result'], fname)

        fname = dir_pth['result'] + f"Results_trial_{trial.number}_epoch_{epoch+1}_opset_{opset}.pkl"
        save_results(model, dataloader, dat, fname)

        return model_score

    # Main code ###############################################################
    # Set the device -----------
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Train test split -----------
    dir_pth = set_paths(config_file, exp_name=exp_name, fold_n=fold_n)
    dat = MET_exc_inh_v2.from_file(dir_pth['MET_data'])
    train_ind, val_ind = dat.train_val_split(fold=fold_n, n_folds=10, seed=0)
    train_dat = dat[train_ind,:]
    val_dat = dat[val_ind,:]

    # Copy the running code into the result dir -----------
    shutil.copy(__file__, dir_pth['result'])

    # Dataset and Dataloader -----------
    train_dataset = MET_dataset_v2(train_dat, device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dataset = MET_dataset_v2(val_dat, device=device)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    dataset = MET_dataset_v2(dat, device=device)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    # Optimization -------------
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    # Load previous optimization study if exist
    storage = 'sqlite:///'+exp_name+'.db'
    
    # if opset==0:
    #     optuna.delete_study(study_name=exp_name, storage=storage)        
    
    study = optuna.create_study(study_name=exp_name,
                                direction="maximize", 
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.HyperbandPruner(),
                                storage=storage, 
                                load_if_exists=True)
    
    if len(study.trials) > 0:
        model_name_to_load = dir_pth['result'] + f"model_trial_{str(study.best_trial.number )}_epoch_{n_epochs}.pt"
    else:
        model_name_to_load = None

    study.optimize(Objective(lambda_tune_T_E_range = lambda_tune_T_E_range, 
                             lambda_tune_E_T_range = lambda_tune_E_T_range, 
                             lambda_tune_T_M_range = lambda_tune_T_M_range,
                             lambda_tune_M_T_range = lambda_tune_M_T_range,
                             lambda_tune_E_M_range = lambda_tune_E_M_range,
                             lambda_tune_M_E_range = lambda_tune_M_E_range, 
                             lambda_tune_T_ME_range = lambda_tune_T_ME_range,
                             lambda_tune_ME_T_range = lambda_tune_ME_T_range,
                             lambda_tune_ME_M_range = lambda_tune_ME_M_range,
                             lambda_tune_M_ME_range = lambda_tune_M_ME_range,
                             lambda_tune_ME_E_range = lambda_tune_ME_E_range,
                             lambda_tune_E_ME_range = lambda_tune_E_ME_range,
                             previous_ML_model_weights_to_load = model_name_to_load), n_trials=n_trials)

    fname = dir_pth['result'] + f"study_{exp_name}_{opset}opset.pkl" 

    savepkl(study, fname)


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))


    
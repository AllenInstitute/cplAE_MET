# From python
import os
import sys
import shutil
import logging
import argparse
import numpy as np
from collections import Counter

# From torch
import torch
from torch.utils.data import DataLoader

# From optuna
import optuna

# From CplAE_MET
from cplAE_MET.utils.dataset import MET_exc_inh
from cplAE_MET.utils.utils import set_paths, save_ckp
from cplAE_MET.models.torch_utils import MET_dataset
from cplAE_MET.models.model_classes import Model_ME_T

from cplAE_MET.models.train_utils import init_losses, save_results, optimizer_to, Criterion
from cplAE_MET.models.optuna_utils import run_classification

from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--config_file',           default='config.toml',  type=str,   help='config file with data paths')
parser.add_argument('--exp_name',              default='MET_50k_stratified_400nmfs_50met_narrower_shorter_M_w_noise',         type=str,   help='Experiment set')
parser.add_argument('--opt_storage_db',        default='MET_50k_stratified_400nmfs_50met_narrower_shorter_M_w_noise.db',      type=str,   help='Optuna study storage database')
parser.add_argument('--variational',           default=False,          type=bool,  help='running a variational autoencoder?')
parser.add_argument('--optimization',          default=True,           type=bool,  help='if False then the hyperparam are read from the input args')
parser.add_argument('--load_model',            default=False,          type=bool,  help='Load weights from an old ML model')
parser.add_argument('--db_load_if_exist',      default=True,           type=bool,  help='True(1) or False(0)')
parser.add_argument('--opset',                 default=0,              type=int,   help='round of operation with n_trials')
parser.add_argument('--opt_n_trials',          default=1,              type=int,   help='number trials for bayesian optimization')
parser.add_argument('--n_epochs',              default=2500,           type=int,   help='Number of epochs to train')
parser.add_argument('--fold_n',                default=0,              type=int,   help='kth fold in 10-fold CV splits')
parser.add_argument('--latent_dim',            default=3,              type=int,   help='Number of latent dims')
parser.add_argument('--batch_size',            default=1000,           type=int,   help='Batch size')
parser.add_argument('--KLD_beta',              default=1.0,            type=float, help='coefficient for KLD term if model is VAE')
parser.add_argument('--alpha_T',               default=1.0,            type=float, help='T reconstruction loss weight')
parser.add_argument('--alpha_E',               default=(-4,2),         type=float, help='E reconstruction loss weight')
parser.add_argument('--alpha_M',               default=(-2,2),         type=float, help='M reconstruction loss weight')
parser.add_argument('--alpha_ME',              default=(-2,2),         type=float, help='ME reconstruction loss weight')
parser.add_argument('--lambda_TE',             default=1.0,            type=float, help='coupling loss weight between T and E')
parser.add_argument('--lambda_TM',             default=1.0,            type=float, help='coupling loss weight between T and M')
parser.add_argument('--lambda_ME',             default=1.0,            type=float, help='coupling loss weight between M and E')
parser.add_argument('--lambda_ME_T',           default=1.0,            type=float, help='coupling loss weight between ME and T')
parser.add_argument('--lambda_ME_M',           default=1.0,            type=float, help='coupling loss weight between ME and M')
parser.add_argument('--lambda_ME_E',           default=1.0,            type=float, help='coupling loss weight between ME and E')
parser.add_argument('--lambda_tune_E_M_range', default=(-2,3),         type=float, help='Tune the directionality of coupling between E and M')
parser.add_argument('--lambda_tune_E_ME_range',default=(-6,-3),        type=float, help='Tune the directionality of coupling between E and ME')
parser.add_argument('--lambda_tune_E_T_range', default=(-6,-3),        type=float, help='Tune the directionality of coupling between E and T')
parser.add_argument('--lambda_tune_ME_E_range',default=(2,6),          type=float, help='Tune the directionality of coupling between ME and E')
parser.add_argument('--lambda_tune_ME_M_range',default=(2,6),          type=float, help='Tune the directionality of coupling between ME and M')
parser.add_argument('--lambda_tune_ME_T_range',default=(-6,-2),        type=float, help='Tune the directionality of coupling between ME and T')
parser.add_argument('--lambda_tune_M_E_range', default=(-6,-2),        type=float, help='Tune the directionality of coupling between M and E')
parser.add_argument('--lambda_tune_M_ME_range',default=(-6,-2),        type=float, help='Tune the directionality of coupling between M and ME')
parser.add_argument('--lambda_tune_M_T_range', default=(-4,0),         type=float, help='Tune the directionality of coupling between M and T')
parser.add_argument('--lambda_tune_T_E_range', default=(3,6),          type=float, help='Tune the directionality of coupling between T and E')
parser.add_argument('--lambda_tune_T_M_range', default=(0,5),          type=float, help='Tune the directionality of coupling between T and M')
parser.add_argument('--lambda_tune_T_ME_range',default=(-1,4),         type=float, help='Tune the directionality of coupling between T and ME')
# If optimization is off
# parser.add_argument('--alpha_E',               default=-2.518659435321765,            type=float, help='E reconstruction loss weight')
# parser.add_argument('--alpha_M',               default=-1.0868595731769475,            type=float, help='M reconstruction loss weight')
# parser.add_argument('--alpha_ME',              default=-0.2584645482925703,            type=float, help='ME reconstruction loss weight')
# parser.add_argument('--lambda_tune_E_M_range', default=2.2794728566646687,            type=float, help='Tune the directionality of coupling between E and M')
# parser.add_argument('--lambda_tune_E_ME_range',default=-3.3272541015401718,            type=float, help='Tune the directionality of coupling between E and ME')
# parser.add_argument('--lambda_tune_E_T_range', default=-5.964269814438085,            type=float, help='Tune the directionality of coupling between E and T')
# parser.add_argument('--lambda_tune_ME_E_range',default=5.79191530020224,            type=float, help='Tune the directionality of coupling between ME and E')
# parser.add_argument('--lambda_tune_ME_M_range',default=5.44841100847209,            type=float, help='Tune the directionality of coupling between ME and M')
# parser.add_argument('--lambda_tune_ME_T_range',default=-4.702109557828222,            type=float, help='Tune the directionality of coupling between ME and T')
# parser.add_argument('--lambda_tune_M_E_range', default=-4.906746754295075,            type=float, help='Tune the directionality of coupling between M and E')
# parser.add_argument('--lambda_tune_M_ME_range',default=-3.0108634557950458,            type=float, help='Tune the directionality of coupling between M and ME')
# parser.add_argument('--lambda_tune_M_T_range', default=-3.5914452978996207,            type=float, help='Tune the directionality of coupling between M and T')
# parser.add_argument('--lambda_tune_T_E_range', default=3.5909869712376783,            type=float, help='Tune the directionality of coupling between T and E')
# parser.add_argument('--lambda_tune_T_M_range', default=2.0566115256506334,            type=float, help='Tune the directionality of coupling between T and M')
# parser.add_argument('--lambda_tune_T_ME_range',default=2.8688827728713155,            type=float, help='Tune the directionality of coupling between T and ME')




def main(exp_name="TEST",
         variational=False,
         load_model=False,
         db_load_if_exist=True,
         optimization=True,
         opt_storage_db="test.db",
         config_file="config.toml", 
         n_epochs=10, 
         fold_n=0, 
         KLD_beta=1.0,
         alpha_T=1.0,
         alpha_M=(-2,2),
         alpha_E=(-2,2),
         alpha_ME=(-2,2),
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
         opt_n_trials=1,
         opset=0):


    
    def build_model(params):
        ''' Config and build the model'''

        model_config = dict(variational=variational,
                            latent_dim=latent_dim, 
                            batch_size=batch_size,
                            KLD_beta=KLD_beta,
                            T=dict(dropout_p=0.2, 
                                   alpha_T=alpha_T),
                            E=dict(gnoise_std=train_dataset.gnoise_e_std, 
                                   gnoise_std_frac=0.05, 
                                   dropout_p=0.2, 
                                   alpha_E=params['alpha_E']),
                            M=dict(gnoise_std=train_dataset.gnoise_m_std, 
                                   gnoise_std_frac=0.005, 
                                   dropout_p=0.2, 
                                   alpha_M=params['alpha_M']),
                            TE=dict(lambda_TE=lambda_TE,
                                    lambda_tune_T_E=params['lambda_tune_T_E'], 
                                    lambda_tune_E_T=params['lambda_tune_E_T']),
                            TM=dict(lambda_TM=lambda_TM, 
                                    lambda_tune_T_M=params['lambda_tune_T_M'], 
                                    lambda_tune_M_T=params['lambda_tune_M_T']),
                            ME=dict(alpha_ME=params['alpha_ME'], 
                                    lambda_ME=lambda_ME, 
                                    lambda_tune_M_E=params['lambda_tune_M_E'], 
                                    lambda_tune_E_M=params['lambda_tune_E_M']),
                            ME_T=dict(lambda_ME_T=lambda_ME_T, 
                                      lambda_tune_ME_T=params['lambda_tune_ME_T'], 
                                      lambda_tune_T_ME=params['lambda_tune_T_ME']),
                            ME_M=dict(lambda_ME_M=lambda_ME_M, 
                                      lambda_tune_ME_M=params['lambda_tune_ME_M'], 
                                      lambda_tune_M_ME=params['lambda_tune_M_ME']), 
                            ME_E=dict(lambda_ME_E=lambda_ME_E, 
                                      lambda_tune_ME_E=params['lambda_tune_ME_E'], 
                                      lambda_tune_E_ME=params['lambda_tune_E_ME'])
                            )  

        model = Model_ME_T(model_config)
        return model, model_config


    class Objective:
        '''Objective class for optimization'''
        def __init__(self,
                     alpha_E=None,
                     alpha_M=None,
                     alpha_ME=None,
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
            
            self.alpha_E = alpha_E
            self.alpha_M = alpha_M      
            self.alpha_ME = alpha_ME   
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
            self.best_model = None
            self._current_model = None
            self.best_optimizer = None
            self._current_optimizer = None

        def __call__(self, trial):
            if optimization:
                params = {'alpha_E': trial.suggest_float('alpha_E', self.alpha_E[0], self.alpha_E[1]),
                        'alpha_M': trial.suggest_float('alpha_M', self.alpha_M[0], self.alpha_M[1]),
                        'alpha_ME': trial.suggest_float('alpha_ME', self.alpha_ME[0], self.alpha_ME[1]),
                        'lambda_tune_T_E': trial.suggest_float('lambda_tune_T_E', self.lambda_tune_T_E_range[0], self.lambda_tune_T_E_range[1]),
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
            else:
                params = {'alpha_E': self.alpha_E,
                    'alpha_M': self.alpha_M,
                    'alpha_ME': self.alpha_ME,
                    'lambda_tune_T_E': self.lambda_tune_T_E_range,
                    'lambda_tune_E_T': self.lambda_tune_E_T_range,
                    'lambda_tune_T_M': self.lambda_tune_T_M_range,
                    'lambda_tune_M_T': self.lambda_tune_M_T_range,
                    'lambda_tune_E_M': self.lambda_tune_E_M_range,
                    'lambda_tune_M_E': self.lambda_tune_M_E_range,
                    'lambda_tune_T_ME': self.lambda_tune_T_ME_range,
                    'lambda_tune_ME_T': self.lambda_tune_ME_T_range,
                    'lambda_tune_ME_M': self.lambda_tune_ME_M_range,
                    'lambda_tune_M_ME':self.lambda_tune_M_ME_range,
                    'lambda_tune_ME_E': self.lambda_tune_ME_E_range,
                    'lambda_tune_E_ME': self.lambda_tune_E_ME_range} 
                
            for k,v in params.items(): 
                    params[k] = np.exp(v) 
            
            model, model_config = build_model(params)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            if self.previous_ML_model_weights_to_load is not None:
                loaded_model = torch.load(self.previous_ML_model_weights_to_load, map_location='cpu')
                model.load_state_dict(loaded_model['state_dict'])
                optimizer.load_state_dict(loaded_model['optimizer'])
                print("loaded previous best model weights and optimizer")
                print(self.previous_ML_model_weights_to_load)

            trained_model, score = train_and_evaluate(model_config, model, optimizer, trial)
            self._current_model = trained_model
            self._current_optimizer = optimizer

            return score

        def callback(self, study, trial):
            if study.best_trial == trial:
                self.best_model = self._current_model
                self.best_optimizer = self._current_optimizer



    def train_and_evaluate(model_config, model, optimizer, trial):
        '''Train and evaluation function, this will be called at each trial and epochs will start from zero'''

        model.to(device)
        optimizer_to(optimizer,device)

        # Training -----------
        for epoch in range(n_epochs):
            # print(epoch)
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
            
            if not optimization:
                if ((epoch % 1000) == 0):
                    fname = dir_pth['result'] + f"checkpoint_epoch_{epoch}.pkl"
                    save_results(model, dataloader, D, fname, train_ind, val_ind)

                # TODO
                # Logging -----------
                tb_writer.add_scalar('Train/MSE_XT', train_loss['rec_t'], epoch)
                tb_writer.add_scalar('Validation/MSE_XT', val_loss['rec_t'], epoch)
                tb_writer.add_scalar('Train/MSE_XM', train_loss['rec_m'], epoch)
                tb_writer.add_scalar('Validation/MSE_XM', val_loss['rec_m'], epoch)
                tb_writer.add_scalar('Train/MSE_XE', train_loss['rec_e'], epoch)
                tb_writer.add_scalar('Validation/MSE_XE', val_loss['rec_e'], epoch)
                tb_writer.add_scalar('Train/MSE_XME', train_loss['rec_m_me'] + train_loss['rec_e_me'], epoch)
                tb_writer.add_scalar('Validation/MSE_XME', val_loss['rec_m_me']+ val_loss['rec_e_me'], epoch)
                tb_writer.add_scalar('Train/cpl_T->E', train_loss['cpl_t->e'], epoch)
                tb_writer.add_scalar('Validation/cpl_T->E', val_loss['cpl_t->e'], epoch)
                tb_writer.add_scalar('Train/cpl_E->T', train_loss['cpl_e->t'], epoch)
                tb_writer.add_scalar('Validation/cpl_E->T', val_loss['cpl_e->t'], epoch)
                tb_writer.add_scalar('Train/cpl_T->M', train_loss['cpl_t->m'], epoch)
                tb_writer.add_scalar('Validation/cpl_T->M', val_loss['cpl_t->m'], epoch)
                tb_writer.add_scalar('Train/cpl_M->T', train_loss['cpl_m->t'], epoch)
                tb_writer.add_scalar('Validation/cpl_M->T', val_loss['cpl_m->t'], epoch)
                tb_writer.add_scalar('Train/cpl_M->E', train_loss['cpl_m->e'], epoch)
                tb_writer.add_scalar('Validation/cpl_M->E', val_loss['cpl_m->e'], epoch)
                tb_writer.add_scalar('Train/cpl_E->M', train_loss['cpl_e->m'], epoch)
                tb_writer.add_scalar('Validation/cpl_E->M', val_loss['cpl_e->m'], epoch)
                tb_writer.add_scalar('Train/cpl_ME->T', train_loss['cpl_me->t'], epoch)
                tb_writer.add_scalar('Validation/cpl_ME->T', val_loss['cpl_me->t'], epoch)
                tb_writer.add_scalar('Train/cpl_T->ME', train_loss['cpl_t->me'], epoch)
                tb_writer.add_scalar('Validation/cpl_T->ME', val_loss['cpl_t->me'], epoch)
                tb_writer.add_scalar('Train/cpl_ME->M', train_loss['cpl_me->m'], epoch)
                tb_writer.add_scalar('Validation/cpl_ME->M', val_loss['cpl_me->m'], epoch)
                tb_writer.add_scalar('Train/cpl_M->ME', train_loss['cpl_m->me'], epoch)
                tb_writer.add_scalar('Validation/cpl_M->ME', val_loss['cpl_me->m'], epoch)
                tb_writer.add_scalar('Train/cpl_ME->E', train_loss['cpl_me->e'], epoch)
                tb_writer.add_scalar('Validation/cpl_ME->E', val_loss['cpl_me->e'], epoch)
                tb_writer.add_scalar('Train/cpl_E->ME', train_loss['cpl_e->me'], epoch)
                tb_writer.add_scalar('Validation/cpl_E->ME', val_loss['cpl_e->me'], epoch)        

        model_score = run_classification(model, dataloader, train_ind, val_ind, T_labels_for_classification)
        return model, model_score

    # Main code ###############################################################
    # Set the device -----------
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # Train test split -----------
    dir_pth = set_paths(config_file, exp_name=exp_name, fold_n=fold_n, opt_storage_db=opt_storage_db, creat_tb_logs= not optimization)
    if not optimization:
        tb_writer = SummaryWriter(log_dir=dir_pth['tb_logs'])
    dat, D = MET_exc_inh.from_file(dir_pth['MET_data'])
    train_ind, val_ind = dat.train_val_split(fold=fold_n, n_folds=10, seed=0)
    train_dat = dat[train_ind,:]
    val_dat = dat[val_ind,:]
    train_dat = dat[train_ind,:]
    T_labels_for_classification = np.array(dat.merged_cluster_label_at50)


    # Copy the running code into the result dir -----------
    shutil.copy(__file__, dir_pth['result'])
 
    # Weighted sampling strategy -----------
    weights = train_dat.make_weights_for_balanced_classes(n_met = 54, met_subclass_id = [2, 3], batch_size=1000)                                                                
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights)) 
    
    # Dataset and Dataloader -----------
    train_dataset = MET_dataset(train_dat, device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler)


    val_dataset = MET_dataset(val_dat, device=device)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    dataset = MET_dataset(dat, device=device)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    # Optimization -------------
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
            
    study = optuna.create_study(study_name=exp_name,
                                direction="maximize", 
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.HyperbandPruner(),
                                storage='sqlite:///'+dir_pth['opt_storage_db'], 
                                load_if_exists=db_load_if_exist)
    if db_load_if_exist:
        if os.path.exists(dir_pth['opt_storage_db']):
            print("Loading the optimization history from:")
            print(dir_pth['opt_storage_db'])

    if load_model:
        assert len(study.trials) > 0, f"sqlite:///{dir_pth['opt_storage_db']}, does not exist"
        model_to_load = dir_pth['result'] + f"model_trial_{str(study.best_trial.number )}_epoch_{n_epochs}.pt"
        print("Going to load the best model from previous optimization study")
        print(model_to_load)
    else:
        model_to_load = None
        print("Starting the model from scratch as there is no model to load")

    objective = Objective(alpha_E = alpha_E,
                          alpha_M = alpha_M,
                          alpha_ME = alpha_ME,
                          lambda_tune_T_E_range = lambda_tune_T_E_range, 
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
                          previous_ML_model_weights_to_load = model_to_load)

    study.optimize(objective, n_trials=opt_n_trials, callbacks=[objective.callback])


    # save the best model checkpoint and the best results at the end of the study -----------
    # This is run only if the best model was happend in this run ----------------------------

    fname = dir_pth['result'] + f"Best_model_trial{study.best_trial.number}" 
    if objective.best_model is not None:
        checkpoint = {
            'state_dict': objective.best_model.state_dict(),
            'optimizer': objective.best_optimizer.state_dict()
            }
        save_ckp(checkpoint, dir_pth['result'], fname)

        fname = dir_pth['result'] + f"Results_trial_{study.best_trial.number}.pkl"

        save_results(objective.best_model, dataloader, D, fname, train_ind, val_ind)

    if not optimization:
        tb_writer.close()

if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))


    

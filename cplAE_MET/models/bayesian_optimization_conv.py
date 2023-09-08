# From python
import os
import sys
import shutil
import logging
import argparse
import numpy as np

# From torch
import torch
from torch.utils.data import DataLoader

# From optuna
import optuna

# From CplAE_MET
from cplAE_MET.utils.dataset import MET_exc_inh
from cplAE_MET.utils.utils import set_paths, save_ckp
from cplAE_MET.models.torch_utils import MET_dataset
from cplAE_MET.models.model_classes import Model_ME_T_conv
from cplAE_MET.models.train_utils import init_losses, save_results, optimizer_to, Criterion
from cplAE_MET.models.optuna_utils import run_classification, LastPlacePruner

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--config_file',           default='config.toml',type=str,   help='config file with data paths')
parser.add_argument('--exp_name',              default='test',       type=str,   help='experiment name') # a folder with this name will be generated in the results folder
parser.add_argument('--opt_storage_db',        default='test.db',    type=str,   help='name of the Optuna study storage database') # this history file will be generated in the exp folder
parser.add_argument('--variational',           default=False,        type=bool,  help='running a variational autoencoder?') # variational AE is implemented but never tested
parser.add_argument('--optimization',          default=True,         type=bool,  help='running an optimization study?')
parser.add_argument('--load_model',            default=False,        type=bool,  help='load weights from an old ML model')
parser.add_argument('--load_best_params',      default=False,        type=bool,  help='load hyperparams of the best run from history file')
parser.add_argument('--use_defined_params',    default=False,        type=bool,  help='use hyperparams that are defined by user')
parser.add_argument('--db_load_if_exist',      default=True,         type=bool,  help='load an optimization database file if exist')
parser.add_argument('--n_epochs',              default=2500,         type=int,   help='number of epochs to train')
parser.add_argument('--warmup_steps',          default=500,          type=int,   help='during optimization, after this many epochs, prunning process starts')
parser.add_argument('--warmup_trials',         default=10,           type=int,   help='during optimization, prunning starts only if this many trials already passed')
parser.add_argument('--pruning_steps',         default=100,          type=int,   help='during optimization, after warmup_steps, every this amount of epochs a prunning will be tried')
parser.add_argument('--prunning_stop',         default=2000,         type=int,   help='during optimization, no prunning will be attempted after this amount of epochs')
parser.add_argument('--fold_n',                default=0,            type=int,   help='which kth fold is running now in 10-fold CV train and test split')
parser.add_argument('--latent_dim',            default=5,            type=int,   help='number of latent dims')
parser.add_argument('--batch_size',            default=1000,         type=int,   help='batch size')
parser.add_argument('--KLD_beta',              default=1.0,          type=float, help='coefficient for KLD term if model is VAE') # variational AE is implemented but never tested
parser.add_argument('--alpha_T',               default=1.0,          type=float, help='T reconstruction loss weight coefficient')
parser.add_argument('--alpha_E',               default=(0, 6),       type=float, help='E reconstruction loss weight coefficient range') # range is used during the optimization
parser.add_argument('--alpha_M',               default=(-3, 3),      type=float, help='M reconstruction loss weight coefficient range')
parser.add_argument('--alpha_ME',              default=(-4, 2),      type=float, help='ME reconstruction loss weight coefficient range')
parser.add_argument('--lambda_tune_E_T_range', default=(-5, 0),      type=float, help='E_T coupling loss weight coefficient range')
parser.add_argument('--lambda_tune_ME_E_range',default=(0, 5),       type=float, help='ME_E coupling loss weight coefficient range')
parser.add_argument('--lambda_tune_ME_M_range',default=(2, 5),       type=float, help='ME_M coupling loss weight coefficient range')
parser.add_argument('--lambda_tune_ME_T_range',default=(-5, -1),     type=float, help='ME_T coupling loss weight coefficient range')
parser.add_argument('--lambda_tune_M_T_range', default=(-5, -2),     type=float, help='M_T coupling loss weight coefficient range')
parser.add_argument('--lambda_tune_T_E_range', default=(3, 6),       type=float, help='T_E coupling loss weight coefficient range')
parser.add_argument('--lambda_tune_T_M_range', default=(-1, 5),      type=float, help='T_M coupling loss weight coefficient range')
parser.add_argument('--lambda_tune_T_ME_range',default=(-1, 4),      type=float, help='T_ME coupling loss weight coefficient range')
parser.add_argument('--lambda_TE',             default=1.0,          type=float, help='is there a coupling between T and E')
parser.add_argument('--lambda_TM',             default=1.0,          type=float, help='is there a coupling between T and M')
parser.add_argument('--lambda_ME_T',           default=1.0,          type=float, help='is there a coupling between T and ME')
parser.add_argument('--lambda_ME_M',           default=1.0,          type=float, help='is there a coupling between ME and M')
parser.add_argument('--lambda_ME_E',           default=1.0,          type=float, help='is there a coupling between ME and E')
parser.add_argument('--arb_dens_shape',        default='120x4x4',    type=str,   help='if using arbor densities directly, please specify the shape here')
# If optimization is off, then lines (alpha_E) to line (lambda_tune_T_ME_range) should be commented out and the following lines 
# should be used instead. the exponential values of whatever you provide below will be used to run a non-optimization model

# parser.add_argument('--alpha_E',               default=3.74,         type=float, help='E reconstruction loss weight coefficient')
# parser.add_argument('--alpha_M',               default=-1.27,        type=float, help='M reconstruction loss weight coefficient')
# parser.add_argument('--alpha_ME',              default=0.99,         type=float, help='ME reconstruction loss weight coefficient')
# parser.add_argument('--lambda_tune_E_T_range', default=-0.47,        type=float, help='E_T coupling loss weight coefficient')
# parser.add_argument('--lambda_tune_ME_E_range',default=5.56,         type=float, help='ME_E coupling loss weight coefficient')
# parser.add_argument('--lambda_tune_ME_M_range',default=5.73,         type=float, help='ME_M coupling loss weight coefficient')
# parser.add_argument('--lambda_tune_ME_T_range',default=-4.72,        type=float, help='ME_T coupling loss weight coefficient')
# parser.add_argument('--lambda_tune_M_T_range', default=-0.63,        type=float, help='M_T coupling loss weight coefficient')
# parser.add_argument('--lambda_tune_T_E_range', default=3.092,        type=float, help='T_E coupling loss weight coefficient')
# parser.add_argument('--lambda_tune_T_M_range', default=3.17,         type=float, help='T_M coupling loss weight coefficient')
# parser.add_argument('--lambda_tune_T_ME_range',default=5.88,         type=float, help='T_ME coupling loss weight coefficient')
                    
def main(exp_name="TEST",
         variational=False,
         load_model=False,
         load_best_params=False,
         use_defined_params=False,
         db_load_if_exist=True,
         optimization=True,
         opt_storage_db="test.db",
         config_file="config.toml", 
         n_epochs=10,
         warmup_steps=500, 
         warmup_trials=10,
         pruning_steps=100,
         prunning_stop=2000,
         fold_n=0, 
         KLD_beta=1.0,
         alpha_T=1.0,
         alpha_M=(-2,2),
         alpha_E=(-2,2),
         alpha_ME=(-2,2),
         lambda_TE=0.0,
         lambda_TM=0.0,
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
         lambda_tune_ME_M_range=(0,2),
         latent_dim=2,
         batch_size=1000,
         arb_dens_shape='120x4x4'):
    
    # Read the model config and set the hyper-params in place
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
                                   gnoise_std_frac=0.05,  
                                   dropout_p=0.2, 
                                   alpha_M=params['alpha_M']),
                            TE=dict(lambda_TE=lambda_TE,
                                    lambda_tune_T_E=params['lambda_tune_T_E'], 
                                    lambda_tune_E_T=params['lambda_tune_E_T']),
                            TM=dict(lambda_TM=lambda_TM, 
                                    lambda_tune_T_M=params['lambda_tune_T_M'], 
                                    lambda_tune_M_T=params['lambda_tune_M_T']),
                            ME=dict(alpha_ME=params['alpha_ME']), 
                            ME_T=dict(lambda_ME_T=lambda_ME_T, 
                                      lambda_tune_ME_T=params['lambda_tune_ME_T'], 
                                      lambda_tune_T_ME=params['lambda_tune_T_ME']),
                            ME_M=dict(lambda_ME_M=lambda_ME_M, 
                                      lambda_tune_ME_M=params['lambda_tune_ME_M']), 
                            ME_E=dict(lambda_ME_E=lambda_ME_E, 
                                      lambda_tune_ME_E=params['lambda_tune_ME_E']) 
                            )  
        model = Model_ME_T_conv(model_config)
        return model, model_config

    # Objective function for the bayesian optimization process
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
                     lambda_tune_ME_E_range=None,
                     lambda_tune_ME_M_range=None,
                     previous_ML_model_weights_to_load=None
                     ):
            
            self.alpha_E = alpha_E
            self.alpha_M = alpha_M      
            self.alpha_ME = alpha_ME   
            self.lambda_tune_T_E_range = lambda_tune_T_E_range
            self.lambda_tune_E_T_range = lambda_tune_E_T_range
            self.lambda_tune_T_M_range = lambda_tune_T_M_range
            self.lambda_tune_M_T_range = lambda_tune_M_T_range
            self.lambda_tune_T_ME_range = lambda_tune_T_ME_range
            self.lambda_tune_ME_T_range = lambda_tune_ME_T_range
            self.lambda_tune_ME_E_range = lambda_tune_ME_E_range
            self.lambda_tune_ME_M_range = lambda_tune_ME_M_range
            self.previous_ML_model_weights_to_load = previous_ML_model_weights_to_load
            self.best_model = None
            self.acceptable_model = None
            self._current_model = None
            self.best_optimizer = None
            self.acceptable_optimizer = None
            self._current_optimizer = None
            self._current_value = None

        def __call__(self, trial):
            # If it is running an optimization process
            if optimization:
                print("hyperparams are chosen in a bayesian optimization manner")
                params = {'alpha_E': trial.suggest_float('alpha_E', self.alpha_E[0], self.alpha_E[1]),
                        'alpha_M': trial.suggest_float('alpha_M', self.alpha_M[0], self.alpha_M[1]),
                        'alpha_ME': trial.suggest_float('alpha_ME', self.alpha_ME[0], self.alpha_ME[1]),
                        'lambda_tune_T_E': trial.suggest_float('lambda_tune_T_E', self.lambda_tune_T_E_range[0], self.lambda_tune_T_E_range[1]),
                        'lambda_tune_E_T': trial.suggest_float('lambda_tune_E_T', self.lambda_tune_E_T_range[0], self.lambda_tune_E_T_range[1]),
                        'lambda_tune_T_M': trial.suggest_float('lambda_tune_T_M', self.lambda_tune_T_M_range[0], self.lambda_tune_T_M_range[1]),
                        'lambda_tune_M_T': trial.suggest_float('lambda_tune_M_T', self.lambda_tune_M_T_range[0], self.lambda_tune_M_T_range[1]),
                        'lambda_tune_T_ME': trial.suggest_float('lambda_tune_T_ME', self.lambda_tune_T_ME_range[0], self.lambda_tune_T_ME_range[1]),
                        'lambda_tune_ME_T': trial.suggest_float('lambda_tune_ME_T', self.lambda_tune_ME_T_range[0], self.lambda_tune_ME_T_range[1]),
                        'lambda_tune_ME_M': trial.suggest_float('lambda_tune_ME_M', self.lambda_tune_ME_M_range[0], self.lambda_tune_ME_M_range[1]),
                        'lambda_tune_ME_E': trial.suggest_float('lambda_tune_ME_E', self.lambda_tune_ME_E_range[0], self.lambda_tune_ME_E_range[1])}
            # If it is not an optimization and user defines the hyper-params instead of providing a range for optuna to pick the hyper-params
            elif use_defined_params:
                print("User defined hyperparam set. This is not an optimization anymore!")
                params = {'alpha_E': self.alpha_E,
                          'alpha_M': self.alpha_M,
                          'alpha_ME': self.alpha_ME,
                          'lambda_tune_T_E': self.lambda_tune_T_E_range,
                          'lambda_tune_E_T': self.lambda_tune_E_T_range,
                          'lambda_tune_T_M': self.lambda_tune_T_M_range,
                          'lambda_tune_M_T': self.lambda_tune_M_T_range,
                          'lambda_tune_T_ME': self.lambda_tune_T_ME_range,
                          'lambda_tune_ME_T': self.lambda_tune_ME_T_range,
                          'lambda_tune_ME_M': self.lambda_tune_ME_M_range,
                          'lambda_tune_ME_E': self.lambda_tune_ME_E_range}
            # If it is not an optimization and the machine is going to load the history file and take the best hyper-params that was
            # found and saved by optuna. Just the hyper-params are read here not the model weights
            elif load_best_params:
                print(f"the best model (Trial:{str(study.best_trial.number )}) hyperparams are used. This is not an optimization anymore!")
                params = study.best_params
                
            else:
                raise ValueError("If it is not an optimization, then hyperparams must be provided")
            
            # The exponential values of the hyper-params are used. This was done for a faster exploration of the hyper-parameter space
            for k,v in params.items(): 
                params[k] = np.exp(v)

            print("Hyper-params used in this model:", params) 
             
            # Take the set of hyper-params and build the ML model configuration  
            model, model_config = build_model(params)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # If loading a previous model weights, then it is done here
            if self.previous_ML_model_weights_to_load is not None:
                loaded_model = torch.load(self.previous_ML_model_weights_to_load, map_location='cpu')
                model.load_state_dict(loaded_model['state_dict'])
                optimizer.load_state_dict(loaded_model['optimizer'])
                print("loaded previous best model weights and optimizer")
                print(self.previous_ML_model_weights_to_load)

            # Train the model and run the evaluation and return the score for optuna 
            trained_model, score = train_and_evaluate(model_config, model, optimizer, trial)
            self._current_model = trained_model
            self._current_optimizer = optimizer
            self._current_value = score

            return score

        # This function checks to see if based on the score the current model is the "best model" or not
        # if it is the best model, we want to save its weights before finishing the run. If it is an
        # average model, then we dont save it. However, to have more results, if the current model is
        # close to the best model, then we call it acceptable model and we save that too. This is just 
        # to check the convergence of the optimization! 
        def callback(self, study, trial):
            if self._current_value is not None:
                if self._current_value >= study.best_value:
                    self.best_model = self._current_model
                    self.best_optimizer = self._current_optimizer
                elif abs(self._current_value - study.best_value) <= 5:
                    self.acceptable_model = self._current_model
                    self.acceptable_optimizer = self._current_optimizer



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
            
            # If it is not optimization run, we save the results every now and then
            if  not optimization:
                if ((epoch % 1000) == 0):
                    fname = dir_pth['result'] + f"trial_{trial.number}_checkpoint_epoch_{epoch}.pkl"
                    save_results(model, dataloader, D, fname, train_ind, val_ind)

                # If it is an optimization run, we dont log all the losses and so on. We only save 
                # everything if it is a specific non-optimization run that we want to monitor things closely.
                # Logging -----------
                tb_writer.add_scalar('Train/MSE_XT', train_loss['rec_t'], epoch)
                tb_writer.add_scalar('Validation/MSE_XT', val_loss['rec_t'], epoch)
                tb_writer.add_scalar('Train/MSE_XM', train_loss['rec_m'], epoch)
                tb_writer.add_scalar('Validation/MSE_XM', val_loss['rec_m'], epoch)
                tb_writer.add_scalar('Train/MSE_XE', train_loss['rec_e'], epoch)
                tb_writer.add_scalar('Validation/MSE_XE', val_loss['rec_e'], epoch)
                tb_writer.add_scalar('Train/MSE_M_XME', train_loss['rec_m_me'], epoch)
                tb_writer.add_scalar('Validation/MSE_M_XME', val_loss['rec_m_me'], epoch)
                tb_writer.add_scalar('Train/MSE_E_XME', train_loss['rec_e_me'], epoch)
                tb_writer.add_scalar('Validation/MSE_E_XME', val_loss['rec_e_me'], epoch)
                tb_writer.add_scalar('Train/cpl_T->E', train_loss['cpl_t->e'], epoch)
                tb_writer.add_scalar('Validation/cpl_T->E', val_loss['cpl_t->e'], epoch)
                tb_writer.add_scalar('Train/cpl_E->T', train_loss['cpl_e->t'], epoch)
                tb_writer.add_scalar('Validation/cpl_E->T', val_loss['cpl_e->t'], epoch)
                tb_writer.add_scalar('Train/cpl_T->M', train_loss['cpl_t->m'], epoch)
                tb_writer.add_scalar('Validation/cpl_T->M', val_loss['cpl_t->m'], epoch)
                tb_writer.add_scalar('Train/cpl_M->T', train_loss['cpl_m->t'], epoch)
                tb_writer.add_scalar('Validation/cpl_M->T', val_loss['cpl_m->t'], epoch)
                tb_writer.add_scalar('Train/cpl_ME->T', train_loss['cpl_me->t'], epoch)
                tb_writer.add_scalar('Validation/cpl_ME->T', val_loss['cpl_me->t'], epoch)
                tb_writer.add_scalar('Train/cpl_T->ME', train_loss['cpl_t->me'], epoch)
                tb_writer.add_scalar('Validation/cpl_T->ME', val_loss['cpl_t->me'], epoch)
                tb_writer.add_scalar('Train/cpl_ME->M', train_loss['cpl_me->m'], epoch)
                tb_writer.add_scalar('Validation/cpl_ME->M', val_loss['cpl_me->m'], epoch)
                tb_writer.add_scalar('Train/cpl_ME->E', train_loss['cpl_me->e'], epoch)
                tb_writer.add_scalar('Validation/cpl_ME->E', val_loss['cpl_me->e'], epoch)    


            if trial.number > warmup_trials and epoch >= warmup_steps and (epoch-warmup_steps) % pruning_steps ==0 and epoch < prunning_stop:
                 intermediate_value = run_classification(model, dataloader, train_ind, val_ind, T_labels_for_classification)
                 trial.report(intermediate_value, epoch)
                 if trial.should_prune():
                     raise optuna.TrialPruned()
        
        model_score = run_classification(model, dataloader, train_ind, val_ind, T_labels_for_classification)
        return model, model_score
    
    # Main code ###############################################################
    # Set the device -----------
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Train test split -----------
    dir_pth = set_paths(config_file, exp_name=exp_name, fold_n=fold_n, opt_storage_db=opt_storage_db, creat_tb_logs= not optimization)

    dat, D = MET_exc_inh.from_file(dir_pth['MET_data'])
    
    # We add one dimension here because later in subnetwork_M, we will be using 3d convolutions
    # This is just to make the image shapes appropriate for conv_3d. Otherwise we could use conv2d
    if arb_dens_shape == '120x4x4':
        dat.XM = np.expand_dims(dat.XM, axis=1)
    elif arb_dens_shape == '120x1x4':
        dat.XM = np.expand_dims(dat.XM, axis=(1,3))
   
    assert len(dat.XM.shape) == 5, "Check the arbor densities shape and the argument input for arb_dens_shape"
    
    dat.Xsd = np.expand_dims(dat.Xsd, axis=1)

    train_ind, val_ind = dat.train_val_split(fold=fold_n, n_folds=10, seed=0)
    train_dat = dat[train_ind,:]
    val_dat = dat[val_ind,:]
    train_dat = dat[train_ind,:]
    T_labels_for_classification = np.array(dat.merged_cluster_label_at50)

    # Copy the running code into the result dir -----------
    shutil.copy(__file__, dir_pth['result'])
    shutil.copy(dir_pth['config_file'], dir_pth['result']) 
 
    # We define weights for each sample in a way that in each batch there are at least 54 met cells from exc and
    # 54 met cells from inh data. 54 was decided based on the previous runs just by observation!
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Optimization -------------
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
            
    study = optuna.create_study(study_name = exp_name,
                                direction = "maximize", 
                                sampler = optuna.samplers.TPESampler(),
                                # pruner=optuna.pruners.HyperbandPruner(),
                                pruner = LastPlacePruner(warmup_steps=warmup_steps, warmup_trials=warmup_trials),
                                storage = 'sqlite:///'+dir_pth['opt_storage_db'], 
                                load_if_exists = db_load_if_exist)
    if db_load_if_exist:
        if os.path.exists(dir_pth['opt_storage_db']):
            print("Loading the optimization history from:")
            print(dir_pth['opt_storage_db'])

    if load_model:
        assert len(study.trials) > 0, f"sqlite:///{dir_pth['opt_storage_db']}, does not exist"
        model_to_load = dir_pth['result'] + f"Best_model_trial{str(study.best_trial.number )}.pt"
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
                          lambda_tune_T_ME_range = lambda_tune_T_ME_range,
                          lambda_tune_ME_T_range = lambda_tune_ME_T_range,
                          lambda_tune_ME_M_range = lambda_tune_ME_M_range,
                          lambda_tune_ME_E_range = lambda_tune_ME_E_range,
                          previous_ML_model_weights_to_load = model_to_load)

    trial_number = str(len(study.trials) + 1)
    if not optimization:
        log_dir = dir_pth['tb_logs'] + "trial" + trial_number + "/"
        tb_writer = SummaryWriter(log_dir=log_dir)
        

    study.optimize(objective, n_trials=1, callbacks=[objective.callback])


    # save the best acceptable checkpoint and results at the end of the study 
    # This is run only if the model is acceptable in this run
    # We accept the models that their score is closer than 5 percernt of the best value so far
    fname = dir_pth['result'] + f"acceptable_model_trial{trial_number}" 

    if objective.acceptable_model is not None:
        checkpoint = {
            'state_dict': objective.acceptable_model.state_dict(),
            'optimizer': objective.acceptable_optimizer.state_dict()
            }
        save_ckp(checkpoint, dir_pth['result'], fname)
        fname = dir_pth['result'] + f"Results_trial_{trial_number}.pkl"
        save_results(objective.acceptable_model, dataloader, D, fname, train_ind, val_ind)
        

    # save the best model checkpoint and the best results at the end of the study
    # This is run only if the best model was happend in this run
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
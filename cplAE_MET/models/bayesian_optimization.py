# From python
import os
import sys
import shutil
import logging
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

# From torch
import torch
from torch.utils.data import DataLoader

# From optuna
import optuna

# From CplAE_MET
from cplAE_MET.utils.utils import savepkl
from cplAE_MET.models.torch_utils import tonumpy
from cplAE_MET.utils.dataset import MET_exc_inh
from cplAE_MET.utils.load_config import load_config
from cplAE_MET.models.torch_utils import MET_dataset
from cplAE_MET.models.model_classes import Model_ME_T
from cplAE_MET.models.train_tempcsfeatures_dev import set_paths, init_losses
from cplAE_MET.models.classification_functions import run_LDA

# For community detection
import networkx as nx
from cdlib import algorithms
from sklearn.neighbors import kneighbors_graph

from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--config_file',           default='config_10k.toml',  type=str,   help='config file with data paths')
parser.add_argument('--exp_name',              default='test',         type=str,   help='Experiment set')
parser.add_argument('--opt_storage_db',        default='test.db',      type=str,   help='Optuna study storage database')
parser.add_argument('--variational',           default=False,          type=bool,  help='running a variational autoencoder?')
parser.add_argument('--optimization',          default=True,           type=bool,  help='if False then the hyperparam are read from the input args')
parser.add_argument('--load_model',            default=False,          type=bool,  help='Load weights from an old ML model')
parser.add_argument('--db_load_if_exist',      default=True,           type=bool,  help='True(1) or False(0)')
parser.add_argument('--opset',                 default=0,              type=int,   help='round of operation with n_trials')
parser.add_argument('--opt_n_trials',          default=1,              type=int,   help='number trials for bayesian optimization')
parser.add_argument('--n_epochs',              default=10000,           type=int,   help='Number of epochs to train')
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
# parser.add_argument('--alpha_M',               default=1.0,            type=float, help='M reconstruction loss weight')
# parser.add_argument('--alpha_E',               default=1.0,            type=float, help='E reconstruction loss weight')
# parser.add_argument('--alpha_ME',              default=1.0,            type=float, help='ME reconstruction loss weight')
# parser.add_argument('--lambda_tune_T_E_range', default=1.0,            type=float, help='Tune the directionality of coupling between T and E')
# parser.add_argument('--lambda_tune_E_T_range', default=0.0,            type=float, help='Tune the directionality of coupling between E and T')
# parser.add_argument('--lambda_tune_T_M_range', default=1.0,            type=float, help='Tune the directionality of coupling between T and M')
# parser.add_argument('--lambda_tune_M_T_range', default=0.0,            type=float, help='Tune the directionality of coupling between M and T')
# parser.add_argument('--lambda_tune_ME_M_range',default=1.0,            type=float, help='Tune the directionality of coupling between ME and M')
# parser.add_argument('--lambda_tune_M_ME_range',default=0.0,            type=float, help='Tune the directionality of coupling between M and ME')
# parser.add_argument('--lambda_tune_ME_E_range',default=1.0,            type=float, help='Tune the directionality of coupling between ME and E')
# parser.add_argument('--lambda_tune_E_ME_range',default=0.0,            type=float, help='Tune the directionality of coupling between E and ME')
# parser.add_argument('--lambda_tune_E_M_range', default=1.0,            type=float, help='Tune the directionality of coupling between E and M')
# parser.add_argument('--lambda_tune_M_E_range', default=0.0,            type=float, help='Tune the directionality of coupling between M and E')
# parser.add_argument('--lambda_tune_T_ME_range',default=0.9,            type=float, help='Tune the directionality of coupling between T and ME')
# parser.add_argument('--lambda_tune_ME_T_range',default=0.1,            type=float, help='Tune the directionality of coupling between ME and T')


def set_paths(config_file=None, exp_name='DEBUG', opt_storage_db="TEST", fold_n=0, optimization=True):
    ''' Set the input and output and the optimization database path.
    Args:
        config_file: Name of the config.toml file which points to the input data
        exp_name: Name of the folder in which all model outputs
        opt_storage_db: Name of the database file to store the optuna study
        fold_n: which fold is running
        optimization: if True then an optimization is running, otherwise the model hyper-param are given
    '''
    paths = load_config(config_file=config_file, verbose=False)
    paths['result'] = f'{str(paths["package_dir"] / "data/results")}/{exp_name}/'
    paths['opt_storage_db'] = f'{str(paths["package_dir"] / "data/results")}/{exp_name}/{opt_storage_db}'
    Path(paths['result']).mkdir(parents=False, exist_ok=True)
    # only if optimization is running, then we save the log file in tensorboard format
    if not optimization:
        paths['tb_logs'] = f'{str(paths["package_dir"] / "data/results")}/tb_logs/{exp_name}/fold_{str(fold_n)}/'
        if os.path.exists(paths['tb_logs']):
            shutil.rmtree(paths['tb_logs'])
        Path(paths['tb_logs']).mkdir(parents=True, exist_ok=False)
    return paths


def save_ckp(state, checkpoint_dir, fname):
    '''
    Save model checkpint.
    Args:
        state: model stat dict
        checkpoint_dir: path to the folder to save the checkpoint
        fname: name of the checkpoint file to output
    '''
    filename = fname + '.pt'
    f_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, f_path)


def init_losses(loss_dict):
    '''
    Initialize the loss dict to zero.
    Args:
        loss_dict: loss dictionary 
    '''
    t_loss = {}
    for k in loss_dict.keys():
        t_loss[k] = 0.
    return t_loss


def rm_emp_end_str(myarray):
    '''
    Remove the empty space at the end of the strings in an array.
    '''
    return np.array([mystr.rstrip() for mystr in myarray])

def calculate_arbor_densities_from_nmfs(rec_nmf, input_datamat):
    '''Reconstruct the arbor densities from the recnstructed nmfs
    Args:
        rec_nmf: reconstructed nmfs 
        input_datamat: input mat file which is loaded
    '''
    min_lim = 0
    rec_channel = {}
    for channel in ['ax', 'de', 'api', 'bas']:
        comp_name = "M_nmf_components_" + channel
        total_var_name = "M_nmf_total_vars_" + channel
        col_limit = (min_lim , min_lim + input_datamat[comp_name].shape[0])
        rec_channel[channel] = (np.dot(rec_nmf[:, col_limit[0]:col_limit[1]] * input_datamat[total_var_name], input_datamat[comp_name])).reshape(-1, 120, 4)
        min_lim = col_limit[1]
    return np.stack((rec_channel['ax'], rec_channel['de'], rec_channel['api'] , rec_channel['bas'] ), axis=3) 


def save_results(model, dataloader, input_datamat, fname, train_ind, val_ind):
    '''
    Takes the model, run it in the evaluation mode to calculate the embeddings and reconstructions for printing out.
    Args:
        model: the model 
        dataloader: the dataloader
        input_datamat: the input mat file that is loaded
        fname: name of the output pkl file that the results going to be saved in
        train_ind: the indices of the tarin cells
        val_ind: the indices of the validation cells
    '''
    model.eval()

    for all_data in iter(dataloader):
        with torch.no_grad():
            _, z_dict, xr_dict = model(all_data)
    
    rec_arbor_density = calculate_arbor_densities_from_nmfs(rec_nmf = tonumpy(xr_dict['xrm']), input_datamat=input_datamat)
    
        
    savedict = {'XT': tonumpy(all_data['xt']),
                'XM': tonumpy(all_data['xm']),
                'XE': tonumpy(all_data['xe']),
                'XrT': tonumpy(xr_dict['xrt']),
                'XrE': tonumpy(xr_dict['xre']),
                'XrM': tonumpy(xr_dict['xrm']),
                'XrM_me_paired': tonumpy(xr_dict['xrm_me_paired']),
                'XrE_me_paired': tonumpy(xr_dict['xre_me_paired']),
                'rec_arbor_density': rec_arbor_density,
                'zm': tonumpy(z_dict['zm']),
                'ze': tonumpy(z_dict['ze']),
                'zt': tonumpy(z_dict['zt']),
                'zme_paired': tonumpy(z_dict['zme_paired']),
                'is_t_1d':tonumpy(all_data['is_t_1d']),
                'is_e_1d':tonumpy(all_data['is_e_1d']),
                'is_m_1d':tonumpy(all_data['is_m_1d']), 
                'cluster_id': input_datamat['cluster_id'],
                'gene_ids': input_datamat['gene_ids'],
                'e_features': input_datamat['E_features'],
                'specimen_id': rm_emp_end_str(input_datamat['specimen_id']),
                'cluster_label': rm_emp_end_str(input_datamat['cluster_label']),
                'merged_cluster_label_at40': rm_emp_end_str(input_datamat['merged_cluster_label_at40']),
                'merged_cluster_label_at50': rm_emp_end_str(input_datamat['merged_cluster_label_at50']),
                'merged_cluster_label_at60': rm_emp_end_str(input_datamat['merged_cluster_label_at60']),
                'cluster_color': rm_emp_end_str(input_datamat['cluster_color']),
                'platform': input_datamat['platform'],
                'class':  input_datamat['class'],
                'class_id': input_datamat['class_id'],
                'group': input_datamat['group'],
                'subgroup': input_datamat['subgroup'],
                'hist_ax_de_api_bas' : input_datamat['hist_ax_de_api_bas'],
                'M_nmf_total_vars_ax': input_datamat['M_nmf_total_vars_ax'],
                'M_nmf_total_vars_de': input_datamat['M_nmf_total_vars_de'],
                'M_nmf_total_vars_api': input_datamat['M_nmf_total_vars_api'],
                'M_nmf_total_vars_bas': input_datamat['M_nmf_total_vars_bas'],
                'M_nmf_components_ax': input_datamat['M_nmf_components_ax'],
                'M_nmf_components_de': input_datamat['M_nmf_components_de'],
                'M_nmf_components_api': input_datamat['M_nmf_components_api'],
                'M_nmf_components_bas': input_datamat['M_nmf_components_bas'],
                'train_ind': train_ind,
                'val_ind': val_ind}

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
                model_config['M']['alpha_M'] * loss_dict['BCELoss_m'] + \
                model_config['ME']['alpha_ME'] * loss_dict['BCELoss_me_m'] + \
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

    if model_config['variational']:
        criterion = criterion + \
                    model_config['KLD_beta'] * loss_dict['KLD_t'] + \
                    model_config['KLD_beta'] * loss_dict['KLD_e'] + \
                    model_config['KLD_beta'] * loss_dict['KLD_m'] + \
                    model_config['KLD_beta'] * loss_dict['KLD_me_paired']      
    return criterion



def Leiden_community_detection(data):
    '''Take the data points, creat a graph with 12 nn and apply leiden community detection
    Args:
        data: the embedding coordinates
    '''
    # Create adj matrix with 12 nn
    A = kneighbors_graph(data, 12, mode='distance', include_self=True)
    # Create a network_x graph
    G = nx.convert_matrix.from_numpy_array(A)
    # Run Leiden community detection algorithm
    comm = algorithms.leiden(G)
    return comm


def run_Leiden_community_detection(model, dataloader):
    '''
    Takes the model and dataloader to run the model in the evaluation mode and compute
    the number of leiden communities on the embedding
    '''
    model.eval()
    for all_data in iter(dataloader):
        _, z_dict, _ = model(all_data) 

    is_t_1d = tonumpy(all_data['is_t_1d'])
    is_e_1d = tonumpy(all_data['is_e_1d'])
    is_m_1d = tonumpy(all_data['is_m_1d'])
    is_me_1d = np.logical_and(is_m_1d, is_e_1d)
    is_met_1d = np.logical_and(is_t_1d, is_me_1d)

    zt = tonumpy(z_dict['zt'])
    zme_paired = tonumpy(z_dict['zme_paired'])
    
    n_t_types = []
    n_me_types = []
    # Instead of running it only once, we run it 10 times and then take the max
    for i in range(10):
        ncomm = len(Leiden_community_detection(zt[is_t_1d]).communities)
        n_t_types.append(ncomm)
        ncomm = len(Leiden_community_detection(zme_paired[is_met_1d]).communities)
        n_me_types.append(ncomm)

    n_t_types = np.median(n_t_types)
    n_me_types = np.median(n_me_types)
    model_score = np.min([n_t_types , n_me_types])
    return model_score


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

    # Classification function #################################################
    def run_classification(model, dataloader):
        '''
        Take the model and the dataloader, to run the model in the evaluation mode, obtain the emd 
        and run the classisification task on the t cells'''

        model.eval()
        for all_data in iter(dataloader):
            _, z_dict, _ = model(all_data) 

        is_t_1d = tonumpy(all_data['is_t_1d'])
        T_labels = np.array(dat.merged_cluster_label_at50)

        zt = tonumpy(z_dict['zt'])
        ze = tonumpy(z_dict['ze'])
        zm = tonumpy(z_dict['zm'])
        zme_paired = tonumpy(z_dict['zme_paired'])
        
        _, _, clf = run_LDA(zt[is_t_1d], 
                            T_labels[is_t_1d],
                            train_test_ids={'train': train_ind, 
                                            'val': val_ind})
        
        te_cpl_score = clf.score(ze[val_ind], T_labels[val_ind]) * 100
        tm_cpl_score = clf.score(zm[val_ind], T_labels[val_ind]) * 100
        met_cpl_score = clf.score(zme_paired[val_ind], T_labels[val_ind]) * 100
        print("te, tm and met classification acc:", te_cpl_score, tm_cpl_score, met_cpl_score)
        return np.min([te_cpl_score, tm_cpl_score, met_cpl_score])
    
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
                
                for k,v in params.items(): 
                    params[k] = np.exp(v) 
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
            
            # if ((epoch + 1) % 100 == 0):
            #     intermediate_value = run_classification(model, dataloader)
            #     trial.report(intermediate_value, epoch + 1)
            #     if trial.should_prune():
            #         raise optuna.TrialPruned()
  
            # if ((epoch % 500) == 0):
            #     fname = dir_pth['result'] + f"checkpoint_epoch_{epoch}.pkl"
            #     save_results(model, dataloader, D, fname, train_ind, val_ind)

        model_score = run_classification(model, dataloader)
        return model, model_score

    # Main code ###############################################################
    # Set the device -----------
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Train test split -----------
    dir_pth = set_paths(config_file, exp_name=exp_name, fold_n=fold_n, opt_storage_db=opt_storage_db)
    dat, D = MET_exc_inh.from_file(dir_pth['MET_data'])
    train_ind, val_ind = dat.train_val_split(fold=fold_n, n_folds=10, seed=0)
    train_dat = dat[train_ind,:]
    val_dat = dat[val_ind,:]
    train_dat = dat[train_ind,:]
    train_classes = train_dat.group
    train_subclasses = train_dat.subgroup

    # Copy the running code into the result dir -----------
    shutil.copy(__file__, dir_pth['result'])

    def make_weights_for_balanced_classes(train_classes, train_subclasses, n_met = 54, met_subclass_id = [2, 3]):

        met_prob = n_met/(batch_size/2)
        others_prob = 1 - met_prob

        dic = {}
        for i, j in zip(train_classes, train_subclasses):
            dic[j] = i

        class_counts = Counter(train_classes)
        subclass_counts = Counter(train_subclasses)
        N = float(sum(class_counts.values()))
        weight_per_class = {}  
        for k, v in class_counts.items():    
            weight_per_class[k] = N/float(v)

        weight_per_subclass = {}   
        for k, v in subclass_counts.items():
            c = dic[k]
            if k in met_subclass_id:
                weight_per_subclass[k] = (weight_per_class[c] * met_prob/v) * class_counts[c]
            else: 
                weight_per_subclass[k] = (weight_per_class[c] * others_prob/v) * class_counts[c]

        weights = [0] * len(train_subclasses)
        for idx, val in enumerate(train_subclasses):                                          
            weights[idx] = weight_per_subclass[val]

        return weights

    
    weights = make_weights_for_balanced_classes(train_classes, train_subclasses)                                                                
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


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))


    

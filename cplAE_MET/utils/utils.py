import csv
import os
import shutil
import torch
import numpy as np
import pickle
from pathlib import Path
from cplAE_MET.utils.load_config import load_config



def write_list_to_csv(path, file):
    with open(path, 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(file)
    print("Done writing!")


def read_list_from_csv(path):
    with open(path, 'r') as myfile:
        reader = csv.reader(myfile)
        data = list(reader)
    return data[0]


def savepkl(ob, fname):
    with open(fname, 'wb') as f:
        pickle.dump(ob, f)
    return


def loadpkl(fname):
    with open(fname, 'rb') as f:
        X = pickle.load(f)
        return X
    
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

def set_preproc_paths(config_file=None):
    '''Set up all the preproc files path, this function is used in the notebooks'''

    paths = load_config(config_file=config_file, verbose=False)
    paths['input'] = f'{str(paths["data_dir"])}'
    paths['t_anno_feather'] = f'{paths["input"]}/{"anno.feather"}'
    paths['t_data_feather'] = f'{paths["input"]}/{"data.feather"}'

    paths['specimen_ids'] = f'{paths["input"]}/{str(paths["specimen_ids_file"])}'
    paths['gene_file'] = f'{paths["input"]}/{str(paths["gene_file"])}'
    paths['t_data'] = f'{paths["input"]}/{str(paths["t_data_output_file"])}'
    paths['t_anno'] = f'{paths["input"]}/{str(paths["t_anno_output_file"])}'
    paths['gene_id'] = f'{paths["input"]}/{str(paths["gene_id_output_file"])}'

    paths['m_data'] = f'{paths["input"]}/{str(paths["m_data_folder"])}'
    paths['m_anno'] = f'{paths["m_data_folder"]}/{str(paths["m_anno"])}'
    paths['hist2d_120x4'] = f'{paths["m_data_folder"]}/{str(paths["hist2d_120x4_folder"])}'
    paths['ivscc_inh_m_features'] = f'{paths["input"]}/{str(paths["ivscc_inh_m_features"])}'
    paths['ivscc_exc_m_features'] = f'{paths["input"]}/{str(paths["ivscc_exc_m_features"])}'
    paths['fmost_exc_m_features'] = f'{paths["input"]}/{str(paths["fmost_exc_m_features"])}'
    paths['arbor_density_PC_vars_file'] = f'{paths["input"]}/{str(paths["arbor_density_PC_vars_file"])}'
    paths['m_data'] = f'{paths["input"]}/{str(paths["m_output_file"])}'


    paths['E_timeseries'] = f'{paths["input"]}/{str(paths["E_timeseries_file"])}'
    paths['ipfx_features'] = f'{paths["input"]}/{str(paths["ipfx_features_file"])}'
    paths['e_data'] = f'{paths["input"]}/{str(paths["e_output_file"])}'

    paths['met_data'] = f'{paths["input"]}/{str(paths["met_output_file"])}'
    return paths



def set_paths(config_file=None, exp_name='DEBUG', opt_storage_db="TEST", fold_n=0, creat_tb_logs=False):
    ''' Set the input and output and the optimization database path.
    Args:
        config_file: Name of the config.toml file which points to the input data
        exp_name: Name of the folder in which all model outputs
        opt_storage_db: Name of the database file to store the optuna study
        fold_n: which fold is running
        optimization: if True then an optimization is running, otherwise the model hyper-param are given
    '''
    paths, config_file_path = load_config(config_file=config_file, verbose=False)
    paths['result'] = f'{str(paths["package_dir"] / "data/results")}/{exp_name}/'
    paths['opt_storage_db'] = f'{str(paths["package_dir"] / "data/results")}/{exp_name}/{opt_storage_db}'
    Path(paths['result']).mkdir(parents=False, exist_ok=True)
    # only if optimization is running, then we save the log file in tensorboard format
    if creat_tb_logs:
        paths['tb_logs'] = f'{str(paths["package_dir"] / "data/results")}/tb_logs/{exp_name}/fold_{str(fold_n)}/'
        if os.path.exists(paths['tb_logs']):
            shutil.rmtree(paths['tb_logs'])
        Path(paths['tb_logs']).mkdir(parents=True, exist_ok=False)
    paths['config_file'] = config_file_path
    return paths


def get_all_1d_mask(dat):
    '''takes the MET_exc_inh object that has isT_1d, isM_1d and isE_1d masks and
    returns all different masks from those three'''

    mask = {}
    mask['is_t_1d'] = dat['is_t_1d']
    mask['is_e_1d'] = dat['is_e_1d']
    mask['is_m_1d'] = dat['is_m_1d']
    mask['is_me_1d'] = np.logical_and(mask['is_m_1d'], mask['is_e_1d'])
    mask['is_te_1d'] = np.logical_and(mask['is_t_1d'], mask['is_e_1d'])
    mask['is_tm_1d'] = np.logical_and(mask['is_t_1d'], mask['is_m_1d'])
    mask['is_met_1d'] = np.logical_and(mask['is_m_1d'], mask['is_te_1d'])

    mask['is_t_only_1d'] = np.logical_and(mask['is_t_1d'], np.logical_and(~mask['is_e_1d'], ~mask['is_m_1d']))
    mask['is_e_only_1d'] = np.logical_and(mask['is_e_1d'], np.logical_and(~mask['is_t_1d'], ~mask['is_m_1d']))
    mask['is_m_only_1d'] = np.logical_and(mask['is_m_1d'], np.logical_and(~mask['is_t_1d'], ~mask['is_e_1d']))
    mask['is_te_only_1d'] = np.logical_and(mask['is_te_1d'], ~mask['is_m_1d'])
    mask['is_tm_only_1d'] = np.logical_and(mask['is_tm_1d'], ~mask['is_e_1d'])
    mask['is_me_only_1d'] = np.logical_and(mask['is_me_1d'], ~mask['is_t_1d'])
    
    dat['platform'] = np.array([i.rstrip() for i in dat['platform']])
    mask['is_patchseq_1d'] = dat['platform']=="patchseq"
    mask['is_fMOST_1d'] = dat['platform']=="fMOST"
    mask['is_ME_1d'] = dat['platform']=="ME"
    mask['is_EM_1d'] = dat['platform']=="EM"

    return mask


def rm_emp_end_str(myarray):
    '''
    Remove the empty space at the end of the strings in an array.
    '''
    return np.array([mystr.rstrip() for mystr in myarray])
    
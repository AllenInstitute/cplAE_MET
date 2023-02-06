import csv
import os
import shutil
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


def set_input_paths(config_file=None, exp_name='DEBUG', opt_storage_db="TEST", fold_n=0, creat_tb_logs=False):
    '''The path to the input data which is read by the model as well as the path to the
    output and tensor board are set here'''

    paths = load_config(config_file=config_file, verbose=False)
    paths['result'] = f'{str(paths["package_dir"] / "data/results")}/{exp_name}/'
    paths['opt_storage_db'] = f'{str(paths["package_dir"] / "data/results")}/{exp_name}/{opt_storage_db}'
    Path(paths['result']).mkdir(parents=False, exist_ok=True)
    paths['tb_logs'] = f'{str(paths["package_dir"] / "data/results")}/tb_logs/{exp_name}/fold_{str(fold_n)}/'
    if creat_tb_logs:
        if os.path.exists(paths['tb_logs']):
            shutil.rmtree(paths['tb_logs'])
        Path(paths['tb_logs']).mkdir(parents=True, exist_ok=False)
    return paths

def get_all_1d_mask(dat):
    '''takes the MET_exc_inh object that has isT_1d, isM_1d and isE_1d masks and
    returns all different masks from those three'''

    mask = {}
    mask['is_t_1d'] = dat.isT_1d
    mask['is_e_1d'] = dat.isE_1d
    mask['is_m_1d'] = dat.isM_1d
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

    return mask

    
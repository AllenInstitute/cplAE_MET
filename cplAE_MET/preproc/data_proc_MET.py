#########################################################
########### Preprocessing M, E and T data ###############
#########################################################
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
from functools import reduce

from cplAE_MET.utils.load_config import load_config


parser = argparse.ArgumentParser()
parser.add_argument('--config_file',    default='config_preproc.toml', type=str,   help='config file with data paths')
parser.add_argument('--pca_th',     default=0.97,  type=float,   help='threshold for pca component')


def set_paths(config_file=None):
 paths = load_config(config_file=config_file, verbose=False)

 paths['input'] = f'{str(paths["package_dir"] / "data/proc/")}'
 paths['output'] = f'{paths["input"]}/{str(paths["m_output_file"])}'

 paths['specimen_ids'] = f'{paths["input"]}/{str(paths["specimen_ids_file"])}'

 paths['m_output'] = f'{paths["input"]}/{str(paths["m_output_file"])}'
 paths['t_output'] = f'{paths["input"]}/{str(paths["t_data_output_file"])}'
 paths['e_output'] = f'{paths["input"]}/{str(paths["e_output_file"])}'
 paths['met_output'] = f'{paths["input"]}/{str(paths["met_output_file"])}'
 paths['gene_id_output'] = f'{paths["input"]}/{str(paths["gene_id_output_file"])}'

 return paths


def main(config_file='config_preproc.toml', pca_th=0.97):

    dir_pth = set_paths(config_file=config_file)

    print("...................................................")
    print("Loading E, T and M data")
    E_data = pd.read_csv(dir_pth['e_output'])
    print("shape of E data:", E_data.shape)
    T_data = pd.read_csv(dir_pth['t_output'])
    print("shape of T data:", T_data.shape)
    M_dat = sio.loadmat(dir_pth['m_output'])
    print("shape of hist_ax_de_api_bas data:", M_dat['hist_ax_de_api_bas'].shape)
    gene_id = pd.read_csv(dir_pth['gene_id_output'])


    print("...................................................")
    print("read specimen ids from m data and align other with that")
    m_anno = pd.DataFrame({"specimen_id": M_dat['specimen_id'][0],
                           "cluster_id": M_dat['cluster_id'][0],
                           "cluster_label": M_dat['cluster_label'],
                           "cluster_color": M_dat['cluster_color']})

    print("...................................................")
    print("Combining M, E and T data")
    ME = reduce(lambda left, right: pd.merge(left, right, on=['specimen_id'], how='left'), [m_anno, E_data])

    MET = reduce(lambda left, right: pd.merge(left, right, on=['specimen_id'], how='left'), [ME, T_data])

    print("...................................................")
    print("Writing the output mat")

    #writing M, E and T data
    model_input_mat = {}
    model_input_mat["E_dat"] = np.array(MET[[c for c in E_data.columns if c != "specimen_id"]])
    model_input_mat["T_dat"] = np.array(MET[[c for c in T_data.columns if c != "specimen_id"]])
    model_input_mat["M_dat"] = np.array(M_dat["hist_ax_de_api_bas"])
    model_input_mat["soma_depth"] = M_dat["soma_depth"][0]
    model_input_mat["gene_ids"] = gene_id["gene_id"].to_list()


    #writing the sample_ids and the masks and some meta data
    model_input_mat["specimen_id"] = MET['specimen_id'].to_list()
    model_input_mat["cluster_id"] = MET.cluster_id.to_list()
    model_input_mat["cluster_color"] = MET.cluster_color.to_list()
    model_input_mat["cluster_label"] = MET.cluster_label.to_list()

    #Saving input mat
    print("Size of M data:", model_input_mat['M_dat'].shape)
    print("Size of M data:", model_input_mat["soma_depth"].shape)
    print("Size of E data:", model_input_mat["E_dat"].shape)
    print("Size of T data:", model_input_mat["T_dat"].shape)
    print("saving!")

    sio.savemat(dir_pth['met_output'], model_input_mat)

    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
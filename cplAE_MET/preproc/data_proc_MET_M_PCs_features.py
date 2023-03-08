#########################################################
########### Preprocessing M, E and T data ###############
#########################################################
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
from functools import reduce

from cplAE_MET.utils.load_config import load_config
from cplAE_MET.utils.analysis_tree_helpers import get_merged_types


parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default='config_preproc.toml', type=str,   help='config file with data paths')


def set_paths(config_file=None):
 paths = load_config(config_file=config_file, verbose=False)

 paths['input'] = f'{str(paths["data_dir"])}'
 paths['arbor_density_file'] = f'{str(paths["arbor_density_file"])}'
#  paths['arbor_density_PC_file'] = f'{paths["input"]}/{str(paths["arbor_density_PC_file"])}'
 paths['arbor_density_PC_vars_file'] = f'{paths["input"]}/{str(paths["arbor_density_PC_vars_file"])}'
 paths['m_output_file'] = f'{paths["input"]}/{str(paths["m_output_file"])}'

 paths['specimen_ids'] = f'{paths["input"]}/{str(paths["specimen_ids_file"])}'

 paths['anno'] = f'{paths["input"]}/{str(paths["t_anno_output_file"])}'
 paths['m_input'] = f'{paths["input"]}/{str(paths["arbor_density_file"])}'
 paths['t_input'] = f'{paths["input"]}/{str(paths["t_data_output_file"])}'
 paths['e_input'] = f'{paths["input"]}/{str(paths["e_output_file"])}'
 paths['met_output'] = f'{paths["input"]}/{str(paths["met_output_file"])}'
 paths['gene_id_input'] = f'{paths["input"]}/{str(paths["gene_id_output_file"])}'

 return paths


def main(config_file='config_preproc.toml'):

    dir_pth = set_paths(config_file=config_file)

    print("...................................................")
    print("Loading specimens")
    cells = pd.read_csv(dir_pth['specimen_ids'])
    print("Loading E, T and M data")
    E_data = pd.read_csv(dir_pth['e_input'])
    print("shape of E data:", E_data.shape)
    T_data = pd.read_csv(dir_pth['t_input'])
    print("shape of T data:", T_data.shape)
    M_data = pd.read_csv(dir_pth['m_output_file'])
    print("shape of m features data:", M_data.shape)
    M_pc_vars = pd.read_csv(dir_pth['arbor_density_PC_vars_file'])
    gene_id = pd.read_csv(dir_pth['gene_id_input'])
    print("Loading T annotations")
    T_ann = pd.read_csv(dir_pth['anno'])


    print("...................................................")
    print("read specimen ids from m data and align other with that")
    M_data['specimen_id'] = [str(i) for i in M_data['specimen_id']]
    m_anno = pd.DataFrame({"specimen_id": np.array([mystr.rstrip() for mystr in M_data['specimen_id']])})
    M_data['specimen_id'] = [mystr.rstrip() for mystr in M_data['specimen_id']]

    m_anno['specimen_id'] = m_anno['specimen_id'].astype(str)
    E_data['specimen_id'] = E_data['specimen_id'].astype(str)
    T_data['specimen_id'] = T_data['specimen_id'].astype(str)
    cells['specimen_id'] = cells['specimen_id'].astype(str)
    T_ann['specimen_id'] = T_ann['specimen_id'].astype(str)
    #dropping the space at the end of the name of the cluster labels
    is_t_1d = np.all(~np.isnan(np.array(T_data.drop(columns=["specimen_id"]))), axis=1)
    T_ann.loc[is_t_1d, 'Tree_first_cl_label'] = np.array([i.rstrip() for i in T_ann[is_t_1d]['Tree_first_cl_label'].to_list()])


    print("...................................................")
    print("Combining M, E and T data and metadata")
    result = reduce(lambda left, right: pd.merge(left, right, on=['specimen_id'], how='left'), [m_anno, M_data])
    result = reduce(lambda left, right: pd.merge(left, right, on=['specimen_id'], how='left'), [result, E_data])
    result = reduce(lambda left, right: pd.merge(left, right, on=['specimen_id'], how='left'), [result, T_data])
    result = reduce(lambda left, right: pd.merge(left, right, on=['specimen_id'], how='left'), [result, cells])
    result = reduce(lambda left, right: pd.merge(left, right, on=['specimen_id'], how='left'), [result, T_ann])

    print("...................................................")
    print("Merging t types and writing the merged types in the mat file")
    # First we need to find the cells that have T data available and then start merging them
    merged_t_at40, _, _  = get_merged_types(htree_file="/home/fahimehb/Local/new_codes/cplAE_MET/tree_20180520.csv",  
                                            cells_labels=np.array(result['Tree_first_cl_label'][is_t_1d]), 
                                            num_classes=40,
                                            ref_leaf=np.unique(result['Tree_first_cl_label'][is_t_1d]),
                                            node="n1")
    
    merged_t_at50, _, _  = get_merged_types(htree_file="/home/fahimehb/Local/new_codes/cplAE_MET/tree_20180520.csv",  
                                            cells_labels=np.array(result['Tree_first_cl_label'][is_t_1d]), 
                                            num_classes=50,
                                            ref_leaf=np.unique(result['Tree_first_cl_label'][is_t_1d]),
                                            node="n1")
    result.loc[is_t_1d, 'merged_types_40'] = merged_t_at40
    result.loc[is_t_1d, 'merged_types_50'] = merged_t_at50
    print("...................................................")
    print("Writing the output mat")

    #writing M, E and T data
    model_input_mat = {}
    model_input_mat["E_dat"] = np.array(result[[c for c in E_data.columns if c != "specimen_id"]])
    model_input_mat["T_dat"] = np.array(result[[c for c in T_data.columns if c != "specimen_id"]])
    model_input_mat["M_dat"] = np.array(result[[c for c in M_data.columns if c not in ["specimen_id"]]])

    #writing the sample_ids and some meta data
    model_input_mat["specimen_id"] = result.specimen_id.to_list()
    model_input_mat["cluster_id"] = result.Tree_first_cl_id.to_list()
    model_input_mat["cluster_color"] = result.Tree_first_cl_color.to_list()
    model_input_mat["cluster_label"] = result.Tree_first_cl_label.to_list()
    model_input_mat["merged_cluster_label_at40"] = result.merged_types_40.to_list()
    model_input_mat["merged_cluster_label_at50"] = result.merged_types_50.to_list()


    #Writing the E_feature and M_features names
    model_input_mat["gene_ids"] = gene_id["gene_id"].to_list()
    model_input_mat['M_features'] = [c for c in M_data.columns if c not in ["specimen_id"]]
    model_input_mat['E_features'] = [c for c in E_data.columns if c not in ["specimen_id"]]

    #Writing the M_feature total variance that was used for scaling 
    model_input_mat['M_features_total_var'] = M_pc_vars

    #Saving input mat
    print("Size of M data:", model_input_mat['M_dat'].shape)
    print("Size of E data:", model_input_mat["E_dat"].shape)
    print("Size of T data:", model_input_mat["T_dat"].shape)
    print("saving!")

    sio.savemat(dir_pth['met_output'], model_input_mat)

    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
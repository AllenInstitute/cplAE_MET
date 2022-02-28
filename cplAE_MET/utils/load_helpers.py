import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict
import numpy as np
import scipy.io as sio
import toml
import cplAE_MET.utils.utils as ut
from cplAE_MET.utils.tree_helpers import HTree, get_merged_ordered_classes, simplify_tree


def get_paths(inputmat_fileid, warn=True, write_toml=False):
    """Get paths for all data used in the analysis. 
    
    Args: 
    warn (bool): Warns when files or directories are missing. 
    write_toml (bool): Writes out a .toml file in the notebooks folder with paths as strings.

    Returns:
    path : a dictionary with many different paths.  
    """
    path = {}
    path['package'] = Path(__file__).parent.parent.parent.absolute()
    

    # Input data
    path['proc_dataset'] = path['package'] / "data/proc/" / inputmat_fileid
    # path['proc_dataset'] = path['package'] / "data/proc/inh_model_input_mat.mat"
    path['proc_E_names'] = path['package'] / "data/proc/E_names.json"
    path['htree'] = path['package'] / "data/proc/dend_RData_Tree_20181220.csv"

    path['proc_DE_gene_dict'] = path['package'] / "data/proc/DE_gene_dict.mat"
    path['anno_33_class_ref_tax'] =path['package'] / "data/proc/anno_33_class_ref_tax.mat"
    path['anno_33_class_consensus'] =path['package'] / "data/proc/anno_33_class_consensus"
    #path['htree_pruned'] = path['package'] / "data/proc/dend_RData_Tree_20181220_pruned.csv"
    for f in ['proc_dataset', 'proc_E_names', 'htree']:
        if (not(path[f].is_file()) and warn):
            print(f'File not found: {path[f]}')

    # Paths to data from different experiments
    # remote_path = Path("/home/rohan/Remote-AI/dat/result")
    remote_path = Path("/Users/fruity/Dropbox/AllenInstitute/CellTypes/dat/proc")
    path['exp_hparam'] = remote_path / "TE_NM/"
    path['exp_hparam_log'] = remote_path / "TE_NM/logs/"
    path['exp_kfold'] = remote_path / "TE_NM/"
    path['exp_repeat_init'] = remote_path / "TE_NM_cc/"
    path['exp_repeat_init_gmm'] = remote_path / "TE_NM_cc/gmm_model_select_cv_0/"

    # Paths to updated dataset (revised version, Gouwens et al. 2021)
    v2_path = Path("/Users/fruity/Dropbox/AllenInstitute/CellTypes/dat/raw/patchseq-inh/")
    path['v2_spc_names'] = v2_path / "spca_components_used_mMET_revision_Apr2020.json"
    path['v2_spc_loadings'] = v2_path / "spca_loadings_mMET_revision_Apr2020_v2.pkl"
    
    for f in ['exp_hparam', 'exp_hparam_log', 'exp_kfold', 'exp_repeat_init', 'exp_repeat_init_gmm']:
        if (not(path[f].is_dir()) and warn):
            print(f'Directory not found: {path[f]}')

    if write_toml:
        with open(path['package'] / "notebooks/config_paths.toml",'w') as f:
            str_path = path.copy()
            for key in str_path.keys():str_path[key] = str(str_path[key])
            toml.dump(str_path,f)
    return path


def load_dataset(inputmat_fileid, subtree_node, min_sample_thr=10):
    """Load input transcriptomic and electrophysiological profiles and label annotations as a dictionary.

    Args:
        min_sample_thr (int): Defaults to 10.

    Returns:
        data(Dict)
    """
    path = get_paths(inputmat_fileid, warn=False,write_toml=False)
    data = sio.loadmat(path['proc_dataset'], squeeze_me=True)
    del_keys = [key for key in data.keys() if '__' in key]
    for key in del_keys:
        data.pop(key)

    with open(path['proc_E_names']) as f:
        ephys_names = json.load(f)
    data['E_pcipfx'] = np.concatenate([data['E_pc_scaled'], data['E_feature']], axis=1)
    data['pcipfx_names'] = np.concatenate([data['pc_name'],data['feature_name']])
    temp = [ephys_names[f] for f in data['pcipfx_names']]
    data['pcipfx_names'] = np.array(temp)
    
    #Get t-types in order as per reference taxonomy
    n_required_classes = np.unique(data['cluster']).size
    _, t_types = get_merged_ordered_classes(data_labels=data['cluster'].copy(),
                                            subtree_node=subtree_node,
                                            htree_file=path['htree'],
                                            n_required_classes=n_required_classes)
    data['unique_sorted_t_types']=np.array(t_types)

    #well-sampled t-types and helpers:
    t_types_well_sampled = []
    for t in t_types:
        if np.sum(data['cluster']==t)>min_sample_thr:
            t_types_well_sampled.append(t)
    data['well_sampled_sorted_t_types'] = np.array(t_types_well_sampled)
    data['well_sampled_bool'] = np.isin(data['cluster'],data['well_sampled_sorted_t_types'])
    data['well_sampled_ind'] = np.flatnonzero(data['well_sampled_bool'])

    #Process E data and mask, standardize names.
    data['XT'] = data['T_dat']
    data['XE'] = np.concatenate([data['E_pc_scaled'],data['E_feature']],axis = 1)
    data['maskE'] = np.ones_like(data['XE'])
    data['maskE'][np.isnan(data['XE'])]=0.0
    data['XE'][np.isnan(data['XE'])]=0.0
    return data


def load_dataset_v2(inputmat_fileid, subtree_node, min_sample_thr=10):
    """Load input transcriptomic and electrophysiological profiles and label annotations as a dictionary.

    Args:
        min_sample_thr (int): Defaults to 10.

    Returns:
        data(Dict)
    """
    path = get_paths(inputmat_fileid, warn=False, write_toml=False)
    data = sio.loadmat(path['proc_dataset'], squeeze_me=True)
    del_keys = [key for key in data.keys() if '__' in key]
    for key in del_keys:
        data.pop(key)

    # Get t-types in order as per reference taxonomy
    n_required_classes = np.unique(data['cluster']).size
    _, t_types = get_merged_ordered_classes(data_labels=data['cluster'].copy(),
                                            subtree_node=subtree_node,
                                            htree_file=path['htree'],
                                            n_required_classes=n_required_classes)
    data['unique_sorted_t_types'] = np.array(t_types)

    # well-sampled t-types and helpers:
    t_types_well_sampled = []
    for t in t_types:
        if np.sum(data['cluster'] == t) > min_sample_thr:
            t_types_well_sampled.append(t)
    data['well_sampled_sorted_t_types'] = np.array(t_types_well_sampled)
    data['well_sampled_bool'] = np.isin(data['cluster'], data['well_sampled_sorted_t_types'])
    data['well_sampled_ind'] = np.flatnonzero(data['well_sampled_bool'])

    # Process E data and mask, standardize names.
    data['XT'] = data['T_dat']
    data['XE'] = data['E_dat']
    data['maskE'] = np.ones_like(data['XE'])
    data['maskE'][np.isnan(data['XE'])] = 0.0
    data['XE'][np.isnan(data['XE'])] = 0.0
    return data


def load_htree_well_sampled(inputmat_fileid, subtree_node, min_sample_thr=10, simplify=True):
    """Loads heirarchical taxonomy for subset of well-sampled inhibitory cell types (leaf nodes).

    Args:
        min_sample_thr (int, optional): n > min_sample_thr qualifies as well-sampled. Defaults to 10.
        simplify (bool, optional): Removes intermediate nodes when only one child node exists. Defaults to True.

    Returns:
        htree: HTree object
    """

    #Get inhibitory tree
    print('Only searching through inhibitory taxonomy (', subtree_node,"')")
    path = get_paths(inputmat_fileid, warn=False, write_toml=False)
    htree = HTree(htree_file=path['htree'])
    subtree = htree.get_subtree(node=subtree_node)

    #Get list of well-sampled cell types
    dataset = load_dataset(min_sample_thr=min_sample_thr)
    kept_classes = dataset['well_sampled_sorted_t_types'].tolist()
    print(f'{len(kept_classes)} cell types retained, with at least {min_sample_thr} samples in the dataset')

    #Get tree with kept_classes:
    kept_tree_nodes = []
    for node in kept_classes:
        kept_tree_nodes.extend(subtree.get_ancestors(node))
        kept_tree_nodes.extend([node])

    kept_htree_df = subtree.obj2df()
    kept_htree_df = kept_htree_df[kept_htree_df['child'].isin(kept_tree_nodes)]
    kept_htree = HTree(htree_df=kept_htree_df)

    #Simplify layout and plot
    if simplify:
        htree, _ = simplify_tree(kept_htree, skip_nodes=None, verbose=False)
        htree.update_layout()
    else:
        htree = kept_htree
    return htree


def taxonomy_assignments(inputmat_fileid, initial_labels, datadict: Dict, n_required_classes: int, merge_on='well_sampled'):
    """Relabels the the input labels according to a lower resolution of the reference taxonomy. 

    Args:
        initial_labels: initial labels. np.array or List.
        datadict (Dict): dictionary with fields `well_sampled_sorted_t_types` or `unique_sorted_t_types`
        n_required_classes (int): assumed to be less than or equal to number of classes in the taxonomy
        merge_on (str, optional): Subset of classes of the taxonomy. Defaults to 'well_sampled'.

    Returns:
        updated_labels (np.array): new labels, same size as initial labels
        n_remain_classes (int)
    """

    reference_labels = None
    if merge_on == 'well_sampled':
        assert 'well_sampled_sorted_t_types' in datadict.keys(), 'Required key missing'
        reference_labels = datadict['well_sampled_sorted_t_types'].copy()
    elif merge_on == 'all':
        assert 'unique_sorted_t_types' in datadict.keys(), 'Required key missing'
        reference_labels = datadict['unique_sorted_t_types'].copy()
    assert reference_labels is not None, "reference_labels not set"

    path = get_paths(inputmat_fileid, warn=False,write_toml=False)
    new_reference_labels, _ = get_merged_ordered_classes(data_labels=reference_labels.copy(),
                                                         htree_file=str(path['htree']),
                                                         n_required_classes=n_required_classes,
                                                         verbose=False)

    n_remain_classes = np.unique(new_reference_labels).size
    updated_labels = np.array(initial_labels.copy())
    updated_labels[~np.isin(updated_labels, reference_labels)] = 'not_well_sampled'
    for (orig, new) in zip(reference_labels, new_reference_labels):
        updated_labels[updated_labels == orig] = new

    return updated_labels, n_remain_classes


def get_fileid(model_id, alpha_T, alpha_E, alpha_M, alpha_sd, lambda_T_EM,
               augment_decoders, E_noise, M_noise, dilate_M, latent_dim, batchsize, n_epochs, run_iter, n_fold):
    '''
    returns the fileid based on the run parameters

    Args:
        model_id: Model-specific id
        alpha_T: T reconstruction loss weight
        alpha_E: E reconstruction loss weight
        alpha_M: M reconstruction loss weight
        alpha_sd: soma depth reconstruction loss weight
        lambda_T_EM: T - EM coupling loss weight
        augment_decoders: 0 or 1 - Train with cross modal reconstruction
        E_noise: the std of the noise added to E data
        M_noise: the std of the noise added to M data
        latent_dim: Number of latent dims
        batchsize: Batch size
        n_epochs: Number of epochs to train
        run_iter: Run-specific id
        n_fold: Fold id in the kfold cross validation training
    '''

    if E_noise is not None:
        if M_noise is not None:
            fileid = (model_id + f'_aT_{str(alpha_T)}_aE_{str(alpha_E)}_aM_{str(alpha_M)}_asd_{str(alpha_sd)}_' +
              f'csT_EM_{str(lambda_T_EM)}_Mnoi_{str(M_noise)}_Enoi_{str(E_noise)}_' +
              f'dil_M_{str(dilate_M)}_ad_{str(augment_decoders)}_' +
              f'ld_{latent_dim:d}_bs_{batchsize:d}_ne_{n_epochs:d}_' +
              f'ri_{run_iter:d}_fold_{n_fold:d}').replace('.', '-')
    else:
        fileid = (model_id + f'_aT_{str(alpha_T)}_aE_{str(alpha_E)}_aM_{str(alpha_M)}_asd_{str(alpha_sd)}_' +
                  f'csT_EM_{str(lambda_T_EM)}_ad_{str(augment_decoders)}_' +
                  f'ld_{latent_dim:d}_bs_{batchsize:d}_ne_{n_epochs:d}_' +
                  f'ri_{run_iter:d}_fold_{n_fold:d}').replace('.', '-')

    return fileid


def get_io_path(package_dir, exp_name, output_fileid=None):
    ''' Returns a dict with all output and input paths

    Args:
         package_dir: path to the package
    '''
    path = {}

    path['input_dir_path'] = os.path.join(package_dir, "data/proc/")
    path['output_dir_path'] = os.path.join(package_dir, "data/results/", exp_name)
    path['logs'] = os.path.join(package_dir, "data/results/", exp_name, "logs/")

    if output_fileid:
        path['output_path'] = os.path.join(path['output_dir_path'], (output_fileid + "_exit-summary.pkl"))
        path['log_path'] = os.path.join(path['output_dir_path'], "logs/", (output_fileid + ".csv"))
    return path

def get_results_run_id(model_id, alpha_T, alpha_E, alpha_M, alpha_sd, lambda_T_EM, augment_decoders, E_noise,
                       M_noise, dilate_M, latent_dim, batchsize, n_epochs, run_iter, n_fold, package_dir, exp_name):

    results = {}
    fileid = get_fileid(model_id, alpha_T, alpha_E, alpha_M, alpha_sd, lambda_T_EM, augment_decoders, E_noise,
                        M_noise, dilate_M, latent_dim, batchsize, n_epochs, run_iter, n_fold)

    io_path = get_io_path(package_dir, exp_name, output_fileid=fileid)
    results["summary"] = ut.loadpkl(io_path["output_path"])
    results["log"] = pd.read_csv(io_path["log_path"])

    return results
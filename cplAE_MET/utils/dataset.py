import json
import torch
import numpy as np
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold


def load_bioarxiv_dataset(data_path):
    """Loads patch-seq data used in the Bioarxiv manuscript 3708 cells: 1252 genes, and 68 (44+24) ephys features.

    Args:
        data_path (str): path to data file

    Returns:
        data (dict)
    """
    data = sio.loadmat(data_path + 'PS_v5_beta_0-4_pc_scaled_ipfx_eqTE.mat', squeeze_me=True)
    with open(data_path + 'E_names.json') as f:
        ephys_names = json.load(f)
    data['E_pcipfx'] = np.concatenate([data['E_pc_scaled'], data['E_feature']], axis=1)
    data['pcipfx_names'] = np.concatenate([data['pc_name'],data['feature_name']])
    temp = [ephys_names[f] for f in data['pcipfx_names']]
    data['pcipfx_names'] = np.array(temp)
    return data

def load_M_inh_dataset(data_path):
    """Loads processed MET data for inhibitory cells.

    Args:
        data_path (str): path to data file

    Returns:
        data (dict)
    """
    data = sio.loadmat(data_path, squeeze_me=True)

    D = {}
    D['XM'] = data['hist_ax_de']
    D['X_sd'] = data['soma_depth']
    D['cluster_label'] = data['cluster_label']
    D['cluster_color'] = data['cluster_color']
    D['cluster_id'] = data['cluster_id']
    D['specimen_id'] = data['specimen_id']
    return D


def load_MET_inh_dataset(data_path, verbose=False):
    """Loads processed MET data for inhibitory cells.

    Args:
        data_path (str): path to data file

    Returns:
        data (dict)
    """
    data = sio.loadmat(data_path, squeeze_me=True)
    if verbose: 
        MET_inh_data_summary(data)

    D = {}
    D['XT'] = data['T_dat']
    D['XE'] = data['E_dat']
    D['XM'] = data['M_dat']
    D['X_sd'] = data['soma_depth']
    D['cluster'] = data['cluster']
    D['cluster_id'] = data['cluster_id'].astype(int)
    D['cluster_color'] = data['cluster_color']
    D['sample_id'] = data['sample_id']
    return D




def MET_inh_data_summary(data):
    """lists shapes and pairing information about the dataset

    Args:
        data (Dict): required keys `T_dat`,`E_dat`, `M_dat`, `soma_depth`
    """
    print(f"T shape {data['T_dat'].shape}")
    print(f"E shape {data['E_dat'].shape}")
    print(f"M shape {data['M_dat'].shape}")
    print(f"sd shape {data['soma_depth'].shape}")

    # find all-nan samples boolean
    def allnans(x): return np.all(
        np.isnan(x.reshape(np.shape(x)[0], -1)), axis=(1))

    # find any-nan samples boolean
    def anynans(x): return np.all(
        np.isnan(x.reshape(np.shape(x)[0], -1)), axis=(1))

    m_T = ~allnans(data['T_dat'])
    m_E = ~allnans(data['E_dat'])
    m_M = ~allnans(data['M_dat'])
    m_sd = ~allnans(data['soma_depth'])

    def paired(x, y): return np.sum(np.logical_and(x, y))
    print('\nPaired samples, allowing for nans in some features')
    print(f'{paired(m_T,m_E)} cells paired in T and E')
    print(f'{paired(m_T,m_M)} cells paired in T and M')
    print(f'{paired(m_E,m_M)} cells paired in E and M')

    m_T = ~anynans(data['T_dat'])
    m_E = ~anynans(data['E_dat'])
    m_M = ~anynans(data['M_dat'])
    m_sd = ~anynans(data['soma_depth'])

    print('\nPaired samples, without nans in any feature (strict)')
    print(f'{paired(m_T,m_E)} cells paired in T and E')
    print(f'{paired(m_T,m_M)} cells paired in T and M')
    print(f'{paired(m_E,m_M)} cells paired in E and M')
    return


def partitions(celltype, n_partitions, seed=0):
    """Create stratified cross validation sets, based on `cluster` annotations. 
    Indices of the n-th fold are `ind_dict[n]['train']` and `ind_dict[n]['val']`.
    Assumes `celltype` has the same samples as in the dataset. 

    Args:
        celltype: numpy array with celltype annotations for all dataset samples
        n_partitions: number of partitions (e.g. cross validation folds)
        seed: random seed used for partitioning.
    
    Returns:
        ind_dict: list with `n_partitions` dict elements. 
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    #Safe to ignore warning - there are celltypes with a low sample number that are not crucial for the analysis.
    with warnings.catch_warnings():    
        skf = StratifiedKFold(n_splits=n_partitions, random_state=0, shuffle=True)

    #Get all partition indices from the sklearn generator:
    ind_dict = [{'train': train_ind, 'val': val_ind} for train_ind, val_ind in
                skf.split(X=np.zeros(shape=celltype.shape), y=celltype)]
    return ind_dict


class MET_Dataset(torch.utils.data.Dataset):
    """Create a torch dataset from inputs XT, XE, XM, sd.  

    Args:
        XT: np.array
        XE: np.array
        XM: np.array
        sd: np.array
    """
    def __init__(self, XT, XE, XM, sd):
        super(MET_Dataset).__init__()
        self.XT = XT
        self.XE = XE
        self.XM = XM
        self.sd = sd
        self.n_samples = XT.shape[0]

    def __getitem__(self, idx):
        sample = {"XT": self.XT[idx, :],
                  "XE": self.XE[idx, :],
                  "XM": self.XM[idx, :],
                  "X_sd": self.sd[idx]}
        return sample

    def __len__(self):
        return self.n_samples


class M_Dataset(torch.utils.data.Dataset):
    """Create a torch dataset from inputs XM, sd.

    Args:
        XM: np.array
        sd: np.array
        target: np.array
    """
    def __init__(self, XM, sd, target, shifts):
        super(M_Dataset).__init__()
        self.XM = XM
        self.sd = sd
        self.target = target
        self.shifts = shifts
        self.n_samples = sd.shape[0]

    def __getitem__(self, idx):
        sample = {"XM": self.XM[idx, :, :, :],
                  "X_sd": self.sd[idx, :],
                  "M_target": self.target[idx],
                  "shifts": self.shifts[idx, :]}
        # return sample
        return self.XM[idx, :],self.sd[idx, :],self.target[idx],self.shifts[idx, :]

    def __len__(self):
        return self.n_samples


class M_AE_Dataset(torch.utils.data.Dataset):
    """Create a torch dataset from inputs XM, sd.

    Args:
        XM: np.array
        sd: np.array
        target: np.array
    """
    def __init__(self, XM, sd, shifts):
        super(M_AE_Dataset).__init__()
        self.XM = XM
        self.sd = sd
        self.shifts = shifts
        self.n_samples = sd.shape[0]

    def __getitem__(self, idx):
        sample = {"XM": self.XM[idx, ...],
                  "X_sd": self.sd[idx, :],
                  "shifts": self.shifts[idx, :]}
        return sample

    def __len__(self):
        return self.n_samples

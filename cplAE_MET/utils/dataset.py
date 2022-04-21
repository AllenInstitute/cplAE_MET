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

def load_M_dataset(data_path):
    """Loads processed MET data.

    Args:
        data_path (str): path to data file

    Returns:
        data (dict)
    """
    data = sio.loadmat(data_path, squeeze_me=True)

    D = {}
    D['XM'] = data['hist_ax_de_api_bas']
    D['X_sd'] = data['soma_depth']
    D['cluster_label'] = data['cluster_label']
    D['cluster_color'] = data['cluster_color']
    D['cluster_id'] = data['cluster_id']
    D['specimen_id'] = data['specimen_id']
    return D

def merge_exc_inh_M_dataset(inh_m_datapath, exc_m_datapath, exc_inh_m_savedatapath):
    """read, merge and save exc and inh M dataset.

        Args:
            inh_m_datapath (str): path to inh data file
            exc_m_datapath (str): path to exc data file
            exc_inh_m_savedatapath (str): path to save the combined file

        Returns:
        """
    inh = sio.loadmat(inh_m_datapath)
    exc = sio.loadmat(exc_m_datapath)
    sio.savemat(exc_inh_m_savedatapath,
                {'hist_ax_de_api_bas': np.concatenate([inh['hist_ax_de_api_bas'], exc['hist_ax_de_api_bas']], axis=0),
                 'soma_depth': np.concatenate([inh['soma_depth'], exc['soma_depth']], axis=1),
                 'cluster_label': np.concatenate([inh['cluster_label'], exc['cluster_label']], axis=1),
                 'cluster_color': np.concatenate([inh['cluster_color'], exc['cluster_color']], axis=1),
                 'cluster_id': np.concatenate([inh['cluster_id'], exc['cluster_id']], axis=1),
                 'specimen_id': np.concatenate([inh['specimen_id'], exc['specimen_id']], axis=1)}, do_compression=True)
    return


def load_MET_dataset(data_path, verbose=False):
    """Loads processed MET data for inhibitory cells.

    Args:
        data_path (str): path to data file

    Returns:
        data (dict)
    """
    data = sio.loadmat(data_path, squeeze_me=True)
    if verbose: 
        MET_data_summary(data)

    D = {}
    D['XT'] = data['T_dat']
    D['XE'] = data['E_dat']
    D['XM'] = data['M_dat']
    D['Xsd'] = data['soma_depth']
    D['cluster_label'] = data['cluster_label']
    D['cluster_id'] = data['cluster_id'].astype(int)
    D['cluster_color'] = data['cluster_color']
    D['specimen_id'] = data['specimen_id']
    D['gene_ids'] = data['gene_ids']

    return D




def MET_data_summary(data):
    """lists shapes and pairing information about the dataset

    Args:
        data (Dict): required keys `T_dat`,`E_dat`, `M_dat`, `soma_depth`
    """
    print(f"T shape {data['XT'].shape}")
    print(f"E shape {data['XE'].shape}")
    print(f"M shape {data['XM'].shape}")
    print(f"sd shape {data['Xsd'].shape}")

    # find all-nan samples boolean
    def allnans(x): return np.all(
        np.isnan(x.reshape(np.shape(x)[0], -1)), axis=(1))

    # find any-nan samples boolean
    def anynans(x): return np.all(
        np.isnan(x.reshape(np.shape(x)[0], -1)), axis=(1))

    m_T = ~allnans(data['XT'])
    m_E = ~allnans(data['XE'])
    m_M = ~allnans(data['XM'])
    m_sd = ~allnans(data['Xsd'])

    def paired(x, y): return np.sum(np.logical_and(x, y))
    print('\nPaired samples, allowing for nans in some features')
    print(f'{paired(m_T,m_E)} cells paired in T and E')
    print(f'{paired(m_T,m_M)} cells paired in T and M')
    print(f'{paired(m_E,m_M)} cells paired in E and M')

    m_T = ~anynans(data['XT'])
    m_E = ~anynans(data['XE'])
    m_M = ~anynans(data['XM'])
    m_sd = ~anynans(data['Xsd'])

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


class T_ME_Dataset(torch.utils.data.Dataset):
    """Create a torch dataset from inputs XT, XE, XM, sd.  

    Args:
        XT: np.array
        XE: np.array
        XM: np.array
        sd: np.array
    """
    def __init__(self, XT, XM, Xsd, XE):
        super(T_ME_Dataset).__init__()
        self.XT = XT
        self.XM = XM
        self.Xsd = Xsd
        self.XE = XE
        self.n_samples = XT.shape[0]

    def __getitem__(self, idx):
        sample = {"XT": self.XT[idx, :],
                  "XM": self.XM[idx, :],
                  "Xsd": self.Xsd[idx],
                  "XE": self.XE[idx, :]}

        return sample

    def __len__(self):
        return self.n_samples


class TE_Dataset(torch.utils.data.Dataset):
    """Create a torch dataset from inputs XT, XE

    Args:
        XT: np.array
        XE: np.array
    """
    def __init__(self, XT, XE):
        super(TE_Dataset).__init__()
        self.XT = XT
        self.XE = XE
        self.n_samples = XT.shape[0]

    def __getitem__(self, idx):
        sample = {"XT": self.XT[idx, :],
                  "XE": self.XE[idx, :]}

        return sample

    def __len__(self):
        return self.n_samples



class M_AE_Dataset(torch.utils.data.Dataset):
    """Create a torch dataset from inputs XM, sd.

    Args:
        XM: np.array
        sd: np.array
        target: np.array
    """

    def __init__(self, XM, sd):
        super(M_AE_Dataset).__init__()
        self.XM = XM
        self.sd = sd
        self.n_samples = sd.shape[0]

    def __getitem__(self, idx):
        sample = {"XM": self.XM[idx, ...],
                  "X_sd": self.sd[idx, :]}
        return sample

    def __len__(self):
        return self.n_samples


class T_AE_Dataset(torch.utils.data.Dataset):
    """Create a torch dataset from inputs XT

    Args:
        XT: np.array
    """
    def __init__(self, XT):
        super(T_AE_Dataset).__init__()
        self.XT = XT
        self.n_samples = XT.shape[0]

    def __getitem__(self, idx):
        sample = {"XT": self.XT[idx, :]}
        return sample

    def __len__(self):
        return self.n_samples


class E_AE_Dataset(torch.utils.data.Dataset):
    """Create a torch dataset from inputs XT

    Args:
        XT: np.array
    """
    def __init__(self, XE):
        super(E_AE_Dataset).__init__()
        self.XE = XE
        self.n_samples = XE.shape[0]

    def __getitem__(self, idx):
        sample = {"XE": self.XE[idx, :]}
        return sample

    def __len__(self):
        return self.n_samples

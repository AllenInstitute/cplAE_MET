import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold


class MET_exc_inh(object):
    def __init__(self, dat):
        self.XT = dat['XT']
        self.XE = dat['XE']
        self.XM = dat['XM']
        self.Xsd = dat['Xsd']
        self.cluster_label = dat['cluster_label']
        self.cluster_id = dat['cluster_id']
        self.cluster_color = dat['cluster_color']
        self.specimen_id = dat['specimen_id']
        self.gene_ids = dat['gene_ids']
        self.E_features = dat['E_features']
        self.norm2px = dat['norm2px']

    @staticmethod
    def from_file(data_path):
        dat = MET_exc_inh.load(data_path)
        dat = MET_exc_inh(dat)
        return dat

    def __getitem__(self, inds):
        # convert a simple indsex x[y] to a tuple for consistency
        if not isinstance(inds, tuple):
            inds = tuple(inds)
        return MET_exc_inh(dict(XT=self.XT[inds[0]],
                                  XE=self.XE[inds[0]],
                                  XM=self.XM[inds[0]],
                                  Xsd=self.Xsd[inds[0]],
                                  cluster_label=self.cluster_label[inds[0]],
                                  cluster_id=self.cluster_id[inds[0]],
                                  cluster_color=self.cluster_color[inds[0]],
                                  specimen_id=self.specimen_id[inds[0]],
                                  gene_ids=self.gene_ids,
                                  E_features=self.E_features,
                                  norm2px=self.norm2px))


    def __repr__(self):
        return f'met data with {self.XT.shape[0]} cells'

    @staticmethod
    def valid_data(x):
        return np.any(~np.isnan(x).reshape(x.shape[0], -1), axis=1)

    @property
    def isT_1d(self): return self.valid_data(self.XT)

    @property
    def isE_1d(self): return self.valid_data(self.XE)

    @property
    def isM_1d(self): return self.valid_data(self.XM)

    @property
    def isT_ind(self): return np.flatnonzero(self.isT_1d)

    @property
    def isE_ind(self): return np.flatnonzero(self.isE_1d)

    @property
    def isM_ind(self): return np.flatnonzero(self.isM_1d)

    @property
    def Xsd_px(self): return self.Xsd * self.norm2px

    @property
    def XM_centered(self):
        return self.soma_center(XM=self.XM, Xsd=self.Xsd, norm2px=self.norm2px, jitter_frac=0)

    @staticmethod
    def load(data_path):
        """Loads processed patchseq MET data.

        Args:
            data_path (str): path to data file

        Returns:
            data (dict)
        """
        data = sio.loadmat(data_path, squeeze_me=True)

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
        D['E_features'] = data['E_features']

        # removing extra whitespaces from strings
        D['cluster_label'] = np.array([c.strip() for c in D['cluster_label']])
        D['cluster_color'] = np.array([c.strip() for c in D['cluster_color']])
        D['gene_ids'] = np.array([c.strip() for c in D['gene_ids']])
        D['E_features'] = np.array([c.strip() for c in D['E_features']])

        # convention for annotations
        isnan = D['cluster_label']=='nan'
        D['cluster_label'][isnan] = 'NA'
        D['cluster_id'][isnan] = np.max(D['cluster_id']) + 1
        D['cluster_color'][isnan] = '#888888'

        # conversion factor to plot soma depth onto density maps
        # in pixel space, 0=pia, 100=white matter; density maps extend below white matter.
        # any missing M data is set to nan (e.g. for inhibitory cells, apical and basal dendrite densities = np.nan)
        D['norm2px'] = 100 
        # If XM is 4 channel arbor density in which for EXC cells two channels are NAN and two channels have nonzero values
        # and for the inh cells the same thing. And for the rest of the cells everything is nan
        # XM_valid = np.apply_over_axes(np.all,
        #                             np.logical_or(D['XM'] == 0, np.isnan(D['XM'])),
        #                             axes = [1, 2])
        # If XM is 4 channel arbor density in which for EXC cells two channels are ZERO and two channels have nonzero values
        # and for the inh cells the same thing. And for the rest of the cells everything is nan
        XM_valid = np.apply_over_axes(np.all,
                                      np.isnan(D['XM']),
                                      axes = [1, 2])

        XM_valid = np.broadcast_to(XM_valid, D['XM'].shape)
        D['XM'][XM_valid] = np.nan
        return D

    def summarize(self):
        print(f"T shape {self.XT.shape}")
        print(f"E shape {self.XE.shape}")
        print(f"M shape {self.XM.shape}")
        print(f"sd shape {self.Xsd.shape}")

        # find all-nan samples boolean
        def allnans(x): return np.all(
            np.isnan(x.reshape(np.shape(x)[0], -1)), axis=(1))

        m_T = ~allnans(self.XT)
        m_E = ~allnans(self.XE)
        m_M = ~allnans(self.XM)

        print('\nSamples with at least one non-nan features')
        print(f'{np.sum(m_T)} cells in T')
        print(f'{np.sum(m_E)} cells in E')
        print(f'{np.sum(m_M)} cells in M')

        def paired(x, y): return np.sum(np.logical_and(x, y))
        print('\nPaired samples, allowing for nans in a strict subset of features in both modalities')
        print(f'{paired(m_T,m_E)} cells paired in T and E')
        print(f'{paired(m_T,m_M)} cells paired in T and M')
        print(f'{paired(m_E,m_M)} cells paired in E and M')
        return

    @staticmethod
    def soma_center(XM, Xsd, norm2px, jitter_frac=0):
        # nans are treated as zeros
        assert (np.nanmax(Xsd) <= 1) and (np.nanmin(Xsd) >= 0), 'Xsd expected in range (0,1)'
        jitter = (np.random.random(Xsd.shape) - 0.5)*jitter_frac
        Xsd_jitter = np.clip(Xsd + jitter, 0, 1)

        Xsd_px = np.round(Xsd_jitter * norm2px)
        Xsd_px = np.nan_to_num(Xsd_px).astype(int)

        # padded density map is double the size of the original in H dimension
        pad = XM.shape[1] // 2
        XM_centered = np.zeros(np.array(XM.shape) + np.array((0, 2*pad, 0, 0)))
        new_zero_px = XM_centered.shape[1] // 2 - Xsd_px
        for i in range(XM_centered.shape[0]):
            XM_centered[i, new_zero_px[i]:new_zero_px[i] +
                        XM.shape[1], :, :] = XM[i, ...]

        setnan = np.apply_over_axes(np.all, np.isnan(XM), axes=[1, 2])
        setnan = np.broadcast_to(setnan, XM_centered.shape)
        XM_centered[setnan] = np.nan
        return XM_centered


    def train_val_split(self, fold, thr=10, n_folds=10, seed=0, verbose=False):
        """ Validation strategy
        - we want to test performance based on classification and cross-modal reconstruction.
        - to have ground truth for comparison, we require cells with (T,M and E) measurements
        - Moreover, evaluation of classification performance requires well-sampled t-types (to train the classifier)
        - there are 1297 such samples (cells)
        - only 38 t-types have >10 such cells

        Select only from these well-sampled t-types using a stratified k-fold approach.

        Args:
            fold (int): fold id
            thr (int, optional): minimum number of cells within t-type that is included in validation set. Defaults to 10.
            n_folds (int, optional): max number of splits. Defaults to 10.
            seed (int): For reproducibility. Defaults to 0.
            verbose (bool): print summary

        Returns:
            train_ind, val_ind: indices to use for training and validation sets
        """

        assert fold < n_folds, f"fold must be int <= {n_folds}"

        # step 1: rename t-types for samples that do not have all measurements
        new_labels = self.cluster_label.copy()
        isMET_1d = np.logical_and(np.logical_and(self.isT_1d, self.isE_1d), self.isM_1d)
        new_labels[~isMET_1d] = 'NA'

        # step 2: rename low-sampled t-types
        for cluster in np.unique(new_labels):
            if np.sum(new_labels == cluster) < thr:
                new_labels[new_labels == cluster] = 'NA'

        # summary
        df = pd.DataFrame({'cluster_labels': new_labels})
        df = df.value_counts().to_frame().rename(columns={0: 'counts'}).reset_index()
        df = df.loc[df['cluster_labels'] != 'NA']
        if verbose:
            print(f'Validation samples to be picked from {df["counts"].sum()} cells across {df.shape[0]} well-sampled t-types')

        # splits
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        inds = list(splitter.split(X=np.arange(new_labels.size), y=new_labels))
        train_ind, val_ind = inds[fold]

        # remove any samples with 'NA' from validation set, and place them in training set
        exclude_ind = val_ind[new_labels[val_ind] == 'NA']
        val_ind = val_ind[~np.isin(val_ind, exclude_ind)]
        train_ind = np.concatenate([train_ind, exclude_ind])
        return train_ind, val_ind


    
class MET_exc_inh_v1(object):
    def __init__(self, dat):
        self.XT = dat['XT']
        self.XE = dat['XE']
        self.XM = dat['XM']
        self.Xsd = dat['Xsd']
        self.cluster_label = dat['cluster_label']
        self.cluster_id = dat['cluster_id']
        self.cluster_color = dat['cluster_color']
        self.specimen_id = dat['specimen_id']
        self.gene_ids = dat['gene_ids']
        self.E_features = dat['E_features']
        self.M_features = dat['M_features']

    @staticmethod
    def from_file(data_path):
        dat = MET_exc_inh_v1.load(data_path)
        dat = MET_exc_inh_v1(dat)
        return dat

    def __getitem__(self, inds):
        # convert a simple indsex x[y] to a tuple for consistency
        if not isinstance(inds, tuple):
            inds = tuple(inds)
        return MET_exc_inh_v1(dict(XT=self.XT[inds[0]],
                                  XE=self.XE[inds[0]],
                                  XM=self.XM[inds[0]],
                                  Xsd=self.Xsd[inds[0]],
                                  cluster_label=self.cluster_label[inds[0]],
                                  cluster_id=self.cluster_id[inds[0]],
                                  cluster_color=self.cluster_color[inds[0]],
                                  specimen_id=self.specimen_id[inds[0]],
                                  gene_ids=self.gene_ids,
                                  E_features=self.E_features, 
                                  M_features=self.M_features))


    def __repr__(self):
        return f'met data with {self.XT.shape[0]} cells'

    @staticmethod
    def valid_data(x):
        return np.any(~np.isnan(x).reshape(x.shape[0], -1), axis=1)

    @property
    def isT_1d(self): return self.valid_data(self.XT)

    @property
    def isE_1d(self): return self.valid_data(self.XE)

    @property
    def isM_1d(self): return self.valid_data(self.XM)

    @property
    def isT_ind(self): return np.flatnonzero(self.isT_1d)

    @property
    def isE_ind(self): return np.flatnonzero(self.isE_1d)

    @property
    def isM_ind(self): return np.flatnonzero(self.isM_1d)


    @staticmethod
    def load(data_path):
        """Loads processed patchseq MET data.

        Args:
            data_path (str): path to data file

        Returns:
            data (dict)
        """
        data = sio.loadmat(data_path, squeeze_me=True)

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
        D['E_features'] = data['E_features']
        D['M_features'] = data['M_features']

        # removing extra whitespaces from strings
        D['cluster_label'] = np.array([c.strip() for c in D['cluster_label']])
        D['cluster_color'] = np.array([c.strip() for c in D['cluster_color']])
        D['gene_ids'] = np.array([c.strip() for c in D['gene_ids']])
        D['E_features'] = np.array([c.strip() for c in D['E_features']])
        D['M_features'] = np.array([c.strip() for c in D['M_features']])

        # convention for annotations
        isnan = D['cluster_label']=='nan'
        D['cluster_label'][isnan] = 'NA'
        D['cluster_id'][isnan] = np.max(D['cluster_id']) + 1
        D['cluster_color'][isnan] = '#888888'
        return D

    def summarize(self):
        print(f"T shape {self.XT.shape}")
        print(f"E shape {self.XE.shape}")
        print(f"M shape {self.XM.shape}")
        print(f"sd shape {self.Xsd.shape}")

        # find all-nan samples boolean
        def allnans(x): return np.all(
            np.isnan(x.reshape(np.shape(x)[0], -1)), axis=(1))

        m_T = ~allnans(self.XT)
        m_E = ~allnans(self.XE)
        m_M = ~allnans(self.XM)

        print('\nSamples with at least one non-nan features')
        print(f'{np.sum(m_T)} cells in T')
        print(f'{np.sum(m_E)} cells in E')
        print(f'{np.sum(m_M)} cells in M')

        def paired(x, y): return np.sum(np.logical_and(x, y))
        print('\nPaired samples, allowing for nans in a strict subset of features in both modalities')
        print(f'{paired(m_T,m_E)} cells paired in T and E')
        print(f'{paired(m_T,m_M)} cells paired in T and M')
        print(f'{paired(m_E,m_M)} cells paired in E and M')
        return    

    def train_val_split(self, fold, thr=10, n_folds=10, seed=0, verbose=False):
        """ Validation strategy
        - we want to test performance based on classification and cross-modal reconstruction.
        - to have ground truth for comparison, we require cells with (T,M and E) measurements
        - Moreover, evaluation of classification performance requires well-sampled t-types (to train the classifier)
        - there are 1297 such samples (cells)
        - only 38 t-types have >10 such cells

        Select only from these well-sampled t-types using a stratified k-fold approach.

        Args:
            fold (int): fold id
            thr (int, optional): minimum number of cells within t-type that is included in validation set. Defaults to 10.
            n_folds (int, optional): max number of splits. Defaults to 10.
            seed (int): For reproducibility. Defaults to 0.
            verbose (bool): print summary

        Returns:
            train_ind, val_ind: indices to use for training and validation sets
        """

        assert fold < n_folds, f"fold must be int <= {n_folds}"

        # step 1: rename t-types for samples that do not have all measurements
        new_labels = self.cluster_label.copy()
        isMET_1d = np.logical_and(np.logical_and(self.isT_1d, self.isE_1d), self.isM_1d)
        new_labels[~isMET_1d] = 'NA'

        # step 2: rename low-sampled t-types
        for cluster in np.unique(new_labels):
            if np.sum(new_labels == cluster) < thr:
                new_labels[new_labels == cluster] = 'NA'

        # summary
        df = pd.DataFrame({'cluster_labels': new_labels})
        df = df.value_counts().to_frame().rename(columns={0: 'counts'}).reset_index()
        df = df.loc[df['cluster_labels'] != 'NA']
        if verbose:
            print(f'Validation samples to be picked from {df["counts"].sum()} cells across {df.shape[0]} well-sampled t-types')

        # splits
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        inds = list(splitter.split(X=np.arange(new_labels.size), y=new_labels))
        train_ind, val_ind = inds[fold]

        # remove any samples with 'NA' from validation set, and place them in training set
        exclude_ind = val_ind[new_labels[val_ind] == 'NA']
        val_ind = val_ind[~np.isin(val_ind, exclude_ind)]
        train_ind = np.concatenate([train_ind, exclude_ind])
        return train_ind, val_ind




class MET_exc_inh_v2(object):
    def __init__(self, dat):
        self.XT = dat['XT']
        self.XE = dat['XE']
        self.XM = dat['XM']
        self.cluster_label = dat['cluster_label']
        self.cluster_id = dat['cluster_id']
        self.cluster_color = dat['cluster_color']
        self.specimen_id = dat['specimen_id']
        self.gene_ids = dat['gene_ids']
        self.E_features = dat['E_features']
        self.M_features = dat['M_features']

    @staticmethod
    def from_file(data_path):
        dat = MET_exc_inh_v2.load(data_path)
        dat = MET_exc_inh_v2(dat)
        return dat

    def __getitem__(self, inds):
        # convert a simple indsex x[y] to a tuple for consistency
        if not isinstance(inds, tuple):
            inds = tuple(inds)
        return MET_exc_inh_v2(dict(XT=self.XT[inds[0]],
                                  XE=self.XE[inds[0]],
                                  XM=self.XM[inds[0]],
                                  cluster_label=self.cluster_label[inds[0]],
                                  cluster_id=self.cluster_id[inds[0]],
                                  cluster_color=self.cluster_color[inds[0]],
                                  specimen_id=self.specimen_id[inds[0]],
                                  gene_ids=self.gene_ids,
                                  E_features=self.E_features, 
                                  M_features=self.M_features))


    def __repr__(self):
        return f'met data with {self.XT.shape[0]} cells'

    @staticmethod
    def valid_data(x):
        return np.any(~np.isnan(x).reshape(x.shape[0], -1), axis=1)

    @property
    def isT_1d(self): return self.valid_data(self.XT)

    @property
    def isE_1d(self): return self.valid_data(self.XE)

    @property
    def isM_1d(self): return self.valid_data(self.XM)

    @property
    def isT_ind(self): return np.flatnonzero(self.isT_1d)

    @property
    def isE_ind(self): return np.flatnonzero(self.isE_1d)

    @property
    def isM_ind(self): return np.flatnonzero(self.isM_1d)


    @staticmethod
    def load(data_path):
        """Loads processed patchseq MET data.

        Args:
            data_path (str): path to data file

        Returns:
            data (dict)
        """
        data = sio.loadmat(data_path, squeeze_me=True)

        D = {}
        D['XT'] = data['T_dat']
        D['XE'] = data['E_dat']
        D['XM'] = data['M_dat']
        D['cluster_label'] = data['cluster_label']
        D['cluster_id'] = data['cluster_id'].astype(int)
        D['cluster_color'] = data['cluster_color']
        D['specimen_id'] = data['specimen_id']
        D['gene_ids'] = data['gene_ids']
        D['E_features'] = data['E_features']
        D['M_features'] = data['M_features']

        # removing extra whitespaces from strings
        D['cluster_label'] = np.array([c.strip() for c in D['cluster_label']])
        D['cluster_color'] = np.array([c.strip() for c in D['cluster_color']])
        D['gene_ids'] = np.array([c.strip() for c in D['gene_ids']])
        D['E_features'] = np.array([c.strip() for c in D['E_features']])
        D['M_features'] = np.array([c.strip() for c in D['M_features']])

        # convention for annotations
        isnan = D['cluster_label']=='nan'
        D['cluster_label'][isnan] = 'NA'
        D['cluster_id'][isnan] = np.max(D['cluster_id']) + 1
        D['cluster_color'][isnan] = '#888888'
        return D

    def summarize(self):
        print(f"T shape {self.XT.shape}")
        print(f"E shape {self.XE.shape}")
        print(f"M shape {self.XM.shape}")

        # find all-nan samples boolean
        def allnans(x): return np.all(
            np.isnan(x.reshape(np.shape(x)[0], -1)), axis=(1))

        m_T = ~allnans(self.XT)
        m_E = ~allnans(self.XE)
        m_M = ~allnans(self.XM)

        print('\nSamples with at least one non-nan features')
        print(f'{np.sum(m_T)} cells in T')
        print(f'{np.sum(m_E)} cells in E')
        print(f'{np.sum(m_M)} cells in M')

        def paired(x, y): return np.sum(np.logical_and(x, y))
        print('\nPaired samples, allowing for nans in a strict subset of features in both modalities')
        print(f'{paired(m_T,m_E)} cells paired in T and E')
        print(f'{paired(m_T,m_M)} cells paired in T and M')
        print(f'{paired(m_E,m_M)} cells paired in E and M')
        return    

    def train_val_split(self, fold, thr=10, n_folds=10, seed=0, verbose=False):
        """ Validation strategy
        - we want to test performance based on classification and cross-modal reconstruction.
        - to have ground truth for comparison, we require cells with (T,M and E) measurements
        - Moreover, evaluation of classification performance requires well-sampled t-types (to train the classifier)
        - there are 1297 such samples (cells)
        - only 38 t-types have >10 such cells

        Select only from these well-sampled t-types using a stratified k-fold approach.

        Args:
            fold (int): fold id
            thr (int, optional): minimum number of cells within t-type that is included in validation set. Defaults to 10.
            n_folds (int, optional): max number of splits. Defaults to 10.
            seed (int): For reproducibility. Defaults to 0.
            verbose (bool): print summary

        Returns:
            train_ind, val_ind: indices to use for training and validation sets
        """

        assert fold < n_folds, f"fold must be int <= {n_folds}"

        # step 1: rename t-types for samples that do not have all measurements
        new_labels = self.cluster_label.copy()
        isMET_1d = np.logical_and(np.logical_and(self.isT_1d, self.isE_1d), self.isM_1d)
        new_labels[~isMET_1d] = 'NA'

        # step 2: rename low-sampled t-types
        for cluster in np.unique(new_labels):
            if np.sum(new_labels == cluster) < thr:
                new_labels[new_labels == cluster] = 'NA'

        # summary
        df = pd.DataFrame({'cluster_labels': new_labels})
        df = df.value_counts().to_frame().rename(columns={0: 'counts'}).reset_index()
        df = df.loc[df['cluster_labels'] != 'NA']
        if verbose:
            print(f'Validation samples to be picked from {df["counts"].sum()} cells across {df.shape[0]} well-sampled t-types')

        # splits
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        inds = list(splitter.split(X=np.arange(new_labels.size), y=new_labels))
        train_ind, val_ind = inds[fold]

        # remove any samples with 'NA' from validation set, and place them in training set
        exclude_ind = val_ind[new_labels[val_ind] == 'NA']
        val_ind = val_ind[~np.isin(val_ind, exclude_ind)]
        train_ind = np.concatenate([train_ind, exclude_ind])
        return train_ind, val_ind
import numpy as np
import pandas as pd
import scipy.io as sio
from collections import Counter
from sklearn.model_selection import StratifiedKFold


class MET_exc_inh(object):
    def __init__(self, dat):
        self.XT = dat['XT']
        self.XE = dat['XE']
        self.XM = dat['XM']
        self.Xsd = dat['Xsd']
        self.cluster_label = dat['cluster_label']
        self.merged_cluster_label_at40 = dat['merged_cluster_label_at40']
        self.merged_cluster_label_at50 = dat['merged_cluster_label_at50']
        self.merged_cluster_label_at60 = dat['merged_cluster_label_at60']
        self.cluster_id = dat['cluster_id']
        self.cluster_color = dat['cluster_color']
        self.specimen_id = dat['specimen_id']
        self.group = dat['group']
        self.subgroup = dat['subgroup']
        self.class_id = dat['class_id']


    @staticmethod
    def from_file(data_path):
        loaded_data = MET_exc_inh.load(data_path)
        dat = MET_exc_inh(loaded_data)
        return dat, loaded_data

    def __getitem__(self, inds):
        # convert a simple indsex x[y] to a tuple for consistency
        if not isinstance(inds, tuple):
            inds = tuple(inds)
        return MET_exc_inh(dict(XT=self.XT[inds[0]],
                                XE=self.XE[inds[0]],
                                XM=self.XM[inds[0]],
                                Xsd=self.Xsd[inds[0]],
                                cluster_label=self.cluster_label[inds[0]],
                                merged_cluster_label_at40=self.merged_cluster_label_at40[inds[0]],
                                merged_cluster_label_at50=self.merged_cluster_label_at50[inds[0]],
                                merged_cluster_label_at60=self.merged_cluster_label_at60[inds[0]],
                                cluster_id=self.cluster_id[inds[0]],
                                cluster_color=self.cluster_color[inds[0]],
                                specimen_id=self.specimen_id[inds[0]],
                                group=self.group[inds[0]],
                                subgroup=self.subgroup[inds[0]],
                                class_id=self.class_id[inds[0]]))


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
        D['merged_cluster_label_at40'] = data['merged_cluster_label_at40']
        D['merged_cluster_label_at50'] = data['merged_cluster_label_at50']
        D['merged_cluster_label_at60'] = data['merged_cluster_label_at60']
        D['cluster_id'] = data['cluster_id'].astype(int)
        D['cluster_color'] = data['cluster_color']
        D['class'] = data['class']
        D['class_id'] = data['class_id']
        D['group'] = data['group']
        D['subgroup'] = data['subgroup']
        D['specimen_id'] = data['specimen_id']
        D['platform'] = data['platform']
        D['gene_ids'] = data['gene_ids']
        D['E_features'] = data['E_features']
        D['M_features'] = data['M_features']
        D['hist_ax_de_api_bas'] = data['hist_ax_de_api_bas']
        # D['M_nmf_total_vars_inh'] = data['M_nmf_total_vars_inh']
        # D['M_nmf_total_vars_exc'] = data['M_nmf_total_vars_exc']
        # D['M_nmf_components_inh'] = data['M_nmf_components_inh']
        # D['M_nmf_components_exc'] = data['M_nmf_components_exc']
        # D['M_nmf_total_vars_ax'] = data['M_nmf_total_vars_ax']
        # D['M_nmf_total_vars_de'] = data['M_nmf_total_vars_de']
        # D['M_nmf_total_vars_api'] = data['M_nmf_total_vars_api']
        # D['M_nmf_total_vars_bas'] = data['M_nmf_total_vars_bas']
        # D['M_nmf_components_ax'] = data['M_nmf_components_ax']
        # D['M_nmf_components_de'] = data['M_nmf_components_de']
        # D['M_nmf_components_api'] = data['M_nmf_components_api']
        # D['M_nmf_components_bas'] = data['M_nmf_components_bas']

        # removing extra whitespaces from strings
        D['cluster_label'] = np.array([c.strip() for c in D['cluster_label']])
        D['merged_cluster_label_at40'] = np.array([c.strip() for c in D['merged_cluster_label_at40']])
        D['merged_cluster_label_at50'] = np.array([c.strip() for c in D['merged_cluster_label_at50']])
        D['merged_cluster_label_at60'] = np.array([c.strip() for c in D['merged_cluster_label_at60']])
        D['cluster_color'] = np.array([c.strip() for c in D['cluster_color']])
        D['gene_ids'] = np.array([c.strip() for c in D['gene_ids']])
        D['E_features'] = np.array([c.strip() for c in D['E_features']])
        D['M_features'] = np.array([c.strip() for c in D['M_features']])

        # convention for annotations
        isnan = D['cluster_label']=='nan'
        isnan_merged_at40 = D['merged_cluster_label_at40'] == 'nan'
        isnan_merged_at50 = D['merged_cluster_label_at50'] == 'nan'
        isnan_merged_at60 = D['merged_cluster_label_at60'] == 'nan'
        assert np.all(isnan == isnan_merged_at40) , f"When t types are merged, nans must be the same"
        assert np.all(isnan == isnan_merged_at50) , f"When t types are merged, nans must be the same"
        assert np.all(isnan == isnan_merged_at60) , f"When t types are merged, nans must be the same"
        D['cluster_label'][isnan] = 'NA'
        D['merged_cluster_label_at40'][isnan] = 'NA'
        D['merged_cluster_label_at50'][isnan] = 'NA'
        D['merged_cluster_label_at60'][isnan] = 'NA'
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
    
    def make_weights_for_balanced_classes(self, n_met, met_subclass_id , batch_size):
        '''Takes class and subclass labels of the cells. and return the weights in such a way that ~50% of the cells
        in each batch are from each class. The weights are in a way that in each class expected 54 cell are met cells. the 
        subclass id of the met cells are 2 and 3. 54 is the expected value of the met cells in the inh side ... 
        We need a balanced amount of met cells from exc and inh classes during the coupling. 
        Args:
            train_classes: the class labels of each cells, either 0 or 1
            train_subclasses: the subclass labels of each cell. If cells are met and their class is 0, then their subclass
            is 2. If they are met and their class is 1, then their subclass if 3. The rest of the cells in class 0 have subclass 
            0 and the rest of the cells in class 1 have subclass 1.
            n_met: number of met cells in each bach from each class
            met_subclass_id: the subclass id for the met cells in each class. from these subclass ids, we will sampling 
            n_met cells. 
            batch_size: batch size

        example: if the batch_size is 1000, roughly 500 will be inh and 500 will be exc. from those 500 in each class, 54
        will be met cells.
        '''
        train_classes = self.group.copy()
        train_subclasses = self.subgroup.copy()

        # desired prob of choosing a met cell from each class
        met_prob = n_met/(batch_size/2)
        others_prob = 1 - met_prob
         
        # class and subclass relationship: dic={0:0, 1:1, 2:0, 3:1}
        dic = {}
        for i, j in zip(train_classes, train_subclasses):
            dic[j] = i

        # count of class and subclasses and the total number of cells  
        class_counts = Counter(train_classes)
        subclass_counts = Counter(train_subclasses)
        N = float(sum(class_counts.values()))
        
        # to each cell, we assign a weight based on their classes. so if we use these weights then we can sample 50% from each class
        weight_per_class = {}  
        for k, v in class_counts.items():    
            weight_per_class[k] = N/float(v)

        # renormalize in such a way so that we still sample 500 from each class but the expected value for met cell sampling is 54 
        # in each class 
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
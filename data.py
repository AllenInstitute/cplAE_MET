import math
import functools

import numpy as np
import torch
from torch.utils.data import IterableDataset
import scipy.io as sio
from sklearn.model_selection import train_test_split

def get_collator(device, dtype):
    def collate(X):
        (X_dict, mask_dict, specimen_ids) = X
        X_torch = {modal: [torch.from_numpy(arr).to(device, dtype = dtype) for arr in tupl] 
                   for (modal, tupl) in X_dict.items()}
        mask_torch = {modal: torch.from_numpy(arr).to(device) for (modal, arr) in mask_dict.items()}
        return (X_torch, mask_torch, specimen_ids)
    return collate

class MET_Data():
    def __init__(self, mat_path):
        self.MET = sio.loadmat(mat_path)
        self.id_map = {spec_id.strip():i for (i, spec_id) in enumerate(self.MET["specimen_id"])}
        self.path = mat_path
        self.exclude = {"__header__", "__version__", "__globals__", "gene_ids", "E_features", "M_features"}

    def __getitem__(self, id_str):
        return self.MET[id_str]
    
    def keys(self):
        return (key for key in self.MET.keys() if key not in self.exclude)
    
    def values(self):
        return (value for (key, value) in self.MET.items() if key not in self.exclude)
    
    def items(self):
        return (item for item in self.MET.items() if item[0] not in self.exclude)

    def get_specimens(self, specimen_ids):
        stripped = [string.strip() for string in specimen_ids]
        data_dict = {}
        for (key, value) in self.items():
            cleaned = [np.squeeze(value)[self.id_map[spec]].reshape([1, -1]) for spec in stripped]
            data_dict[key] = np.concatenate(cleaned)
        return data_dict
    
    def get_stratified_split(self, test_frac, seed = 42):
        strat_cats = ["platform", "class", "cluster_label"]
        labels = functools.reduce(np.char.add, [np.char.strip(self[cat]) for cat in strat_cats])
        (values, counts) = np.unique(labels, return_counts = True)
        singleton_labels = values[counts == 1]
        if singleton_labels.size > 1:
            labels[np.isin(labels, singleton_labels)] = "_singleton"
        else:
            labels[np.isin(labels, singleton_labels)] = values[np.argmax(counts)]
        (train_ids, test_ids) = train_test_split(self["specimen_id"], test_size = test_frac, random_state = seed, stratify = labels)
        return (train_ids, test_ids)

class MET_Dataset(IterableDataset):
    def __init__(self, met_data, batch_size, modal_frac, allowed_specimen_ids = None):
        self.MET = met_data
        allowed_specimen_ids = (self.MET["specimen_id"] if allowed_specimen_ids is None else allowed_specimen_ids)
        (self.modal_indices, self.is_xt, self.is_xe, self.is_xm) = self.get_modal_indices(allowed_specimen_ids)
        (self.repeaters, self.counts) = self.get_repeaters(batch_size, modal_frac)
        self.num_batches = self.get_num_batches()

    def get_modal_indices(self, allowed_specimen_ids):
        num_cells = self.MET["specimen_id"].size
        is_xt = ~np.isnan(self.MET["T_dat"].reshape([num_cells, -1])).any(1)
        is_xe = ~np.isnan(self.MET["E_dat"].reshape([num_cells, -1])).any(1)
        is_xm = ~np.isnan(self.MET["M_dat"].reshape([num_cells, -1])).any(1)
        allowed = np.isin(self.MET["specimen_id"], allowed_specimen_ids)
        modal_bool = {
            "T":   allowed &  is_xt & ~is_xe & ~is_xm,
            "E":   allowed & ~is_xt &  is_xe & ~is_xm,
            "M":   allowed & ~is_xt & ~is_xe &  is_xm,
            "TE":  allowed &  is_xt &  is_xe & ~is_xm,
            "TM":  allowed &  is_xt & ~is_xe &  is_xm,
            "EM":  allowed & ~is_xt &  is_xe &  is_xm,
            "MET": allowed &  is_xt &  is_xe &  is_xm
        }
        all_indices = np.arange(num_cells)
        modal_indices = {modal: all_indices[mask] for (modal, mask) in modal_bool.items()}
        return (modal_indices, is_xt, is_xe, is_xm)

    def get_repeaters(self, batch_size, modal_frac):
        processed_frac = self.process_frac(modal_frac)
        frac_list = list(processed_frac.items())
        frac_list.sort(key = lambda tupl: tupl[1])
        round_up = [math.ceil(frac*batch_size) for (modal, frac) in frac_list[:-1]]
        counts_list = round_up + [(batch_size - sum(round_up)) if round_up else batch_size]
        counts = {modal: count for ((modal, frac), count) in zip(frac_list, counts_list) if count > 0}
        repeaters = {modal: RepeatingRandomIndex(self.modal_indices[modal]) for modal in counts}
        return (repeaters, counts)

    def get_num_batches(self):
        batches_needed = []
        for (modal, count) in self.counts.items():
            num_samples = len(self.modal_indices[modal])
            needed = math.ceil(num_samples / count)
            batches_needed.append(needed)
        num_batches = max(batches_needed)
        return num_batches
    
    def process_frac(self, modal_frac):
        num_cells = sum([len(indices) for indices in self.modal_indices])
        given_frac = {modal: frac for (modal, frac) in modal_frac.items() if frac != "native"}
        given_cuml = sum([frac for frac in given_frac.values()])
        if given_cuml > 1:
            raise ValueError(f"Modal fractions provided sum to {given_cuml} > 1.")
        native_modal = {modal for (modal, frac) in modal_frac.items() if frac == "native"}
        native_frac = {modal: len(self.modal_indices[modal]) / num_cells for modal in native_modal}
        native_cuml = sum([frac for frac in native_frac.values()])
        scaled_frac = {modal: frac*(1-given_cuml)/native_cuml for (modal, frac) in native_frac.items()}
        process_frac = {**given_frac, **scaled_frac}
        return process_frac
    
    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            indices = []
            for (modal, repeating_index) in self.repeaters.items():
                indices += repeating_index.get(self.counts[modal])
            (xt, xe, xm) = (self.MET["T_dat"][indices], self.MET["E_dat"][indices], self.MET["M_dat"][indices])
            (is_xt, is_xe, is_xm) = (self.is_xt[indices], self.is_xe[indices], self.is_xm[indices])
            specimen_ids = self.MET["specimen_id"][indices]
            outputs = ({"T": (xt,), "E": (xe,), "M": (xm,), "EM": (xe, xm)}, 
                       {"T": is_xt, "E": is_xe, "M": is_xm, "EM": is_xe & is_xm}, 
                       specimen_ids)
            yield outputs

class RepeatingRandomIndex():
    def __init__(self, indices):
        self.indices = torch.as_tensor(indices)
        self.order = torch.randperm(len(self.indices))
        self.step = 0
    
    def get(self, count):
        indices = []
        for _ in range(count):
            if self.step == len(self.indices):
                self.order = torch.randperm(len(self.indices))
                self.step = 0
            indices.append(self.indices[self.order[self.step]])
            self.step += 1
        return indices
    
if __name__ == "__main__":
    met_data = MET_Data("data/raw/MET_M120x4_50k_4Apr23.mat")
    batch_size = 20
    modal_frac = {"T": 0.5, "M": 0.5}
    met = MET_Dataset(met_data, batch_size, modal_frac, met_data["specimen_id"][:1000])
    print(met.counts)
    print(met.num_batches)
    print([(modal, len(indices)) for (modal, indices) in met.modal_indices.items()])
    # x = next(iter(met))
    # print(np.unique(np.unique(x[-1], return_counts = True)[-1], return_counts = True))
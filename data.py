import math
import functools
import zipfile
import itertools
import operator

import numpy as np
import torch
from torch.utils.data import IterableDataset
from sklearn.model_selection import train_test_split, StratifiedKFold

import utils

def powerset(iterable):
    seq = list(iterable)
    combinations = (itertools.combinations(seq, i) for i in range(1, len(seq) + 1))
    chained = itertools.chain.from_iterable(combinations)
    frozen_sets = [frozenset(comb) for comb in chained]
    return frozen_sets

def get_collator(device, dtype):
    def collate(X):
        (X_dict, mask_dict, specimen_ids) = X
        X_torch = {modal: {form: torch.from_numpy(arr).to(device, dtype = dtype) for (form, arr) in forms.items()}
                   for (modal, forms) in X_dict.items()}
        mask_torch = {modal: torch.from_numpy(arr).to(device) for (modal, arr) in mask_dict.items()}
        return (X_torch, mask_torch, specimen_ids)
    return collate

def filter_specimens(met_data, specimen_ids, config):
    platforms = config["select"]["platforms"]
    specimens = met_data.query(specimen_ids, platforms = platforms)["specimen_id"]
    return specimens

def get_transformation_function(transform_dict):
    functions = [transform_functions[func](params) for (func, params) in transform_dict.items()]
    def chain(data):
        output = functools.reduce(lambda x, func: func(x), functions, data)
        return output
    return chain

def get_shapes_lazy(npz_path):
    shapes = {}
    with zipfile.ZipFile(npz_path) as archive:
        for array_name in archive.namelist():
            with archive.open(array_name) as arr_file:
                version = np.lib.format.read_magic(arr_file)[0]
                head_func = np.lib.format.read_array_header_1_0 if version == 1 else np.lib.format.read_array_header_2_0
                shapes[array_name[:-4]] = head_func(arr_file)[0]
    return shapes

class MET_Data():
    def __init__(self, npz_path):
        self.MET = np.load(npz_path)
        self.shapes = get_shapes_lazy(npz_path)
        ids = self.MET["specimen_id"]
        self.id_map = {spec_id.strip():i for (i, spec_id) in enumerate(ids)}
        self.meta = {key for key in self.MET.keys() if self.shapes[key][0] != ids.shape[0]}
        self.path = npz_path
        self.cached = {}

    def __getitem__(self, id_str):
        if id_str not in self.cached:
            self.cached[id_str] = self.MET[id_str]
        data = self.cached[id_str]
        return data
    
    def keys(self):
        return (key for key in self.MET.keys() if key not in self.meta)
    
    def values(self):
        return (self[key] for key in self.MET if key not in self.meta)
    
    def items(self):
        return ((key, self[key]) for key in self.MET if key not in self.meta)

    def query(self, specimen_ids = None, formats = None, exclude_formats = None, platforms = None, classes = None):
        specimen_ids = (self["specimen_id"] if specimen_ids is None else specimen_ids)
        platforms = (np.char.strip(np.unique(self["platform"])) if platforms is None else platforms)
        classes = (np.char.strip(np.unique(self["class"])) if classes is None else classes)
        valid = np.isin(np.char.strip(self["specimen_id"]), np.char.strip(specimen_ids))
        valid = valid & np.isin(np.char.strip(self["platform"]), platforms)
        valid = valid & np.isin(np.char.strip(self["class"]), classes)
        if formats is not None:
            format_mask = np.full_like(valid, False)
            for form_tuple in formats:
                tupl_mask = np.full_like(valid, True)
                for form in form_tuple:
                    data = np.squeeze(self[form])
                    tupl_mask = tupl_mask & ~np.isnan(data).reshape([data.shape[0], -1]).all(1)
                format_mask = format_mask | tupl_mask
            valid = valid & format_mask
        if exclude_formats is not None:
            exclude_mask = np.full_like(valid, True)
            for form in exclude_formats:
                data = np.squeeze(self[form])
                exclude_mask = exclude_mask & np.isnan(data).reshape([data.shape[0], -1]).all(1)
            valid = valid & exclude_mask
        valid_specimens = self["specimen_id"][valid]
        data_dict = self.get_specimens(valid_specimens)
        return data_dict

    def get_specimens(self, specimen_ids):
        stripped = [string.strip() for string in specimen_ids]
        data_dict = {}
        for (key, value) in self.items():
            cleaned = [np.squeeze(value)[None, self.id_map[spec]] for spec in stripped]
            data_dict[key] = np.concatenate(cleaned) if cleaned else value[:0]
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
    
    def get_stratified_KFold(self, folds, seed = 42):
        strat_cats = ["platform", "class", "cluster_label"]
        labels = functools.reduce(np.char.add, [np.char.strip(self[cat]) for cat in strat_cats])
        splitter = StratifiedKFold(folds, shuffle = True, random_state = seed)
        for (train_ids, test_ids) in splitter.split(self["specimen_id"], labels):
            (train_spec, test_spec) = (self["specimen_id"][train_ids], self["specimen_id"][test_ids])
            yield (train_spec, test_spec)

class MET_Simulated():
    def __init__(self, config):
        (latent_samples, simulated_data, labels) = self.simulate_data(config)
        num_samples = labels.size
        self.MET = {
            "specimen_id": np.arange(num_samples).astype("str"),
            "platform": np.full([num_samples], "simulated"),
            "class": np.full([num_samples], "null"),
            "cluster_label": labels.astype("str"),
            "latent": latent_samples.reshape([num_samples, -1]),
            "logcpm": simulated_data["logcpm"].reshape([num_samples, -1]),
            "pca-ipfx": simulated_data["pca-ipfx"].reshape([num_samples, -1]),
            "arbors": simulated_data["arbors"].reshape([num_samples, 120, 4, 4]),
            "ivscc": simulated_data["ivscc"].reshape([num_samples, -1])
        }
        ids = self.MET["specimen_id"]
        self.id_map = {spec_id.strip():i for (i, spec_id) in enumerate(ids)}
        self.cached = {}

    def simulate_data(self, config):
        (counts, params) = (config["simulate"]["counts"], config["simulate"])
        (latent_dim, num_clusters) = (config["latent_dim"], params["num_clusters"])
        rng = np.random.default_rng(config["seed"])
        format_strings = np.asarray(list(counts.keys()))
        format_indices = np.concatenate([np.full([counts], i) for (i, counts) in enumerate(counts.values())])
        rng.shuffle(format_indices)
        formats = format_strings[format_indices]
        cluster_indices = np.arange(formats.size)
        index_segments = [cluster_indices[i::num_clusters] for i in range(num_clusters)]
        for (i, segment) in enumerate(index_segments):
            cluster_indices[segment] = i
        cluster_centroids = rng.normal(0, params["category_std"], [num_clusters, latent_dim])
        latent_samples = rng.normal(cluster_centroids[cluster_indices], params["cluster_std"])
        models = {form: utils.load_jit_folds(path)["folds"][1]["best"] for (form, path) in params["model_paths"].items()}
        simulated_data = {}
        for (form, model) in models.items():
            mask = (np.char.find(formats, form) == -1)
            decoder = next(iter(model.values()))["dec"]
            with torch.no_grad():
                hidden_variables = rng.normal(0, params["category_std"], [latent_samples.shape[0], 10 - latent_dim])
                latent_input = np.concatenate([latent_samples, hidden_variables], 1)
                output = decoder(torch.from_numpy(latent_input).float())[form].numpy()
            output[mask] = None
            simulated_data[form] = output
        return (latent_samples, simulated_data, cluster_indices)

    def __getitem__(self, id_str):
        if id_str not in self.cached:
            self.cached[id_str] = self.MET[id_str]
        data = self.cached[id_str]
        return data
    
    def keys(self):
        return (key for key in self.MET.keys())
    
    def values(self):
        return (self[key] for key in self.MET)
    
    def items(self):
        return ((key, self[key]) for key in self.MET)

    def query(self, specimen_ids = None, formats = None, exclude_formats = None, platforms = None, classes = None):
        specimen_ids = (self["specimen_id"] if specimen_ids is None else specimen_ids)
        platforms = (np.char.strip(np.unique(self["platform"])) if platforms is None else platforms)
        classes = (np.char.strip(np.unique(self["class"])) if classes is None else classes)
        valid = np.isin(np.char.strip(self["specimen_id"]), np.char.strip(specimen_ids))
        valid = valid & np.isin(np.char.strip(self["platform"]), platforms)
        valid = valid & np.isin(np.char.strip(self["class"]), classes)
        if formats is not None:
            format_mask = np.full_like(valid, False)
            for form_tuple in formats:
                tupl_mask = np.full_like(valid, True)
                for form in form_tuple:
                    data = np.squeeze(self[form])
                    tupl_mask = tupl_mask & ~np.isnan(data).reshape([data.shape[0], -1]).all(1)
                format_mask = format_mask | tupl_mask
            valid = valid & format_mask
        if exclude_formats is not None:
            exclude_mask = np.full_like(valid, True)
            for form in exclude_formats:
                data = np.squeeze(self[form])
                exclude_mask = exclude_mask & np.isnan(data).reshape([data.shape[0], -1]).all(1)
            valid = valid & exclude_mask
        valid_specimens = self["specimen_id"][valid]
        data_dict = self.get_specimens(valid_specimens)
        return data_dict

    def get_specimens(self, specimen_ids):
        stripped = [string.strip() for string in specimen_ids]
        data_dict = {}
        for (key, value) in self.items():
            cleaned = [np.squeeze(value)[None, self.id_map[spec]] for spec in stripped]
            data_dict[key] = np.concatenate(cleaned) if cleaned else value[:0]
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
    
    def get_stratified_KFold(self, folds, seed = 42):
        strat_cats = ["platform", "class", "cluster_label"]
        labels = functools.reduce(np.char.add, [np.char.strip(self[cat]) for cat in strat_cats])
        splitter = StratifiedKFold(folds, shuffle = True, random_state = seed)
        for (train_ids, test_ids) in splitter.split(self["specimen_id"], labels):
            (train_spec, test_spec) = (self["specimen_id"][train_ids], self["specimen_id"][test_ids])
            yield (train_spec, test_spec)

class MET_Decoupled():
    def __init__(self, npz_path, config):
        orig_MET = MET_Data(npz_path)
        self.MET = self.get_decoupled_met(config, orig_MET)
        self.id_map = {spec_id.strip():i for (i, spec_id) in enumerate(self["specimen_id"])}

    def get_decoupled_met(self, config, met):
        num_cells = met["specimen_id"].size
        rng = np.random.default_rng(config["seed"])
        masks = {form: np.any(~np.isnan(met[form]).reshape(num_cells, -1), 1)
                      for form in ["logcpm", "pca-ipfx", "arbors", "ivscc"]}
        used_specimens = met["specimen_id"][:0]
        data = {form: [] for form in masks}
        form_counts = list(config["decouple"]["counts"].items())
        form_counts.sort(key = lambda tupl: len(tupl[0].split("_")), reverse = True)
        for (comp_form, count) in form_counts:
            mask_list = [masks[form] for form in comp_form.split("_")]
            comp_mask = functools.reduce(np.logical_and, mask_list, True)
            comp_mask = comp_mask & (met["platform"] == "patchseq")
            comp_mask = comp_mask & ~np.isin(met["specimen_id"], used_specimens)
            valid_specimens = met["specimen_id"][comp_mask]
            if valid_specimens.size < count:
                raise RuntimeError(f"Not enough cells to generate decoupled forms.")
            specimens = rng.choice(valid_specimens, count, replace = False)
            used_specimens = np.concatenate([used_specimens, specimens])
            for (form, data_list) in data.items():
                raw = met.get_specimens(specimens)[form]
                if form not in comp_form:
                    raw = np.full_like(raw, np.nan)
                data_list.append(raw)
        data_arrays = {form: np.concatenate(data_list) for (form, data_list) in data.items()}
        all_data = {**met.get_specimens(used_specimens), **data_arrays, "specimen_id": used_specimens}
        return all_data

    def __getitem__(self, id_str):
        data = self.MET[id_str]
        return data
    
    def keys(self):
        return (key for key in self.MET.keys() if key not in self.meta)
    
    def values(self):
        return (self[key] for key in self.MET if key not in self.meta)
    
    def items(self):
        return ((key, self[key]) for key in self.MET if key not in self.meta)

    def query(self, specimen_ids = None, formats = None, exclude_formats = None, platforms = None, classes = None):
        specimen_ids = (self["specimen_id"] if specimen_ids is None else specimen_ids)
        platforms = (np.char.strip(np.unique(self["platform"])) if platforms is None else platforms)
        classes = (np.char.strip(np.unique(self["class"])) if classes is None else classes)
        valid = np.isin(np.char.strip(self["specimen_id"]), np.char.strip(specimen_ids))
        valid = valid & np.isin(np.char.strip(self["platform"]), platforms)
        valid = valid & np.isin(np.char.strip(self["class"]), classes)
        if formats is not None:
            format_mask = np.full_like(valid, False)
            for form_tuple in formats:
                tupl_mask = np.full_like(valid, True)
                for form in form_tuple:
                    data = np.squeeze(self[form])
                    tupl_mask = tupl_mask & ~np.isnan(data).reshape([data.shape[0], -1]).all(1)
                format_mask = format_mask | tupl_mask
            valid = valid & format_mask
        if exclude_formats is not None:
            exclude_mask = np.full_like(valid, True)
            for form in exclude_formats:
                data = np.squeeze(self[form])
                exclude_mask = exclude_mask & np.isnan(data).reshape([data.shape[0], -1]).all(1)
            valid = valid & exclude_mask
        valid_specimens = self["specimen_id"][valid]
        data_dict = self.get_specimens(valid_specimens)
        return data_dict

    def get_specimens(self, specimen_ids):
        stripped = [string.strip() for string in specimen_ids]
        data_dict = {}
        for (key, value) in self.items():
            cleaned = [np.squeeze(value)[None, self.id_map[spec]] for spec in stripped]
            data_dict[key] = np.concatenate(cleaned) if cleaned else value[:0]
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
    
    def get_stratified_KFold(self, folds, seed = 42):
        strat_cats = ["platform", "class", "cluster_label"]
        labels = functools.reduce(np.char.add, [np.char.strip(self[cat]) for cat in strat_cats])
        splitter = StratifiedKFold(folds, shuffle = True, random_state = seed)
        for (train_ids, test_ids) in splitter.split(self["specimen_id"], labels):
            (train_spec, test_spec) = (self["specimen_id"][train_ids], self["specimen_id"][test_ids])
            yield (train_spec, test_spec)

class DeterministicDataset(IterableDataset):
    def __init__(self, met_data, batch_size, modal_formats, modal_frac, transformations, allowed_specimen_ids = None):
        self.MET = met_data
        self.allowed_specimen_ids = (self.MET["specimen_id"] if allowed_specimen_ids is None else allowed_specimen_ids)
        if transformations:
            self.transform = {form: get_transformation_function(transform_dict) 
                            for (form, transform_dict) in transformations.items()}
        else:
            self.transform = {}
        (self.data, self.modal_masks) = self.get_data(modal_formats, self.transform)
        indices = self.filter_by_modal(modal_frac, self.modal_masks)
        num_batches = max(indices.size // batch_size, 1)
        self.batch_indices = [indices[i::num_batches] for i in range(num_batches)]

    def filter_by_modal(self, modal_frac, modal_masks):
        indices = np.arange(self.allowed_specimen_ids.size)
        cuml_mask = np.full_like(indices, False, "bool")
        for (modal_string, frac) in modal_frac.items():
            if frac == "native" or frac > 0:
                masks = [modal_masks[modal] for modal in modal_string]
                combined_mask = functools.reduce(np.logical_and, masks, np.full_like(cuml_mask, True))
                cuml_mask = cuml_mask | combined_mask
        filtered_indices = indices[cuml_mask]
        return filtered_indices

    def get_data(self, modal_formats, transform):
        (data, masks) = ({}, {})
        allowed_data = self.MET.query(self.allowed_specimen_ids)
        for (modal, formats) in modal_formats.items():
            raw_data = {form: allowed_data[form] for form in formats}
            transformed = {form: transf_func(raw_data[form]) for (form, transf_func) in transform.items()
                           if form in raw_data}
            data[modal] = {**raw_data, **transformed}
            num_cells = self.allowed_specimen_ids.size
            mask = np.full([num_cells], True)
            for array in raw_data.values():
                mask = mask & ~np.isnan(array.reshape([num_cells, -1])).all(1)
            masks[modal] = mask
        return (data, masks)
    
    def __len__(self):
        return len(self.batch_indices)

    def __iter__(self):
        for indices in self.batch_indices:
            specimen_ids = self.allowed_specimen_ids[indices]
            data = {modal: {form: array[indices] for (form, array) in formats.items()}
                    for (modal, formats) in self.data.items()}
            masks = {modal: mask[indices] for (modal, mask) in self.modal_masks.items()}
            outputs = (data, masks, specimen_ids)
            yield outputs

class RandomizedDataset(IterableDataset):
    def __init__(self, met_data, batch_size, modal_formats, modal_frac, transformations, allowed_specimen_ids = None):
        self.MET = met_data
        self.allowed_specimen_ids = (self.MET["specimen_id"] if allowed_specimen_ids is None else allowed_specimen_ids)
        if transformations:
            self.transform = {form: get_transformation_function(transform_dict) 
                            for (form, transform_dict) in transformations.items()}
        else:
            self.transform = {}
        self.data = self.get_data(modal_formats, self.transform)
        (self.modal_indices, self.modal_masks) = self.get_modal_indices(self.data, self.allowed_specimen_ids)
        (self.repeaters, self.counts) = self.get_repeaters(batch_size, modal_frac)
        self.num_batches = self.get_num_batches()

    def get_data(self, modal_formats, transform):
        data = {}
        for (modal, formats) in modal_formats.items():
            raw_data = {form: self.MET[form] for form in formats}
            transformed = {form: transf_func(raw_data[form]) for (form, transf_func) in transform.items()
                           if form in raw_data}
            data[modal] = {**raw_data, **transformed}
        return data

    def get_modal_indices(self, data, allowed_specimen_ids):
        num_cells = self.MET["specimen_id"].size
        allowed = np.isin(np.char.strip(self.MET["specimen_id"]), np.char.strip(allowed_specimen_ids))
        modal_masks = {}
        for (modal, formats) in data.items():
            mask = np.full([num_cells], True)
            for array in formats.values():
                mask = mask & ~np.isnan(array.reshape([num_cells, -1])).all(1)
            modal_masks[modal] = mask
        modal_sets = powerset(modal_masks.keys())
        set_masks = {}
        for modal_set in modal_sets:
            masks = [(mask if modal in modal_set else ~mask) 
                     for (modal, mask) in modal_masks.items()]
            set_masks[modal_set] = functools.reduce(operator.and_, masks, allowed)
        all_indices = np.arange(num_cells)
        modal_set_indices = {modal_set: all_indices[mask] for (modal_set, mask) in set_masks.items()}
        return (modal_set_indices, modal_masks)

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
        modal_frac = {frozenset(key): value for (key, value) in modal_frac.items()
                      if frozenset(key) in self.modal_indices}
        given_frac = {modal: frac for (modal, frac) in modal_frac.items() if frac != "native"}
        given_cuml = sum([frac for frac in given_frac.values()])
        if given_cuml > 1:
            raise ValueError(f"Modal fractions provided sum to {given_cuml} > 1.")
        native_modal = {modal for (modal, frac) in modal_frac.items() if frac == "native"}
        native_counts = {modal: len(self.modal_indices[modal]) for modal in native_modal}
        native_cuml = sum(native_counts.values())
        scaled_frac = {modal: count*(1-given_cuml)/native_cuml for (modal, count) in native_counts.items()}
        process_frac = {**given_frac, **scaled_frac}
        return process_frac
    
    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            indices = []
            for (modal, repeating_index) in self.repeaters.items():
                indices += repeating_index.get(self.counts[modal])
            specimen_ids = self.MET["specimen_id"][indices]
            data = {modal: {form: array[indices] for (form, array) in formats.items()} 
                    for (modal, formats) in self.data.items()}
            masks = {modal: mask[indices] for (modal, mask) in self.modal_masks.items()}
            outputs = (data, masks, specimen_ids)
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

def binarize(threshold):
    def transform(data):
        binarized = (data > threshold).astype(data.dtype)
        binarized[np.isnan(data)] = np.nan
        return binarized
    return transform

def replace_nan(replacement):
    def transform(data):
        is_nan = np.isnan(data)
        all_nan = is_nan.reshape([is_nan.shape[0], -1]).all(1, keepdims = True)
        cleaned = data.copy()
        cleaned[is_nan & ~all_nan] = replacement
        return cleaned
    return transform

def standardize(params = None):
    def transform(data):
        centered = data - np.nanmean(data, 0, keepdims = True)
        stds = np.nanstd(centered, 0, keepdims = True)
        stds[stds == 0] = 1
        scaled = centered / stds
        return scaled
    return transform

def random(params = None):
    def transform(data):
        data = data + np.random.normal(size = data.shape)
        return data
    return transform

transform_functions = {
    "binarize": binarize,
    "replace_nan": replace_nan,
    "standardize": standardize,
    "random": random
}

if __name__ == "__main__":

    config = {
        "seed": 42,
        "decouple": {
            "counts": {
                "logcpm": 611,
                "logcpm_arbors": 163,
                "logcpm_pca-ipfx": 4384,
                "logcpm_pca-ipfx_arbors": 1394
            }
        }
    }
    met_data = MET_Decoupled("data/raw/MET_full_data.npz", config)
    print(met_data.MET)
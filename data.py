import math
import functools
import zipfile
import itertools
import operator
import pathlib
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import IterableDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
import h5py

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
    formats = config["select"]["formats"]
    specimens = met_data.query(specimen_ids, formats = formats, platforms = platforms, outputs = ["specimen_id"])["specimen_id"]
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

def get_specimens_data(hdf5_files, data_keys):
    (specimens, sp_indices, data) = ({}, {}, {})
    for (form_name, file_key_pairs) in data_keys.items():
        form_specimens = []
        for (file_name, key) in file_key_pairs:
            data.setdefault(form_name, {})[file_name] = hdf5_files[file_name][key]["data"]
            file_specimens = np.char.decode(hdf5_files[file_name][key]["specimens"][:])
            form_specimens.append(file_specimens)
            file_indices = {sp_id: (file_name, i) for (i, sp_id) in enumerate(file_specimens)}
            sp_indices.setdefault(form_name, {}).update(file_indices)
        specimens[form_name] = np.concatenate(form_specimens)
    all_specimens = np.unique(np.concatenate(list(specimens.values())))
    valid = {form: np.isin(all_specimens, form_sp) for (form, form_sp) in specimens.items()}
    all_indices = {sp_id: i for (i, sp_id) in enumerate(all_specimens)}
    ind_map = {form: {all_indices[sp_id]: val for (sp_id, val) in file_ind.items()}
               for (form, file_ind) in sp_indices.items()}
    return (all_specimens, all_indices, valid, ind_map, data)

def get_meta(hdf5_files, all_specimens):
    (all_meta_data, categories) = ({}, set())
    for hdf5_file in hdf5_files:
        file_data = {key: np.char.decode(array[:]) for (key, array) in hdf5_file.get("meta", {}).items()}
        file_specimens = file_data.pop("specimens", [])
        categories.update(file_data.keys())
        for (i, sp_id) in enumerate(file_specimens):
            all_meta_data.setdefault(sp_id, {}).update({key: arr[i] for (key, arr) in file_data.items()})
    cat_meta_data = {cat: np.asarray([all_meta_data.get(sp_id, {}).get(cat, np.nan) 
                                      for sp_id in all_specimens])
                     for cat in categories}
    cat_meta_data["specimen_id"] = all_specimens
    return cat_meta_data

class Yielder():
    def __init__(self, data, id_maps, num_specimens):
        self.data_shape = next(iter(data.values())).shape[1:]
        self.cached_indices = np.full(num_specimens, num_specimens)
        self.cached_data = np.zeros((0, ) + self.data_shape)
        self.num_specimens = num_specimens
        self.data = data
        self.id_maps = id_maps

    def __call__(self, specimen_idxs):
        sp_indices = np.atleast_1d(specimen_idxs)
        cached_indices = self.cached_indices[sp_indices]
        is_cached = cached_indices < self.num_specimens
        samples = np.zeros(sp_indices.shape + self.data_shape)
        samples[is_cached] = self.cached_data[cached_indices[is_cached]]
        if np.any(~is_cached):
            samples[~is_cached] = self._get_uncached(sp_indices[~is_cached])
        return samples
    
    def __getitem__(self, specimen_idxs):
        return self(specimen_idxs)

    def _get_uncached(self, specimen_idxs):
        form_mapping = [self.id_maps.get(global_idx, (None, None)) for global_idx in specimen_idxs]
        uncached_data = [self.data[file_name][idx] if file_name else np.full(self.data_shape, np.nan) 
                         for (file_name, idx) in form_mapping]
        return np.stack(uncached_data, 0)

    def cache_data(self, specimen_idxs):
        self.cached_data = self._get_uncached(specimen_idxs)
        self.cached_indices = np.full(self.num_specimens, self.num_specimens)
        self.cached_indices[specimen_idxs] = np.arange(specimen_idxs.size)           

class MET_Data():
    def __init__(self, hdf5_paths, **data_keys):
        hdf5_files = {name: h5py.File(path, "r") for (name, path) in hdf5_paths.items()}
        (self.specimens, self.id_map, self.valid, self.local_id_map, self.data) = get_specimens_data(hdf5_files, data_keys)
        self._meta = get_meta(hdf5_files.values(), self.specimens)
        self._data_funcs = {form: Yielder(self.data[form], self.local_id_map[form], len(self.specimens)) for form in data_keys}

    def __getitem__(self, id_str):
        if id_str in self._meta:
            value = self._meta[id_str]
        elif id_str in self._data_funcs:
            value = self._data_funcs[id_str]
        else:
            raise KeyError(f'Key "{id_str}" not found.')
        return value

    def keys(self):
        return itertools.chain(self._meta.keys(), self._data_funcs.keys())
    
    def values(self):
        return itertools.chain(self._meta.values(), self._data_funcs.values())
    
    def items(self):
        return itertools.chain(self._meta.items(), self._data_funcs.items())

    def query(self, specimen_ids = None, formats = None, exclude_formats = None, platforms = None, classes = None, outputs = None):
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
                    tupl_mask = tupl_mask & self.valid[form]
                format_mask = format_mask | tupl_mask
            valid = valid & format_mask
        if exclude_formats is not None:
            exclude_mask = np.full_like(valid, True)
            for form in exclude_formats:
                exclude_mask = exclude_mask & ~self.valid[form]
            valid = valid & exclude_mask
        valid_specimens = self["specimen_id"][valid]
        outputs = self.keys() if outputs is None else outputs
        data_dict = self.get_specimens(valid_specimens, outputs)
        return data_dict
    
    def get_specimens(self, specimen_ids, outputs = None):
        outputs = self.keys() if outputs is None else outputs
        indices = [self.id_map[sp_id.strip()] for sp_id in specimen_ids]
        data_dict = {key: value[indices] for (key, value) in self.items()
                     if key in outputs}
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

    def cache_data(self, form, specimen_ids = None, verbose = True):
        specimen_ids = self.specimens if specimen_ids is None else specimen_ids
        specimen_idxs = np.asarray([self.id_map[sp_id.strip()] for sp_id in specimen_ids])
        if verbose:
            print(f"Caching {form}...")
        self._data_funcs[form].cache_data(specimen_idxs)

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
    def __init__(self, hdf5_path, counts, seed, platforms, **data_keys):
        hdf5 = h5py.File(hdf5_path)
        self.specimens = np.char.decode(hdf5["specimens"][:])
        self.id_map = {sp_id.strip():i for (i, sp_id) in enumerate(self.specimens)}
        self.data = {form: hdf5["modalities"][key] for (form, key) in data_keys.items()}
        self._meta = {key: np.char.decode(value) for (key, value) in hdf5["meta"].items()}
        self._other = {key: np.char.decode(value) for (key, value) in hdf5["other"].items()}
        self._data_funcs = {name: Yielder(name, self) for name in self.data}
        self._cached_indices = {name: np.full(self.specimens.size, self.specimens.size) for name in self.data}
        self._cached_data = {name: np.zeros((0, ) + array.shape[1:]) for (name, array) in self.data.items()}

        orig_valid = {form: hdf5["valid"][key][:] for (form, key) in data_keys.items()}
        self.valid = self.get_decoupled_valid(orig_valid, counts, seed, platforms)

    def get_decoupled_valid(self, orig_valid, counts, seed, platforms):
        rng = np.random.default_rng(seed)
        used_specimens = self.specimens[:0]
        form_counts = list(counts.items())
        form_counts.sort(key = lambda tupl: len(tupl[0].split("_")), reverse = True)
        decoupled_valid = orig_valid.copy()
        for (comp_form, count) in form_counts:
            mask_list = [orig_valid[form] for form in comp_form.split("_")]
            comp_mask = functools.reduce(np.logical_and, mask_list, True)
            if platforms:
                comp_mask = comp_mask & np.isin(self._meta["platform"], platforms)
            comp_mask = comp_mask & ~np.isin(self.specimens, used_specimens)
            valid_indices = np.arange(comp_mask.size)[comp_mask]
            if valid_indices.size < count:
                raise RuntimeError(f"Not enough cells to generate decoupled forms.")
            chosen_indices = rng.choice(valid_indices, count, replace = False)
            used_specimens = np.concatenate([used_specimens, self.specimens[chosen_indices]])
            for (form, is_valid) in decoupled_valid.items():
                if form not in comp_form:
                    is_valid[chosen_indices] = False
        not_used = ~np.isin(self.specimens, used_specimens)
        for (form, is_valid) in decoupled_valid.items():
            is_valid[not_used] = False
        return decoupled_valid

    def __getitem__(self, id_str):
        if id_str in self._meta:
            value = self._meta[id_str]
        elif id_str in self._data_funcs:
            value = self._data_funcs[id_str]
        elif id_str in self._other:
            value = self._other[id_str]
        else:
            raise KeyError(f'Key "{id_str}" not found.')
        return value

    def keys(self):
        return itertools.chain(self._meta.keys(), self._data_funcs.keys())
    
    def values(self):
        return itertools.chain(self._meta.values(), self._data_funcs.values())
    
    def items(self):
        return itertools.chain(self._meta.items(), self._data_funcs.items())

    def query(self, specimen_ids = None, formats = None, exclude_formats = None, platforms = None, classes = None, outputs = None):
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
                    tupl_mask = tupl_mask & self.valid[form]
                format_mask = format_mask | tupl_mask
            valid = valid & format_mask
        if exclude_formats is not None:
            exclude_mask = np.full_like(valid, True)
            for form in exclude_formats:
                exclude_mask = exclude_mask & ~self.valid[form]
            valid = valid & exclude_mask
        valid_specimens = self["specimen_id"][valid]
        outputs = self.keys() if outputs is None else outputs
        data_dict = self.get_specimens(valid_specimens, outputs)
        return data_dict
    
    def get_specimens(self, specimen_ids, outputs = None):
        outputs = self.keys() if outputs is None else outputs
        indices = [self.id_map[sp_id.strip()] for sp_id in specimen_ids]
        data_dict = {key: value[indices] for (key, value) in self.items()
                     if key in outputs}
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

    def cache_data(self, dataset_name, specimen_ids = None, verbose = True):
        specimen_ids = self.specimens if specimen_ids is None else specimen_ids
        sp_indices = np.asarray([self.id_map[sp_id.strip()] for sp_id in specimen_ids])
        print(f"Caching {dataset_name}...")
        sorted_indices = np.sort(sp_indices)
        data_array = self.data[dataset_name][sorted_indices]
        self._cached_data[dataset_name] = data_array
        self._cached_indices[dataset_name] = np.full(self.specimens.size, self.specimens.size)
        self._cached_indices[dataset_name][sorted_indices] = np.arange(specimen_ids.size)

class DeterministicDataset(IterableDataset):
    def __init__(self, met_data, batch_size, modal_formats, modal_frac, transformations, unpack, allowed_specimen_ids = None):
        self.MET = met_data
        self.allowed_specimen_ids = (self.MET["specimen_id"] if allowed_specimen_ids is None else allowed_specimen_ids)
        self.allowed_specimen_indices = np.asarray([self.MET.id_map[sp_id.strip()] for sp_id in self.allowed_specimen_ids])
        if transformations:
            self.transform = {form: get_transformation_function(transform_dict) 
                            for (form, transform_dict) in transformations.items()}
        else:
            self.transform = {}
        (self.data_funcs, self.modal_masks) = self.get_data_funcs(modal_formats, self.transform)
        indices = self.filter_by_modal(modal_frac, self.modal_masks)
        num_batches = max(indices.size // batch_size, 1)
        self.batch_indices = [indices[i::num_batches] for i in range(num_batches)]
        unpack_counts = [(form, met_data.data[form].shape[1]) for (form, is_packed) in unpack.items() if is_packed]
        self.unpack_forms = [form for (form, _) in unpack_counts]
        self.unpack_indices = list(itertools.product(*[range(count) for (_, count) in unpack_counts]))

    def filter_by_modal(self, modal_frac, modal_masks):
        cuml_mask = np.full_like(self.MET["specimen_id"], False, "bool")
        for (modal_string, frac) in modal_frac.items():
            if frac == "native" or frac > 0:
                masks = [(mask if modal in modal_string else ~mask) 
                         for (modal, mask) in modal_masks.items()]
                # masks = [modal_masks[modal] for modal in modal_string]
                combined_mask = functools.reduce(np.logical_and, masks, np.full_like(cuml_mask, True))
                cuml_mask = cuml_mask | combined_mask
        filtered_indices = self.allowed_specimen_indices[cuml_mask[self.allowed_specimen_indices]]
        return filtered_indices

    def get_data_funcs(self, modal_formats, transform):
        (data_funcs, masks) = ({}, {})
        for (modal, formats) in modal_formats.items():
            for form in formats:
                data_func = self.MET[form]
                if form in transform:
                    transf_func = transform[form]
                    def data_func(indices, transf = transf_func, raw = data_func):
                        return transf(raw(indices))
                data_funcs.setdefault(modal, {})[form] = data_func
            mask = np.full_like(self.MET["specimen_id"], True, bool)
            for form in formats:
                mask = mask & self.MET.valid[form]
            masks[modal] = mask
        return (data_funcs, masks)
    
    def unpack_output(self, data):
        if not self.unpack_forms:
            yield data
        else:
            for indices in self.unpack_indices:
                unpacked_data = {modal: {form: data[modal][form][:, index] for (form, index) in zip(self.unpack_forms, indices)
                                         if form in modal_forms}
                                 for (modal, modal_forms) in data.items()}
                unpacked_data = {modal: {**data[modal], **unpack_dict} for (modal, unpack_dict) in unpacked_data.items()}     
                yield unpacked_data

    def __len__(self):
        return len(self.batch_indices) * len(self.unpack_indices)

    def __iter__(self):
        for indices in self.batch_indices:
            specimen_ids = self.MET["specimen_id"][indices]
            data = {modal: {form: func(indices) for (form, func) in formats.items()}
                    for (modal, formats) in self.data_funcs.items()}
            masks = {modal: mask[indices] for (modal, mask) in self.modal_masks.items()}
            for unpacked_data in self.unpack_output(data):
                yield (unpacked_data, masks, specimen_ids)

class RandomizedDataset(IterableDataset):
    def __init__(self, met_data, batch_size, modal_formats, modal_frac, transformations, unpack, allowed_specimen_ids = None):
        self.MET = met_data
        self.allowed_specimen_ids = (self.MET["specimen_id"] if allowed_specimen_ids is None else allowed_specimen_ids)
        if transformations:
            self.transform = {form: get_transformation_function(transform_dict) 
                            for (form, transform_dict) in transformations.items()}
        else:
            self.transform = {}
        self.data_funcs = self.get_data_funcs(modal_formats, self.transform)
        (self.modal_indices, self.modal_masks) = self.get_modal_indices(modal_formats, self.allowed_specimen_ids)
        (self.repeaters, self.counts) = self.get_repeaters(batch_size, modal_frac)
        self.num_batches = self.get_num_batches()
        unpack_counts = [(form, met_data.data[form].shape[1]) for (form, is_packed) in unpack.items() if is_packed]
        self.unpack_forms = [form for (form, _) in unpack_counts]
        self.unpack_indices = list(itertools.product(*[range(count) for (_, count) in unpack_counts]))

    def get_data_funcs(self, modal_formats, transform):
        data_funcs = {}
        for (modal, formats) in modal_formats.items():
            for form in formats:
                data_func = self.MET[form]
                if form in transform:
                    transf_func = transform[form]
                    def data_func(indices, transf = transf_func, raw = data_func):
                        return transf(raw(indices))
                data_funcs.setdefault(modal, {})[form] = data_func
        return data_funcs

    def get_modal_indices(self, modal_formats, allowed_specimen_ids):
        num_cells = self.MET["specimen_id"].size
        allowed = np.isin(np.char.strip(self.MET["specimen_id"]), np.char.strip(allowed_specimen_ids))
        modal_masks = {}
        for (modal, formats) in modal_formats.items():
            mask = np.full([num_cells], True)
            for form in formats:
                mask = mask & self.MET.valid[form]
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
    
    def unpack_output(self, data):
        if not self.unpack_forms:
            yield data
        else:
            for indices in self.unpack_indices:
                unpacked_data = {modal: {form: data[modal][form][:, index] for (form, index) in zip(self.unpack_forms, indices)
                                         if form in modal_forms}
                                 for (modal, modal_forms) in data.items()}
                unpacked_data = {modal: {**data[modal], **unpack_dict} for (modal, unpack_dict) in unpacked_data.items()}     
                yield unpacked_data

    def __len__(self):
        return self.num_batches * len(self.unpack_indices)

    def __iter__(self):
        for _ in range(self.num_batches):
            indices = []
            for (modal, repeating_index) in self.repeaters.items():
                indices += repeating_index.get(self.counts[modal])
            specimen_ids = self.MET["specimen_id"][indices]
            data = {modal: {form: func(indices) for (form, func) in formats.items()} 
                    for (modal, formats) in self.data_funcs.items()}
            masks = {modal: mask[indices] for (modal, mask) in self.modal_masks.items()}
            for unpacked_data in self.unpack_output(data):
                yield (unpacked_data, masks, specimen_ids)

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

    # config = {
    #     "seed": 42,
    #     "decouple": {
    #         "counts": {
    #             "logcpm": 611,
    #             "logcpm_arbors": 163,
    #             "logcpm_pca-ipfx": 4384,
    #             "logcpm_pca-ipfx_arbors": 1394
    #         }
    #     }
    # }
    # met_data = MET_Decoupled("data/raw/MET_full_data.npz", config)
    # print(met_data.MET)

    import yaml

    with open("configs/config_template.yaml", "r") as target:
        data_config = yaml.safe_load(target)["data_config"]

    data_keys = {key: dct["keys"] for (key, dct) in data_config["formats"].items()}
    met = MET_Data(data_config["data_paths"], **data_keys)
    for form in ["logcpm", "pca-ipfx", "arbors"]:
        met.cache_data(form)
    sp_ids = met["specimen_id"]

    specimens = met.query(platforms = ["patchseq"], outputs = ["specimen_id"])["specimen_id"]
    
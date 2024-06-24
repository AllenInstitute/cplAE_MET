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
    specimens = met_data.query(specimen_ids, platforms = platforms, outputs = ["specimen_id"])["specimen_id"]
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

class Yielder():
    def __init__(self, dataset_name, met_instance):
        self.name = dataset_name
        self.met = met_instance

    def __call__(self, sp_indices):
        sp_indices = np.atleast_1d(sp_indices)
        cached_indices = self.met._cached_indices[self.name][sp_indices]
        cached_data = self.met._cached_data[self.name]
        index_bound = self.met["specimen_id"].size
        if np.all(cached_indices < index_bound):
            return cached_data[cached_indices]
        else:
            samples = []
            for (index, cache_index) in zip(sp_indices, cached_indices):
                if cache_index < index_bound:
                    samples.append(cached_data[cache_index])
                else:
                    samples.append(self.met._load_or_nan(self.name, index))
            return np.stack(samples, 0)
    
    def __getitem__(self, indices):
        return self(indices)        

class MET_Data():
    def __init__(self, specimens_path, **dataset_folders):
        dataset_folders = {name: pathlib.Path(path) for (name, path) in dataset_folders.items()}
        spec_data = np.loadtxt(specimens_path, str, delimiter = ",")
        sp_ids = spec_data[:, 0]
        self.id_map = {sp_id.strip():i for (i, sp_id) in enumerate(sp_ids)}
        self.data_paths = dataset_folders
        self.data_shapes = {name: self._get_data_shape(folder)
                            for (name, folder) in dataset_folders.items()}
        self.valid = {name: self._get_valid(sp_ids, folder)
                      for (name, folder) in dataset_folders.items()}
        self._meta = {
            "specimen_id": sp_ids,
            "platform": spec_data[:, 1],
            "class": spec_data[:, 2],
            "cluster_label": spec_data[:, 3],
        }
        self._data_funcs = {name: Yielder(name, self) for name in dataset_folders}
        self._cached_indices = {name: np.full(self["specimen_id"].size, self["specimen_id"].size) for name in dataset_folders}
        self._cached_data = {name: np.zeros((0, ) + self.data_shapes[name]) for name in dataset_folders}

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

    def cache_data(self, dataset_name, specimen_ids = None, verbose = True):
        specimen_ids = self["specimen_id"] if specimen_ids is None else specimen_ids
        sp_indices = np.asarray([self.id_map[sp_id.strip()] for sp_id in specimen_ids])
        dataset_shape = sp_indices.shape + self.data_shapes[dataset_name]
        data_array = np.zeros(dataset_shape)
        print(f"Caching {dataset_name}...")
        for (i, sp_index) in enumerate(tqdm(sp_indices, miniters = 1, disable = not verbose)):
            data_array[i] = self._load_or_nan(dataset_name, sp_index)
        self._cached_data[dataset_name] = data_array
        self._cached_indices[dataset_name] = np.full(self["specimen_id"].size, self["specimen_id"].size)
        self._cached_indices[dataset_name][sp_indices] = np.arange(specimen_ids.size) 

    def _get_valid(self, sp_ids, data_path):
        valid = np.array([path.stem for path in data_path.iterdir()])
        sp_ids_valid = np.isin(sp_ids, valid)
        return sp_ids_valid
    
    def _get_data_shape(self, data_path):
        example = np.load(next(data_path.iterdir()))
        return example.shape

    def _load_or_nan(self, dataset_name, index):
        data_shape = self.data_shapes[dataset_name]
        if self.valid[dataset_name][index]:
            sp_id = self["specimen_id"][index]
            path = self.data_paths[dataset_name] / f"{sp_id}.npy"
            data = np.load(path)
        else:
            data = np.full(data_shape, np.nan)
        return data
    
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
        self.meta = orig_MET.meta

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
        try:
            next(splitter.split(self["specimen_id"], labels))
            fold_iter = splitter.split(self["specimen_id"], labels)
        except ValueError:
            print("Stratification failed. Using un-stratified folds.")
            fold_iter = splitter.split(self["specimen_id"], np.ones_like(labels))
        for (train_ids, test_ids) in fold_iter:
            (train_spec, test_spec) = (self["specimen_id"][train_ids], self["specimen_id"][test_ids])
            yield (train_spec, test_spec)

class DeterministicDataset(IterableDataset):
    def __init__(self, met_data, batch_size, modal_formats, modal_frac, transformations, allowed_specimen_ids = None):
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
    
    def __len__(self):
        return len(self.batch_indices)

    def __iter__(self):
        for indices in self.batch_indices:
            specimen_ids = self.MET["specimen_id"][indices]
            data = {modal: {form: func(indices) for (form, func) in formats.items()}
                    for (modal, formats) in self.data_funcs.items()}
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
        self.data_funcs = self.get_data_funcs(modal_formats, self.transform)
        (self.modal_indices, self.modal_masks) = self.get_modal_indices(modal_formats, self.allowed_specimen_ids)
        (self.repeaters, self.counts) = self.get_repeaters(batch_size, modal_frac)
        self.num_batches = self.get_num_batches()

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
    
    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            indices = []
            for (modal, repeating_index) in self.repeaters.items():
                indices += repeating_index.get(self.counts[modal])
            specimen_ids = self.MET["specimen_id"][indices]
            data = {modal: {form: func(indices) for (form, func) in formats.items()} 
                    for (modal, formats) in self.data_funcs.items()}
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

    fracs = {
        "T": "native",
        "E": "native",
        "M": "native",
        "TE": "native",
        "TM": "native",
        "EM": "native",
        "MET": "native"
    }

    dataset_folders = {
        "logcpm": "data/transcriptomics",
        "pca-ipfx": "data/electrophysiology",
        "arbors": "data/morphology/densities/120_4_4_old"
    }
    met = MET_Data("data/meta/specimens.csv", **dataset_folders)
    for form in ["logcpm", "pca-ipfx", "arbors"]:
        met.cache_data(form)
    input()
    specimens = met.query(platforms = ["patchseq"], outputs = ["specimen_id"])["specimen_id"]
    data_iter = DeterministicDataset(met, 128, {"T": ["logcpm"], "E": ["pca-ipfx"], "M": ["arbors"]}, 
                                  fracs, {}, specimens)
    next(iter(data_iter))
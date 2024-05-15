import pathlib
import yaml
import pickle as pk

import numpy as np
import pandas as pd
import torch

from pca_cca import PCA_CCA

project_dir = pathlib.Path("/Users/ian.convy/code/cplAE_MET")

class VariationalWrapper(torch.nn.Module):
    def __init__(self, model_dict, mappers, decoder_cov):
        super().__init__()
        self.mappers = mappers
        self.model_dict = model_dict
        for (modal, arm) in model_dict.items():
            setattr(self, f"{modal}_enc", arm["enc"])
            setattr(self, f"{modal}_dec", arm["dec"])
        self.decoder_cov = decoder_cov

    def forward(self, x_forms, in_modal, out_modals):
        orig_latent = self[in_modal]["enc"](x_forms)[0]
        outputs = {}
        for modal in out_modals:
            if self.mappers and modal != in_modal:
                latent = self.mappers[f"{in_modal}-{modal}"](orig_latent)[0]
            else:
                latent = orig_latent
            outputs[modal] = self[modal]["dec"](latent)
        return outputs

    def infer(self, x_forms, in_modal, out_modals, sample = True):
        (mean, transf) = self[in_modal]["enc"](x_forms)
        outputs = {}
        for modal in out_modals:
            out_mean = out_transf = None
            latent = self.z_sample(mean, transf) if sample else mean
            orig_sample = latent
            if self.mappers and modal != in_modal:
                (out_mean, out_transf) = self.mappers[f"{in_modal}-{modal}"](orig_sample)
                latent = self.z_sample(out_mean, out_transf) if sample else out_mean
            recon = self[modal]["dec"](latent)
            outputs[modal] = (latent, recon, mean, transf, out_mean, out_transf, orig_sample)
        return outputs

    def z_sample(self, mean, transf, num_samples):
        expanded_mean = mean[:, None].expand(-1, num_samples, -1)
        noise = torch.einsum("nij,nsj->nsi", transf, torch.randn_like(expanded_mean))
        z_sampled = expanded_mean + noise
        return z_sampled
    
    def cross_z_sample(self, in_modal, out_modal, in_mean, in_transf, num_samples):
        mapper = self.mappers[f"{in_modal}-{out_modal}"]
        direct_sample = self.z_sample(in_mean, in_transf, 1)[:, 0]
        (out_mean, out_transf) = mapper(direct_sample)
        cross_sample = self.z_sample(out_mean, out_transf, num_samples)
        return (direct_sample, cross_sample, out_mean, out_transf)

    def __getitem__(self, modal):
        return self.model_dict[modal]
    
    def keys(self):
        return self.model_dict.keys()
    
    def values(self):
        return self.model_dict.values()
    
    def items(self):
        return self.model_dict.items()
    
class CouplerWrapper(torch.nn.Module):
    def __init__(self, model_dict):
        super().__init__()
        self.model_dict = model_dict
        for (modal_str, module) in model_dict.items():
            setattr(self, modal_str.replace("-", "_"), module)
        self.modal_arms = {}

    def forward(self, x_forms, in_modal, out_modals):
        if not self.modal_arms:
            raise RuntimeError("Auto-encoders must be loaded to generate reconstructions.")
        latent = self.modal_arms[in_modal]["enc"](x_forms)[0]
        outputs = {}
        for modal in out_modals:
            if modal != in_modal:
                transf_latent = self[f"{in_modal}-{modal}"](latent)
                outputs[modal] = self.modal_arms[modal]["dec"](transf_latent)
            else:
                outputs[modal] = self.modal_arms[modal]["dec"](latent)
        return outputs 

    def load_autoencoder(self, model_dict):
        self.modal_arms = model_dict

    def __getitem__(self, modal):
        return self.model_dict[modal]
    
    def keys(self):
        return self.model_dict.keys()
    
    def values(self):
        return self.model_dict.values()
    
    def items(self):
        return self.model_dict.items()

def assemble_jit(jit_path, device = "cpu"):
    jit_path = pathlib.Path(jit_path)
    model = {}
    modalities = [path.stem for path in (jit_path / "encoder").iterdir()]
    for modal in modalities:
        model[modal] = {}
        model[modal]["enc"] = torch.jit.load(jit_path / "encoder" / f"{modal}.pt", map_location = device)
        model[modal]["dec"] = torch.jit.load(jit_path / "decoder" / f"{modal}.pt", map_location = device)
    mapper_path = jit_path / "mapper"
    mappers = {path.stem: torch.jit.load(path, device) for path in mapper_path.iterdir()} if mapper_path.exists() else None
    model = VariationalWrapper(model, mappers, None)
    return model

def assemble_coupler(jit_path, wrap = True, device = "cpu"):
    trace_paths = list(pathlib.Path(jit_path).iterdir())
    model = {path.stem: torch.jit.load(path, map_location = device)
             for path in trace_paths}
    if wrap:
        model = CouplerWrapper(model)
    return model

def save_trace(path, model, config, dataset):
    path = pathlib.Path(path)
    encoder_path = path / "encoder"
    decoder_path = path / "decoder"
    encoder_path.mkdir(parents = True)
    decoder_path.mkdir(parents = True)
    if model.mappers:
        mapper_path = path / "mapper"
        mapper_path.mkdir(parents = True)
    was_training = model.training
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for (modal, arm) in model.items():
            encoder_input = {}
            for form in config["formats"][modal]:
                raw_data = torch.from_numpy(dataset.MET[form][:1])
                encoder_input[form] = torch.nan_to_num(raw_data).to(device, dtype = torch.float32)
            encoder_trace = torch.jit.trace(arm["enc"], encoder_input, strict = False)
            decoder_input = encoder_trace(encoder_input)[0]
            decoder_trace = torch.jit.trace(arm["dec"], decoder_input, strict = False)
            encoder_trace.save(encoder_path /  f"{modal}.pt")
            decoder_trace.save(decoder_path / f"{modal}.pt")
            if model.mappers:
                for (modal_string, mapper) in model.mappers.items():
                    if modal_string[0] == modal:
                        mapper_trace = torch.jit.trace(mapper, decoder_input, strict = False)
                        mapper_trace.save(mapper_path / f"{modal_string}.pt")
            if model.decoder_cov:
                cov_trace = torch.jit.trace(model.decoder_cov, tuple())
                cov_trace.save(path / f"decoder_cov.pt")
    if was_training:
        model.train()

def load_jit_folds(exp_path, folds = None, get_checkpoints = False, check_step = 1):
    exp_path = pathlib.Path(exp_path)
    results = {}
    with open(exp_path / "config.yaml", "r") as target:
        results["config"] = yaml.safe_load(target)
    fold_paths = list(exp_path.glob("fold_*"))
    fold_paths.sort(key = lambda path: int(path.stem.split("_")[-1]))
    if folds is not None:
        fold_paths = [path for path in fold_paths if int(path.name.split("_")[-1]) in folds]
    results["folds"] = {}
    for fold_path in fold_paths:
        fold = int(fold_path.stem.split("_")[-1])
        info_dict = {}
        specimen_ids = np.load(fold_path / "train_test_ids.npz")
        info_dict["train_ids"] = np.char.strip(specimen_ids["train"])
        info_dict["test_ids"] = np.char.strip(specimen_ids["test"])
        info_dict["best"] = assemble_jit(fold_path / "best")
        if get_checkpoints:
            info_dict["checkpoints"] = {}
            paths = list((fold_path / "checkpoints").iterdir())
            paths.sort(key = lambda path: int(path.stem.split("_")[-1]))
            for path in paths[::check_step]:
                epoch = int(path.stem.split("_")[-1])
                state = assemble_jit(path)
                info_dict["checkpoints"][epoch] = state
        results["folds"][fold] = info_dict
    return results

def load_coupler_folds(exp_path, folds = None, get_checkpoints = False):
    exp_path = pathlib.Path(exp_path)
    results = {}
    with open(exp_path / "config.yaml", "r") as target:
        results["config"] = yaml.safe_load(target)
    autoencoder_paths = {modal: pathlib.Path(spec["model_path"]) / "fold_1" / "best"
                         for (modal, spec) in results["config"]["modal_specs"].items()}
    autoencoders = {modal: assemble_jit(path)[modal] 
                    for (modal, path) in autoencoder_paths.items()}
    fold_paths = list(exp_path.glob("fold_*"))
    fold_paths.sort(key = lambda path: int(path.stem.split("_")[-1]))
    if folds is not None:
        fold_paths = [path for path in fold_paths if int(path.name.split("_")[-1]) in folds]
    results["folds"] = {}
    for fold_path in fold_paths:
        fold = int(fold_path.stem.split("_")[-1])
        info_dict = {}
        specimen_ids = np.load(fold_path / "train_test_ids.npz")
        info_dict["train_ids"] = np.char.strip(specimen_ids["train"])
        info_dict["test_ids"] = np.char.strip(specimen_ids["test"])
        info_dict["best"] = assemble_coupler(fold_path / "best")
        info_dict["best"].load_autoencoder(autoencoders)
        if get_checkpoints:
            info_dict["checkpoints"] = {}
            paths = list((fold_path / "checkpoints").iterdir())
            paths.sort(key = lambda path: int(path.stem.split("_")[-1]))
            for path in paths:
                epoch = int(path.stem.split("_")[-1])
                state = assemble_coupler(path)
                state.load_autoencoder(autoencoders)
                info_dict["checkpoints"][epoch] = state
        results["folds"][fold] = info_dict
    return results

def load_pca_cca(exp_dir):
    exp_path = project_dir / "data" / exp_dir
    results = {}
    with open(exp_path / "config.yaml", "r") as target:
        results["config"] = yaml.safe_load(target)
    fold_paths = list(exp_path.glob("fold_*"))
    fold_paths.sort()
    results["folds"] = []
    for fold_path in fold_paths:
        info_dict = {}
        pca_cca = PCA_CCA(results["config"])
        pca_cca.load(fold_path)
        info_dict["model"] = pca_cca
        specimen_ids = np.load(fold_path / "train_test_ids.npz")
        info_dict["train_ids"] = specimen_ids["train"]
        info_dict["test_ids"] = specimen_ids["test"]
        results["folds"].append(info_dict)
    return results

def get_tree_merge_map(tree_csv_path, top_node):
    data = pd.read_csv(tree_csv_path)
    children = {}
    for (node, parent) in data[["label", "parent"]].itertuples(False):
        children.setdefault(parent, []).append(node)
    valid_nodes = {top_node}
    while True:
        new_valid_nodes = valid_nodes.union({node for (node, parent) in data[["label", "parent"]].itertuples(False) 
                           if parent in valid_nodes})
        if len(new_valid_nodes) == len(valid_nodes):
            break
        else:
            valid_nodes = new_valid_nodes
    data = data[data["label"].isin(valid_nodes)]
    nodes_with_children = data[data["y"] > 0].sort_values("y")
    cuml_map = {label: [label] for label in data["label"]}
    for label in nodes_with_children["label"]:
        children = data[data["parent"] == label]["label"].to_list()
        for map_outs in cuml_map.values():
            curr_out = map_outs[-1]
            map_outs.append(label if curr_out in children else curr_out)
    cuml_map = {label.strip(): [out.strip() for out in map_outs]
                for (label, map_outs) in cuml_map.items()}
    return cuml_map

def get_forest_AE(base_dir, exp_path, exp_name, merge):
    exp_dict = {
        "config": {},
        "folds": {}}
    base_dir = pathlib.Path("../data/forest_baselines")
    encoder_dict = load_jit_folds(exp_path, get_checkpoints = False) if exp_path else None
    modalities = encoder_dict["config"]["modalities"] if exp_path else ["T", "E", "M"]
    formats = encoder_dict["config"]["formats"] if exp_path else {"T": ["logcpm"], "E": ["pca-ipfx"], "M": ["arbors"]}
    tree_path = base_dir / "models" / "latent_encoders" / exp_name if exp_path else base_dir / "models" / "raw_encoders"
    for dec_fold_path in (base_dir / "models" / "decoders").iterdir():
        fold = int(dec_fold_path.name)
        if exp_path:
            encoders = {modal: encoder_dict["folds"][fold]["best"][modal]["enc"] for modal in modalities}
        else:
            encoders = {modal: lambda form_dict: (torch.flatten(next(iter(form_dict.values())), start_dim = 1),) for modal in modalities}
        trees = {}
        for modal in modalities:
            with open(tree_path / str(fold) / str(merge) / f"{modal}.pk", "rb") as target:
                trees[modal] = pk.load(target)
        means_dict = {}
        for form_path in dec_fold_path.glob("*.pk"):
            with open(form_path, "rb") as target:
                means_dict[form_path.stem] = pk.load(target)
        decoders = {form: lambda labels,means=means: np.asarray([means[label] for label in labels]) 
                    for (form, means) in means_dict.items()}
        def model(form_dict, in_modal, out_modals):
            latent = encoders[in_modal](form_dict)[0].numpy(force = True)
            labels = trees[in_modal].predict(latent)
            recons = {}
            for out_modal in out_modals:
                recons[out_modal] = {form: torch.from_numpy(decoders[form](labels)).float() for form in formats[out_modal]}
            return recons
        exp_dict["folds"].setdefault(int(fold), {})["best"] = model
        specimen_ids = np.load(dec_fold_path / "train_test_ids.npz")
        exp_dict["folds"][int(fold)]["train_ids"] = np.char.strip(specimen_ids["train"])
        exp_dict["folds"][int(fold)]["test_ids"] = np.char.strip(specimen_ids["test"])
    exp_dict["config"]["modalities"] = modalities
    exp_dict["config"]["formats"] = formats
    return exp_dict

if __name__ == "__main__":
    x = get_tree_merge_map("data/raw/dend_RData_Tree_20181220.csv", "n4")
    # pd.DataFrame(x).to_csv("test2.csv")
# # Retracing operation
#     class EncWrapper(torch.nn.Module):
#         def __init__(self, trace, form):
#             super().__init__()
#             self.trace = trace
#             self.form = form

#         def forward(self, x_forms):
#             mean = self.trace(x_forms)
#             transf = mean[:, None] * torch.zeros_like(mean)[..., None]
#             return (mean, transf) 
        
#     class DecWrapper(torch.nn.Module):
#         def __init__(self, trace, form):
#             super().__init__()
#             self.trace = trace
#             self.form = form

#         def forward(self, z):
#             return self.trace(z)

#     from data import MET_Data
#     import shutil
#     folders =  ["data/binary"]#["data/full", "data/smartseq", "data/binary_old"] #["../archive/2-24/all-mse", "../archive/2-24/patchseq-mse"]
#     dest = pathlib.Path("data/retraced_variational") #pathlib.Path("../archive/2-24_variational")
#     form_map = {"T": ["logcpm"], "E": ["pca-ipfx"], "M": ["arbors", "ivscc"]}

#     met = MET_Data("data/raw/MET_full_data.npz")
#     for folder in folders:
#         folder_path = pathlib.Path(folder)
#         for exp_path in folder_path.iterdir():
#             if exp_path.is_file(): continue
#             dest_exp_path = dest / folder_path.name / exp_path.name
#             if dest_exp_path.exists(): continue
#             dest_exp_path.mkdir(exist_ok = True, parents = True)
#             shutil.copy2(exp_path / "config.yaml", dest_exp_path / "config.yaml")
#             shutil.copy2(exp_path / "git_hash.txt", dest_exp_path / "git_hash.txt")
#             shutil.copy2(exp_path / "terminal.out", dest_exp_path / "terminal.out")
#             for fold_path in exp_path.glob("fold_*"):
#                 dest_fold_path = dest_exp_path / fold_path.name
#                 dest_fold_path.mkdir(exist_ok = True)
#                 shutil.copy2(fold_path / "best_params.pt", dest_fold_path / "best_params.pt")
#                 shutil.copy2(fold_path / "train_test_ids.npz", dest_fold_path / "train_test_ids.npz")
#                 shutil.copytree(fold_path / "tn_board", dest_fold_path / "tn_board")
#                 model_paths = [fold_path / "best"] + list((fold_path / "checkpoints").iterdir())
#                 for model_path in model_paths:
#                     model = assemble_jit(model_path)
#                     output_path = dest / "/".join(model_path.parts[1:])
#                     output_path.mkdir(exist_ok = True, parents = True)
#                     encoder_path = output_path / "encoder"
#                     decoder_path = output_path / "decoder"
#                     encoder_path.mkdir(parents = True)
#                     decoder_path.mkdir(parents = True)
#                     for (modal, arm) in model.items():
#                         forms = form_map[modal]
#                         forms_data = [torch.from_numpy(met[form][:1]) for form in forms]
#                         encoder_input = {form: torch.nan_to_num(raw_data).to("cpu", dtype = torch.float32)
#                                          for (form, raw_data) in zip(forms, forms_data)}
#                         encoder_trace = torch.jit.trace(EncWrapper(arm["enc"], forms[0]), encoder_input, strict = False)
#                         decoder_input = encoder_trace(encoder_input)[0]
#                         decoder_trace = torch.jit.trace(DecWrapper(arm["dec"], forms[0]), decoder_input, strict = False)
#                         encoder_trace.save(encoder_path / f"{modal}.pt")
#                         decoder_trace.save(decoder_path / f"{modal}.pt")
#         input("Done")

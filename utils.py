import pathlib
import yaml

import numpy as np
import torch

from pca_cca import PCA_CCA

project_dir = pathlib.Path("/Users/ian.convy/code/cplAE_MET")

class ModelWrapper(torch.nn.Module):
    def __init__(self, model_dict):
        super().__init__()
        self.model_dict = model_dict
        for (modal, arm) in model_dict.items():
            setattr(self, f"{modal}_enc", arm["enc"])
            setattr(self, f"{modal}_dec", arm["dec"])
    
    def __getitem__(self, modal):
        return self.model_dict[modal]
    
    def keys(self):
        return self.model_dict.keys()
    
    def values(self):
        return self.model_dict.values()
    
    def items(self):
        return self.model_dict.items()

def assemble_jit(jit_path, device = "cpu"):
    model = {}
    modalities = [path.stem for path in (jit_path / "encoder").iterdir()]
    for modal in modalities:
        model[modal] = {}
        model[modal]["enc"] = torch.jit.load(jit_path / "encoder" / f"{modal}.pt", map_location = device)
        model[modal]["dec"] = torch.jit.load(jit_path / "decoder" / f"{modal}.pt", map_location = device)
    return model

def save_trace(path, model, config, dataset):
    path = pathlib.Path(path)
    encoder_path = path / "encoder"
    decoder_path = path / "decoder"
    encoder_path.mkdir(parents = True)
    decoder_path.mkdir(parents = True)
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
            decoder_input = encoder_trace(encoder_input)
            decoder_trace = torch.jit.trace(arm["dec"], decoder_input, strict = False)
            encoder_trace.save(encoder_path /  f"{modal}.pt")
            decoder_trace.save(decoder_path / f"{modal}.pt")
    if was_training:
        model.train()

def load_jit_folds(exp_path, folds = None, get_checkpoints = False):
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
            for path in paths:
                epoch = int(path.stem.split("_")[-1])
                state = assemble_jit(path)
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
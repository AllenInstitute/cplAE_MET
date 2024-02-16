import pathlib
import yaml

import numpy as np
import torch

from cplAE_MET.models.subnetworks_T import get_t_arm
from cplAE_MET.models.subnetworks_E import get_e_arm
from cplAE_MET.models.subnetworks_M import get_m_arm

from pca_cca import PCA_CCA

model_builders = {
    "T": get_t_arm,
    "E": get_e_arm,
    "M": get_m_arm,
}

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
        return self.model_dict.keys()
    
    def items(self):
        return self.model_dict.items()

def get_model(config):
    model = {modal: model_builders[modal](config) for modal in config["modalities"]}
    return model

def assemble_jit(jit_path, device = "cpu"):
    model = {}
    modalities = [path.stem for path in (jit_path / "encoder").iterdir()]
    for modal in modalities:
        model[modal] = {}
        model[modal]["enc"] = torch.jit.load(jit_path / "encoder" / f"{modal}.pt", map_location = device)
        model[modal]["dec"] = torch.jit.load(jit_path / "decoder" / f"{modal}.pt", map_location = device)
    return model

def save_trace(path, model, dataset):
    path = pathlib.Path(path)
    encoder_path = path / "encoder"
    decoder_path = path / "decoder"
    encoder_path.mkdir(parents = True)
    decoder_path.mkdir(parents = True)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for (modal, arm) in model.items():
            example_input = torch.as_tensor(dataset.MET[f"{modal}_dat"][:1]).float()
            encoder_trace = torch.jit.trace(arm["enc"], (example_input,))
            decoder_input = (encoder_trace(example_input), )
            decoder_trace = torch.jit.trace(arm["dec"], decoder_input)
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
        info_dict["train_ids"] = specimen_ids["train"]
        info_dict["test_ids"] = specimen_ids["test"]
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
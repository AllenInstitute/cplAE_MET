import pathlib
import yaml

import pandas as pd
import numpy as np
import torch

from cplAE_MET.models.model_classes import MultiModal
from data import MET_Data

project_dir = pathlib.Path("/Users/ian.convy/code/cplAE_MET")

# This functions aligns a target array with the passed annotation dataframe using specimen ids
def align(array, specimen_ids, anno):
    aligned_array = np.stack([np.full_like(array[0], np.nan)] * len(anno))
    id_df = pd.DataFrame(data = {"specimen_id": specimen_ids, "alignment_index": range(len(specimen_ids))})
    id_df["specimen_id"] = id_df["specimen_id"].str.strip()
    aligned_indices = pd.merge(anno, id_df, on = "specimen_id", how = "inner")
    aligned_array[aligned_indices["row_index"].to_numpy()] =  array[aligned_indices["alignment_index"].to_numpy()]
    return aligned_array

# This function selects data from the target array based on which rows are present in the passed annotation
def select(array, anno):
    selected = array[anno["row_index"]]
    return selected

# This function samples from the passed dataframes to ensure that there is an equal number of samples
# from each experiment across excitatory cells and inhibitory cells
def normalize_sample_count(exp1, exp2, anno):
    unnormalized = anno.query("platform == @exp1 or platform == @exp2")
    exp1_exc = unnormalized.query("platform == @exp1 and `class` == 'exc'")
    exp1_inh = unnormalized.query("platform == @exp1 and `class` == 'inh'")
    exp2_exc = unnormalized.query("platform == @exp2 and `class` == 'exc'")
    exp2_inh = unnormalized.query("platform == @exp2 and `class` == 'inh'")
    if len(exp1_exc) > len(exp2_exc):
       exp1_exc = exp1_exc.sample(len(exp2_exc), replace = False)
    else:
        exp2_exc = exp2_exc.sample(len(exp1_exc), replace = False)
    if len(exp1_inh) > len(exp2_inh):
       exp1_inh = exp1_inh.sample(len(exp2_inh), replace = False)
    else:
        exp2_inh = exp2_inh.sample(len(exp1_inh), replace = False)
    anno_samp = pd.concat([exp1_exc, exp1_inh, exp2_exc, exp2_inh])
    return (unnormalized, anno_samp)

def load_data(experiment):
    data_filename = pathlib.Path(experiment["config"]["data_file"]).name
    mat_path = project_dir / "data" / "raw" / data_filename
    data = MET_Data(mat_path)
    return data

def load_experiment(exp_dir, get_checkpoints = True):
    exp_path = project_dir / "data" / exp_dir
    specimen_ids = np.load(exp_path / "train_test_ids.npz")
    experiment = {}
    with open(exp_path / "config.yaml", "r") as target:
        experiment["config"] = yaml.safe_load(target)
    experiment["train_ids"] = specimen_ids["train"]
    experiment["test_ids"] = specimen_ids["test"]
    experiment["best"] = torch.load(exp_path / "best_params.pt", torch.device("cpu"))
    if get_checkpoints:
        experiment["checkpoints"] = {}
        for path in (exp_path / "checkpoints").iterdir():
            epoch = int(path.stem.split("_")[-1])
            state = torch.load(path, torch.device("cpu"))
            experiment["checkpoints"][epoch] = state
    return experiment
    
def load_model(config, state_dict):
    config = config.copy()
    config["gauss_e_baseline"] = np.ones([1, 82])
    config["gauss_m_baseline"] = np.ones([1, 120, 4, 4])
    model = MultiModal(config)
    model.load_state_dict(state_dict)
    return model

def load_cross_validation(exp_dir, get_checkpoints = False):
    exp_path = project_dir / "data" / exp_dir
    results = {}
    with open(exp_path / "config.yaml", "r") as target:
        results["config"] = yaml.safe_load(target)
    results["data"] = load_data(results)
    fold_paths = list(exp_path.glob("trial_*"))
    fold_paths.sort()
    results["folds"] = []
    for path in fold_paths:
        info_dict = {}
        specimen_ids = np.load(path / "train_test_ids.npz")
        info_dict["train_ids"] = specimen_ids["train"]
        info_dict["test_ids"] = specimen_ids["test"]
        info_dict["best"] = torch.load(exp_path / "best_params.pt", torch.device("cpu"))
        if get_checkpoints:
            info_dict["checkpoints"] = {}
            for path in (exp_path / "checkpoints").iterdir():
                epoch = int(path.stem.split("_")[-1])
                state = torch.load(path, torch.device("cpu"))
                info_dict["checkpoints"][epoch] = state
        results["folds"].append(info_dict)
    return results
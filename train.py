import yaml
import pathlib
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np

from data import MET_Data, MET_Dataset, get_collator
from losses import loss_classes, min_var_loss
import utils

class EarlyStopping():
    # This class keeps track of the passed loss value and saves the model
    # which minimizes it across an experiment. If the loss value is not
    # improved by more than a specified minimum fraction within a 
    # "patience" period, the experiment is halted early.
     
    def __init__(self, exp_dir, patience, min_improvement_fraction):
        self.exp_dir = exp_dir
        self.patience = patience
        self.frac = min_improvement_fraction
        self.counter = 0
        self.best_epoch = 0
        self.min_loss = np.inf

    def stop_check(self, loss, model, epoch):
        # When this method is called, the stopper compares the passed loss
        # value to the minimum value that is has observed. If the new loss is
        # better, the passed model is saved and "False" is returned, If the loss 
        # has not improved and the patience period elapses, "True" is returned.
         
        if loss < (1 - self.frac) * self.min_loss:
            self.counter = 0
            self.min_loss = loss
            torch.save(model.state_dict(), self.exp_dir / f"best_params.pt")
            self.best_epoch = epoch
            print(f"New best model, loss {loss:.4g}")
        else:
            self.counter += 1
        stop = self.counter > self.patience
        return stop
    
    def load_best_parameters(self, model):
        # This method takes the passed model and loads the
        # parameters which minimized the loss during the experiment.

        best_state = torch.load(self.exp_dir / "best_params.pt")
        model.load_state_dict(best_state)

def combine_losses(total_loss, cuml_losses, *new_losses):
    # This function takes an existing dictionary of cumulative loses and adds
    # a set of new loss values to it, matching across the different loss keys. 

    incremented = {key:value + cuml_losses.get(key, 0) 
                   for l_dict in new_losses for (key, value) in l_dict.items()}
    cuml_total = cuml_losses.get("total", 0) + total_loss
    new_cuml = {**cuml_losses, **incremented, "total": cuml_total}
    return new_cuml

def compute_weighted_loss(config, *loss_dicts):
    # This function takes a set of component losses and combines them into a 
    # a single scalar loss value using weighrs specified in the config YAML file.

    weighted = [sum([config[modal]*loss_value for (modal, loss_value) in l_dict.items()]) 
                for l_dict in loss_dicts]
    total_loss = sum(weighted)
    return total_loss

def get_gauss_baselines(dataset, modal):
    # This function computes the variances of the morphological and 
    # eletrophysiological modalities. It selects samples with the correct
    # modalities by using the indices of the passed MET_Dataset object,
    # which has already separated the samples by modality combination.

    all_modalities = ["T", "E", "M", "TE", "TM", "EM", "MET"]
    indices = np.concatenate([dataset.modal_indices[string] for string in all_modalities if modal in string])
    x = dataset.MET[f"{modal}_dat"][indices]
    std = np.std(x, 0, keepdims = True)
    return std

def filter_specimens(met_data, specimen_ids, config):
    platforms = config["select"]["platforms"]
    specimens = met_data.query(specimen_ids, platforms = platforms)["specimen_id"]
    return specimens

def build_model(config, train_dataset):
    # This function builds the model specified in the config YAML file. It completes
    # the specification by computing the baseline variances of the morphological and
    # electro-physiological data.

    model_config = config.copy()
    if "E" in config["modalities"]:
        std_e = get_gauss_baselines(train_dataset, "E")
        model_config["gauss_e_baseline"] = std_e.astype("float32")
    if "M" in config["modalities"]:
        std_m = get_gauss_baselines(train_dataset, "M")
        model_config["gauss_m_baseline"] = std_m.astype("float32")
    model_dict = utils.get_model(model_config)
    model = utils.ModelWrapper(model_dict)
    return model

def log_tensorboard(tb_writer, train_loss, val_loss, epoch):
    # This function takes the training/validation losses and logs them
    # in Tensoboard. The component losses are reported without any scaling,
    # alongside the weighted sum of the losses.

    tb_writer.add_scalars("Weighted Loss", {"Train": train_loss["total"], "Validation": val_loss["total"]}, epoch)
    tb_writer.add_scalars("R2/Train", 
        {key: 1 - value for (key, value) in train_loss.items() if key in {"T", "E", "M"}}, epoch)
    tb_writer.add_scalars("R2/Validation", 
        {key: 1 - value for (key, value) in val_loss.items() if key in {"T", "E", "M"}}, epoch)
    tb_writer.add_scalars("Cross-R2/Train", 
        {key: 1 - value for (key, value) in train_loss.items() if "=" in key}, epoch)
    tb_writer.add_scalars("Cross-R2/Validation", 
        {key: 1 - value for (key, value) in val_loss.items() if "=" in key}, epoch)
    tb_writer.add_scalars("Coupling/Train", 
        {key: value for (key, value) in train_loss.items() if "-" in key}, epoch)
    tb_writer.add_scalars("Coupling/Validation",
        {key:value for (key, value) in val_loss.items() if "-" in key}, epoch)
    print(f"Epoch {epoch} -- Train: {train_loss['total']:.4e} | Val: {val_loss['total']:.4e}")

def process_batch(model, X_dict, mask_dict, config, loss_funcs):
    # This function processes a single batch during model optimization. It takes as
    # argument the target model, a dictionary of data from different modalities, a
    # dictionary of masks specifying which samples hold valid data for each modality,
    # and the experiment configuration dictionary. For each modality, the latent space
    # and reconstruction are calculated, along with the the self-modal R2 loss. The function
    # then iterates through any previous modalities and computes the latent space coupling loss
    # and the cross-modal R2 loss. The modality masks are combined in order to select data for
    # pairs of modalities.

    (latent_dict, recon_dict, recon_loss_dict, coupling_dict, cross_dict) = ({}, {}, {}, {}, {})
    for modal in config["modalities"]:
        (arm, x, mask) = (model[modal], X_dict[modal], mask_dict[modal])
        z = arm["enc"](x[mask])
        xr = arm["dec"](z)
        (latent_dict[modal], recon_dict[modal]) = (z, xr)
        recon_loss_dict[modal] = loss_funcs[modal].loss(x[mask], xr, modal)
        for (prev_modal, prev_z) in list(latent_dict.items())[:-1]:
            prev_mask = mask_dict[prev_modal]
            if torch.any(prev_mask[mask]):
                (z_masked, prev_masked) = (z[prev_mask[mask]], prev_z[mask[prev_mask]])
                coupling_dict[f"{prev_modal}-{modal}"] = min_var_loss(z_masked, prev_masked.detach())
                coupling_dict[f"{modal}-{prev_modal}"] = min_var_loss(z_masked.detach(), prev_masked)
                (x_masked, x_prev_masked) = (x[mask & prev_mask], X_dict[prev_modal][mask & prev_mask])
                cross_dict[f"{modal}={prev_modal}"] = loss_funcs[prev_modal].cross_loss(model, x_prev_masked, z_masked, prev_modal)
                cross_dict[f"{prev_modal}={modal}"] = loss_funcs[modal].cross_loss(model, x_masked, prev_masked, modal)
    return (latent_dict, recon_dict, recon_loss_dict, coupling_dict, cross_dict)

def train_and_evaluate(exp_dir, config, train_dataset, val_dataset):
    # This function takes trains a model as specified in the passed configuration
    # dictionary (loaded from a config YAML file), using the provided training and
    # validation datasets. It monitors the loss improvement using and EarlyStopping
    # instance, and also saves the model at regular intervals. At the end of each
    # epoch the training and validation losses are logged in Tensorboard.

    model = build_model(config, train_dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr = config["learning_rate"])
    model.to(config["device"])
    tb_writer = SummaryWriter(log_dir = exp_dir / "tn_board")
    stopper = EarlyStopping(exp_dir, config["patience"], config["improvement_frac"])
    collate = get_collator(config["device"], torch.float32) # Converts tensors to desired device and type
    train_loader = DataLoader(train_dataset, batch_size = None, collate_fn = collate)
    val_loader = DataLoader(val_dataset, batch_size = None, collate_fn = collate)
    (exp_dir / "checkpoints").mkdir(exist_ok = True)
    loss_funcs = {modal: loss_classes[loss](config, train_dataset.MET, train_dataset.allowed_specimen_ids) 
                  for (modal, loss) in config["losses"].items()}
    for epoch in range(config["num_epochs"]):
        # Training -----------
        cuml_losses = {}
        model.train()
        if config["check_step"] > 0 and epoch % config["check_step"] == 0:
            utils.save_trace(exp_dir / "checkpoints" / f"model_{epoch}", model, train_dataset)
        for (X_dict, mask_dict, specimen_ids) in train_loader:
            optimizer.zero_grad()
            (latent_dict, recon_dict, recon_loss_dict, coupling_dict, cross_dict) = process_batch(model, X_dict, mask_dict, config, loss_funcs)
            loss = compute_weighted_loss(config, recon_loss_dict, coupling_dict, cross_dict)
            loss.backward()
            optimizer.step()
            cuml_losses = combine_losses(loss, cuml_losses, recon_loss_dict, coupling_dict, cross_dict)
        avg_losses = {key: value / len(train_dataset) for (key, value) in cuml_losses.items()}
        # Validation -----------
        with torch.no_grad():
            cuml_val_losses = {}
            for (X_val, mask_val, _) in val_loader:
                model.eval()
                (latent_val, recon_val, recon_loss_val, coupling_val, cross_val) = process_batch(model, X_val, mask_val, config, loss_funcs)
                val_loss = compute_weighted_loss(config, recon_loss_val, coupling_val, cross_val)
                cuml_val_losses = combine_losses(val_loss, cuml_val_losses, recon_loss_val, coupling_val, cross_val)
            avg_val_losses = {key: value / len(val_dataset) for (key, value) in cuml_val_losses.items()}
        log_tensorboard(tb_writer, avg_losses, avg_val_losses, epoch + 1)
        if stopper.stop_check(avg_val_losses["total"], model, epoch) or (epoch + 1 == config["num_epochs"]):
            stopper.load_best_parameters(model)
            print(f"Best model was epoch {stopper.best_epoch} with loss {stopper.min_loss:.4g}")
            break
    utils.save_trace(exp_dir / "best", model, train_dataset)
    tb_writer.close()
    return model

def train_model(config, exp_dir):
    met_data = MET_Data(config["data_file"])
    num_folds = config["folds"]
    if num_folds > 0:
        indices = list(met_data.get_stratified_KFold(config["folds"], seed = config["seed"]))
    else:
        (train_ids, test_ids) = met_data.get_stratified_split(config["val_split"], seed = config["seed"])
        indices = [(train_ids, test_ids)]
        num_folds = 1
    fold_list = config["fold_list"] if config["fold_list"] else range(1, num_folds + 1)
    for fold in fold_list:
        (train_ids, test_ids) = indices[fold - 1]
        print(f"Processing fold {fold} / {num_folds}.")
        exp_fold_dir = exp_dir / f"fold_{fold}"
        exp_fold_dir.mkdir(exist_ok = True)
        filtered_train_ids = filter_specimens(met_data, train_ids, config)
        filtered_test_ids = filter_specimens(met_data, test_ids, config)
        train_dataset = MET_Dataset(met_data, config["batch_size"], config["modal_frac"], filtered_train_ids)
        test_dataset = MET_Dataset(met_data, config["batch_size"], config["modal_frac"], filtered_test_ids)
        np.savez_compressed(exp_fold_dir / "train_test_ids.npz", **{"train": train_ids, "test": test_ids})
        train_and_evaluate(exp_fold_dir, config, train_dataset, test_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help = "Name of experiment.")
    parser.add_argument("config_path", help = "path to config yaml file")
    args = parser.parse_args()
    with open(args.config_path, "r") as target:
        config = yaml.safe_load(target)
    exp_dir = pathlib.Path(config["output_dir"]) / args.exp_name
    exp_dir.mkdir(exist_ok = True)
    train_model(config, exp_dir)

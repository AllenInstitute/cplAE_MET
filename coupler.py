import yaml
import pathlib
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np

from data import MET_Data, MET_Simulated, DeterministicDataset, get_collator, filter_specimens
from losses import MSE
import utils
import subnetworks

class EarlyStopping():
    # This class keeps track of the passed loss value and saves the model
    # which minimizes it across an experiment. If the loss value is not
    # improved by more than a specified minimum fraction within a 
    # "patience" period, the experiment is halted early.
     
    def __init__(self, exp_dir, name, patience, min_improvement_fraction):
        self.exp_dir = exp_dir
        self.patience = patience
        self.frac = min_improvement_fraction
        self.counter = 0
        self.best_epoch = 0
        self.min_loss = np.inf
        self.name = name
        self.stopped = False
        self.exp_dir.mkdir(exist_ok = True)

    def stop_check(self, loss, model, epoch):
        # When this method is called, the stopper compares the passed loss
        # value to the minimum value that is has observed. If the new loss is
        # better, the passed model is saved and "False" is returned, If the loss 
        # has not improved and the patience period elapses, "True" is returned.
        if not self.stopped: 
            if loss < (1 - self.frac) * self.min_loss:
                self.counter = 0
                self.min_loss = loss
                torch.save(model.state_dict(), self.exp_dir / f"{self.name}.pt")
                self.best_epoch = epoch
            else:
                self.counter += 1
            stop = self.counter > self.patience
            if stop:
                self.stopped = stop
                print(f"Model {self.name} stopped, loss {loss:.4g}")
        else:
            stop = True
        return stop
    
    def load_best_parameters(self, model):
        # This method takes the passed model and loads the
        # parameters which minimized the loss during the experiment.

        best_state = torch.load(self.exp_dir / f"{self.name}.pt")
        model.load_state_dict(best_state)

def apply_mask(dct, mask):
    masked = {key: value[mask] for (key, value) in dct.items()}
    return masked

def combine_losses(cuml_losses, new_losses):
    # This function takes an existing dictionary of cumulative loses and adds
    # a set of new loss values to it, matching across the different loss keys. 

    incremented = {key:value + cuml_losses.get(key, 0) for (key, value) in new_losses.items()}
    new_cuml = {**cuml_losses, **incremented}
    return new_cuml

def build_model(config, train_dataset):
    # This function builds the model specified in the config YAML file. It completes
    # the specification by computing the baseline variances of the morphological and
    # electro-physiological data.

    model_dict = subnetworks.get_coupler(config, train_dataset)
    model = utils.CouplerWrapper(model_dict)
    return model

def save_coupler(path, model, config):
    path = pathlib.Path(path)
    path.mkdir(exist_ok = True)
    was_training = model.training
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for (modal_str, module) in model.items():
            in_latent_dim = config["modal_specs"][modal_str[0]]["latent_dim"]
            x_input = torch.randn([1, in_latent_dim]).to(device, dtype = torch.float32)
            trace = torch.jit.trace(module, x_input)
            trace.save(path / f"{modal_str}.pt")
    if was_training:
        model.train()

def train_setup(exp_dir, config, train_dataset, val_dataset):
    model = build_model(config, train_dataset)
    (optimizers, stoppers, encoders) = ({}, {}, {})
    for in_modal in config["modalities"]:
        trace_path = pathlib.Path(config["modal_specs"][in_modal]["model_path"]) / "fold_1" / "best"
        encoders[in_modal] = utils.assemble_jit(trace_path, config["device"])[in_modal]["enc"]
        for out_modal in config["modalities"]:
            if in_modal == out_modal:
                continue
            modal_str = f"{in_modal}-{out_modal}"
            optimizers[modal_str] = torch.optim.Adam(model[modal_str].parameters(), lr = config["learning_rate"])
            stoppers[modal_str] = EarlyStopping(exp_dir / "states", modal_str, config["patience"], config["improvement_frac"])
    tb_writer = SummaryWriter(log_dir = exp_dir / "tn_board")
    collate = get_collator(config["device"], torch.float32) # Converts tensors to desired device and type
    train_loader = DataLoader(train_dataset, batch_size = None, collate_fn = collate)
    val_loader = DataLoader(val_dataset, batch_size = None, collate_fn = collate)
    loss_func = MSE(config, train_dataset.MET, train_dataset.allowed_specimen_ids)
    return (model, encoders, optimizers, tb_writer, stoppers, train_loader, val_loader, loss_func)

def log_tensorboard(tb_writer, train_loss, val_loss, epoch):
    tb_writer.add_scalars("Coupling/Train", train_loss, epoch)
    tb_writer.add_scalars("Coupling/Validation", val_loss, epoch)
    (train_comb, val_comb) = (sum(train_loss.values()), sum(val_loss.values()))
    print(f"Epoch {epoch} -- Train: {train_comb:.4e} | Val: {val_comb:.4e}")

def process_batch(model, encoders, X_dict, mask_dict, config, loss_func, stoppers):
    (latent_dict, loss_dict) = ({}, {})
    for modal in config["modalities"]:
        (encoder, x_forms, mask) = (encoders[modal], X_dict[modal], mask_dict[modal])
        x_masked = apply_mask(x_forms, mask)
        with torch.no_grad():
            z = encoder(x_masked)
        latent_dict[modal] = z
        for (prev_modal, prev_z) in list(latent_dict.items())[:-1]:
            prev_mask = mask_dict[prev_modal]
            if torch.any(prev_mask[mask]):
                (z_masked, prev_z_masked) = (z[prev_mask[mask]], prev_z[mask[prev_mask]])
                (curr_prev, prev_curr) = (f"{modal}-{prev_modal}", f"{prev_modal}-{modal}")
                if not stoppers[curr_prev].stopped:
                    recon_prev_z = model[curr_prev](z_masked)
                    loss_dict[curr_prev] = loss_func(recon_prev_z, prev_z_masked, None)
                if not stoppers[prev_curr].stopped:
                    recon_z = model[prev_curr](prev_z_masked)
                    loss_dict[prev_curr] = loss_func(recon_z, z_masked, None)
    return (latent_dict, loss_dict)

def train_and_evaluate(exp_dir, config, train_dataset, val_dataset):
    # This function takes trains a model as specified in the passed configuration
    # dictionary (loaded from a config YAML file), using the provided training and
    # validation datasets. It monitors the loss improvement using and EarlyStopping
    # instance, and also saves the model at regular intervals. At the end of each
    # epoch the training and validation losses are logged in Tensorboard.
    
    (model, encoders, optimizers, tb_writer, stoppers, train_loader, val_loader, loss_func) = train_setup(
        exp_dir, config, train_dataset, val_dataset)
    model.to(config["device"])
    # input(model)
    for epoch in range(config["num_epochs"]):
        # Training -----------
        cuml_losses = {}
        model.train()
        if config["check_step"] > 0 and epoch % config["check_step"] == 0:
            save_coupler(exp_dir / "checkpoints" / f"model_{epoch}", model, config)
        for (X_dict, mask_dict, specimen_ids) in train_loader:
            for optimizer in optimizers.values():
                optimizer.zero_grad()
            (latent_dict, loss_dict) = process_batch(model, encoders, X_dict, mask_dict, config, loss_func, stoppers)
            for (key, loss) in loss_dict.items():
                loss.backward()
                optimizers[key].step()
            cuml_losses = combine_losses(cuml_losses, loss_dict)
        avg_losses = {key: value / len(train_dataset) for (key, value) in cuml_losses.items()}
        # Validation -----------
        with torch.no_grad():
            cuml_val_losses = {}
            for (X_val, mask_val, _) in val_loader:
                model.eval()
                (latent_val, val_loss_dict) = process_batch(model, encoders, X_val, mask_val, config, loss_func, stoppers)
                cuml_val_losses = combine_losses(cuml_val_losses, val_loss_dict)
            avg_val_losses = {key: value / len(val_dataset) for (key, value) in cuml_val_losses.items()}
        log_tensorboard(tb_writer, avg_losses, avg_val_losses, epoch + 1)
        for (key, loss) in avg_val_losses.items():
            stoppers[key].stop_check(loss, model[key], epoch)
        if all([stopper.stopped for stopper in stoppers.values()]):
            break
    for (key, stopper) in stoppers.items():
        stopper.load_best_parameters(model[key])
    save_coupler(exp_dir / "best", model, config)
    tb_writer.close()
    return model

def train_model(config, exp_dir):
    met_data = MET_Data(config["data_file"]) if "simulate" not in config else MET_Simulated(config)
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
        (exp_fold_dir / "checkpoints").mkdir(exist_ok = True)
        filtered_train_ids = filter_specimens(met_data, train_ids, config)
        filtered_test_ids = filter_specimens(met_data, test_ids, config)
        train_dataset = DeterministicDataset(met_data, config["batch_size"], config["formats"], config["modal_frac"], config["transform"], filtered_train_ids)
        test_dataset = DeterministicDataset(met_data, config["batch_size"], config["formats"], config["modal_frac"], config["transform"], filtered_test_ids)
        np.savez_compressed(exp_fold_dir / "train_test_ids.npz", **{"train": train_ids, "test": test_ids})
        train_and_evaluate(exp_fold_dir, config, train_dataset, test_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_path", help = "Name of experiment.")
    parser.add_argument("config_path", help = "path to config yaml file")
    args = parser.parse_args()
    with open(args.config_path, "r") as target:
        config = yaml.safe_load(target)
    exp_dir = pathlib.Path(args.exp_path)
    exp_dir.mkdir(exist_ok = True)
    train_model(config, exp_dir)

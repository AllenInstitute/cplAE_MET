import yaml
import pathlib
import argparse
import pickle as pk

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np

from data import MET_Data, DeterministicDataset, RandomizedDataset, get_collator, filter_specimens
from losses import ReconstructionLoss, VariationalLoss
import utils
import subnetworks

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
        self.min_loss = np.inf

    def stop_check(self, loss, model):
        # When this method is called, the stopper compares the passed loss
        # value to the minimum value that is has observed. If the new loss is
        # better, the passed model is saved and "False" is returned, If the loss 
        # has not improved and the patience period elapses, "True" is returned.
         
        if loss < (1 - self.frac) * self.min_loss:
            self.counter = 0
            self.min_loss = loss
            print(f"New best loss {loss:.4g}")
        else:
            self.counter += 1
        stop = self.counter > self.patience
        if stop:
            torch.save(model.state_dict(), self.exp_dir / f"best_params.pt")
        return stop
    
    def load_best_parameters(self, model):
        # This method takes the passed model and loads the
        # parameters which minimized the loss during the experiment.

        best_state = torch.load(self.exp_dir / "best_params.pt")
        model.load_state_dict(best_state)

def apply_mask(dct, mask):
    masked = {key: value[mask] for (key, value) in dct.items()}
    return masked

def combine_losses(total_loss, cuml_losses, new_losses):
    # This function takes an existing dictionary of cumulative loses and adds
    # a set of new loss values to it, matching across the different loss keys. 

    incremented = {key:value + cuml_losses.get(key, 0) for (key, value) in new_losses.items()}
    cuml_total = cuml_losses.get("total", 0) + total_loss
    new_cuml = {**cuml_losses, **incremented, "total": cuml_total}
    return new_cuml

def build_model(config, train_dataset):
    # This function builds the model specified in the config YAML file. It completes
    # the specification by computing the baseline variances of the morphological and
    # electro-physiological data.
    
    model_dict = subnetworks.get_model(config, train_dataset)
    mappers = subnetworks.get_mapper(config, train_dataset) if config["inference"] else None
    classifiers = subnetworks.get_classifiers(config, train_dataset)
    model = utils.VariationalWrapper(model_dict, mappers, classifiers)
    # from torchinfo import summary
    # summary(model, input_data = [{"m0": torch.zeros([2, 28, 28, 3])}], in_modal = "A", out_modals = ["A"])
    # input()
    return model

def train_setup(exp_dir, config, train_dataset, val_dataset):
    model = build_model(config, train_dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr = config["learning_rate"])
    tb_writer = SummaryWriter(log_dir = exp_dir / "tn_board")
    stopper = EarlyStopping(exp_dir, config["patience"], config["improvement_frac"])
    collate = get_collator(config["device"], torch.float32) # Converts tensors to desired device and type
    train_loader = DataLoader(train_dataset, batch_size = None, collate_fn = collate)
    val_loader = DataLoader(val_dataset, batch_size = None, collate_fn = collate)
    if config["inference"]:
        loss_class = VariationalLoss
    # elif config["contrastive"]["active"]:
    #     loss_class = ContrastiveLoss
    else:
        loss_class = ReconstructionLoss
    loss_handler = loss_class(config, train_dataset.MET, train_dataset.allowed_specimen_ids)
    return (model, optimizer, tb_writer, stopper, train_loader, val_loader, loss_handler)

def train_and_evaluate(exp_dir, config, train_dataset, val_dataset):
    # This function takes trains a model as specified in the passed configuration
    # dictionary (loaded from a config YAML file), using the provided training and
    # validation datasets. It monitors the loss improvement using and EarlyStopping
    # instance, and also saves the model at regular intervals. At the end of each
    # epoch the training and validation losses are logged in Tensorboard.
    
    (model, optimizer, tb_writer, stopper, train_loader, val_loader, loss_handler) = train_setup(
        exp_dir, config, train_dataset, val_dataset)
    model.to(config["device"])
    for epoch in range(config["num_epochs"]):
        # Training -----------
        (cuml_losses, cuml_acc) = ({}, {})
        model.train()
        if config["check_step"] > 0 and epoch % config["check_step"] == 0:
            utils.save_trace(exp_dir / "checkpoints" / f"model_{epoch}", model, config, train_dataset)
        for (X_dict, mask_dict, _, labels) in train_loader:
            optimizer.zero_grad()
            (loss_dict, loss, acc) = loss_handler.process_batch(model, X_dict, mask_dict, labels)
            loss.backward()
            optimizer.step()
            cuml_losses = combine_losses(loss, cuml_losses, loss_dict)
            cuml_acc = combine_losses(0, cuml_acc, acc)
        avg_losses = {key: value / len(train_dataset) for (key, value) in cuml_losses.items()}
        avg_accs = {key: value / len(train_dataset) for (key, value) in cuml_acc.items()}
        # Validation -----------
        with torch.no_grad():
            (cuml_val_losses, cuml_val_acc) = ({}, {})
            for (X_val, mask_val, _, labels) in val_loader:
                model.eval()
                (val_loss_dict, val_loss, val_acc) = loss_handler.process_batch(model, X_val, mask_val, labels)
                cuml_val_losses = combine_losses(val_loss, cuml_val_losses, val_loss_dict)
                cuml_val_acc = combine_losses(0, cuml_val_acc, val_acc)
            avg_val_losses = {key: value / len(val_dataset) for (key, value) in cuml_val_losses.items()}
            avg_val_accs = {key: value / len(val_dataset) for (key, value) in cuml_val_acc.items()}
        loss_handler.log(tb_writer, avg_losses, avg_val_losses, avg_accs, avg_val_accs, epoch + 1)
        print(f"Epoch {epoch} -- Train: {avg_losses['total']:.4e} | Val: {avg_val_losses['total']:.4e}")
        if config["tracked_loss"] and stopper.stop_check(avg_val_losses[config["tracked_loss"]], model):
            break
    utils.save_trace(exp_dir / "best", model, config, train_dataset)
    tb_writer.close()
    return model

def train_model(config, exp_dir):
    data_keys = {form: data_config["keys"] for (form, data_config) in config["data_config"]["formats"].items()}
    hdf5_paths = config["data_config"]["data_paths"]
    met_data = MET_Data(hdf5_paths, **data_keys)
    label_configs = config["variational"]["classifier"]["label"]
    label_funcs = {label_type: utils.label_functions[conf["name"]](*conf["args"])
                  for (label_type, conf) in label_configs.items()}
    label_encoders = met_data.set_labels(label_funcs)
    (num_reps, num_folds) = (config["fold_reps"], config["folds"])
    for rep in range(num_reps):
        if num_folds > 0:
            indices = list(met_data.get_stratified_KFold(config["folds"], seed = config["seed"] + rep))
        else:
            (train_ids, test_ids) = met_data.get_stratified_split(config["val_split"], seed = config["seed"] + rep)
            indices = [(train_ids, test_ids)]
            num_folds = 1
        fold_list = config["fold_list"] if config["fold_list"] else range(1, num_folds + 1)
        for fold in fold_list:
            (train_ids, test_ids) = indices[fold - 1]
            print(f"Processing fold {fold + rep*num_folds} / {num_reps*num_folds}.")
            exp_fold_dir = exp_dir / f"fold_{fold + rep*num_folds}"
            exp_fold_dir.mkdir(exist_ok = True)
            (exp_fold_dir / "checkpoints").mkdir(exist_ok = True)
            filtered_train_ids = filter_specimens(met_data, train_ids, config)
            filtered_test_ids = filter_specimens(met_data, test_ids, config)
            for (form, data_config) in config["data_config"]["formats"].items():
                if data_config["cache"]:
                    met_data.cache_data(form, np.concatenate([filtered_train_ids, filtered_test_ids]), verbose = True)
            unpack = {form: data_config["unpack"] for (form, data_config) in config["data_config"]["formats"].items()}
            train_dataset = RandomizedDataset(met_data, config["batch_size"], config["formats"], config["modal_frac"], config["transform"], unpack, filtered_train_ids)
            test_dataset = DeterministicDataset(met_data, config["batch_size"], config["formats"], config["modal_frac"], config["transform"], unpack, filtered_test_ids)
            np.savez_compressed(exp_fold_dir / "train_test_ids.npz", **{"train": train_ids, "test": test_ids})
            with open(exp_fold_dir / "label_encoder.pk", "wb") as target:
                pk.dump(label_encoders, target)
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

import yaml
import pathlib
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np

from data import MET_Data, MET_Dataset, get_collator
from cplAE_MET.models.model_classes import MultiModal

class EarlyStopping():
    def __init__(self, exp_dir, patience, min_improvement_fraction):
        self.exp_dir = exp_dir
        self.patience = patience
        self.frac = min_improvement_fraction
        self.counter = 0
        self.best_epoch = 0
        self.min_loss = np.inf

    def stop_check(self, loss, model, epoch):
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
        best_state = torch.load(self.exp_dir / "best_params.pt")
        model.load_state_dict(best_state)

def min_var_loss(zi, zj):
    batch_size = zj.shape[0]
    zj_centered = zj - torch.mean(zj, 0, True)
    try:
        min_eig = torch.min(torch.linalg.svdvals(zj_centered))
    except torch._C._LinAlgError:
        print("SVD failed.")
        min_eig = torch.as_tensor((batch_size - 1)**0.5).to(zj)
    min_var_zj = torch.square(min_eig)/(batch_size-1)
    zi_centered = zi - torch.mean(zi, 0, True)
    try:
        min_eig = torch.min(torch.linalg.svdvals(zi_centered))
    except torch._C._LinAlgError:
        print("SVD failed.")
        min_eig = torch.as_tensor((batch_size - 1)**0.5).to(zi)
    min_var_zi = torch.square(min_eig)/(batch_size-1)
    zi_zj_mse = torch.mean(torch.sum(torch.square(zi-zj), 1))
    loss_ij = zi_zj_mse/torch.squeeze(torch.minimum(min_var_zi, min_var_zj))
    return loss_ij

def squared_error_loss(x_tupl, xr_tupl, mask):
    squares = (torch.square(x[mask] - xr).mean() for (x, xr) in zip(x_tupl, xr_tupl))
    mean_squared_error = sum(squares) / len(xr_tupl)
    return mean_squared_error

def combine_losses(cuml_losses, recon_losses, coupling_losses, total_loss):
    cuml_recon = {key: value + cuml_losses.get(key, 0) for (key, value) in recon_losses.items()}
    cuml_coupling = {key: value + cuml_losses.get(key, 0) for (key, value) in coupling_losses.items()}
    new_cuml = {**cuml_losses, **cuml_recon, **cuml_coupling, "total": cuml_losses.get("total", 0) + total_loss}
    return new_cuml

def compute_weighted_loss(config, recon_loss_dict, coupling_dict):
    recon_loss = sum([config[modal]*loss_value for (modal, loss_value) in recon_loss_dict.items()])
    coupl_loss = sum([config[modal]*loss_value for (modal, loss_value) in coupling_dict.items()])
    weighted_loss = recon_loss + coupl_loss
    return weighted_loss

def get_gauss_baselines(dataset):
    e_indices = np.concatenate(
        [dataset.modal_indices["E"], dataset.modal_indices["TE"], dataset.modal_indices["EM"],
        dataset.modal_indices["MET"]])
    m_indices = np.concatenate(
        [dataset.modal_indices["M"], dataset.modal_indices["TM"], dataset.modal_indices["EM"],
        dataset.modal_indices["MET"]])
    (xe, xm) = (dataset.MET["E_dat"][e_indices], dataset.MET["M_dat"][m_indices])
    (std_e, std_m) = (np.std(xe, 0, keepdims = True), np.std(xm, 0, keepdims = True))
    return (std_e, std_m)

def build_model(config, train_dataset):
    model_config = config.copy()
    (std_e, std_m) = get_gauss_baselines(train_dataset)
    model_config["gauss_e_baseline"] = std_e.astype("float32")
    model_config["gauss_m_baseline"] = std_m.astype("float32")
    model = MultiModal(model_config)
    return model

def log_tensorboard(tb_writer, train_loss, val_loss, epoch):
    tb_writer.add_scalars("Weighted Loss", {"Train": train_loss["total"], "Validation": val_loss["total"]}, epoch)
    for key in train_loss:
        if "-" not in key and key != "total":
            tb_writer.add_scalars(f"MSE/{key}", {"Train": train_loss[key], "Validation": val_loss[key]}, epoch)
    tb_writer.add_scalars("Coupling/Train", 
        {key:value for (key, value) in train_loss.items() if "-" in key}, epoch)
    tb_writer.add_scalars("Coupling/Validation",
        {key:value for (key, value) in val_loss.items() if "-" in key}, epoch)
    print(f"Epoch {epoch} -- Train: {train_loss['total']:.4e} | Val: {val_loss['total']:.4e}")

def process_batch(model, X_dict, mask_dict, config):
    (latent_dict, recon_dict, recon_loss_dict, coupling_dict) = ({}, {}, {}, {})
    for modal in config["modalities"]:
        (arm, x_tupl, mask) = (model.modal_arms[modal], X_dict[modal], mask_dict[modal])
        (z, xr) = arm(*(x[mask] for x in x_tupl))
        xr_tupl = (xr,) if type(xr) != tuple else xr
        latent_dict[modal] = z
        recon_dict[modal] = xr_tupl
        recon_loss_dict[modal] = squared_error_loss(x_tupl, xr_tupl, mask)
        for (prev_modal, prev_z) in list(latent_dict.items())[:-1]:
            prev_mask = mask_dict[prev_modal]
            if torch.any(prev_mask[mask]):
                (z_masked, prev_masked) = (z[prev_mask[mask]], prev_z[mask[prev_mask]])
                coupling_dict[f"{prev_modal}-{modal}"] = min_var_loss(z_masked, prev_masked.detach())
                coupling_dict[f"{modal}-{prev_modal}"] = min_var_loss(z_masked.detach(), prev_masked)
    return (latent_dict, recon_dict, recon_loss_dict, coupling_dict)

def train_and_evaluate(exp_dir, config, train_dataset, val_dataset):
    model = build_model(config, train_dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr = config["learning_rate"])
    model.to(config["device"])
    tb_writer = SummaryWriter(log_dir = exp_dir / "tn_board")
    stopper = EarlyStopping(exp_dir, config["patience"], config["improvement_frac"])
    collate = get_collator(config["device"], torch.float32)
    train_loader = DataLoader(train_dataset, batch_size = None, collate_fn = collate)
    val_loader = DataLoader(val_dataset, batch_size = None, collate_fn = collate)
    for epoch in range(config["num_epochs"]):
        # Training -----------
        cuml_losses = {}
        model.train()
        if config["check_step"] > 0 and epoch % config["check_step"] == 0:
            torch.save(model.state_dict(), exp_dir / "checkpoints" / f"model_{epoch}.pt")
        for (X_dict, mask_dict, specimen_ids) in train_loader:
            optimizer.zero_grad()
            (latent_dict, recon_dict, recon_loss_dict, coupling_dict) = process_batch(model, X_dict, mask_dict, config)
            loss = compute_weighted_loss(config, recon_loss_dict, coupling_dict)
            loss.backward()
            optimizer.step()
            cuml_losses = combine_losses(cuml_losses, recon_loss_dict, coupling_dict, loss)
        avg_losses = {key: value / len(train_dataset) for (key, value) in cuml_losses.items()}
        # Validation -----------
        with torch.no_grad():
            cuml_val_losses = {}
            for (X_val, mask_val, _) in val_loader:
                model.eval()
                (latent_val, recon_val, recon_loss_val, coupling_val) = process_batch(model, X_val, mask_val, config)
                val_loss = compute_weighted_loss(config, recon_loss_val, coupling_val)
                cuml_val_losses = combine_losses(cuml_val_losses, recon_loss_val, coupling_val, val_loss)
            avg_val_losses = {key: value / len(val_dataset) for (key, value) in cuml_val_losses.items()}
        log_tensorboard(tb_writer, avg_losses, avg_val_losses, epoch + 1)
        if stopper.stop_check(avg_val_losses["total"], model, epoch) or (epoch + 1 == config["num_epochs"]):
            stopper.load_best_parameters(model)
            print(f"Best model was epoch {stopper.best_epoch} with loss {stopper.min_loss:.4g}")
            break
    tb_writer.close()
    return model

def train_model(config, exp_dir):
    met_data = MET_Data(config["data_file"])
    (train_ids, test_ids) = met_data.get_stratified_split(config["val_split"], seed = 42)
    train_dataset = MET_Dataset(met_data, config["batch_size"], config["modal_frac"], train_ids)
    test_dataset = MET_Dataset(met_data, config["batch_size"], config["modal_frac"], test_ids)
    np.savez_compressed(exp_dir / "train_test_ids.npz", **{"train": train_ids, "test": test_ids})
    trained_model = train_and_evaluate(exp_dir, config, train_dataset, test_dataset)
    return trained_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help = "Name of experiment.")
    parser.add_argument("config_path", help = "path to config yaml file")
    args = parser.parse_args()
    with open(args.config_path, "r") as target:
        config = yaml.safe_load(target)
    exp_dir = pathlib.Path(config["output_dir"]) / args.exp_name
    exp_dir.mkdir(exist_ok = True)
    (exp_dir / "checkpoints").mkdir(exist_ok = True)
    train_model(config, exp_dir)

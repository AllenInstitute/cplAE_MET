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

def min_var_loss(zi, zj):
    # This function computes a loss which penalizes differences
    # between the passed latent vectors (from different modalities).
    # The value is computed by taking the L2 distance between 
    # the vectors, and then dividing this value by the smallest 
    # singular value of the latent space covariance matrices 
    # (approximated using the passed batch of latent vectors). This
    # scaling helps prevent the latent spaces from collpasing into
    # the origin or into a low-dimensional subspace. 

    batch_size = zj.shape[0]
    zj_centered = zj - torch.mean(zj, 0, True)
    try: # If SVD fails, do not scale L2 distance
        min_eig = torch.min(torch.linalg.svdvals(zj_centered))
    except torch._C._LinAlgError:
        print("SVD failed.")
        min_eig = torch.as_tensor((batch_size - 1)**0.5).to(zj)
    min_var_zj = torch.square(min_eig)/(batch_size-1)
    zi_centered = zi - torch.mean(zi, 0, True)
    try: # If SVD fails, do not scale L2 distance
        min_eig = torch.min(torch.linalg.svdvals(zi_centered))
    except torch._C._LinAlgError:
        print("SVD failed.")
        min_eig = torch.as_tensor((batch_size - 1)**0.5).to(zi)
    min_var_zi = torch.square(min_eig)/(batch_size-1)
    zi_zj_mse = torch.mean(torch.sum(torch.square(zi-zj), 1))
    loss_ij = zi_zj_mse/torch.squeeze(torch.minimum(min_var_zi, min_var_zj))
    return loss_ij

def cross_r2_loss(model, x, z, out_modal, variances):
    # This function computes the 1 - R2 score of the cross-modality 
    # reconstruction, which is the ratio of the model's reconstruction error 
    # with that of a dummy model which always outputs the mean of the data,
    # computed for each feature and then averaged.

    xr = model.modal_arms[out_modal].decoder(z)
    squares = torch.square(x - xr).sum(0)
    r2_error = (squares / (variances*z.shape[0])).mean()
    return r2_error

def r2_loss(x_tupl, xr_tupl, mask, variances):
    # This function computes the 1 - R2 score of the self-modality 
    # reconstruction, which compares the model's reconstruction error 
    # to that of a dummy model which always outputs the mean of the data,
    # computed for each feature and then averaged.

    squares = (torch.square(x[mask] - xr).sum(0) for (x, xr) in zip(x_tupl, xr_tupl))
    r2_error = sum([(square / (var*xr_tupl[0].shape[0])).mean() for (square, var) in zip(squares, variances)]) 
    return r2_error

def mse_loss(x_tupl, xr_tupl, mask):
    squares = (torch.square(x[mask] - xr).mean() for (x, xr) in zip(x_tupl, xr_tupl))
    mean_squared_error = sum(squares) / len(xr_tupl)
    return mean_squared_error

def combine_losses(cuml_losses, recon_losses, coupling_losses, cross_losses, total_loss):
    # This function takes an existing dictionary of cumulative loses and adds
    # a set of new loss values to it, matching across the different loss keys. 

    cuml_recon = {key: value + cuml_losses.get(key, 0) for (key, value) in recon_losses.items()}
    cuml_coupling = {key: value + cuml_losses.get(key, 0) for (key, value) in coupling_losses.items()}
    cuml_cross = {key: value + cuml_losses.get(key, 0) for (key, value) in cross_losses.items()}
    cuml_total = cuml_losses.get("total", 0) + total_loss
    new_cuml = {**cuml_losses, **cuml_recon, **cuml_coupling, **cuml_cross, "total": cuml_total}
    return new_cuml

def compute_weighted_loss(config, recon_loss_dict, coupling_dict, cross_dict):
    # This function takes a set of component losses and combines them into a 
    # a single scalar loss value using weighrs specified in the config YAML file.

    recon_loss = sum([config[modal]*loss_value for (modal, loss_value) in recon_loss_dict.items()])
    coupl_loss = sum([config[modal]*loss_value for (modal, loss_value) in coupling_dict.items()])
    cross_loss = sum([config[modal]*loss_value for (modal, loss_value) in cross_dict.items()])
    weighted_loss = recon_loss + coupl_loss + cross_loss
    return weighted_loss

def get_gauss_baselines(dataset):
    # This function computes the variances of the morphological and 
    # eletrophysiological modalities. It selects samples with the correct
    # modalities by using the indices of the passed MET_Dataset object,
    # which has already separated the samples by modality combination.

    e_indices = np.concatenate(
        [dataset.modal_indices["E"], dataset.modal_indices["TE"], dataset.modal_indices["EM"],
        dataset.modal_indices["MET"]])
    m_indices = np.concatenate(
        [dataset.modal_indices["M"], dataset.modal_indices["TM"], dataset.modal_indices["EM"],
        dataset.modal_indices["MET"]])
    (xe, xm) = (dataset.MET["E_dat"][e_indices], dataset.MET["M_dat"][m_indices])
    (std_e, std_m) = (np.std(xe, 0, keepdims = True), np.std(xm, 0, keepdims = True))
    return (std_e, std_m)

def get_variances(dataset, modalities, device, dtype, reg = 1e-2):
    variances = {}
    for modal in modalities:
        data = dataset.MET.query(dataset.allowed_specimen_ids, [modal])[f"{modal}_dat"]
        var = (np.square(data - data.mean(0, keepdims = True)).sum(0) + reg) / data.shape[0] 
        variances[modal] = torch.from_numpy(var).to(device, dtype)
    return variances

def filter_specimens(met_data, specimen_ids, config):
    platforms = config["select"]["platforms"]
    specimens = met_data.query(specimen_ids, platforms = platforms)["specimen_id"]
    return specimens

def build_model(config, train_dataset):
    # This function builds the model specified in the config YAML file. It completes
    # the specification by computing the baseline variances of the morphological and
    # electro-physiological data.

    model_config = config.copy()
    (std_e, std_m) = get_gauss_baselines(train_dataset)
    model_config["gauss_e_baseline"] = std_e.astype("float32")
    model_config["gauss_m_baseline"] = std_m.astype("float32")
    model = MultiModal(model_config)
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

def process_batch(model, X_dict, mask_dict, config, var_dict):
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
        (arm, x_tupl, mask) = (model.modal_arms[modal], X_dict[modal], mask_dict[modal])
        (z, xr) = arm(*(x[mask] for x in x_tupl))
        xr_tupl = (xr,) if type(xr) != tuple else xr
        latent_dict[modal] = z
        recon_dict[modal] = xr_tupl
        recon_loss_dict[modal] = r2_loss(x_tupl, xr_tupl, mask, (var_dict[modal],))
        for (prev_modal, prev_z) in list(latent_dict.items())[:-1]:
            prev_mask = mask_dict[prev_modal]
            if torch.any(prev_mask[mask]):
                (z_masked, prev_masked) = (z[prev_mask[mask]], prev_z[mask[prev_mask]])
                coupling_dict[f"{prev_modal}-{modal}"] = min_var_loss(z_masked, prev_masked.detach())
                coupling_dict[f"{modal}-{prev_modal}"] = min_var_loss(z_masked.detach(), prev_masked)
                (x_masked, x_prev_masked) = (X_dict[modal][0][mask & prev_mask], X_dict[prev_modal][0][mask & prev_mask])
                cross_dict[f"{modal}={prev_modal}"] = cross_r2_loss(model, x_prev_masked, z_masked.detach(), prev_modal, var_dict[prev_modal])
                cross_dict[f"{prev_modal}={modal}"] = cross_r2_loss(model, x_masked, prev_masked.detach(), modal, var_dict[modal])
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
    train_variances = get_variances(train_dataset, model.modal_arms, config["device"], torch.float32)
    val_variances = get_variances(val_dataset, model.modal_arms, config["device"], torch.float32)
    (exp_dir / "checkpoints").mkdir(exist_ok = True)
    for epoch in range(config["num_epochs"]):
        # Training -----------
        cuml_losses = {}
        model.train()
        if config["check_step"] > 0 and epoch % config["check_step"] == 0:
            torch.save(model.state_dict(), exp_dir / "checkpoints" / f"model_{epoch}.pt")
        for (X_dict, mask_dict, specimen_ids) in train_loader:
            optimizer.zero_grad()
            (latent_dict, recon_dict, recon_loss_dict, coupling_dict, cross_dict) = process_batch(model, X_dict, mask_dict, config, train_variances)
            loss = compute_weighted_loss(config, recon_loss_dict, coupling_dict, cross_dict)
            loss.backward()
            optimizer.step()
            cuml_losses = combine_losses(cuml_losses, recon_loss_dict, coupling_dict, cross_dict, loss)
        avg_losses = {key: value / len(train_dataset) for (key, value) in cuml_losses.items()}
        # Validation -----------
        with torch.no_grad():
            cuml_val_losses = {}
            for (X_val, mask_val, _) in val_loader:
                model.eval()
                (latent_val, recon_val, recon_loss_val, coupling_val, cross_val) = process_batch(model, X_val, mask_val, config, val_variances)
                val_loss = compute_weighted_loss(config, recon_loss_val, coupling_val, cross_val)
                cuml_val_losses = combine_losses(cuml_val_losses, recon_loss_val, coupling_val, cross_val, val_loss)
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
    num_folds = config["folds"]
    if num_folds > 0:
        indices = list(met_data.get_stratified_KFold(config["folds"], seed = config["seed"]))
    else:
        (train_ids, test_ids) = met_data.get_stratified_split(config["val_split"], seed = config["seed"])
        indices = [(train_ids, test_ids)]
        num_folds = 1
    for (fold, (train_ids, test_ids)) in enumerate(indices, 1):
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

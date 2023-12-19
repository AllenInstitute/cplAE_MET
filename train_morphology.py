import subprocess
import os
import yaml
import pathlib
import argparse
import shutil

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import scipy.io as sio

from cplAE_MET.models.subnetworks_M import AE_M
from cplAE_MET.models.train_utils import optimizer_to
from cplAE_MET.utils.utils import save_ckp

class MorphoDatasetAE(torch.utils.data.Dataset):
    def __init__(self, arbor_mat_path):
        super().__init__()
        data_dict = sio.loadmat(arbor_mat_path)
        all_arbors = data_dict["hist_ax_de_api_bas"]
        not_nan = ~np.any(np.isnan(all_arbors), 0)
        self.arbors = torch.as_tensor(all_arbors[not_nan])
        self.specimen_ids = np.asarray(data_dict["specimen_id"])[not_nan]
        self.gnoise_m_std = torch.var(self.arbors, dim = 0, keepdim = True).sqrt()

    def __len__(self):
        return self.arbors.shape[0]

    def __getitem__(self, idx):
        return (self.arbors[idx], self.specimen_ids[idx])

def get_git_hash():
    try:
        git_hash = subprocess.run(
            (["powershell"] if os.name == "nt" else []) + ["git", "rev-parse", "--short", "HEAD"],
            capture_output = True 
        ).stdout.decode().strip()
    except Exception:
        git_hash = ""
    return git_hash

def record_settings(exp_dir, config_path):
    git_hash = get_git_hash()
    if git_hash:
        with open(exp_dir / "git_hash.txt", "w") as target:
            target.write(git_hash)
    else:
        print("Git hash not saved.")
    shutil.copy(config_path, exp_dir / "config.yaml")

def clear_experiment(exp_dir):
    model_path = exp_dir / "model.pt"
    if model_path.exists():
        model_path.unlink()
    output_path = exp_dir / "output.pkl"
    if output_path.exists():
        output_path.unlink()
    tensorboard_path = exp_dir / "tn_board"
    if tensorboard_path.exists():
        for path in tensorboard_path.iterdir():
            path.unlink()

def build_model(config, train_dataset):
    model_config = dict(
        variational = False,
        latent_dim = config["latent_dim"], 
        batch_size = config["batch_size"],
        KLD_beta = 1.0,
        M = dict(gnoise_std = train_dataset.gnoise_m_std,
                gnoise_std_frac = config["gauss_var_frac"],  
                dropout_p = config["dropout"]),
        )  
    model = AE_M(model_config)
    return model

def train(num_epochs, exp_dir, model, optimizer, train_dataloader, device):
    model.to(device)
    optimizer_to(optimizer, device)
    tb_writer = SummaryWriter(log_dir = exp_dir / "tn_board")
    # Training -----------
    for epoch in range(num_epochs):
        print(epoch)
        model.train()
        culm_loss = 0
        for (arbors, specimen_ids) in iter(train_dataloader):
            optimizer.zero_grad()
            # forward pass -----------
            (zm_int_enc, zm, zm_int_dec, xrm, mu, log_sigma) = model(arbors.cuda())
            loss = torch.square(arbors - zm).sum() / arbors.shape[0]
            culm_loss += loss.item()
            loss.backward()
            optimizer.step()            
        # Average losses over batches -----------
        avg_loss = culm_loss / len(train_dataloader)
        # Logging -----------
        tb_writer.add_scalar('Train/MSE_XM', avg_loss, epoch)
    tb_writer.close()
    # Save outputs -----------
    model.eval()
    outputs = [(model(arbors)[1], specimen_ids) for (arbors, specimen_ids) in iter(train_dataloader)]
    latent_space = torch.cat([arbors for (arbors, ids) in outputs], 0).numpy()
    specimen_ids = np.concatenate([ids for (arbors, ids) in outputs], 0)
    np.savez_compressed(exp_dir / "outputs.npz", specimen_ids = specimen_ids, latent_space = latent_space)
    return model

def train_model(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = MorphoDatasetAE(config["arbor_mat_file"])
    train_dataloader = DataLoader(train_dataset, batch_size = config["batch_size"], shuffle = True)
    model = build_model(config, train_dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr = config["learning_rate"])
    trained_model = train(config["num_epochs"], exp_dir, model, optimizer, train_dataloader, device)
    checkpoint = {'state_dict': trained_model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_ckp(checkpoint, exp_dir, "model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help = "path to config yaml file")
    args = parser.parse_args()
    with open(args.config_path, "r") as target:
        config = yaml.safe_load(target)
    exp_dir = pathlib.Path(config["output_dir"]) / config["experiment_name"]
    if exp_dir.exists():
        clear_experiment(exp_dir)
    else:
        exp_dir.mkdir()
    record_settings(exp_dir, args.config_path)
    train_model(config)
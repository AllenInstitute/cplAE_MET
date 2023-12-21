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
    def __init__(self, arbor_mat_path, combine):
        super().__init__()
        data_dict = sio.loadmat(arbor_mat_path)
        all_arbors = data_dict["hist_ax_de_api_bas"]
        not_nan = ~np.any(np.isnan(all_arbors), (1, 2, 3))
        self.arbors = torch.as_tensor(all_arbors[not_nan])[:, None].float()
        if combine:
            self.arbors = self.arbors[..., :2] + self.arbors[..., 2:]
        self.specimen_ids = np.asarray(data_dict["specimen_id"])[not_nan]
        self.gnoise_m_std = torch.var(self.arbors, dim = 0, keepdim = True).sqrt().float()

    def __len__(self):
        return self.arbors.shape[0]

    def __getitem__(self, idx):
        return (self.arbors[idx], self.specimen_ids[idx])

def build_model(config, train_dataset):
    model_config = dict(
        variational = False,
        combine = config["combine_types"],
        latent_dim = config["latent_dim"], 
        batch_size = config["batch_size"],
        KLD_beta = 1.0,
        M = dict(gnoise_std = train_dataset.gnoise_m_std,
                gnoise_std_frac = config["gauss_var_frac"],  
                dropout_p = config["dropout"]),
        )  
    model = AE_M(model_config)
    return model

def train(num_epochs, exp_dir, model, optimizer, train_dataloader, device, print_step = False):
    model.to(device)
    optimizer_to(optimizer, device)
    tb_writer = SummaryWriter(log_dir = exp_dir / "tn_board")
    # Training -----------
    for epoch in range(num_epochs):
        model.train()
        culm_loss = 0
        for (step, (arbors, specimen_ids)) in enumerate(iter(train_dataloader)):
            if print_step:
                print(f"Epoch {epoch + 1}: {step + 1} / {len(train_dataloader)}", end = "\r")
            optimizer.zero_grad()
            arbors = arbors.to(device)
            # forward pass -----------
            (zm_int_enc, zm, zm_int_dec, xrm, mu, log_sigma) = model(arbors)
            loss = torch.square(arbors - xrm).sum() / arbors.shape[0]
            culm_loss += loss.item()
            loss.backward()
            optimizer.step()            
        # Average losses over batches -----------
        avg_loss = culm_loss / len(train_dataloader)
        print(f"\nEpoch {epoch + 1}: Avg Loss {avg_loss:.4f}")
        # Logging -----------
        tb_writer.add_scalar('Train/MSE_XM', avg_loss, epoch)
    tb_writer.close()
    # Save outputs -----------
    model.eval()
    with torch.no_grad():
        outputs = [(model(arbors.to(device))[1], specimen_ids) for (arbors, specimen_ids) in iter(train_dataloader)]
    latent_space = torch.cat([arbors.cpu() for (arbors, ids) in outputs], 0).numpy()
    specimen_ids = np.concatenate([ids for (arbors, ids) in outputs], 0)
    np.savez_compressed(exp_dir / "outputs.npz", specimen_ids = specimen_ids, latent_space = latent_space)
    return model

def train_model(config, exp_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = MorphoDatasetAE(config["arbor_mat_file"], combine = config["combine_types"])
    train_dataloader = DataLoader(train_dataset, batch_size = config["batch_size"], shuffle = True)
    model = build_model(config, train_dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr = config["learning_rate"])
    trained_model = train(config["num_epochs"], exp_dir, model, optimizer, train_dataloader, device)
    checkpoint = {'state_dict': trained_model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_ckp(checkpoint, exp_dir, "model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help = "Name of experiment.")
    parser.add_argument("config_path", help = "Path to config YAML file.")
    args = parser.parse_args()
    with open(args.config_path, "r") as target:
        config = yaml.safe_load(target)
    exp_path = pathlib.Path(config["output_dir"]) / args.exp_name
    exp_path.mkdir(exist_ok = True)
    train_model(config, exp_path)
# Dataset class that can load data at different levels of preprocessing, and select
# for different modalities, platforms, etc.

# Model class that can be initialized with different autoencoder branches, where a 
# branch is only loaded if it couples to another branch.

# Loss generating class/function that can be used to easily produce a loss function
# with the desired parameters.

# Training function should have a consistent form across different model types, 
# abstracting away particular of architecture and loss function.

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

from cplAE_MET.utils.dataset import MET_exc_inh
from cplAE_MET.models.torch_utils import MET_dataset
from cplAE_MET.models.model_classes import Model_ME_T_conv, MultiModal
from cplAE_MET.models.train_utils import init_losses, save_results, optimizer_to, Criterion
from cplAE_MET.models.optuna_utils import run_classification
from cplAE_MET.utils.utils import save_ckp

def get_dataloaders(dat, device):
    dat.XM = np.expand_dims(dat.XM, axis = 1) # TODO check if 1D Conv could be used here
    (train_ind, val_ind) = dat.train_val_split(fold = config["fold"], n_folds = 10, seed = 0)
    (train_dat, val_dat) = (dat[train_ind,:], dat[val_ind,:])
    # We define weights for each sample in a way that in each batch there are at least 54 met cells from exc and
    # 54 met cells from inh data. 54 was decided based on the previous runs just by observation!
    # Weighted sampling strategy -----------
    weights = train_dat.make_weights_for_balanced_classes(n_met = 54, met_subclass_id = [2, 3], batch_size = 1000)                                   
    sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights), len(weights)) 
    # Dataset and Dataloader -----------
    train_dataset = MET_dataset(train_dat, device = device)
    train_dataloader = DataLoader(train_dataset, batch_size = config["batch_size"], 
                                  shuffle = False, drop_last = True, sampler = sampler)
    val_dataset = MET_dataset(val_dat, device = device)
    val_dataloader = DataLoader(val_dataset, batch_size = len(val_dataset), shuffle = False)
    full_dataset = MET_dataset(dat, device = device)
    full_dataloader = DataLoader(full_dataset, batch_size = config["batch_size"], shuffle = False, 
                            drop_last = False)
    return (train_ind, val_ind, train_dataloader, val_dataloader, full_dataloader)

def build_model(config, train_dataset):
    model_config = config.copy()
    model_config["gauss_e_baseline"] = train_dataset.gnoise_e_std
    model_config["gauss_m_baseline"] = train_dataset.gnoise_m_std
    model = MultiModal(model_config)
    return model

def log_tensorboard(tb_writer, train_loss, val_loss, epoch):
    tb_writer.add_scalars("Train/MSE", {
        "XT": train_loss.get('rec_t', 0), "XM": train_loss.get('rec_m', 0),
        "XE": train_loss.get('rec_e', 0), "XM_ME": train_loss.get('rec_m_me', 0),
        "XE_ME": train_loss.get('rec_e_me', 0)
    }, epoch)
    tb_writer.add_scalars("Validation/MSE", {
        "XT": val_loss.get('rec_t', 0), "XM": val_loss.get('rec_m', 0),
        "XE": val_loss.get('rec_e', 0), "XM_ME": val_loss.get('rec_m_me', 0),
        "XE_ME": val_loss.get('rec_e_me', 0)
    }, epoch)
    tb_writer.add_scalars("Train/cpl", {
        "T-E": train_loss.get('cpl_t->e', 0), "E-T": train_loss.get('cpl_e->t', 0),
        "T-M": train_loss.get('cpl_t->m', 0), "M-T": train_loss.get('cpl_m->t', 0),
        "ME-T": train_loss.get('cpl_me->t', 0), "T-ME": train_loss.get('cpl_t->me', 0),
        "ME-M": train_loss.get('cpl_me->m', 0), "ME-E": train_loss.get('cpl_me->e', 0)
    }, epoch)
    tb_writer.add_scalars("Validation/cpl", {
        "T-E": val_loss.get('cpl_t->e', 0), "E-T": val_loss.get('cpl_e->t', 0),
        "T-M": val_loss.get('cpl_t->m', 0), "M-T": val_loss.get('cpl_m->t', 0),
        "ME-T": val_loss.get('cpl_me->t', 0), "T-ME": val_loss.get('cpl_t->me', 0),
        "ME-M": val_loss.get('cpl_me->m', 0), "ME-E": val_loss.get('cpl_me->e', 0)
    }, epoch)

    # tb_writer.add_scalar('Train/MSE_XT', train_loss.get('rec_t', 0), epoch)
    # tb_writer.add_scalar('Validation/MSE_XT', val_loss.get('rec_t', 0), epoch)
    # tb_writer.add_scalar('Train/MSE_XM', train_loss.get('rec_m', 0), epoch)
    # tb_writer.add_scalar('Validation/MSE_XM', val_loss.get('rec_m', 0), epoch)
    # tb_writer.add_scalar('Train/MSE_XE', train_loss.get('rec_e', 0), epoch)
    # tb_writer.add_scalar('Validation/MSE_XE', val_loss.get('rec_e', 0), epoch)
    # tb_writer.add_scalar('Train/MSE_M_XME', train_loss.get('rec_m_me', 0), epoch)
    # tb_writer.add_scalar('Validation/MSE_M_XME', val_loss.get('rec_m_me', 0), epoch)
    # tb_writer.add_scalar('Train/MSE_E_XME', train_loss.get('rec_e_me', 0), epoch)
    # tb_writer.add_scalar('Validation/MSE_E_XME', val_loss.get('rec_e_me', 0), epoch)
    # tb_writer.add_scalar('Train/cpl_T->E', train_loss.get('cpl_t->e', 0), epoch)
    # tb_writer.add_scalar('Validation/cpl_T->E', val_loss.get('cpl_t->e', 0), epoch)
    # tb_writer.add_scalar('Train/cpl_E->T', train_loss.get('cpl_e->t', 0), epoch)
    # tb_writer.add_scalar('Validation/cpl_E->T', val_loss.get('cpl_e->t', 0), epoch)
    # tb_writer.add_scalar('Train/cpl_T->M', train_loss.get('cpl_t->m', 0), epoch)
    # tb_writer.add_scalar('Validation/cpl_T->M', val_loss.get('cpl_t->m', 0), epoch)
    # tb_writer.add_scalar('Train/cpl_M->T', train_loss.get('cpl_m->t', 0), epoch)
    # tb_writer.add_scalar('Validation/cpl_M->T', val_loss.get('cpl_m->t', 0), epoch)
    # tb_writer.add_scalar('Train/cpl_ME->T', train_loss.get('cpl_me->t', 0), epoch)
    # tb_writer.add_scalar('Validation/cpl_ME->T', val_loss.get('cpl_me->t', 0), epoch)
    # tb_writer.add_scalar('Train/cpl_T->ME', train_loss.get('cpl_t->me', 0), epoch)
    # tb_writer.add_scalar('Validation/cpl_T->ME', val_loss.get('cpl_t->me', 0), epoch)
    # tb_writer.add_scalar('Train/cpl_ME->M', train_loss.get('cpl_me->m', 0), epoch)
    # tb_writer.add_scalar('Validation/cpl_ME->M', val_loss.get('cpl_me->m', 0), epoch)
    # tb_writer.add_scalar('Train/cpl_ME->E', train_loss.get('cpl_me->e', 0), epoch)
    # tb_writer.add_scalar('Validation/cpl_ME->E', val_loss.get('cpl_me->e', 0), epoch)

def train_and_evaluate(num_epochs, exp_dir, model_config, model, optimizer, train_dataloader, val_dataloader, device):
    '''Train and evaluation function, this will be called at each trial and epochs will start from zero'''
    model.to(device)
    tb_writer = SummaryWriter(log_dir = exp_dir / "tn_board")
    # Training -----------
    for epoch in range(num_epochs):
        print(epoch + 1)
        model.train()
        for step, batch in enumerate(iter(train_dataloader)):
            optimizer.zero_grad()
            # forward pass -----------
            loss_dict, _, _ = model(batch)
            loss = Criterion(model_config, loss_dict)
            loss.backward()
            optimizer.step()
            if step == 0:
                train_loss = init_losses(loss_dict)
            # track loss over batches -----------
            for k, v in loss_dict.items():
                train_loss[k] += v
        # Average losses over batches -----------
        for k, v in train_loss.items():
            train_loss[k] = train_loss[k] / len(train_dataloader)
        # Validation -----------
        with torch.no_grad():
            for val_batch in iter(val_dataloader):
                model.eval()
                val_loss, _, _ = model(val_batch)
        log_tensorboard(tb_writer, train_loss, val_loss, epoch)
    tb_writer.close()
    return model

def train_model(config, exp_dir):
    (dat, D) = MET_exc_inh.from_file(config["data_file"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    (train_ind, val_ind, train_dataloader, val_dataloader, full_dataloader) = get_dataloaders(dat, device)
    model = build_model(config, train_dataloader.dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr = config["learning_rate"])
    trained_model = train_and_evaluate(config["num_epochs"], exp_dir, config, model, optimizer, train_dataloader, val_dataloader, device)
    save_results(trained_model, full_dataloader, D, exp_dir / "output.pkl", train_ind, val_ind)
    checkpoint = {
            'state_dict': trained_model.state_dict(),
            'optimizer': optimizer.state_dict()
            }
    save_ckp(checkpoint, exp_dir, "model")

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
    
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
from cplAE_MET.models.model_classes import Model_ME_T_conv
from cplAE_MET.models.train_utils import init_losses, save_results, optimizer_to, Criterion
from cplAE_MET.models.optuna_utils import run_classification
from cplAE_MET.utils.utils import save_ckp

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
        T = dict(dropout_p = config["dropout"], 
                alpha_T = config["alpha_T"]),
        E = dict(gnoise_std = train_dataset.gnoise_e_std, 
                gnoise_std_frac = config["gauss_var_frac"], 
                dropout_p = config["dropout"], 
                alpha_E = config['alpha_E']),
        M = dict(gnoise_std = train_dataset.gnoise_m_std,
                gnoise_std_frac = config["gauss_var_frac"],  
                dropout_p = config["dropout"], 
                alpha_M = config['alpha_M']),
        TE = dict(lambda_TE = 1.0,
                lambda_tune_T_E = config['lambda_T_E'], 
                lambda_tune_E_T = config['lambda_E_T']),
        TM=dict(lambda_TM = 1.0, 
                lambda_tune_T_M = config['lambda_T_M'], 
                lambda_tune_M_T = config['lambda_M_T']),
        ME = dict(alpha_ME = config['alpha_ME']), 
        ME_T = dict(lambda_ME_T = 1.0, 
                    lambda_tune_ME_T = config['lambda_ME_T'], 
                    lambda_tune_T_ME = config['lambda_T_ME']),
        ME_M = dict(lambda_ME_M = 1.0, 
                    lambda_tune_ME_M = config['lambda_ME_M']), 
        ME_E=dict(lambda_ME_E = 1.0, 
                    lambda_tune_ME_E = config['lambda_ME_E']) 
        )  
    model = Model_ME_T_conv(model_config)
    return model, model_config

def train_and_evaluate(num_epochs, exp_dir, model_config, model, optimizer, train_dataloader, val_dataloader, device):
    '''Train and evaluation function, this will be called at each trial and epochs will start from zero'''

    model.to(device)
    optimizer_to(optimizer, device)

    tb_writer = SummaryWriter(log_dir = exp_dir / "tn_board")

    # Training -----------
    for epoch in range(num_epochs):
        print(epoch)
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
        
        # If it is an optimization run, we dont log all the losses and so on. We only save 
        # everything if it is a specific non-optimization run that we want to monitor things closely.
        # Logging -----------
        tb_writer.add_scalar('Train/MSE_XT', train_loss['rec_t'], epoch)
        tb_writer.add_scalar('Validation/MSE_XT', val_loss['rec_t'], epoch)
        tb_writer.add_scalar('Train/MSE_XM', train_loss['rec_m'], epoch)
        tb_writer.add_scalar('Validation/MSE_XM', val_loss['rec_m'], epoch)
        tb_writer.add_scalar('Train/MSE_XE', train_loss['rec_e'], epoch)
        tb_writer.add_scalar('Validation/MSE_XE', val_loss['rec_e'], epoch)
        tb_writer.add_scalar('Train/MSE_M_XME', train_loss['rec_m_me'], epoch)
        tb_writer.add_scalar('Validation/MSE_M_XME', val_loss['rec_m_me'], epoch)
        tb_writer.add_scalar('Train/MSE_E_XME', train_loss['rec_e_me'], epoch)
        tb_writer.add_scalar('Validation/MSE_E_XME', val_loss['rec_e_me'], epoch)
        tb_writer.add_scalar('Train/cpl_T->E', train_loss['cpl_t->e'], epoch)
        tb_writer.add_scalar('Validation/cpl_T->E', val_loss['cpl_t->e'], epoch)
        tb_writer.add_scalar('Train/cpl_E->T', train_loss['cpl_e->t'], epoch)
        tb_writer.add_scalar('Validation/cpl_E->T', val_loss['cpl_e->t'], epoch)
        tb_writer.add_scalar('Train/cpl_T->M', train_loss['cpl_t->m'], epoch)
        tb_writer.add_scalar('Validation/cpl_T->M', val_loss['cpl_t->m'], epoch)
        tb_writer.add_scalar('Train/cpl_M->T', train_loss['cpl_m->t'], epoch)
        tb_writer.add_scalar('Validation/cpl_M->T', val_loss['cpl_m->t'], epoch)
        tb_writer.add_scalar('Train/cpl_ME->T', train_loss['cpl_me->t'], epoch)
        tb_writer.add_scalar('Validation/cpl_ME->T', val_loss['cpl_me->t'], epoch)
        tb_writer.add_scalar('Train/cpl_T->ME', train_loss['cpl_t->me'], epoch)
        tb_writer.add_scalar('Validation/cpl_T->ME', val_loss['cpl_t->me'], epoch)
        tb_writer.add_scalar('Train/cpl_ME->M', train_loss['cpl_me->m'], epoch)
        tb_writer.add_scalar('Validation/cpl_ME->M', val_loss['cpl_me->m'], epoch)
        tb_writer.add_scalar('Train/cpl_ME->E', train_loss['cpl_me->e'], epoch)
        tb_writer.add_scalar('Validation/cpl_ME->E', val_loss['cpl_me->e'], epoch)    
    
    tb_writer.close()
    return model

def train_model(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    (dat, D) = MET_exc_inh.from_file(config["data_file"])
    dat.XM = np.expand_dims(dat.XM, axis = 1) # TODO check if 1D Conv could be used here
    
    (train_ind, val_ind) = dat.train_val_split(fold = config["fold"], n_folds = 10, seed = 0)
    train_dat = dat[train_ind,:]
    val_dat = dat[val_ind,:]
    train_dat = dat[train_ind,:]
    T_labels_for_classification = np.array(dat.merged_cluster_label_at50)

    # We define weights for each sample in a way that in each batch there are at least 54 met cells from exc and
    # 54 met cells from inh data. 54 was decided based on the previous runs just by observation!
    # Weighted sampling strategy -----------
    weights = train_dat.make_weights_for_balanced_classes(n_met = 54, met_subclass_id = [2, 3], batch_size = 1000)                                                                
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights)) 
    
    # Dataset and Dataloader -----------
    train_dataset = MET_dataset(train_dat, device = device)
    train_dataloader = DataLoader(train_dataset, batch_size = config["batch_size"], 
                                  shuffle = False, drop_last = True, sampler = sampler)

    val_dataset = MET_dataset(val_dat, device = device)
    val_dataloader = DataLoader(val_dataset, batch_size = len(val_dataset), shuffle = False)

    dataset = MET_dataset(dat, device = device)
    dataloader = DataLoader(dataset, batch_size = config["batch_size"], shuffle = False, 
                            drop_last = False)
    
    (model, model_config) = build_model(config, train_dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr = config["learning_rate"])

    trained_model = train_and_evaluate(config["num_epochs"], exp_dir, model_config, model, optimizer, train_dataloader, val_dataloader, device)
    model_score = run_classification(model, dataloader, train_ind, val_ind, T_labels_for_classification)

    save_results(trained_model, dataloader, D, exp_dir / "output.pkl", train_ind, val_ind)
    checkpoint = {
            'state_dict': trained_model.state_dict(),
            'optimizer': optimizer.state_dict()
            }
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
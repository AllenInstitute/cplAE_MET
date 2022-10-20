# %%
import os
import torch
import shutil
import argparse
import numpy as np
from pathlib import Path

from torch.utils.data import DataLoader
from cplAE_MET.models.model_classes import Model_M
from cplAE_MET.utils.load_config import load_config
from cplAE_MET.utils.dataset import MET_exc_inh
from cplAE_MET.utils.utils import savepkl
from torch_utils import add_noise, scale_depth, center_resize, tonumpy

from torch.utils.tensorboard import SummaryWriter

from cplAE_MET.models.torch_utils import MET_dataset

parser = argparse.ArgumentParser()
# TODO
parser.add_argument('--exp_name',        default='Monly_v0',                  type=str,   help='Experiment set')
parser.add_argument('--alpha_M',         default=1.0,                         type=float, help='M reconstruction loss weight')
parser.add_argument('--alpha_sd',        default=1.0,                         type=float, help='soma depth reconstruction loss weight')
parser.add_argument('--scale_by',        default=0.3,                         type=float, help='scaling factor for M_data interpolation')
parser.add_argument('--config_file',     default='config.toml',               type=str,   help='config file with data paths')
parser.add_argument('--n_epochs',        default=50000,                       type=int,   help='Number of epochs to train')
parser.add_argument('--fold_n',          default=0,                           type=int,   help='kth fold in 10-fold CV splits')
parser.add_argument('--latent_dim',      default=2,                           type=int,   help='Number of latent dims')
parser.add_argument('--batch_size',      default=1000,                        type=int,   help='Batch size')



def set_paths(config_file=None, exp_name='DEBUG', fold_n=0):
    paths = load_config(config_file=config_file, verbose=False)
    paths['result'] = f'{str(paths["package_dir"] / "data/results")}/{exp_name}/'
    Path(paths['result']).mkdir(parents=False, exist_ok=True)
    paths['tb_logs'] = f'{str(paths["package_dir"] / "data/results")}/tb_logs/{exp_name}/fold_{str(fold_n)}/'
    if os.path.exists(paths['tb_logs']):
        shutil.rmtree(paths['tb_logs'])
    Path(paths['tb_logs']).mkdir(parents=True, exist_ok=False)
    return paths

def save_ckp(state, checkpoint_dir, fname):
    filename = fname + '.pt'
    f_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, f_path)

def init_losses(loss_dict):
    t_loss = {}
    for k in loss_dict.keys():
        t_loss[k] = 0.
    return t_loss

def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val

def save_results(model, dataloader, dat, fname):
    '''
    Takes the model, run it in the evaluation mode to calculate the embeddings and reconstructions for printing out.
    Also classification is run inside this function
    '''
    # Run the model in the evaluation mode
    # TODO
    model.eval()

    for all_data in iter(dataloader):
        with torch.no_grad():
            loss_dict, z_dict, xr_dict = model(all_data) 

    # TODO
    savedict = {'XM': tonumpy(all_data['xm']),
                'Xsd': tonumpy(all_data['xsd']),
                'Xrsd': tonumpy(xr_dict['xrsd']),
                'XrM': tonumpy(xr_dict['xrm']),
                'specimen_id': np.array([mystr.rstrip() for mystr in dat.specimen_id]),
                'cluster_label': np.array([mystr.rstrip() for mystr in dat.cluster_label]),
                'cluster_color': np.array([mystr.rstrip() for mystr in dat.cluster_color]),
                'cluster_id': dat.cluster_id,
                'gene_ids': dat.gene_ids,
                'e_features': dat.E_features,
                'loss_rec_xm': tonumpy(loss_dict['rec_m']),
                'loss_rec_xsd': tonumpy(loss_dict['rec_sd']),
                'zm': tonumpy(z_dict['zm']),
                'is_m_1d':tonumpy(all_data['is_m_1d'])}


    savepkl(savedict, fname)

    model.train()

    return

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def main(exp_name="DEBUG",
         config_file="config.toml", 
         n_epochs=10000, 
         fold_n=0, 
         alpha_M=1.0,
         alpha_sd=1.0,
         augment_decoders=0,
         scale_by=0.1,
         latent_dim=2,
         batch_size=1000):

    # Set the device -----------
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Train test split -----------
    dir_pth = set_paths(config_file, exp_name=exp_name, fold_n=fold_n)
    tb_writer = SummaryWriter(log_dir=dir_pth['tb_logs'])
    dat = MET_exc_inh.from_file(dir_pth['MET_data'])
    train_ind, val_ind = dat.train_val_split(fold=fold_n, n_folds=10, seed=0)
    train_dat = dat[train_ind,:]
    val_dat = dat[val_ind,:]

    # Copy the running code into the result dir -----------
    shutil.copy(__file__, dir_pth['result'])

    # Dataset and Dataloader -----------
    train_dataset = MET_dataset(train_dat, device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dataset = MET_dataset(val_dat, device=device)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    dataset = MET_dataset(dat, device=device)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    # TODO
    # Model init -----------
    model_config = dict(latent_dim=latent_dim,
                        batch_size=batch_size,
                        M=dict(gnoise_std_frac=0.1,
                               scale_by=scale_by, 
                               interpolation_mode="nearest",
                               random=True,
                               alpha_M=alpha_M,
                               alpha_sd=alpha_sd)
                        )
    
    
    model = Model_M(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    fileid = ( f"aM_{str(model_config['M']['alpha_M'])}_" +
               f"asd_{str(model_config['M']['alpha_sd'])}_"+
               f"Mnoise_{model_config['M']['gnoise_std_frac']}_"+
               f"Mscale_{str(model_config['M']['scale_by'])}_"+
               f"ld_{model_config['latent_dim']:d}_ne_{n_epochs:d}_"+
               f"fold_{fold_n:d}").replace('.', '-')


    # Training -----------
    for epoch in range(n_epochs):
        model.train()
        for step, batch in enumerate(iter(train_dataloader)):
            xm=batch['xm']
            xsd=batch['xsd']
            is_m_1d=batch['is_m_1d']

            optimizer.zero_grad()

            # apply M augmentations on m_cells ----------- 
            noisy_xsd = add_noise(xsd, clamp_min=0.)
            noisy_xm = add_noise(xm, sd=model_config['M']['gnoise_std_frac'], clamp_min=0., scale_by_x=True)

            aug_xm_mcells = torch.zeros(xm[is_m_1d].shape).to(device=device)
            target_size = xm.shape[2:]
            for sample in range(is_m_1d.sum()):
                _, scaled_xm = scale_depth(torch.unsqueeze(noisy_xm[is_m_1d][sample, ...], dim=0), 
                                           scale_by= model_config['M']['scale_by'], 
                                           interpolation_mode=model_config['M']['interpolation_mode'],
                                           random=model_config['M']['random'])
                scaled_xm = torch.squeeze(scaled_xm)
                aug_xm_mcells[sample, 0, ...] = center_resize(scaled_xm, target_size)
            aug_xm = torch.zeros(xm.shape).to(device=device)
            aug_xm[is_m_1d] = aug_xm_mcells

            batch['aug_xm'] = aug_xm
            batch['noisy_xsd'] = noisy_xsd
            
            # TODO
            # forward pass T, E, M -----------
            loss_dict, _, _ = model(batch)

            loss = model_config['M']['alpha_M'] * loss_dict['rec_m'] + \
                   model_config['M']['alpha_sd'] * loss_dict['rec_sd'] 
       
            loss.backward()
            optimizer.step()

            if step == 0:
                train_loss = init_losses(loss_dict)

            # track loss over batches -----------
            for k, v in loss_dict.items():
                train_loss[k] += v

        # Validation -----------
        with torch.no_grad():
            for val_batch in iter(val_dataloader):
                model.eval()
                val_loss, _, _ = model(val_batch)

        model.train()

        # Average losses over batches -----------
        for k, v in train_loss.items():
            train_loss[k] = train_loss[k] / len(train_dataloader)

        # printing logs -----------
        for k, v in train_loss.items():
            print(f'epoch {epoch:04d},  Train {k}: {v:.5f}')
        
        for k, v in val_loss.items():
            print(f'epoch {epoch:04d} ----- Val {k}: {v:.5f}')

        # TODO
        # Logging -----------
        tb_writer.add_scalar('Train/MSE_XM', train_loss['rec_m'], epoch)
        tb_writer.add_scalar('Validation/MSE_XM', val_loss['rec_m'], epoch)
        tb_writer.add_scalar('Train/MSE_Xsd', train_loss['rec_sd'], epoch)
        tb_writer.add_scalar('Validation/MSE_Xsd', val_loss['rec_sd'], epoch)
        
        # TODO
        # save model -----------
        if (((epoch) % 500) == 0):
            fname = dir_pth['result'] + f"checkpoint_ep_{epoch}_" + fileid 
            save_results(model, dataloader, dat, fname+".pkl")
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }
            save_ckp(checkpoint, dir_pth['result'], fname)

    fname = dir_pth['result'] + "exit_summary_" + fileid + ".pkl"
    save_results(model, dataloader, dat, fname)
    tb_writer.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))

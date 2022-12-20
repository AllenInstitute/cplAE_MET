# %%
import os
import torch
import shutil
import argparse
import numpy as np
from pathlib import Path

from torch.utils.data import DataLoader
from cplAE_MET.models.model_classes import Model_ME_T
from cplAE_MET.models.torch_utils import MET_dataset
from cplAE_MET.utils.load_config import load_config
from cplAE_MET.utils.dataset import MET_exc_inh
from cplAE_MET.utils.utils import savepkl
from torch_utils import add_noise, scale_depth, center_resize, tonumpy

from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
# TODO
parser.add_argument('--exp_name',        default='Tonly_Eonly_Monly_8643_MNarbor_l2_v0',  type=str,   help='Experiment set')
parser.add_argument('--alpha_T',         default=1.0,                         type=float, help='T reconstruction loss weight')
parser.add_argument('--alpha_M',         default=1.0,                         type=float, help='M reconstruction loss weight')
parser.add_argument('--alpha_sd',        default=1.0,                         type=float, help='soma depth reconstruction loss weight')
parser.add_argument('--alpha_E',         default=1.0,                         type=float, help='E reconstruction loss weight')
parser.add_argument('--alpha_ME',        default=1.0,                         type=float, help='ME reconstruction loss weight')
parser.add_argument('--lambda_TE',       default=0.0,                         type=float, help='coupling loss weight between T and E')
parser.add_argument('--lambda_TM',       default=0.0,                         type=float, help='coupling loss weight between T and M')
parser.add_argument('--lambda_ME',       default=0.0,                         type=float, help='coupling loss weight between M and E')
parser.add_argument('--lambda_ME_T',     default=0.0,                         type=float, help='coupling loss weight between ME and T')
parser.add_argument('--lambda_ME_M',     default=0.0,                         type=float, help='coupling loss weight between ME and M')
parser.add_argument('--lambda_ME_E',     default=0.0,                         type=float, help='coupling loss weight between ME and E')
parser.add_argument('--lambda_tune_T_E', default=0.0,                         type=float, help='Tune the directionality of coupling between T and E')
parser.add_argument('--lambda_tune_E_T', default=0.0,                         type=float, help='Tune the directionality of coupling between T and E')
parser.add_argument('--lambda_tune_T_M', default=0.0,                         type=float, help='Tune the directionality of coupling between T and M')
parser.add_argument('--lambda_tune_M_T', default=0.0,                         type=float, help='Tune the directionality of coupling between T and M')
parser.add_argument('--lambda_tune_E_M', default=0.0,                         type=float, help='Tune the directionality of coupling between M and E')
parser.add_argument('--lambda_tune_M_E', default=0.0,                         type=float, help='Tune the directionality of coupling between M and E')
parser.add_argument('--lambda_tune_T_ME',default=0.0,                         type=float, help='Tune the directionality of coupling between ME and T')
parser.add_argument('--lambda_tune_ME_T',default=0.0,                         type=float, help='Tune the directionality of coupling between ME and T')
parser.add_argument('--lambda_tune_ME_M',default=0.0,                         type=float, help='Tune the directionality of coupling between ME and M')
parser.add_argument('--lambda_tune_M_ME',default=0.0,                         type=float, help='Tune the directionality of coupling between ME and M')
parser.add_argument('--lambda_tune_ME_E',default=0.0,                         type=float, help='Tune the directionality of coupling between ME and E')
parser.add_argument('--lambda_tune_E_ME',default=0.0,                         type=float, help='Tune the directionality of coupling between ME and E')
parser.add_argument("--augment_decoders",default=0,                           type=int,   help="0 or 1 : Train with cross modal reconstruction")
parser.add_argument('--scale_by',        default=0.3,                         type=float, help='scaling factor for M_data interpolation')
parser.add_argument('--config_file',     default='config.toml',               type=str,   help='config file with data paths')
parser.add_argument('--n_epochs',        default=10000,                       type=int,   help='Number of epochs to train')
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
    savedict = {'XT': tonumpy(all_data['xt']),
                'XM': tonumpy(all_data['xm']),
                'Xsd': tonumpy(all_data['xsd']),
                'XE': tonumpy(all_data['xe']),
                'XrT': tonumpy(xr_dict['xrt']),
                'XrE': tonumpy(xr_dict['xre']),
                'Xrsd': tonumpy(xr_dict['xrsd']),
                'XrM': tonumpy(xr_dict['xrm']),
                'XrM_me_paired': tonumpy(xr_dict['xrm_me_paired']),
                'XrE_me_paired': tonumpy(xr_dict['xre_me_paired']),
                'specimen_id': np.array([mystr.rstrip() for mystr in dat.specimen_id]),
                'cluster_label': np.array([mystr.rstrip() for mystr in dat.cluster_label]),
                'cluster_color': np.array([mystr.rstrip() for mystr in dat.cluster_color]),
                'cluster_id': dat.cluster_id,
                'gene_ids': dat.gene_ids,
                'e_features': dat.E_features,
                'loss_rec_xt': tonumpy(loss_dict['rec_t']),
                'loss_rec_xe': tonumpy(loss_dict['rec_e']),
                'loss_rec_xm': tonumpy(loss_dict['rec_m']),
                'loss_rec_xsd': tonumpy(loss_dict['rec_sd']),
                'loss_rec_xme_paired': tonumpy(loss_dict['rec_m_me']+loss_dict['rec_e_me']+loss_dict['rec_sd_me']),
                'loss_cpl_me->t': tonumpy(loss_dict['cpl_me->t']),
                'loss_cpl_t->me': tonumpy(loss_dict['cpl_t->me']),
                'loss_cpl_me->m': tonumpy(loss_dict['cpl_me->m']),
                'loss_cpl_m->me': tonumpy(loss_dict['cpl_m->me']),
                'loss_cpl_me->e': tonumpy(loss_dict['cpl_me->e']),
                'loss_cpl_e->me': tonumpy(loss_dict['cpl_e->me']),
                'loss_cpl_t->e': tonumpy(loss_dict['cpl_t->e']),
                'loss_cpl_e->t': tonumpy(loss_dict['cpl_e->t']),
                'loss_cpl_t->m': tonumpy(loss_dict['cpl_t->m']),
                'loss_cpl_m->t': tonumpy(loss_dict['cpl_m->t']),
                'loss_cpl_e->m': tonumpy(loss_dict['cpl_e->m']),
                'loss_cpl_m->e': tonumpy(loss_dict['cpl_m->e']),
                'zm': tonumpy(z_dict['zm']),
                'ze': tonumpy(z_dict['ze']),
                'zt': tonumpy(z_dict['zt']),
                'zme_paired': tonumpy(z_dict['zme_paired']),
                'is_t_1d':tonumpy(all_data['is_t_1d']),
                'is_e_1d':tonumpy(all_data['is_e_1d']),
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
         alpha_T=1.0,
         alpha_M=1.0,
         alpha_sd=1.0,
         alpha_E=1.0,
         alpha_ME=0.0,
         lambda_TE=0.0,
         lambda_TM=0.0,
         lambda_ME=0.0,
         lambda_ME_T=0.0,
         lambda_ME_M=0.0,
         lambda_ME_E=0.0,
         lambda_tune_T_E=0.0,
         lambda_tune_E_T=0.0,
         lambda_tune_T_M=0.0,
         lambda_tune_M_T=0.0,
         lambda_tune_M_E=0.0,
         lambda_tune_E_M=0.0,
         lambda_tune_ME_T=0.0,
         lambda_tune_T_ME=0.0,
         lambda_tune_ME_M=0.0,
         lambda_tune_M_ME=0.0,
         lambda_tune_ME_E=0.0,
         lambda_tune_E_ME=0.0,
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
                        augment_decoders=augment_decoders,
                        T=dict(dropout_p=0.2, 
                               alpha_T=alpha_T),
                        E=dict(gnoise_std=train_dataset.gnoise_std,
                               gnoise_std_frac=0.05, 
                               dropout_p=0.2, 
                               alpha_E=alpha_E),
                        M=dict(gnoise_std_frac=0.1,
                               scale_by=scale_by, 
                               interpolation_mode="nearest",
                               random=True,
                               alpha_M=alpha_M,
                               alpha_sd=alpha_sd),
                        TE=dict(lambda_TE=lambda_TE, lambda_tune_T_E=lambda_tune_T_E, lambda_tune_E_T=lambda_tune_E_T),
                        TM=dict(lambda_TM=lambda_TM, lambda_tune_T_M=lambda_tune_T_M, lambda_tune_M_T=lambda_tune_M_T),
                        ME=dict(alpha_ME=alpha_ME, lambda_ME=lambda_ME, lambda_tune_M_E=lambda_tune_M_E, lambda_tune_E_M=lambda_tune_E_M),
                        ME_T=dict(lambda_ME_T=lambda_ME_T, lambda_tune_ME_T=lambda_tune_ME_T, lambda_tune_T_ME=lambda_tune_T_ME),
                        ME_M=dict(lambda_ME_M=lambda_ME_M, lambda_tune_ME_M=lambda_tune_ME_M, lambda_tune_M_ME=lambda_tune_M_ME), 
                        ME_E=dict(lambda_ME_E=lambda_ME_E, lambda_tune_ME_E=lambda_tune_ME_E, lambda_tune_E_ME=lambda_tune_E_ME),
                        )
    
    
    model = Model_ME_T(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # loaded_model = torch.load("/home/fahimehb/Local/new_codes/cplAE_MET/data/results/connections_v0/checkpoint_ep_9500_test.pt", map_location='cpu')
    # model.load_state_dict(loaded_model['state_dict'])
    # optimizer.load_state_dict(loaded_model['optimizer'])

    model.to(device)
    # optimizer_to(optimizer,device)

    fileid = "test"
    # fileid = ( f"aT_{str(model_config['T']['alpha_T'])}_" +
    #            f"aM_{str(model_config['M']['alpha_M'])}_" +
    #            f"asd_{str(model_config['M']['alpha_sd'])}_" +
    #            f"aE_{str(model_config['E']['alpha_E'])}_"+
    #            f"aME_{str(model_config['ME']['alpha_ME'])}_" +
    #            f"lambda_ME_{str(model_config['ME']['lambda_ME'])}_" +
    #            f"lambda_tune_ME_{str(model_config['ME']['lambda_tune_ME'])}_" +
    #            f"lambda_TE_{str(model_config['TE']['lambda_TE'])}_" +
    #            f"lambda_tune_TE_{str(model_config['TE']['lambda_tune_TE'])}_" +
    #            f"lambda_TM_{str(model_config['TM']['lambda_TM'])}_" +
    #            f"lambda_tune_TM_{str(model_config['TM']['lambda_tune_TM'])}_" +
    #            f"lambda_ME_T_{str(model_config['ME_T']['lambda_ME_T'])}_"+
    #            f"lambda_tune_ME_T_{str(model_config['ME_T']['lambda_tune_ME_T'])}_"+
    #            f"lambda_ME_M_{str(model_config['ME_M']['lambda_ME_M'])}_"+
    #            f"lambda_tune_ME_M_{str(model_config['ME_M']['lambda_tune_ME_M'])}_"+
    #            f"lambda_ME_E_{str(model_config['ME_E']['lambda_ME_E'])}_"+
    #            f"lambda_tune_ME_E_{str(model_config['ME_E']['lambda_tune_ME_E'])}_"+
    #            f"aug_dec_{str(model_config['augment_decoders'])}_" +
    #            f"Enoise_{str(model_config['E']['gnoise_std_frac'])}_Mnoise_{model_config['M']['gnoise_std_frac']}_"+
    #            f"Mscale_{str(model_config['M']['scale_by'])}_ld_{model_config['latent_dim']:d}_ne_{n_epochs:d}_"+
    #            f"fold_{fold_n:d}").replace('.', '-')


    # Training -----------
    for epoch in range(n_epochs):
        model.train()
        for step, batch in enumerate(iter(train_dataloader)):
            xm=batch['xm']
            xsd=batch['xsd']
            is_m_1d=batch['is_m_1d']

            optimizer.zero_grad()

            # apply M augmentations on m_cells ----------- 
            aug_xm_mcells = torch.zeros(xm[is_m_1d].shape).to(device=device)
            target_size = xm.shape[2:]
            for sample in range(is_m_1d.sum()):
                _, scaled_xm = scale_depth(torch.unsqueeze(xm[is_m_1d][sample, ...], dim=0), 
                                           scale_by= model_config['M']['scale_by'], 
                                           interpolation_mode=model_config['M']['interpolation_mode'],
                                           random=model_config['M']['random'])
                scaled_xm = torch.squeeze(scaled_xm)
                aug_xm_mcells[sample, 0, ...] = center_resize(scaled_xm, target_size)
            aug_xm = torch.zeros(xm.shape).to(device=device)
            aug_xm[is_m_1d] = aug_xm_mcells

            noisy_aug_xm = add_noise(aug_xm, sd=model_config['M']['gnoise_std_frac'], clamp_min=0., scale_by_x=True)
            noisy_xsd = add_noise(xsd, clamp_min=0.)
            
            batch['aug_xm'] = aug_xm
            batch['noisy_aug_xm'] = noisy_aug_xm
            batch['noisy_xsd'] = noisy_xsd
            
            # TODO
            # forward pass T, E, M -----------
            loss_dict, _, _ = model(batch)

            loss =  model_config['T']['alpha_T'] * loss_dict['rec_t'] + \
                    model_config['E']['alpha_E'] * loss_dict['rec_e'] + \
                    model_config['M']['alpha_M'] * loss_dict['rec_m'] + \
                    model_config['M']['alpha_sd'] * loss_dict['rec_sd'] + \
                    model_config['ME']['alpha_ME'] * (loss_dict['rec_m_me'] + loss_dict['rec_e_me'] + loss_dict['rec_sd_me']) + \
                    model_config['TE']['lambda_TE'] * model_config['TE']['lambda_tune_T_E'] * loss_dict['cpl_t->e'] + \
                    model_config['TE']['lambda_TE'] * model_config['TE']['lambda_tune_E_T'] * loss_dict['cpl_e->t'] + \
                    model_config['TM']['lambda_TM'] * model_config['TM']['lambda_tune_T_M'] * loss_dict['cpl_t->m'] + \
                    model_config['TM']['lambda_TM'] * model_config['TM']['lambda_tune_M_T'] * loss_dict['cpl_m->t'] + \
                    model_config['ME']['lambda_ME'] * model_config['ME']['lambda_tune_E_M'] * loss_dict['cpl_e->m'] + \
                    model_config['ME']['lambda_ME'] * model_config['ME']['lambda_tune_M_E'] * loss_dict['cpl_m->e'] + \
                    model_config['ME_T']['lambda_ME_T'] * model_config['ME_T']['lambda_tune_T_ME'] * loss_dict['cpl_t->me'] + \
                    model_config['ME_T']['lambda_ME_T'] * model_config['ME_T']['lambda_tune_ME_T'] * loss_dict['cpl_me->t'] + \
                    model_config['ME_M']['lambda_ME_M'] * model_config['ME_M']['lambda_tune_ME_M'] * loss_dict['cpl_me->m'] + \
                    model_config['ME_M']['lambda_ME_M'] * model_config['ME_M']['lambda_tune_M_ME']*  loss_dict['cpl_m->me'] + \
                    model_config['ME_E']['lambda_ME_E'] * model_config['ME_E']['lambda_tune_ME_E'] * loss_dict['cpl_me->e'] + \
                    model_config['ME_E']['lambda_ME_E'] * model_config['ME_E']['lambda_tune_E_ME'] * loss_dict['cpl_e->me']    

            if model_config['augment_decoders']:
                loss =  loss + model_config['T']['alpha_T'] * loss_dict['rec_t_from_zme_paired'] + \
                        model_config['E']['alpha_E'] * loss_dict['rec_e_from_zt'] + \
                        model_config['E']['alpha_E'] * loss_dict['rec_e_from_zme_paired'] + \
                        model_config['M']['alpha_M'] * loss_dict['rec_m_from_zme_paired'] + \
                        model_config['M']['alpha_M'] * loss_dict['rec_m_from_ze'] + \
                        model_config['M']['alpha_M'] * loss_dict['rec_m_from_zt'] + \
                        model_config['M']['alpha_sd'] * loss_dict['rec_sd_from_zme_paired'] + \
                        model_config['M']['alpha_sd'] * loss_dict['rec_sd_from_ze'] + \
                        model_config['M']['alpha_sd'] * loss_dict['rec_sd_from_zt'] + \
                        model_config['ME']['alpha_ME'] * (loss_dict['rec_e_me_paired_from_zt'] + loss_dict['rec_m_me_paired_from_zt'] + loss_dict['rec_sd_me_paired_from_zt']) 


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
        tb_writer.add_scalar('Train/MSE_XT', train_loss['rec_t'], epoch)
        tb_writer.add_scalar('Validation/MSE_XT', val_loss['rec_t'], epoch)
        tb_writer.add_scalar('Train/MSE_XM', train_loss['rec_m'], epoch)
        tb_writer.add_scalar('Validation/MSE_XM', val_loss['rec_m'], epoch)
        tb_writer.add_scalar('Train/MSE_Xsd', train_loss['rec_sd'], epoch)
        tb_writer.add_scalar('Validation/MSE_Xsd', val_loss['rec_sd'], epoch)
        tb_writer.add_scalar('Train/MSE_XE', train_loss['rec_e'], epoch)
        tb_writer.add_scalar('Validation/MSE_XE', val_loss['rec_e'], epoch)
        tb_writer.add_scalar('Train/MSE_XME', train_loss['rec_m_me']+ train_loss['rec_sd_me'] + train_loss['rec_e_me'], epoch)
        tb_writer.add_scalar('Validation/MSE_XME', val_loss['rec_m_me']+ val_loss['rec_sd_me'] + val_loss['rec_e_me'], epoch)
        tb_writer.add_scalar('Train/cpl_T->E', train_loss['cpl_t->e'], epoch)
        tb_writer.add_scalar('Validation/cpl_T->E', val_loss['cpl_t->e'], epoch)
        tb_writer.add_scalar('Train/cpl_E->T', train_loss['cpl_e->t'], epoch)
        tb_writer.add_scalar('Validation/cpl_E->T', val_loss['cpl_e->t'], epoch)
        tb_writer.add_scalar('Train/cpl_T->M', train_loss['cpl_t->m'], epoch)
        tb_writer.add_scalar('Validation/cpl_T->M', val_loss['cpl_t->m'], epoch)
        tb_writer.add_scalar('Train/cpl_M->T', train_loss['cpl_m->t'], epoch)
        tb_writer.add_scalar('Validation/cpl_M->T', val_loss['cpl_m->t'], epoch)
        tb_writer.add_scalar('Train/cpl_M->E', train_loss['cpl_m->e'], epoch)
        tb_writer.add_scalar('Validation/cpl_M->E', val_loss['cpl_m->e'], epoch)
        tb_writer.add_scalar('Train/cpl_E->M', train_loss['cpl_e->m'], epoch)
        tb_writer.add_scalar('Validation/cpl_E->M', val_loss['cpl_e->m'], epoch)
        tb_writer.add_scalar('Train/cpl_ME->T', train_loss['cpl_me->t'], epoch)
        tb_writer.add_scalar('Validation/cpl_ME->T', val_loss['cpl_me->t'], epoch)
        tb_writer.add_scalar('Train/cpl_T->ME', train_loss['cpl_t->me'], epoch)
        tb_writer.add_scalar('Validation/cpl_T->ME', val_loss['cpl_t->me'], epoch)
        tb_writer.add_scalar('Train/cpl_ME->M', train_loss['cpl_me->m'], epoch)
        tb_writer.add_scalar('Validation/cpl_ME->M', val_loss['cpl_me->m'], epoch)
        tb_writer.add_scalar('Train/cpl_M->ME', train_loss['cpl_m->me'], epoch)
        tb_writer.add_scalar('Validation/cpl_M->ME', val_loss['cpl_me->m'], epoch)
        tb_writer.add_scalar('Train/cpl_ME->E', train_loss['cpl_me->e'], epoch)
        tb_writer.add_scalar('Validation/cpl_ME->E', val_loss['cpl_me->e'], epoch)
        tb_writer.add_scalar('Train/cpl_E->ME', train_loss['cpl_e->me'], epoch)
        tb_writer.add_scalar('Validation/cpl_E->ME', val_loss['cpl_e->me'], epoch)
        if model_config['augment_decoders']:
            tb_writer.add_scalar('Train/MSE_XT_from_zme_paired', train_loss['rec_t_from_zme_paired'], epoch)
            tb_writer.add_scalar('Train/MSE_XE_from_zme_paired', train_loss['rec_e_from_zme_paired'], epoch)
            tb_writer.add_scalar('Train/MSE_XE_from_zt', train_loss['rec_e_from_zt'], epoch)
            tb_writer.add_scalar('Train/MSE_XM_from_zme_paired', train_loss['rec_m_from_zme_paired'], epoch)
            tb_writer.add_scalar('Train/MSE_XM_from_zt', train_loss['rec_m_from_zt'], epoch)
            tb_writer.add_scalar('Train/MSE_XM_from_ze', train_loss['rec_m_from_ze'], epoch)
            tb_writer.add_scalar('Train/MSE_Xsd_from_zme_paired', train_loss['rec_sd_from_zme_paired'], epoch)
            tb_writer.add_scalar('Train/MSE_Xsd_from_zt', train_loss['rec_sd_from_zt'], epoch)
            tb_writer.add_scalar('Train/MSE_Xsd_from_ze', train_loss['rec_sd_from_ze'], epoch)
            tb_writer.add_scalar('Train/MSE_XME_paired_from_zt', (train_loss['rec_e_me_paired_from_zt'] + train_loss['rec_m_me_paired_from_zt'] + train_loss['rec_sd_me_paired_from_zt']), epoch)

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

# %%
import os
import torch
import shutil
import argparse
import numpy as np
from pathlib import Path

from torch.utils.data import DataLoader
from cplAE_MET.models.model_classes import Model_ME_T
from cplAE_MET.utils.load_config import load_config
from cplAE_MET.utils.dataset import MET_exc_inh
from cplAE_MET.utils.utils import savepkl
from torch_utils import add_noise, scale_depth, center_resize, tonumpy

from torch.utils.tensorboard import SummaryWriter

from cplAE_MET.models.torch_utils import MET_dataset

parser = argparse.ArgumentParser()
# TODO
parser.add_argument('--exp_name',        default='T_ME_M_ME_E_ME_separate_decoders_symT_ME_v1',    type=str,   help='Experiment set')
parser.add_argument('--alpha_T',         default=1.0,                         type=float, help='T reconstruction loss weight')
parser.add_argument('--alpha_M',         default=1.0,                         type=float, help='M reconstruction loss weight')
parser.add_argument('--alpha_sd',        default=1.0,                         type=float, help='soma depth reconstruction loss weight')
parser.add_argument('--alpha_E',         default=1.0,                         type=float, help='E reconstruction loss weight')
parser.add_argument('--alpha_ME',        default=1.0,                         type=float, help='ME reconstruction loss weight')
parser.add_argument('--lambda_TE',       default=0.0,                         type=float, help='coupling loss weight between T and E')
parser.add_argument('--lambda_TM',       default=0.0,                         type=float, help='coupling loss weight between T and M')
parser.add_argument('--lambda_ME',       default=0.0,                         type=float, help='coupling loss weight between M and E')
parser.add_argument('--lambda_ME_T',     default=1.0,                         type=float, help='coupling loss weight between ME and T')
parser.add_argument('--lambda_tune_ME_T',default=0.0,                         type=float, help='Tune the directionality of coupling between ME and T')
parser.add_argument('--lambda_ME_M',     default=1.0,                         type=float, help='coupling loss weight between ME and M')
parser.add_argument('--lambda_ME_E',     default=1.0,                         type=float, help='coupling loss weight between ME and E')
parser.add_argument("--augment_decoders",default=0,                           type=int,   help="0 or 1 : Train with cross modal reconstruction")
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
                'loss_cpl_me_t': tonumpy(loss_dict['cpl_me_t']),
                'loss_cpl_me_m': tonumpy(loss_dict['cpl_me_m']),
                'loss_cpl_me_e': tonumpy(loss_dict['cpl_me_e']),
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
         lambda_tune_ME_T=0.0,
         lambda_ME_M=0.0,
         lambda_ME_E=0.0,
         augment_decoders=0,
         scale_by=0.1,
         latent_dim=2,
         batch_size=1000):

    # Set the device -----------
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

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
                        TM=dict(lambda_TM=lambda_TM),
                        TE=dict(lambda_TE=lambda_TE),
                        ME=dict(alpha_ME=alpha_ME, lambda_ME=lambda_ME),
                        ME_T=dict(lambda_ME_T=lambda_ME_T, 
                                  lambda_tune_ME_T=lambda_tune_ME_T),
                        ME_M=dict(lambda_ME_M=lambda_ME_M), 
                        ME_E=dict(lambda_ME_E=lambda_ME_E),
                        )
    
    
    model = Model_ME_T(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # loaded_model = torch.load("/home/fahimehb/Local/code/cplAE_MET/data/results/T_ME_corrected_conv_v1/checkpoint_ep_49500_aT_1-0_aM_1-0_asd_1-0_aE_1-0_aME_1-0_lambda_ME_T_1-0_lambda_tune_ME_T_0-0_lambda_ME_M_0-0_lambda_ME_E_0-0_aug_dec_0_Enoise_0-05_Mnoise_0-1_Mscale_0-3_ld_2_ne_50000_fold_0.pt", map_location='cpu')
    # model.load_state_dict(loaded_model['state_dict'])
    # optimizer.load_state_dict(loaded_model['optimizer'])

    model.to(device)
    # optimizer_to(optimizer,device)

    fileid = ( f"aT_{str(model_config['T']['alpha_T'])}_aM_{str(model_config['M']['alpha_M'])}_" +
               f"asd_{str(model_config['M']['alpha_sd'])}_aE_{str(model_config['E']['alpha_E'])}_"+
               f"aME_{str(model_config['ME']['alpha_ME'])}_lambda_ME_T_{str(model_config['ME_T']['lambda_ME_T'])}_"+
               f"lambda_tune_ME_T_{str(model_config['ME_T']['lambda_tune_ME_T'])}_"+
               f"lambda_ME_M_{str(model_config['ME_M']['lambda_ME_M'])}_"+
               f"lambda_ME_E_{str(model_config['ME_E']['lambda_ME_E'])}_aug_dec_{str(model_config['augment_decoders'])}_" +
               f"Enoise_{str(model_config['E']['gnoise_std_frac'])}_Mnoise_{model_config['M']['gnoise_std_frac']}_"+
               f"Mscale_{str(model_config['M']['scale_by'])}_ld_{model_config['latent_dim']:d}_ne_{n_epochs:d}_"+
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

            loss =  model_config['T']['alpha_T'] * loss_dict['rec_t'] + \
                    model_config['E']['alpha_E'] * loss_dict['rec_e'] + \
                    model_config['M']['alpha_M'] * loss_dict['rec_m'] + \
                    model_config['M']['alpha_sd'] * loss_dict['rec_sd'] + \
                    model_config['ME']['alpha_ME'] * (loss_dict['rec_m_me'] + loss_dict['rec_e_me'] + loss_dict['rec_sd_me']) + \
                    model_config['ME_T']['lambda_ME_T'] * loss_dict['cpl_me_t'] + \
                    model_config['ME_M']['lambda_ME_M'] * loss_dict['cpl_me_m'] + \
                    model_config['ME_E']['lambda_ME_E'] * loss_dict['cpl_me_e']

            # if model_config['augment_decoders']:
            #     loss = loss + model_config['ME_T']['lambda_ME_T'] * loss_dict['rec_t_from_zme_paired'] + \
            #                   model_config['ME_T']['lambda_ME_T'] * loss_dict['rec_me_from_zt']
       

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
        tb_writer.add_scalar('Train/cpl_ME_T', train_loss['cpl_me_t'], epoch)
        tb_writer.add_scalar('Validation/cpl_ME_T', val_loss['cpl_me_t'], epoch)
        tb_writer.add_scalar('Train/cpl_ME_M', train_loss['cpl_me_m'], epoch)
        tb_writer.add_scalar('Validation/cpl_ME_M', val_loss['cpl_me_m'], epoch)
        tb_writer.add_scalar('Train/cpl_ME_E', train_loss['cpl_me_e'], epoch)
        tb_writer.add_scalar('Validation/cpl_ME_E', val_loss['cpl_me_e'], epoch)
        # if model_config['augment_decoders']:
        #     tb_writer.add_scalar('Train/MSE_XT_from_zme', train_loss['rec_t_from_zme_paired'], epoch)
        #     tb_writer.add_scalar('Validation/MSE_XT_from_zme', val_loss['rec_t_from_zme_paired'], epoch)
        #     tb_writer.add_scalar('Train/MSE_XME_from_zt', train_loss['rec_me_from_zt'], epoch)
        #     tb_writer.add_scalar('Validation/MSE_XME_from_zt', val_loss['rec_me_from_zt'], epoch)

        # tb_writer.add_histogram('eM/Weight/enc_xm_to_zm_int.fc_0', model.ae_m.enc_xm_to_zm_int.fc_0.weight, epoch)
        # tb_writer.add_histogram('eM/Weight/enc_xm_to_zm_int.fc_1', model.ae_m.enc_xm_to_zm_int.fc_1.weight, epoch)
        # tb_writer.add_histogram('eM/Weight/enc_xm_to_zm_int.conv_0', model.ae_m.enc_xm_to_zm_int.conv_0.weight, epoch)
        # tb_writer.add_histogram('eM/Weight/enc_xm_to_zm_int.conv_1', model.ae_m.enc_xm_to_zm_int.conv_1.weight, epoch)

        # tb_writer.add_histogram('eM/Weight/enc_zm_int_to_zm.fc_0', model.ae_m.enc_zm_int_to_zm.fc_0.weight, epoch)

        # tb_writer.add_histogram('eM/Weight/dec_zm_to_zm_int.fc_0', model.ae_m.dec_zm_to_zm_int.fc_0.weight, epoch)
        # tb_writer.add_histogram('eM/Weight/dec_zm_to_zm_int.fc_1', model.ae_m.dec_zm_to_zm_int.fc_1.weight, epoch)

        # tb_writer.add_histogram('eM/Weight/dec_zm_int_to_xm.convT_0', model.ae_m.dec_zm_int_to_xm.convT_0.weight, epoch)
        # tb_writer.add_histogram('eM/Weight/dec_zm_int_to_xm.convT_1', model.ae_m.dec_zm_int_to_xm.convT_1.weight, epoch)
        # tb_writer.add_histogram('eM/Weight/dec_zm_int_to_xm.fc_0', model.ae_m.dec_zm_int_to_xm.fc_0.weight, epoch)

        # tb_writer.add_histogram('eE/Weight/enc_xe_to_ze_int.fc_0', model.ae_e.enc_xe_to_ze_int.fc_0.weight, epoch)
        # tb_writer.add_histogram('eE/Weight/enc_xe_to_ze_int.fc_1', model.ae_e.enc_xe_to_ze_int.fc_1.weight, epoch)
        # tb_writer.add_histogram('eE/Weight/enc_xe_to_ze_int.fc_2', model.ae_e.enc_xe_to_ze_int.fc_2.weight, epoch)
        # tb_writer.add_histogram('eE/Weight/enc_xe_to_ze_int.fc_3', model.ae_e.enc_xe_to_ze_int.fc_3.weight, epoch)

        # tb_writer.add_histogram('eE/Weight/enc_ze_int_to_ze.fc_0', model.ae_e.enc_ze_int_to_ze.fc_0.weight, epoch)
        # tb_writer.add_histogram('eE/Weight/enc_ze_int_to_ze.fc_1', model.ae_e.enc_ze_int_to_ze.fc_1.weight, epoch)

        # tb_writer.add_histogram('eE/Weight/dec_ze_to_ze_int.fc_0', model.ae_e.dec_ze_to_ze_int.fc_0.weight, epoch)
        # tb_writer.add_histogram('eE/Weight/dec_ze_to_ze_int.fc_1', model.ae_e.dec_ze_to_ze_int.fc_1.weight, epoch)

        # tb_writer.add_histogram('eE/Weight/dec_ze_int_to_xe.fc_0', model.ae_e.dec_ze_int_to_xe.fc_0.weight, epoch)
        # tb_writer.add_histogram('eE/Weight/dec_ze_int_to_xe.fc_1', model.ae_e.dec_ze_int_to_xe.fc_1.weight, epoch)
        # tb_writer.add_histogram('eE/Weight/dec_ze_int_to_xe.fc_2', model.ae_e.dec_ze_int_to_xe.fc_2.weight, epoch)
        # tb_writer.add_histogram('eE/Weight/dec_ze_int_to_xe.fc_3', model.ae_e.dec_ze_int_to_xe.fc_3.weight, epoch)
        

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

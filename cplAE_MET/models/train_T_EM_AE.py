from typing import Dict, Any

import torch
import argparse
from pathlib import Path
from functools import partial

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from cplAE_MET.utils.utils import savepkl
from cplAE_MET.utils.log_helpers import Log_model_weights_histogram
from cplAE_MET.utils.load_config import load_config
from cplAE_MET.models.pytorch_models import Model_T_ME
from cplAE_MET.models.classification_functions import *
from cplAE_MET.models.torch_helpers import astensor, tonumpy
from cplAE_MET.utils.dataset import T_ME_Dataset, load_MET_dataset, partitions
from cplAE_MET.models.augmentations import get_padded_im, get_soma_aligned_im


parser = argparse.ArgumentParser()
parser.add_argument('--alpha_T',         default=1.0,          type=float, help='T reconstruction loss weight')
parser.add_argument('--alpha_M',         default=0.0,          type=float, help='M reconstruction loss weight')
parser.add_argument('--alpha_E',         default=0.0,          type=float, help='E reconstruction loss weight')
parser.add_argument('--alpha_ME',        default=1.0,          type=float, help='ME reconstruction loss weight')
parser.add_argument('--lambda_ME_T',     default=1.0,          type=float, help='coupling loss weight between ME and T')
parser.add_argument('--lambda_tune_ME_T',default=1.0,          type=float, help='Tune the directionality of coupling between ME and T')
parser.add_argument('--lambda_ME_M',     default=0.0,          type=float, help='coupling loss weight between ME and M')
parser.add_argument('--lambda_ME_E',     default=0.0,          type=float, help='coupling loss weight between ME and E')
parser.add_argument('--scale_factor',    default=0.3,          type=float, help='scaling factor for M_data interpolation')
parser.add_argument('--latent_dim',      default=5,            type=int,   help='Number of latent dims')
parser.add_argument('--M_noise',         default=0.0,          type=float, help='std of the gaussian noise added to M data')
parser.add_argument('--E_noise',         default=0.05,         type=float, help='std of the gaussian noise added to E data')
parser.add_argument('--n_epochs',        default=50000,        type=int,   help='Number of epochs to train')
parser.add_argument('--n_fold',          default=0,            type=int,   help='kth fold in 10-fold CV splits')
parser.add_argument('--run_iter',        default=0,            type=int,   help='Run-specific id')
parser.add_argument('--config_file',     default='config.toml',type=str,   help='config file with data paths')
parser.add_argument('--model_id',        default='ME_T',       type=str,   help='Model-specific id')
parser.add_argument('--exp_name',        default='DEBUG',      type=str,   help='Experiment set')
parser.add_argument('--log_weights',     default=False,        type=bool,  help='To log the model w')



def set_paths(config_file=None, exp_name='TEMP'):
    paths = load_config(config_file=config_file, verbose=False)
    paths['result'] = f'{str(paths["package_dir"] / "data/results")}/{exp_name}/'
    paths['tb_logs'] = f'{paths["result"]}tb_logs/{exp_name}'
    Path(paths['tb_logs']).mkdir(parents=True, exist_ok=True)
    return paths


def main(alpha_T=1.0,
         alpha_M=1.0,
         alpha_E=1.0,
         alpha_ME=1.0,
         lambda_ME_T=1.0,
         lambda_tune_ME_T=1.0,
         lambda_ME_M=1.0,
         lambda_ME_E=1.0,
         scale_factor=0.3,
         M_noise=0.0,
         E_noise=0.05,
         latent_dim=3,
         n_epochs=500,
         config_file='config.toml',
         n_fold=0,
         run_iter=0,
         model_id='T_EM',
         exp_name='DEBUG',
         log_weights=False):


    global epoch, val_loss, train_loss
    dir_pth = set_paths(config_file=config_file, exp_name=exp_name)
    tb_writer = SummaryWriter(log_dir=dir_pth['tb_logs'])

    fileid = (model_id + f'_aT_{str(alpha_T)}_aM_{str(alpha_M)}_aE_{str(alpha_E)}_aME_{str(alpha_ME)}_' +
              f'lambda_ME_T_{str(lambda_ME_T)}_lambda_tune_ME_T_{str(lambda_tune_ME_T)}_lambda_ME_M_{str(lambda_ME_M)}_'
              f'lambda_ME_E_{str(lambda_ME_E)}_' +
              f'Enoise_{str(E_noise)}_Mnoise_{str(M_noise)}_scale_{str(scale_factor)}_' +
              f'ld_{latent_dim:d}_ne_{n_epochs:d}_ri_{run_iter:d}_fold_{n_fold:d}').replace('.', '-')

    def save_results(model, data, fname, n_fold, splits, tb_writer, epoch):
        # Run the model in the evaluation mode
        model.eval()
        with torch.no_grad():
            loss_dict, z_dict, xr_dict, mask_dict = model((astensor_(data['XT']),
                                                           astensor_(data['XM']),
                                                           astensor_(data['Xsd']),
                                                           astensor_(data['XE'])))
            # convert model output tensors to numpy
            for dict in [z_dict, xr_dict, mask_dict]:
                for k, v in dict.items():
                    dict[k] = tonumpy(v)


            # Run classification task
            classification_acc = {}
            n_class = {}
            for (z, mask, key) in zip(
                            [z_dict['zt'], z_dict['zm'], z_dict['ze'], z_dict['zme']],
                            [mask_dict['valid_T'], mask_dict['valid_M'], mask_dict['valid_E'], mask_dict['valid_ME']],
                            ["zt", "zm", "ze", "zme"]):

                classification_acc[key], n_class[key] = run_QDA(X=z,
                                                                y=data['cluster_label'][mask],
                                                                test_size=0.1,
                                                                min_label_size=7)

                # Logging
                out_key = "Classification_acc_" + key
                tb_writer.add_scalar(out_key, classification_acc[key], epoch)
                print(f'epoch {epoch:04d} ----- {out_key} {classification_acc[key]:.2f} ----- Number of types {n_class[key]}')


        model.train()

        savedict = {'XT': data['XT'],
                    'XM': data['XM'],
                    'Xsd': data['Xsd'],
                    'XE': data['XE'],
                    'specimen_id': data['specimen_id'],
                    'cluster_label': data['cluster_label'],
                    'cluster_color': data['cluster_color'],
                    'cluster_id': data['cluster_id'],
                    'gene_ids': data['gene_ids'],
                    'classification_acc_zt': classification_acc["zt"],
                    'classification_acc_zm': classification_acc["zm"],
                    'classification_acc_ze': classification_acc["ze"],
                    'classification_acc_zme': classification_acc["zme"],
                    'T_class': n_class['zt'],
                    'M_class': n_class['zm'],
                    'E_class': n_class['ze'],
                    'ME_class': n_class['zme']}

        # saving the embeddings, masks and classification_acc
        for dict in [z_dict, xr_dict, mask_dict]:
            for key, value in dict.items():
                savedict[key] = value

        savedict.update(splits[n_fold])
        savepkl(savedict, fname)
        return

    def set_requires_grad(module, val):
        for p in module.parameters():
            p.requires_grad = val

    def init_losses(loss_dict):
        train_loss = {}
        val_loss = {}
        for k in loss_dict.keys():
            train_loss[k] = 0.
            val_loss[k] = 0.
        return train_loss, val_loss

    # Data selection============================
    D = load_MET_dataset(dir_pth['MET_data'])
    D['XM'] = np.expand_dims(D['XM'], axis=1)
    D['Xsd'] = np.expand_dims(D['X_sd'], axis=1)

    # soma depth is range (0,1) <-- check this
    pad = 60
    norm2pixel_factor = 100
    padded_soma_coord = np.squeeze(D['Xsd'] * norm2pixel_factor + pad)
    D['XM'] = get_padded_im(im=D['XM'], pad=pad)
    D['XM'] = get_soma_aligned_im(im=D['XM'], soma_H=padded_soma_coord)


    splits = partitions(celltype=D['cluster_label'], n_partitions=10, seed=0)

    # Helpers ==========================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    astensor_ = partial(astensor, device=device)

    # Dataset and dataloaders
    train_ind = splits[n_fold]['train']
    val_ind = splits[n_fold]['val']
    batchsize = len(train_ind)

    train_dataset = T_ME_Dataset(XT=D['XT'][train_ind, ...],
                                 XM=D['XM'][train_ind, ...],
                                 Xsd=D['Xsd'][train_ind],
                                 XE=D['XE'][train_ind, ...])

    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

    # Model ============================
    model = Model_T_ME(alpha_T=alpha_T,
                       alpha_M=alpha_M,
                       alpha_E=alpha_E,
                       alpha_ME=alpha_ME,
                       lambda_ME_T=lambda_ME_T,
                       lambda_ME_M=lambda_ME_M,
                       lambda_ME_E=lambda_ME_E,
                       lambda_tune_ME_T=lambda_tune_ME_T,
                       scale_factor=scale_factor,
                       E_noise=E_noise * np.nanstd(train_dataset.XE, axis=0),
                       M_noise=M_noise,
                       latent_dim=latent_dim)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)


    # Train ============================
    for epoch in range(n_epochs):
        for step, batch in enumerate(iter(train_dataloader)):
            # zero + forward + backward + update
            optimizer.zero_grad()
            loss_dict, *_ = model((
                astensor_(batch['XT']),
                astensor_(batch['XM']),
                astensor_(batch['Xsd']),
                astensor_(batch['XE'])))

            loss = model.alpha_T * loss_dict['recon_T'] + \
                   model.alpha_M * loss_dict['recon_M'] + \
                   model.alpha_M * loss_dict['recon_sd'] + \
                   model.alpha_E * loss_dict['recon_E'] + \
                   model.alpha_ME * loss_dict['recon_ME'] + \
                   model.lambda_ME_T * model.lambda_tune_ME_T * loss_dict['cpl_ME_T'] + \
                   model.lambda_ME_T * (1 - model.lambda_tune_ME_T) * loss_dict['cpl_T_ME'] + \
                   loss_dict['cpl_ME_M'] * model.lambda_ME_M + \
                   loss_dict['cpl_ME_E'] * model.lambda_ME_E


            # set require grad for the shared module in the M and in the E equal to False
            # This way, we will not update shared modules in the M or in the E autoencoder
            set_requires_grad(model.dE_shared_copy, False)
            set_requires_grad(model.dM_shared_copy, False)

            loss.backward()
            optimizer.step()

            # copy the weights of the shared modules from the ME part
            model.dM_shared_copy.load_state_dict(model.dM_shared.state_dict())
            model.dE_shared_copy.load_state_dict(model.dE_shared.state_dict())

            if step == 0:
                train_loss, val_loss = init_losses(loss_dict)

            # track loss over batches:
            for k, v in loss_dict.items():
                train_loss[k] += loss_dict[k]

        if log_weights:
            Log_model_weights_histogram(model=model, tensorb_writer=tb_writer, epoch=epoch)

        # validation
        model.eval()
        with torch.no_grad():
            loss_dict, *_ = model((
                astensor_(D['XT'][val_ind, ...]),
                astensor_(D['XM'][val_ind, ...]),
                astensor_(D['Xsd'][val_ind]),
                astensor_(D['XE'][val_ind])))


        for k, v in loss_dict.items():
            val_loss[k] += loss_dict[k]

        model.train()

        # Average losses over batches
        for k, v in train_loss.items():
            train_loss[k] = train_loss[k] / len(train_dataloader)

        # printing logs
        for k, v in train_loss.items():
            print(f'epoch {epoch:04d},  Train {k}: {v:.5f}')

        for k, v in val_loss.items():
            print(f'epoch {epoch:04d} ----- Val {k}: {v:.5f}')

        # Logging ==============
        tb_writer.add_scalar('Train/MSE_XT', train_loss['recon_T'], epoch)
        tb_writer.add_scalar('Validation/MSE_XT', val_loss['recon_T'], epoch)
        tb_writer.add_scalar('Train/MSE_XM', train_loss['recon_M'], epoch)
        tb_writer.add_scalar('Validation/MSE_XM', val_loss['recon_M'], epoch)
        tb_writer.add_scalar('Train/MSE_Xsd', train_loss['recon_sd'], epoch)
        tb_writer.add_scalar('Validation/MSE_Xsd', val_loss['recon_sd'], epoch)
        tb_writer.add_scalar('Train/MSE_XE', train_loss['recon_E'], epoch)
        tb_writer.add_scalar('Validation/MSE_XE', val_loss['recon_E'], epoch)
        tb_writer.add_scalar('Train/MSE_XME', train_loss['recon_ME'], epoch)
        tb_writer.add_scalar('Validation/MSE_XME', val_loss['recon_ME'], epoch)
        tb_writer.add_scalar('Train/cpl_ME_T', train_loss['cpl_ME_T'], epoch)
        tb_writer.add_scalar('Validation/cpl_ME_T', val_loss['cpl_ME_T'], epoch)
        tb_writer.add_scalar('Train/cpl_ME_M', train_loss['cpl_ME_M'], epoch)
        tb_writer.add_scalar('Validation/cpl_ME_M', val_loss['cpl_ME_M'], epoch)
        tb_writer.add_scalar('Train/cpl_ME_E', train_loss['cpl_ME_E'], epoch)
        tb_writer.add_scalar('Validation/cpl_ME_E', val_loss['cpl_ME_E'], epoch)


        #Save checkpoint
        if (epoch) % 10000 == 0:
            fname = dir_pth['result'] + f"checkpoint_ep_{epoch}_" + fileid + ".pkl"
            save_results(model, D, fname, n_fold, splits, tb_writer, epoch)

    #Save final results
    fname = dir_pth['result'] + "exit_summary_" + fileid + ".pkl"
    save_results(model, D, fname, n_fold, splits, tb_writer, epoch)
    tb_writer.close()
    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))

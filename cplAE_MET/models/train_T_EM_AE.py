########################################## T_ME_version_0.0 ################################################
# This code is for training Model_T_ME in which T arm is coupled with ME and ME is coupled with E and with M,
# The newest part that I was trying to implement was to have the same set of validation cells in the model
# and the T_type classifier. This is being implemented inside the save_results function.
# Also I just realized I did not copy the E-features name in the output, for example we have gene ids and all
# the T metadata copied to the output but not the E-features name. Rohan please do this.
############################################################################################################

import os
import torch
import shutil
import argparse
from pathlib import Path
from functools import partial
from torch import profiler

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from cplAE_MET.utils.utils import savepkl
from cplAE_MET.utils.load_config import load_config
from cplAE_MET.models.pytorch_models import Model_T_ME
from cplAE_MET.models.classification_functions import *
from cplAE_MET.models.torch_helpers import astensor, tonumpy
from cplAE_MET.utils.log_helpers import Log_model_weights_histogram
from cplAE_MET.utils.dataset import T_ME_Dataset, load_MET_dataset, partitions
from cplAE_MET.models.augmentations import get_padded_im, get_soma_aligned_im


parser = argparse.ArgumentParser()
parser.add_argument('--alpha_T',         default=1.0,          type=float, help='T reconstruction loss weight')
parser.add_argument('--alpha_M',         default=1.0,          type=float, help='M reconstruction loss weight')
parser.add_argument('--alpha_sd',        default=1.0,          type=float, help='soma depth reconstruction loss weight')
parser.add_argument('--alpha_E',         default=1.0,          type=float, help='E reconstruction loss weight')
parser.add_argument('--alpha_ME',        default=1.0,          type=float, help='ME reconstruction loss weight')
parser.add_argument('--lambda_ME_T',     default=1.0,          type=float, help='coupling loss weight between ME and T')
parser.add_argument('--lambda_tune_ME_T',default=0.5,          type=float, help='Tune the directionality of coupling between ME and T')
parser.add_argument('--lambda_ME_M',     default=1.0,          type=float, help='coupling loss weight between ME and M')
parser.add_argument('--lambda_ME_E',     default=1.0,          type=float, help='coupling loss weight between ME and E')
parser.add_argument("--augment_decoders",default=1,            type=int,   help="0 or 1 : Train with cross modal reconstruction")
parser.add_argument('--scale_factor',    default=0.3,          type=float, help='scaling factor for M_data interpolation')
parser.add_argument('--latent_dim',      default=5,            type=int,   help='Number of latent dims')
parser.add_argument('--M_noise',         default=0.0,          type=float, help='std of the gaussian noise added to M data')
parser.add_argument('--E_noise',         default=0.05,         type=float, help='std of the gaussian noise added to E data')
parser.add_argument('--n_epochs',        default=10,           type=int,   help='Number of epochs to train')
parser.add_argument('--n_fold',          default=0,            type=int,   help='kth fold in 10-fold CV splits')
parser.add_argument('--run_iter',        default=0,            type=int,   help='Run-specific id')
parser.add_argument('--config_file',     default='config.toml',type=str,   help='config file with data paths')
parser.add_argument('--model_id',        default='ME_T',       type=str,   help='Model-specific id')
parser.add_argument('--exp_name',        default='DEBUG',      type=str,   help='Experiment set')
parser.add_argument('--log_weights',     default=False,        type=bool,  help='To log the model w')



def set_paths(config_file=None, exp_name='TEMP', fold=0):
    paths = load_config(config_file=config_file, verbose=False)
    paths['result'] = f'{str(paths["package_dir"] / "data/results")}/{exp_name}/'
    paths['tb_logs'] = f'{paths["result"]}tb_logs/{exp_name}/fold_{str(fold)}/'
    Path(paths['tb_logs']).mkdir(parents=True, exist_ok=True)
    return paths


def main(alpha_T=1.0,
         alpha_M=1.0,
         alpha_sd=1.0,
         alpha_E=1.0,
         alpha_ME=1.0,
         lambda_ME_T=1.0,
         lambda_tune_ME_T=1.0,
         lambda_ME_M=1.0,
         lambda_ME_E=1.0,
         augment_decoders=1.0,
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

    if lambda_ME_T == 0.0:
        augment_decoders = 0

    dir_pth = set_paths(config_file=config_file, exp_name=exp_name, fold=n_fold)
    tb_writer = SummaryWriter(log_dir=dir_pth['tb_logs'])

    fileid = (model_id + f'_aT_{str(alpha_T)}_aM_{str(alpha_M)}_asd_{str(alpha_sd)}_aE_{str(alpha_E)}_aME_{str(alpha_ME)}_' +
              f'lambda_ME_T_{str(lambda_ME_T)}_lambda_tune_ME_T_{str(lambda_tune_ME_T)}_lambda_ME_M_{str(lambda_ME_M)}_'
              f'lambda_ME_E_{str(lambda_ME_E)}_aug_dec_{str(augment_decoders)}_' +
              f'Enoise_{str(E_noise)}_Mnoise_{str(M_noise)}_scale_{str(scale_factor)}_' +
              f'ld_{latent_dim:d}_ne_{n_epochs:d}_ri_{run_iter:d}_fold_{n_fold:d}').replace('.', '-')

    # Convert int to boolean
    augment_decoders = augment_decoders > 0
    # alpha_tune_ME is a factor that will be used to make everything symmetric in the loss function. If augment
    # decoder is on, then we will have 2 terms for T, 2 terms of M and 2 terms for E reconstruction error. However
    # we have 4 terms for ME recon loss. For that we multiply that term by 0.5 to make everything symmetric.
    alpha_tune_ME = 0.5 if augment_decoders else 1.0

    # TODO: this function is too big now
    def save_results(model, data, fname, n_fold, splits, tb_writer, epoch):
        '''
        Takes the model, run it in the evaluation mode to calculate the embeddings and reconstructions for printing out.
        Also classification is run inside this function
        '''
        # Run the model in the evaluation mode
        model.eval()

        XT = astensor_(data['XT'])
        XM = astensor_(data['XM'])
        Xsd = astensor_(data['Xsd'])
        XE = astensor_(data['XE'])

        with torch.no_grad():
            loss_dict, z_dict, xr_dict, mask_dict = model((XT, XM, Xsd, XE))

            # convert model output tensors to numpy
            for dict in [z_dict, xr_dict, mask_dict]:
                for k, v in dict.items():
                    dict[k] = tonumpy(v)

            # This part is new:
            # Run classification task
            classification_acc = {}
            n_class = {}
            for (z, mask, key) in zip(
                            [z_dict['zt'], z_dict['zm'][mask_dict['MT_M']], z_dict['ze'][mask_dict['TE_E']], z_dict['zme'][mask_dict['MET_ME']]],
                            [mask_dict['T_tot'], mask_dict['MT_tot'], mask_dict['TE_tot'], mask_dict['MET_tot']],
                            ["zt", "zm", "ze", "zme"]):

                # in the next steps, create a dictionary that has the train and test cells indices for a specific modality such as M, E or T
                # For now the indices of the train and test cells are for all the data and not the specific modality
                # 1- index of specific modality cells out of all cells
                MET_cell_ind_tot = np.where(mask)[0]

                # 2- index of the train or val cells out of all cells
                train_cell_ind_tot = splits[n_fold]['train']
                val_cell_ind_tot = splits[n_fold]['val']

                # 3- index of train or val cells of the specific modality out of all cells
                MET_train_cell_ind_tot = np.array([i for i in MET_cell_ind_tot if i in train_cell_ind_tot])
                MET_val_cell_ind_tot = np.array([i for i in MET_cell_ind_tot if i in val_cell_ind_tot])

                # 4- index of train cells of the specific modality out of that modality cells
                MET_train_cell_ind_modality = np.array([np.where(MET_cell_ind_tot == i)[0][0] for i in MET_train_cell_ind_tot])
                MET_val_cell_ind_modality = np.array([np.where(MET_cell_ind_tot == i)[0][0] for i in MET_val_cell_ind_tot])

                train_test_ids = {"train": MET_train_cell_ind_modality, "val": MET_val_cell_ind_modality}

                classification_acc[key], n_class[key], _ = run_QDA(X=z,
                                                                   y=data['cluster_label'][mask],
                                                                   test_size=0.1,
                                                                   min_label_size=7,
                                                                   train_test_ids=train_test_ids)

                # Logging
                out_key = "Classification_acc_" + key
                tb_writer.add_scalar(out_key, classification_acc[key], epoch)
                #(f'epoch {epoch:04d} ----- {out_key} {classification_acc[key]:.2f} ----- Number of types {n_class[key]}')

        savedict = {'XT': data['XT'],
                    'XM': data['XM'],
                    'Xsd': data['Xsd'],
                    'XE': data['XE'],
                    'specimen_id': data['specimen_id'],
                    'cluster_label': np.array([mystr.rstrip() for mystr in data['cluster_label']]),
                    'cluster_color': np.array([mystr.rstrip() for mystr in data['cluster_color']]),
                    'cluster_id': data['cluster_id'],
                    'gene_ids': data['gene_ids'],
                    'recon_loss_xt': tonumpy(loss_dict['recon_T']),
                    'recon_loss_xe': tonumpy(loss_dict['recon_E']),
                    'recon_loss_xm': tonumpy(loss_dict['recon_M']),
                    'recon_loss_xsd': tonumpy(loss_dict['recon_sd']),
                    'recon_loss_xme': tonumpy(loss_dict['recon_ME']),
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
        model.train()

        return

    def save_ckp(state, checkpoint_dir, n_fold):
        filename = 'checkpoint_' + str(n_fold) + '.pt'
        f_path = os.path.join(checkpoint_dir, filename)
        torch.save(state, f_path)
        filename = 'best_model_' + str(n_fold) + '.pt'
        best_fpath = os.path.join(checkpoint_dir, filename)
        shutil.copyfile(f_path, best_fpath)

    def set_requires_grad(module, val):
        for p in module.parameters():
            p.requires_grad = val

    def init_losses(loss_dict):
        t_loss = {}
        v_loss = {}
        for k in loss_dict.keys():
            t_loss[k] = 0.
            v_loss[k] = 0.
        return t_loss, v_loss

    # Data selection============================
    D = load_MET_dataset(dir_pth['MET_data'])
    D['XM'] = np.expand_dims(D['XM'], axis=1)
    D['Xsd'] = np.expand_dims(D['Xsd'], axis=1)

    # soma depth is range (0,1) <-- check this
    # check out augmentation_debug.ipynb
    # We are adding a pad of 60 pixels on the top and 60 pixels on the bottom of each arbor density image
    # so the height is going to be 240 instead of 120, this is becasue we are going to stretch or squeeze
    # the image during augmentation and we do not want that image get out of the frame
    #
    # next, we are soma centering the images. For that we need to now the exact location of soma, Olga said
    # we need to multiply the soma depth values by 100 to get the correct soma depth value. If she gave us new
    # cells we need to double check if the soma depth has been calculated as the previous cells or not to know
    # if we have to multiply their soma depth with 100 or 120
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
                       alpha_sd=alpha_sd,
                       alpha_E=alpha_E,
                       alpha_ME=alpha_ME,
                       alpha_tune_ME=alpha_tune_ME,
                       lambda_ME_T=lambda_ME_T,
                       lambda_ME_M=lambda_ME_M,
                       lambda_ME_E=lambda_ME_E,
                       lambda_tune_ME_T=lambda_tune_ME_T,
                       augment_decoders=augment_decoders,
                       scale_factor=scale_factor,
                       E_noise=E_noise * np.nanstd(train_dataset.XE, axis=0),
                       M_noise=M_noise,
                       latent_dim=latent_dim,
                       E_features=D['XE'].shape[1])


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)


    # Train ============================
    for epoch in range(n_epochs):
        for step, batch in enumerate(iter(train_dataloader)):
            # zero + forward + backward + update
            optimizer.zero_grad()
            with profiler.profile(with_stack=True, profile_memory=True) as prof:
                loss_dict, *_ = model((
                    astensor_(batch['XT']),
                    astensor_(batch['XM']),
                    astensor_(batch['Xsd']),
                    astensor_(batch['XE'])))


            loss = model.alpha_T * loss_dict['recon_T'] + \
                   model.alpha_M * loss_dict['recon_M'] + \
                   model.alpha_sd * loss_dict['recon_sd'] + \
                   model.alpha_E * loss_dict['recon_E'] + \
                   model.alpha_ME * model.alpha_tune_ME * loss_dict['recon_ME'] + \
                   model.lambda_ME_T * model.lambda_tune_ME_T * loss_dict['cpl_T->ME'] + \
                   model.lambda_ME_T * (1 - model.lambda_tune_ME_T) * loss_dict['cpl_ME->T'] + \
                   model.lambda_ME_M * loss_dict['cpl_ME->M'] + \
                   model.lambda_ME_E * loss_dict['cpl_ME->E']

            if model.augment_decoders:
                loss += model.alpha_T * loss_dict['aug_recon_T_from_zme'] + \
                        model.alpha_ME * alpha_tune_ME * loss_dict['aug_recon_ME_from_zt'] + \
                        model.alpha_ME * alpha_tune_ME * loss_dict['aug_recon_ME_from_ze'] + \
                        model.alpha_ME * alpha_tune_ME * loss_dict['aug_recon_ME_from_zm'] + \
                        model.alpha_M * loss_dict['aug_recon_M_from_zme'] + \
                        model.alpha_sd * loss_dict['aug_recon_sd_from_zme'] + \
                        model.alpha_E * loss_dict['aug_recon_E_from_zme']

            # set require grad for the shared module in the M and in the E equal to False
            # This way, we will not update shared modules in the M or in the E autoencoders
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
        tb_writer.add_scalar('Train/cpl_ME->T', train_loss['cpl_ME->T'], epoch)
        tb_writer.add_scalar('Validation/cpl_ME->T', val_loss['cpl_ME->T'], epoch)
        tb_writer.add_scalar('Train/cpl_T->ME', train_loss['cpl_T->ME'], epoch)
        tb_writer.add_scalar('Validation/cpl_T->ME', val_loss['cpl_T->ME'], epoch)
        tb_writer.add_scalar('Train/cpl_ME->M', train_loss['cpl_ME->M'], epoch)
        tb_writer.add_scalar('Validation/cpl_ME->M', val_loss['cpl_ME->M'], epoch)
        tb_writer.add_scalar('Train/cpl_ME->E', train_loss['cpl_ME->E'], epoch)
        tb_writer.add_scalar('Validation/cpl_ME->E', val_loss['cpl_ME->E'], epoch)

        #Save checkpoint
        if (epoch) % 1000 == 0:
            fname = dir_pth['result'] + f"checkpoint_ep_{epoch}_" + fileid + ".pkl"
            save_results(model, D, fname, n_fold, splits, tb_writer, epoch)
            #save model
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_ckp(checkpoint, dir_pth['result'], n_fold)


    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=1000))

    #Save final results
    fname = dir_pth['result'] + "exit_summary_" + fileid + ".pkl"
    save_results(model, D, fname, n_fold, splits, tb_writer, epoch)
    tb_writer.close()
    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))

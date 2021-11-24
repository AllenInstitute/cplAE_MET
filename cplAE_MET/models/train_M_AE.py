import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import shutil
import inspect
from torch.utils.data import DataLoader
from cplAE_MET.models.pytorch_models import Model_M_AE
from cplAE_MET.models.torch_helpers import astensor, tonumpy
from cplAE_MET.utils.dataset import partitions, load_M_inh_dataset, M_AE_Dataset
from cplAE_MET.utils.load_config import load_config
from cplAE_MET.utils.utils import savepkl
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--batchsize',        default=500,             type=int,   help='Batch size')
parser.add_argument('--alpha_M',          default=1.0,             type=float, help='M reconstruction loss weight')
parser.add_argument('--alpha_sd',         default=5.0,             type=float, help='Soma depth reconstruction loss weight')
parser.add_argument('--M_noise',          default=0.0,             type=float, help='std of the gaussian noise added to M data')
parser.add_argument('--dilate_M',         default=0,               type=int,   help='dilating M images')
parser.add_argument('--augment_decoders', default=0,               type=int,   help='0 or 1 - Train with cross modal reconstruction')
parser.add_argument('--latent_dim',       default=3,               type=int,   help='Number of latent dims')
parser.add_argument('--n_epochs',         default=50000,            type=int,   help='Number of epochs to train')
parser.add_argument('--n_fold',           default=0,               type=int,   help='Fold number in the kfold cross validation training')
parser.add_argument('--config_file',      default='config.toml',   type=str,   help='config file with data paths')
parser.add_argument('--run_iter',         default=0,               type=int,   help='Run-specific id')
parser.add_argument('--model_id',         default='run5',          type=str,   help='Model-specific id')
parser.add_argument('--exp_name',         default='M_AutoEncoder_tests',     type=str,   help='Experiment set')
parser.add_argument('--scale_im_factor',  default=0.1,     type=float,   help='scaling factor for interpolation')


def set_paths(config_file=None, exp_name='TEMP'):
    paths = load_config(config_file=config_file, verbose=False)
    paths['result'] = f'{str(paths["package_dir"] / "data/results")}/{exp_name}/'
    paths['logs'] = f'{paths["result"]}logs/'
    Path(paths['logs']).mkdir(parents=True, exist_ok=True)
    return paths


def get_padding_up_and_down(soma_depth, im):
    soma_shift = np.round(60 - soma_depth).astype(int).squeeze()
    upper_edge = np.zeros(soma_shift.shape)
    lower_edge = np.zeros(soma_shift.shape)
    n_cells = im.shape[0]
    for c in range(n_cells):
        select = np.nonzero(im[c, 0, :, :, :])
        upper_edge[c] = np.min(select[0]).item()
        lower_edge[c] = np.max(select[0]).item()
    mask_to_move_up = soma_shift < 0
    mask_to_move_down = soma_shift > 0
    upper_pad = np.max(abs(soma_shift[mask_to_move_up]) - upper_edge[mask_to_move_up])
    lower_pad = np.min(120 - lower_edge[mask_to_move_down] - soma_shift[mask_to_move_down])
    pad_lower_and_upper = max(abs(upper_pad), abs(lower_pad))
    return (np.ceil(pad_lower_and_upper/10) * 10).astype(int)


def get_padded_im(im, pad):
    shape = im.shape
    padded_im = np.zeros((shape[0], shape[1], shape[2] + pad * 2, shape[3], shape[4]))
    n_cells = im.shape[0]
    for c in range(n_cells):
        padded_im[c, 0, pad:-pad, :, :] = im[c, 0, ...]
    return padded_im

def shift3d(arr, num, fill_value=0):
    result = np.empty_like(arr)
    if num > 0:  # moving down
        result[:num, :, :] = fill_value
        result[num:, :, :] = arr[:-num, :, :]
    elif num < 0:  # moving up
        result[num:, :, :] = fill_value
        result[:num, :, :] = arr[-num:, :, :]
    else:
        result = arr
    return result

def get_soma_aligned_im(padded_soma_depth, im):
    shifted_im = np.empty_like(im)
    center = int(im.shape[2]/2)
    move_by = (center - padded_soma_depth).astype(int)
    n_cells = im.shape[0]
    for c in range(n_cells):
        shifted_im[c, 0, ...] = shift3d(im[c, 0, ...], move_by[c].item())
    return shifted_im

def main(alpha_M=1.0, alpha_sd=1.0, augment_decoders=1, dilate_M =0, M_noise=0.0, scale_im_factor=0.0, batchsize=500,
         latent_dim=3, n_epochs=5000, n_fold=0, run_iter=0, config_file='config_exc_MET.toml', model_id='T_EM',
         exp_name='T_EM_torch'):

    dir_pth = set_paths(config_file=config_file, exp_name=exp_name)
    fileid = (model_id + f'_aM_{str(alpha_M)}_asd_{str(alpha_sd)}_' +
              f'Mnoi_{str(M_noise)}_dil_M_{str(dilate_M)}_' +
              f'imscale_{str(scale_im_factor)}_' +
              f'ad_{str(augment_decoders)}_ld_{latent_dim:d}_' +
              f'bs_{batchsize:d}_ne_{n_epochs:d}_' +
              f'ri_{run_iter:d}_fold_{n_fold:d}').replace('.', '-')

    augment_decoders = augment_decoders > 0

    def save_results(model, data, fname, n_fold, splits):
        # Run the model in the evaluation mode
        model.eval()
        with torch.no_grad():
            XrM, Xr_sd, loss_dict = model((astensor_(data['XM']), astensor_(data['X_sd']), None))

            # Get the crossmodal reconstructions
            z, _, _, _, _ = model.eM(astensor_(data['XM']), astensor_(data['X_sd']), None)
        model.train()
        # Save into a mat file
        savedict = {'z': tonumpy(z),
                   'XrM': tonumpy(XrM),
                   'Xr_sd': tonumpy(Xr_sd),
                   'XM': data['XM'],
                   'X_sd': data['X_sd'],
                   'specimen_id': data['specimen_id'],
                   'cluster_label': data['cluster_label'],
                   'cluster_color': data['cluster_color'],
                   'cluster_id': data['cluster_id']}

        # Save the train and validation indices
        savedict.update(splits[n_fold])
        savepkl(savedict, fname)
        return

    # Data selection============================
    D = load_M_inh_dataset(dir_pth['M_inh_data'])
    D['XM'] = np.expand_dims(D['XM'], axis=1)
    D['X_sd'] = np.expand_dims(D['X_sd'], axis=1)

    # Standardazing X-sd
    # D['X_sd'] = D['X_sd'] - np.min(D['X_sd'])
    # D['X_sd'] = D['X_sd'] / np.max(D['X_sd'])

    # padding the images
    soma_depth = D['X_sd'] * 100
    # pad = get_padding_up_and_down(soma_depth, D['XM'])
    pad = 60
    D['XM'] = get_padded_im(D['XM'], pad)

    # soma aligning the images
    padded_soma_depth = soma_depth + pad
    D['XM'] = get_soma_aligned_im(padded_soma_depth, D['XM'])


    #setting the shift aug
    n_cells = D['cluster_label'].shape[0]
    D['shifts'] = np.zeros((n_cells, 2), dtype=int)
    D['shifts'] = np.full(D['shifts'].shape, 0.)

    # create noise image
    # noise = np.zeros((1000, 120, 4))
    # for i in range(noise.shape[0]):
    #     n = np.random.rand(120, 4)
    #     n = cv2.blur(n, (3, 3))
    #     noise[i, :, :] = (n - np.mean(n[:])) / (np.max(n[:]) - np.min(n[:])) * 2 * 0.8 + 1
    # print(np.mean(noise[:]), np.min(noise[:]), np.max(noise[:]), noise.shape)

    # calculate shifts per cell type
    # soma_depth_range = np.zeros((len(classes), 2))
    # for i, t in enumerate(classes):
    #     select = np.where(D['cluster_label'] == t)
    #     soma_depth_range[i, :] = np.array([np.min(D['X_sd'][select]), np.max(D['X_sd'][select])])
    # # for every cell calculate shift range (low and high)
    # n_cells = D['cluster_label'].shape[0]
    # D['shifts'] = np.zeros((n_cells, 2), dtype=int)
    # for i in range(n_cells):
    #     sd_range = np.where(classes==D['cluster_label'][i])[0].item()
    #     D['shifts'][i, :] = np.round((soma_depth_range[sd_range] - D['X_sd'][i]) * 120).astype(int)


    # shift_up = 120.
    # shift_down = 120.
    # for c in range(n_cells):
    #     select = np.nonzero(D['XM'][c, 0, :, :, :])
    #     zrange = np.min(select[0]).item(), np.max(select[0]).item()
    #     shift_up = np.minimum(shift_up, zrange[0])
    #     shift_down = np.minimum(shift_down, zrange[1])

    # print(shift_down, shift_up)
    # create soma depth feature
    # mean_sd = np.mean(D['X_sd'])
    # D['X_sd'] = D['X_sd'] - np.min(D['X_sd'])
    # D['X_sd'] = D['X_sd'] / np.linalg.norm(D['X_sd'], ord=2) * 100  # unit variance, scale up
    # scaling_factor = np.mean(D['X_sd']) / mean_sd
    # scaling_factor = 1.
    # stratified kfold splits ===================

    splits = partitions(celltype=D['cluster_label'], n_partitions=10, seed=0)

   # Number of types in each validation set across folds
    for i in range(10):
        print(len(np.unique(D['cluster_label'][splits[i]['train']])))



    # Helpers ==========================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    astensor_ = partial(astensor, device=device)
    print("Done!")

    # copy this file and the file that contains the model class into the result folder
    shutil.copyfile(__file__, dir_pth['result'] + "trainer_code.py")
    shutil.copyfile(inspect.getfile(Model_M_AE), dir_pth['result'] + "model.py")

    # Dataset and dataloaders
    train_ind = splits[n_fold]['train']
    val_ind = splits[n_fold]['val']
    batchsize = len(train_ind)

    train_dataset = M_AE_Dataset(XM=D['XM'][train_ind, ...],
                                 sd=D['X_sd'][train_ind],
                                 shifts=D['shifts'][train_ind])

    # train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False)

    # train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False)
    # train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False,
                                  # sampler=train_sampler, drop_last=True, pin_memory=True)


    # Model ============================
    model = Model_M_AE(M_noise=M_noise, latent_dim=latent_dim, alpha_M=alpha_M,
                       alpha_sd=alpha_sd, augment_decoders=augment_decoders, scale_im_factor=scale_im_factor)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    # Train ============================
    train_loss_xm = np.zeros(n_epochs)
    train_loss_xsd = np.zeros(n_epochs)
    val_loss_xm = np.zeros(n_epochs)
    val_loss_xsd = np.zeros(n_epochs)

    for epoch in range(n_epochs):
        epoch_train_loss_xm = 0.
        epoch_train_loss_xsd = 0.
        epoch_val_loss_xm = 0.
        epoch_val_loss_xsd = 0.
        for batch in iter(train_dataloader):
            # zero + forward + backward + udpate
            optimizer.zero_grad()
            XrM, Xr_sd, loss_dict = model((astensor_(batch['XM']),
                                           astensor_(batch['X_sd']),
                                           # astensor_(noise),
                                           astensor_(batch['shifts'])))

            model.loss.backward()
            optimizer.step()

            # track loss over batches:
            epoch_train_loss_xm += loss_dict["recon_M"]
            epoch_train_loss_xsd += loss_dict["recon_sd"]

        # validation
        model.eval()
        with torch.no_grad():
            XrM, Xr_sd, loss_dict = model((astensor_(D['XM'][val_ind, ...]),
                                           astensor_(D['X_sd'][val_ind]),
                                           None))

        epoch_val_loss_xm += loss_dict["recon_M"]
        epoch_val_loss_xsd += loss_dict["recon_sd"]
        model.train()

        # Averaging losses over batches
        train_loss_xm[epoch] = epoch_train_loss_xm / len(train_ind)
        train_loss_xsd[epoch] = epoch_train_loss_xsd / len(train_ind)
        val_loss_xm[epoch] = epoch_val_loss_xm / len(val_ind)
        val_loss_xsd[epoch] = epoch_val_loss_xsd / len(val_ind)

        print(f'epoch {epoch:04d},  Train xm {train_loss_xm[epoch]:.5f} Train xsd {train_loss_xsd[epoch]:.5f}')
        print(f'epoch {epoch:04d} ----- Val xm {val_loss_xm[epoch]:.5f} Val xsd {val_loss_xsd[epoch]:.5f}')
        # Logging ==============
        with open(dir_pth['logs'] + f'{fileid}.csv', 'a') as f:
            writer = csv.writer(f, delimiter=',')
            if epoch == 0:
                writer.writerow(['epoch'] +
                                ['train_loss_xm'] +
                                ['train_loss_xsd'] +
                                ['val_loss_xm'] +
                                ['val_loss_xsd'])

            writer.writerow([epoch + 1,
                             train_loss_xm[epoch],
                             train_loss_xsd[epoch],
                             val_loss_xm[epoch],
                             val_loss_xsd[epoch]])

        if (epoch % 200) == 0:
            fname = dir_pth['result'] + "exit_summary_" + fileid + ".pkl"
            save_results(model, D, fname, n_fold, splits)

    fname = dir_pth['result'] + "exit_summary_" + fileid + ".pkl"
    save_results(model, D, fname, n_fold, splits)
    print("Done!")
    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))

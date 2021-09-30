import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import shutil
import inspect
import scipy
from torch.utils.data import DataLoader
from cplAE_MET.models.torch_cplAE import Model_T_EM
from cplAE_MET.models.torch_helpers import astensor, tonumpy
from cplAE_MET.utils.dataset import partitions, MET_Dataset, load_MET_inh_dataset
from cplAE_MET.utils.load_config import load_config
from cplAE_MET.utils.utils import savepkl
from functools import partial


parser = argparse.ArgumentParser()
parser.add_argument('--batchsize',        default=500,             type=int,   help='Batch size')
parser.add_argument('--alpha_T',          default=0.0,             type=float, help='T reconstruction loss weight')
parser.add_argument('--alpha_E',          default=0.0,             type=float, help='E reconstruction loss weight')
parser.add_argument('--alpha_M',          default=1.0,             type=float, help='M reconstruction loss weight')
parser.add_argument('--alpha_sd',         default=0.0,             type=float, help='Soma depth reconstruction loss weight')
parser.add_argument('--lambda_T_EM',      default=0.0,             type=float, help='T - EM coupling loss weight')
parser.add_argument('--M_noise',          default=0.1,             type=float, help='std of the gaussian noise added to M data')
parser.add_argument('--E_noise',          default=0.05,            type=float, help='std of the gaussian noise added to E data')
parser.add_argument('--dilate_M',         default=1,               type=int,   help='dilating M images')
parser.add_argument('--augment_decoders', default=0,               type=int,   help='0 or 1 - Train with cross modal reconstruction')
parser.add_argument('--latent_dim',       default=3,               type=int,   help='Number of latent dims')
parser.add_argument('--n_epochs',         default=5000,            type=int,   help='Number of epochs to train')
parser.add_argument('--n_fold',           default=0,               type=int,   help='Fold number in the kfold cross validation training')
parser.add_argument('--config_file',      default='config.toml',   type=str,   help='config file with data paths')
parser.add_argument('--run_iter',         default=0,               type=int,   help='Run-specific id')
parser.add_argument('--model_id',         default='AE_M',          type=str,   help='Model-specific id')
parser.add_argument('--exp_name',         default='AE_M',     type=str,   help='Experiment set')


def set_paths(config_file=None, exp_name='TEMP'):
    paths = load_config(config_file=config_file, verbose=False)
    paths['result'] = f'{str(paths["package_dir"] / "data/results")}/{exp_name}/'
    paths['logs'] = f'{paths["result"]}logs/'
    Path(paths['logs']).mkdir(parents=True, exist_ok=True)
    return paths


def main(alpha_T=1.0, alpha_E=1.0, alpha_M=1.0, alpha_sd=1.0, lambda_T_EM=1.0,
         augment_decoders=1, dilate_M =0, M_noise=0.02, E_noise=0.05,
         batchsize=500, latent_dim=3,
         n_epochs=5000, n_fold=0, run_iter=0,
         config_file='config_exc_MET.toml',
         model_id='T_EM', exp_name='T_EM_torch'):

    dir_pth = set_paths(config_file=config_file, exp_name=exp_name)
    fileid = (model_id + f'_aT_{str(alpha_T)}_aE_{str(alpha_E)}_aM_{str(alpha_M)}_asd_{str(alpha_sd)}_' +
              f'csT_EM_{str(lambda_T_EM)}_Mnoi_{str(M_noise)}_Enoi_{str(E_noise)}_' +
              f'_dil_M_{str(dilate_M)}_ad_{str(augment_decoders)}_' +
              f'ld_{latent_dim:d}_bs_{batchsize:d}_ne_{n_epochs:d}_' +
              f'ri_{run_iter:d}').replace('.', '-')

    augment_decoders = augment_decoders > 0

    # Data selection===================
    D = load_MET_inh_dataset(dir_pth['MET_inh_data'])
    n_genes = D['XT'].shape[1]
    n_E_features = D['XE'].shape[1]

    def Binary_dilate_M(x):
        x = np.nan_to_num(x, nan=0.0) #set all nans to zero otherwise nans will be considered as real values
        x_dil = np.full(x.shape, np.nan) #initiate images with all nans
        for i, x_i in enumerate(x):  #spply binary dilation on the 2d arrays
            for j, x_j in enumerate(x_i):
                x_dil[i][j] = scipy.ndimage.binary_dilation(x_j)
        return np.where(x_dil, x, np.nan)

    if dilate_M:
        D['XM'] = Binary_dilate_M(D['XM'])

    splits = partitions(celltype=D['cluster'], n_partitions=10, seed=0)

    # Helpers ==========================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    astensor_ = partial(astensor, device=device)

    def report_losses(loss_dict, partition):
        for key in loss_dict.keys():
            if key != 'steps':
                loss_dict[key] = loss_dict[key] / loss_dict['steps']
        return '   '.join([f'{partition}_{key} : {loss_dict[key]:0.5f}' for key in loss_dict])

    def collect_losses(model_loss, tracked_loss):
        # Init if keys do not exist
        if not tracked_loss:
            tracked_loss = {key: 0 for key in model_loss}
            tracked_loss['steps'] = 0.0

        # Add info
        tracked_loss['steps'] += 1
        for key in model_loss:
            tracked_loss[key] += tonumpy(model_loss[key])
        return tracked_loss

    def save_results(model, data, fname, n_fold, splits):
        # Run the model in the evaluation mode
        model.eval()
        with torch.no_grad():
            zT, zEM, XrT, XrE, XrM, Xr_sd, _ = model((astensor_(data['XT']),
                                                   astensor_(data['XE']),
                                                   astensor_(data['XM']),
                                                   astensor_(data['X_sd'])))

            # Get the crossmodal reconstructions
            XrT_from_zEM = model.dT(zEM)
            XrE_from_zT, XrM_from_zT, Xr_sd_from_zT = model.dEM(zT)

        # Save into a mat file
        savedict = {'zT': tonumpy(zT),
                   'zEM': tonumpy(zEM),
                   'XrT': tonumpy(XrT),
                   'XrE': tonumpy(XrE),
                   'XrM': tonumpy(XrM),
                   'Xr_sd': tonumpy(Xr_sd),
                   'XrT_from_zEM': tonumpy(XrT_from_zEM),
                   'XrE_from_zT': tonumpy(XrE_from_zT),
                   'XrM_from_zT': tonumpy(XrM_from_zT),
                   'Xr_sd_from_zT': tonumpy(Xr_sd_from_zT),
                   'XT': data['XT'],
                   'XE': data['XE'],
                   'XM': data['XM'],
                   'X_sd': data['X_sd'],
                   'sample_id': data['sample_id'],
                   'cluster_label': data['cluster'],
                   'cluster_id': data['cluster_id'],
                   'cluster_color': data['cluster_color']}

        # Save the train and validation indices
        savedict.update(splits[n_fold])
        savepkl(savedict, fname)
        return

    # Training data
    fold_fileid = fileid + '_fold_' + str(n_fold)
    train_ind = splits[n_fold]['train']
    val_ind = splits[n_fold]['val']
    train_dataset = MET_Dataset(XT=D['XT'][train_ind, :],
                                XE=D['XE'][train_ind, :],
                                XM=D['XM'][train_ind, :],
                                sd=D['X_sd'][train_ind])
    train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False,
                                  sampler=train_sampler, drop_last=True, pin_memory=True)

    # Model ============================
    model = Model_T_EM(T_dim=n_genes, T_int_dim=50, T_dropout=0.2,
                       E_dim=n_E_features, E_int_dim=50, EM_int_dim=5,
                       E_dropout=0.2, E_noise=E_noise * np.nanstd(train_dataset.XE, axis=0), M_noise=M_noise,
                       latent_dim=latent_dim,
                       alpha_T=alpha_T, alpha_E=alpha_E, alpha_M=alpha_M, alpha_sd=alpha_sd,
                       lambda_T_EM=lambda_T_EM, augment_decoders=augment_decoders)
    optimizer = torch.optim.Adam(model.parameters())

    model.to(device)

    # Training loop ====================
    best_loss = np.inf
    best_epoch = 0
    monitor_loss = []
    persistence = 100
    min_wait = 1000

    for epoch in range(n_epochs):
        train_loss_dict = {}
        for batch in iter(train_dataloader):
            # zero + forward + backward + udpate
            optimizer.zero_grad()
            zT, zEM, XrT, XrE, XrM, Xr_sd, loss_dict = model((astensor_(batch['XT']),
                       astensor_(batch['XE']),
                       astensor_(batch['XM']),
                       astensor_(batch['X_sd'])))

            model.loss.backward()
            optimizer.step()

            # track loss over batches:
            train_loss_dict = collect_losses(model_loss=loss_dict, tracked_loss=train_loss_dict)

        # Operations after each epoch: 
        # Validation: train mode -> eval mode + no_grad + eval mode -> train mode
        model.eval()
        with torch.no_grad():
            zT, zEM, XrT, XrE, XrM, Xr_sd, loss_dict = model((astensor_(D['XT'][val_ind, :]),
                       astensor_(D['XE'][val_ind, :]),
                       astensor_(D['XM'][val_ind, :]),
                       astensor_(D['X_sd'][val_ind])))

        val_loss_dict = collect_losses(model_loss=loss_dict, tracked_loss={})
        model.train()

        # Averaging losses over batches
        train_loss = report_losses(loss_dict=train_loss_dict, partition='train')
        val_loss = report_losses(loss_dict=val_loss_dict, partition='val')
        print(f'epoch {epoch:04d} Train {train_loss}')
        print(f'epoch {epoch:04d} ----- Val {val_loss}')

        # Logging ==============
        with open(dir_pth['logs'] + f'{fold_fileid}.csv', 'a') as f:
            writer = csv.writer(f, delimiter=',')
            if epoch == 0:
                writer.writerow(['epoch'] +
                                ['train_' + k for k in train_loss_dict] +
                                ['val_' + k for k in val_loss_dict])
            writer.writerow([epoch + 1, *train_loss_dict.values(), *val_loss_dict.values()])

        monitor_loss.append(val_loss_dict['recon_T'] +
                            val_loss_dict['recon_E'] +
                            val_loss_dict['recon_M'] +
                            val_loss_dict['recon_sd'] +
                            val_loss_dict['cpl_T_EM'])

        # Checkpoint ===========
        if (monitor_loss[-1] < best_loss) and (epoch > best_epoch + persistence) and epoch > min_wait:
            best_loss = monitor_loss[-1]
            torch.save({'epoch': epoch,
                        'hparams': model.get_hparams(),
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss, }, dir_pth['result'] + f'{fold_fileid}_weights-best.pt')

    print('\nTraining completed for fold:', n_fold)

    # Save model weights on exit
    torch.save({'epoch': epoch,
                'hparams': model.get_hparams(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss}, dir_pth['result'] + f'{fold_fileid}_weights-exit.pt')

    save_fname = dir_pth['result'] + f'{fold_fileid}_exit'
    save_results(model=model,
                 data=D.copy(),
                 fname=f'{save_fname}-summary.pkl',
                 n_fold=n_fold, splits=splits)

    #copy this file and the file that contains the model class into the result folder
    shutil.copyfile(__file__, dir_pth['result']+"trainer_code.py")
    shutil.copyfile(inspect.getfile(Model_T_EM), dir_pth['result'] + "model.py")
    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))

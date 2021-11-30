import argparse
import inspect
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import torch
from cplAE_MET.models.pytorch_models import Model_M_AE
from cplAE_MET.models.torch_helpers import astensor, tonumpy
from cplAE_MET.models.augmentations import get_padded_im, get_soma_aligned_im, get_celltype_specific_shifts
from cplAE_MET.utils.dataset import M_AE_Dataset, load_M_inh_dataset, partitions
from cplAE_MET.utils.load_config import load_config
from cplAE_MET.utils.utils import savepkl
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--alpha_M',         default=1.0,           type=float, help='M reconstruction loss weight')
parser.add_argument('--alpha_sd',        default=5.0,           type=float, help='Soma depth reconstruction loss weight')
parser.add_argument('--M_noise',         default=0.0,           type=float, help='std of the gaussian noise added to M data')
parser.add_argument('--dilate_M',        default=0,             type=int,   help='dilating M images')
parser.add_argument('--scale_factor',    default=0.1,           type=float, help='scaling factor for interpolation')
parser.add_argument('--latent_dim',      default=3,             type=int,   help='Number of latent dims')
parser.add_argument('--n_epochs',        default=500,           type=int,   help='Number of epochs to train')
parser.add_argument('--config_file',     default='config.toml', type=str,   help='config file with data paths')
parser.add_argument('--n_fold',          default=0,             type=int,   help='kth fold in 10-fold CV splits')
parser.add_argument('--run_iter',        default=0,             type=int,   help='Run-specific id')
parser.add_argument('--model_id',        default='MAE',         type=str,   help='Model-specific id')
parser.add_argument('--exp_name',        default='DEBUG',       type=str,   help='Experiment set')



def set_paths(config_file=None, exp_name='TEMP'):
    paths = load_config(config_file=config_file, verbose=False)
    paths['result'] = f'{str(paths["package_dir"] / "data/results")}/{exp_name}/'
    paths['tb_logs'] = f'{paths["result"]}tb_logs/{exp_name}'
    Path(paths['tb_logs']).mkdir(parents=True, exist_ok=True)
    return paths


def main(alpha_M=1.0,
         alpha_sd=5.0,
         M_noise=0.0,
         dilate_M=0,
         scale_factor=0.1,
         latent_dim=3,
         n_epochs=500,
         config_file='config.toml',
         n_fold=0,
         run_iter=0,
         model_id='MAE',
         exp_name='DEBUG'):

    dir_pth = set_paths(config_file=config_file, exp_name=exp_name)
    tb_writer = SummaryWriter(log_dir=dir_pth['tb_logs'])

    fileid = (model_id + f'_aM_{str(alpha_M)}_asd_{str(alpha_sd)}_' +
              f'noise_{str(M_noise)}_dilate_{str(dilate_M)}_scale_{str(scale_factor)}_' +
              f'ld_{latent_dim:d}_ne_{n_epochs:d}_' +
              f'ri_{run_iter:d}_fold_{n_fold:d}').replace('.', '-')

    def save_results(model, data, fname, n_fold, splits):
        # Run the model in the evaluation mode
        model.eval()
        with torch.no_grad():
            XrM, Xr_sd, _ = model((astensor_(data['XM']), 
                                          astensor_(data['X_sd'])))

            # Get the crossmodal reconstructions
            z, _, _, _, _ = model.eM(astensor_(data['XM']),
                                     astensor_(data['X_sd']),
                                     None)
        model.train()

        savedict = {'z': tonumpy(z),
                    'XrM': tonumpy(XrM),
                    'Xr_sd': tonumpy(Xr_sd),
                    'XM': data['XM'],
                    'X_sd': data['X_sd'],
                    'specimen_id': data['specimen_id'],
                    'cluster_label': data['cluster_label'],
                    'cluster_color': data['cluster_color'],
                    'cluster_id': data['cluster_id']}
        savedict.update(splits[n_fold])
        savepkl(savedict, fname)
        return

    # Data selection============================
    D = load_M_inh_dataset(dir_pth['M_inh_data'])
    D['XM'] = np.expand_dims(D['XM'], axis=1)
    D['X_sd'] = np.expand_dims(D['X_sd'], axis=1)

    # soma depth is range (0,1) <-- check this
    pad = 60
    norm2pixel_factor = 100
    padded_soma_coord = np.squeeze(D['X_sd'] * norm2pixel_factor + pad)
    D['XM'] = get_padded_im(im=D['XM'], pad=pad)
    D['XM'] = get_soma_aligned_im(im=D['XM'], soma_H=padded_soma_coord)
    D['shifts'] = get_celltype_specific_shifts(ctype=D['cluster_label'], dummy=True)


    splits = partitions(celltype=D['cluster_label'], n_partitions=10, seed=0)

    # Helpers ==========================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    astensor_ = partial(astensor, device=device)

    # copy this file and the file that contains the model class into the result folder
    shutil.copyfile(__file__, dir_pth['result'] + "trainer_code.py")
    shutil.copyfile(inspect.getfile(Model_M_AE), dir_pth['result'] + "model.py")

    # Dataset and dataloaders
    train_ind = splits[n_fold]['train']
    val_ind = splits[n_fold]['val']
    batchsize = len(train_ind)

    train_dataset = M_AE_Dataset(XM=D['XM'][train_ind, ...],
                                 sd=D['X_sd'][train_ind])

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batchsize, shuffle=False)

    # Model ============================
    model = Model_M_AE(M_noise=M_noise,
                       scale_factor=scale_factor,
                       latent_dim=latent_dim,
                       alpha_M=alpha_M,
                       alpha_sd=alpha_sd)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    # Train ============================
    for epoch in range(n_epochs):
        train_loss_xm = 0.
        train_loss_xsd = 0.
        val_loss_xm = 0.
        val_loss_xsd = 0.

        for batch in iter(train_dataloader):
            # zero + forward + backward + udpate
            optimizer.zero_grad()
            _, _, loss_dict = model((astensor_(batch['XM']),
                                    astensor_(batch['X_sd'])))

            loss = model.alpha_M * loss_dict['recon_M'] + model.alpha_sd*loss_dict['recon_sd']
            loss.backward()
            optimizer.step()

            # track loss over batches:
            train_loss_xm += loss_dict["recon_M"]
            train_loss_xsd += loss_dict["recon_sd"]

        # validation
        model.eval()
        with torch.no_grad():
            _, _, loss_dict = model((astensor_(D['XM'][val_ind, ...]),
                                     astensor_(D['X_sd'][val_ind])))

        val_loss_xm += loss_dict["recon_M"]
        val_loss_xsd += loss_dict["recon_sd"]
        model.train()

        # Average losses over batches
        train_loss_xm = train_loss_xm / len(train_dataloader)
        train_loss_xsd = train_loss_xsd / len(train_dataloader)
        val_loss_xm = val_loss_xm
        val_loss_xsd = val_loss_xsd

        print(f'epoch {epoch:04d},  Train xm {train_loss_xm:.5f} Train xsd {train_loss_xsd:.5f}')
        print(f'epoch {epoch:04d} ----- Val xm {val_loss_xm:.5f} Val xsd {val_loss_xsd:.5f}')

        # Logging ==============
        tb_writer.add_scalar('Train/MSE_XM', train_loss_xm, epoch)
        tb_writer.add_scalar('Train/MSE_Xsd', train_loss_xm, epoch)
        tb_writer.add_scalar('Validation/MSE_XM', train_loss_xsd, epoch)
        tb_writer.add_scalar('Validation/MSE_Xsd', val_loss_xsd, epoch)

        #Save checkpoint
        if (epoch+1) % 200 == 0:
            fname = dir_pth['result'] + f"checkpoint_ep_{epoch}_" + fileid + ".pkl"
            save_results(model, D, fname, n_fold, splits)

    #Save final results
    fname = dir_pth['result'] + "exit_summary_" + fileid + ".pkl"
    save_results(model, D, fname, n_fold, splits)
    tb_writer.close()
    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))

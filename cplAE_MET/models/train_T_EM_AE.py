import torch
import argparse
from pathlib import Path
from functools import partial

from cplAE_MET.utils.utils import savepkl
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from cplAE_MET.utils.load_config import load_config

from cplAE_MET.models.pytorch_models import Model_T_ME
from cplAE_MET.models.classification_functions import *
from cplAE_MET.models.torch_helpers import astensor, tonumpy
from cplAE_MET.utils.dataset import T_ME_Dataset, load_MET_dataset, partitions
from cplAE_MET.models.augmentations import get_padded_im, get_soma_aligned_im


parser = argparse.ArgumentParser()
parser.add_argument('--alpha_T',         default=1.0,           type=float, help='T reconstruction loss weight')
parser.add_argument('--alpha_ME',         default=1.0,           type=float, help='ME reconstruction loss weight')
parser.add_argument('--lambda_ME_T',     default=1.0,           type=float, help='coupling term between ME and T')
parser.add_argument('--scale_factor',    default=0.3,           type=float, help='scaling factor for interpolation')
parser.add_argument('--latent_dim',      default=5,             type=int,   help='Number of latent dims')
parser.add_argument('--M_noise',         default=0.0,           type=float, help='std of the gaussian noise added to M data')
parser.add_argument('--E_noise',         default=0.05,          type=float, help='std of the gaussian noise added to E data')
parser.add_argument('--n_epochs',        default=50000,         type=int,   help='Number of epochs to train')
parser.add_argument('--n_fold',          default=0,             type=int,   help='kth fold in 10-fold CV splits')
parser.add_argument('--run_iter',        default=0,             type=int,   help='Run-specific id')
parser.add_argument('--config_file',     default='config.toml', type=str,   help='config file with data paths')
parser.add_argument('--model_id',        default='ME_T',     type=str,   help='Model-specific id')
parser.add_argument('--exp_name',        default='DEBUG',       type=str,   help='Experiment set')



def set_paths(config_file=None, exp_name='TEMP'):
    paths = load_config(config_file=config_file, verbose=False)
    paths['result'] = f'{str(paths["package_dir"] / "data/results")}/{exp_name}/'
    paths['tb_logs'] = f'{paths["result"]}tb_logs/{exp_name}'
    Path(paths['tb_logs']).mkdir(parents=True, exist_ok=True)
    return paths


def main(alpha_T=1.0,
         alpha_ME=1.0,
         lambda_ME_T=1.0,
         scale_factor=0.3,
         M_noise=0.0,
         E_noise=0.05,
         latent_dim=3,
         n_epochs=500,
         config_file='config.toml',
         n_fold=0,
         run_iter=0,
         model_id='T_EM',
         exp_name='DEBUG'):


    dir_pth = set_paths(config_file=config_file, exp_name=exp_name)
    tb_writer = SummaryWriter(log_dir=dir_pth['tb_logs'])

    fileid = (model_id + f'_aT_{str(alpha_T)}_aME_{str(alpha_ME)}_' +
              f'lambda_ME_T_{str(lambda_ME_T)}_' +
              f'noise_{str(M_noise)}_scale_{str(scale_factor)}_' +
              f'ld_{latent_dim:d}_ne_{n_epochs:d}_' +
              f'ri_{run_iter:d}_fold_{n_fold:d}').replace('.', '-')

    def save_results(model, data, fname, n_fold, splits, tb_writer, epoch):
        # Run the model in the evaluation mode
        model.eval()
        with torch.no_grad():
            loss_dict, zt, zm, ze, zme, XrT, XrM_from_zme, Xrsd_from_zme, XrE_from_zme, valid_T, valid_M, valid_E, valid_ME = \
                model((astensor_(data['XT']),
                       astensor_(data['XM']),
                       astensor_(data['Xsd']),
                       astensor_(data['XE'])))

            zt = tonumpy(zt)
            zme = tonumpy(zme)
            XrT = tonumpy(XrT)
            XrM_from_zme = tonumpy(XrM_from_zme)
            Xrsd_from_zme = tonumpy(Xrsd_from_zme)
            XrE_from_zme = tonumpy(XrE_from_zme)
            valid_T = tonumpy(valid_T)
            valid_M = tonumpy(valid_M)
            valid_E = tonumpy(valid_E)
            valid_ME = tonumpy(valid_ME)


            # Run classification task
            classification_acc = {}
            n_class = {}
            for (z, mask, key) in zip([zt, zme], [valid_T, valid_ME], ["zt", "zme"]):
                masked_labels = data['cluster_label'][mask]
                small_types_mask = get_small_types_mask(masked_labels, 7)
                X = z[small_types_mask]
                _, y = np.unique(masked_labels[small_types_mask], return_inverse=True)
                classification_acc[key], n_class[key] = run_LogisticRegression(X, y, y, 0.1)
                out_key = "Classification_acc_" + key
                tb_writer.add_scalar(out_key, classification_acc[key], epoch)
                print(f'epoch {epoch:04d} ----- {out_key} {classification_acc[key]:.2f} ----- Number of types {n_class[key]}')

        model.train()

        savedict = {'zt': zt,
                    'zme': zme,
                    'XT': data['XT'],
                    'XM': data['XM'],
                    'Xsd': data['Xsd'],
                    'XE': data['XE'],
                    'XrT': XrT,
                    'XrE_from_zme': XrE_from_zme,
                    'XrM_from_zme': XrM_from_zme,
                    'Xrsd_from_zme': Xrsd_from_zme,
                    'valid_T': valid_T,
                    'valid_M': valid_M,
                    'valid_E': valid_E,
                    'specimen_id': data['specimen_id'],
                    'cluster_label': data['cluster_label'],
                    'cluster_color': data['cluster_color'],
                    'cluster_id': data['cluster_id'],
                    'gene_ids': data['gene_ids'],
                    'classification_acc_zt': classification_acc["zt"],
                    'classification_acc_zme': classification_acc["zme"],
                    'T_class': n_class['zt'],
                    'ME_class': n_class['zme']}

        savedict.update(splits[n_fold])
        savepkl(savedict, fname)
        return

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
                       alpha_ME=alpha_ME,
                       lambda_ME_T=lambda_ME_T,
                       scale_factor=scale_factor,
                       E_noise=E_noise * np.nanstd(train_dataset.XE, axis=0),
                       M_noise=M_noise,
                       latent_dim=latent_dim)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    # Train ============================
    for epoch in range(n_epochs):
        train_loss_xt = 0.
        train_loss_xme = 0.
        train_loss_cpl_met = 0.
        val_loss_xt = 0.
        val_loss_xme = 0.
        val_loss_cpl_met = 0.

        for batch in iter(train_dataloader):
            # zero + forward + backward + update
            optimizer.zero_grad()
            loss_dict, *_ = model((
                astensor_(batch['XT']),
                astensor_(batch['XM']),
                astensor_(batch['Xsd']),
                astensor_(batch['XE'])))

            loss = model.alpha_T * loss_dict['recon_T'] + \
                   model.alpha_ME * loss_dict['recon_ME'] + \
                   model.lambda_ME_T * loss_dict['cpl_ME_T']


            loss.backward()
            optimizer.step()

            # track loss over batches:
            train_loss_xt += loss_dict["recon_T"]
            train_loss_xme += loss_dict["recon_ME"]
            train_loss_cpl_met += loss_dict['cpl_ME_T']

        # validation
        model.eval()
        with torch.no_grad():
            loss_dict, *_ = model((
                astensor_(D['XT'][val_ind, ...]),
                astensor_(D['XM'][val_ind, ...]),
                astensor_(D['Xsd'][val_ind]),
                astensor_(D['XE'][val_ind])))

        val_loss_xt += loss_dict["recon_T"]
        val_loss_xme += loss_dict["recon_ME"]
        val_loss_cpl_met += loss_dict['cpl_ME_T']
        model.train()

        # Average losses over batches
        train_loss_xt = train_loss_xt / len(train_dataloader)
        train_loss_xme = train_loss_xme / len(train_dataloader)
        train_loss_cpl_met = train_loss_cpl_met / len(train_dataloader)

        val_loss_xt = val_loss_xt
        val_loss_xme = val_loss_xme
        val_loss_cpl_met = val_loss_cpl_met


        print(f'epoch {epoch:04d},  Train xt {train_loss_xt:.5f}')
        print(f'epoch {epoch:04d} ----- Val xt {val_loss_xt:.5f}')
        print(f'epoch {epoch:04d},  Train xme {train_loss_xme:.5f}')
        print(f'epoch {epoch:04d} ----- Val xme {val_loss_xme:.5f}')

        # Logging ==============
        tb_writer.add_scalar('Train/MSE_XT', train_loss_xt, epoch)
        tb_writer.add_scalar('Validation/MSE_XT', val_loss_xt, epoch)
        tb_writer.add_scalar('Train/MSE_XME', train_loss_xme, epoch)
        tb_writer.add_scalar('Validation/MSE_XME', val_loss_xme, epoch)
        tb_writer.add_scalar('Train/cpl_ME_T', train_loss_cpl_met, epoch)
        tb_writer.add_scalar('Validation/cpl_ME_T', val_loss_cpl_met, epoch)


        # Save checkpoint
        if (epoch) % 1000 == 0:
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

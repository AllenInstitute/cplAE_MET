import argparse
from functools import partial
from pathlib import Path

import torch
from cplAE_MET.models.pytorch_models import Model_T_ME
from cplAE_MET.models.classification_functions import *
from cplAE_MET.models.torch_helpers import astensor, tonumpy
from cplAE_MET.utils.dataset import T_ME_Dataset, load_MET_dataset, partitions
from cplAE_MET.models.augmentations import get_padded_im, get_soma_aligned_im, get_celltype_specific_shifts

from cplAE_MET.utils.load_config import load_config
from cplAE_MET.utils.utils import savepkl
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--alpha_T',         default=1.0,           type=float, help='T reconstruction loss weight')
parser.add_argument('--alpha_M',         default=1.0,           type=float, help='M reconstruction loss weight')
parser.add_argument('--alpha_sd',        default=1.0,           type=float, help='Soma depth reconstruction loss weight')
parser.add_argument('--alpha_E',         default=1.0,           type=float, help='E reconstruction loss weight')
parser.add_argument('--lambda_ME_M',        default=1.0,           type=float, help='coupling term between ME and M')
parser.add_argument('--lambda_ME_E',        default=1.0,           type=float, help='coupling term between ME and E')
parser.add_argument('--lambda_ME_T',        default=1.0,           type=float, help='coupling term between ME and T')
parser.add_argument('--lambda_T_M',        default=1.0,           type=float, help='coupling term between M and T')
parser.add_argument('--lambda_T_E',        default=1.0,           type=float, help='coupling term between T and E')
parser.add_argument('--scale_factor',    default=0.3,           type=float, help='scaling factor for interpolation')
parser.add_argument('--latent_dim',      default=5,             type=int,   help='Number of latent dims')
parser.add_argument('--M_noise',         default=0.0,           type=float, help='std of the gaussian noise added to M data')
parser.add_argument('--E_noise',          default=0.05,            type=float, help='std of the gaussian noise added to E data')
parser.add_argument('--n_epochs',        default=50000,           type=int,   help='Number of epochs to train')
parser.add_argument('--config_file',     default='config.toml', type=str,   help='config file with data paths')
parser.add_argument('--n_fold',          default=0,             type=int,   help='kth fold in 10-fold CV splits')
parser.add_argument('--run_iter',        default=0,             type=int,   help='Run-specific id')
parser.add_argument('--model_id',        default='T_EM_AE',         type=str,   help='Model-specific id')
parser.add_argument('--exp_name',        default='DEBUG',       type=str,   help='Experiment set')



def set_paths(config_file=None, exp_name='TEMP'):
    paths = load_config(config_file=config_file, verbose=False)
    paths['result'] = f'{str(paths["package_dir"] / "data/results")}/{exp_name}/'
    paths['tb_logs'] = f'{paths["result"]}tb_logs/{exp_name}'
    Path(paths['tb_logs']).mkdir(parents=True, exist_ok=True)
    return paths


def main(alpha_T=1.0,
         alpha_M=1.0,
         alpha_sd=1.0,
         alpha_E=1.0,
         lambda_ME_M=1.0,
         lambda_ME_E=1.0,
         lambda_ME_T=1.0,
         lambda_T_M=1.0,
         lambda_T_E=1.0,
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

    fileid = (model_id + f'_aT_{str(alpha_T)}_aM_{str(alpha_M)}_asd_{str(alpha_sd)}_aE_{str(alpha_E)}_' +
              f'lambda_MEM_{str(lambda_ME_M)}_lambda_MEE_{str(lambda_ME_E)}_lambda_MET_{str(lambda_ME_T)}_' +
              f'lambda_TM_{str(lambda_T_M)}_lambda_TE_{str(lambda_T_E)}_' +
              f'noise_{str(M_noise)}_scale_{str(scale_factor)}_' +
              f'ld_{latent_dim:d}_ne_{n_epochs:d}_' +
              f'ri_{run_iter:d}_fold_{n_fold:d}').replace('.', '-')

    def save_results(model, data, fname, n_fold, splits, tb_writer, epoch):
        # Run the model in the evaluation mode
        model.eval()
        with torch.no_grad():
            loss_dict, zt, zm, ze, zme, XrT, XrM, Xrsd, XrE, valid_T, valid_M, valid_E, valid_ME = \
                model((astensor_(data['XT']),
                       astensor_(data['XM']),
                       astensor_(data['Xsd']),
                       astensor_(data['XE'])))

            zt = tonumpy(zt)
            zm = tonumpy(zm)
            ze = tonumpy(ze)
            zme = tonumpy(zme)
            XrT = tonumpy(XrT)
            XrM = tonumpy(XrM)
            Xrsd = tonumpy(Xrsd)
            XrE = tonumpy(XrE)
            valid_T = tonumpy(valid_T)
            valid_M = tonumpy(valid_M)
            valid_E = tonumpy(valid_E)
            valid_ME = tonumpy(valid_ME)


            # Run classification task
            classification_acc = {}
            n_class = {}
            for (z, mask, key) in zip([zt, zm, ze, zme], [valid_T, valid_M, valid_E, valid_ME], ["zt", "zm", "ze", "zme"]):
                masked_labels = data['cluster_label'][mask]
                small_types_mask = get_small_types_mask(masked_labels, 7)
                X = z[small_types_mask]
                n_class[key], y = np.unique(masked_labels[small_types_mask], return_inverse=True)
                classification_acc[key] = run_LogisticRegression(X, y, y, 0.1)
                out_key = "Classification_acc_" + key
                tb_writer.add_scalar(out_key, classification_acc[key], epoch)
                print(f'epoch {epoch:04d} ----- {out_key} {classification_acc[key]:.2f} ----- Number of types {len(n_class[key])}')

        model.train()

        savedict = {'zt': zt,
                    'zm': zm,
                    'ze': ze,
                    'zme': zme,
                    'XrT': XrT,
                    'XT': data['XT'],
                    'XrM': XrM,
                    'XM': data['XM'],
                    'Xrsd': Xrsd,
                    'Xsd': data['Xsd'],
                    'XrE': XrE,
                    'XE': data['XE'],
                    'valid_T': valid_T,
                    'valid_M': valid_M,
                    'valid_E': valid_E,
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
                       alpha_M=alpha_M,
                       alpha_sd=alpha_sd,
                       alpha_E=alpha_E,
                       scale_factor=scale_factor,
                       E_noise=E_noise * np.nanstd(train_dataset.XE, axis=0),
                       M_noise=M_noise,
                       latent_dim=latent_dim)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    # Train ============================
    for epoch in range(n_epochs):
        train_loss_xt = 0.
        train_loss_xm = 0.
        train_loss_xsd = 0.
        train_loss_xe = 0.
        train_loss_cpl_mem = 0.
        train_loss_cpl_mee = 0.
        train_loss_cpl_met = 0.
        train_loss_cpl_tm = 0.
        train_loss_cpl_te = 0.
        val_loss_xt = 0.
        val_loss_xm = 0.
        val_loss_xsd = 0.
        val_loss_xe = 0.
        val_loss_cpl_mem = 0.
        val_loss_cpl_mee = 0.
        val_loss_cpl_met = 0.
        val_loss_cpl_tm = 0.
        val_loss_cpl_te = 0.

        for batch in iter(train_dataloader):
            # zero + forward + backward + udpate
            optimizer.zero_grad()
            loss_dict, _, _, _, _, _, _, _, _, _, _, _, _ = model((
                astensor_(batch['XT']), astensor_(batch['XM']), astensor_(batch['Xsd']), astensor_(batch['XE'])))

            loss = model.alpha_T * loss_dict['recon_T'] + \
                   model.alpha_M * loss_dict['recon_M'] + \
                   model.alpha_sd * loss_dict['recon_sd'] + \
                   model.alpha_E * loss_dict['recon_E'] + \
                   lambda_ME_M * loss_dict['cpl_ME_M'] + \
                   lambda_ME_E * loss_dict['cpl_ME_E'] +\
                   lambda_ME_E * loss_dict['cpl_ME_T'] +\
                   lambda_T_M * loss_dict['cpl_TM'] + \
                   lambda_T_E * loss_dict['cpl_TE']

            loss.backward()
            optimizer.step()

            # track loss over batches:
            train_loss_xt += loss_dict["recon_T"]
            train_loss_xm += loss_dict["recon_M"]
            train_loss_xsd += loss_dict['recon_sd']
            train_loss_xe += loss_dict["recon_E"]
            train_loss_cpl_mem += loss_dict['cpl_ME_M']
            train_loss_cpl_mee += loss_dict['cpl_ME_E']
            train_loss_cpl_met += loss_dict['cpl_ME_T']
            train_loss_cpl_tm += loss_dict['cpl_TM']
            train_loss_cpl_te += loss_dict['cpl_TE']

        # validation
        model.eval()
        with torch.no_grad():
            loss_dict, _, _, _, _, _, _, _, _, _, _, _, _ = model((
                astensor_(D['XT'][val_ind, ...]), astensor_(D['XM'][val_ind, ...]), astensor_(D['Xsd'][val_ind]), astensor_(D['XE'][val_ind])))

        val_loss_xt += loss_dict["recon_T"]
        val_loss_xm += loss_dict["recon_M"]
        val_loss_xsd += loss_dict['recon_sd']
        val_loss_xe += loss_dict["recon_E"]
        val_loss_cpl_mem += loss_dict['cpl_ME_M']
        val_loss_cpl_mee += loss_dict['cpl_ME_E']
        val_loss_cpl_met += loss_dict['cpl_ME_T']
        val_loss_cpl_tm += loss_dict['cpl_TM']
        val_loss_cpl_te += loss_dict['cpl_TE']
        model.train()

        # Average losses over batches
        train_loss_xt = train_loss_xt / len(train_dataloader)
        train_loss_xm = train_loss_xm / len(train_dataloader)
        train_loss_xsd = train_loss_xsd / len(train_dataloader)
        train_loss_xe = train_loss_xe / len(train_dataloader)
        train_loss_cpl_mee = train_loss_cpl_mee / len(train_dataloader)
        train_loss_cpl_mem = train_loss_cpl_mem / len(train_dataloader)
        train_loss_cpl_met = train_loss_cpl_met / len(train_dataloader)
        train_loss_cpl_tm = train_loss_cpl_tm / len(train_dataloader)
        train_loss_cpl_te = train_loss_cpl_te / len(train_dataloader)

        val_loss_xt = val_loss_xt
        val_loss_xm = val_loss_xm
        val_loss_xsd = val_loss_xsd
        val_loss_xe = val_loss_xe
        val_loss_cpl_mem = val_loss_cpl_mem
        val_loss_cpl_mee = val_loss_cpl_mee
        val_loss_cpl_met = val_loss_cpl_met
        val_loss_cpl_tm = val_loss_cpl_tm
        val_loss_cpl_te = val_loss_cpl_te

        print(f'epoch {epoch:04d},  Train xt {train_loss_xt:.5f}')
        print(f'epoch {epoch:04d} ----- Val xt {val_loss_xt:.5f}')
        print(f'epoch {epoch:04d},  Train xm {train_loss_xm:.5f}')
        print(f'epoch {epoch:04d} ----- Val xm {val_loss_xm:.5f}')
        print(f'epoch {epoch:04d},  Train xsd {train_loss_xsd:.5f}')
        print(f'epoch {epoch:04d} ----- Val xsd {val_loss_xsd:.5f}')
        print(f'epoch {epoch:04d},  Train xe {train_loss_xe:.5f}')
        print(f'epoch {epoch:04d} ----- Val xe {val_loss_xe:.5f}')
        # Logging ==============
        tb_writer.add_scalar('Train/MSE_XT', train_loss_xt, epoch)
        tb_writer.add_scalar('Validation/MSE_XT', val_loss_xt, epoch)
        tb_writer.add_scalar('Train/MSE_XM', train_loss_xm, epoch)
        tb_writer.add_scalar('Validation/MSE_XM', val_loss_xm, epoch)
        tb_writer.add_scalar('Train/MSE_Xsd', train_loss_xsd, epoch)
        tb_writer.add_scalar('Validation/MSE_Xsd', val_loss_xsd, epoch)
        tb_writer.add_scalar('Train/MSE_XE', train_loss_xe, epoch)
        tb_writer.add_scalar('Validation/MSE_XE', val_loss_xe, epoch)

        tb_writer.add_scalar('Train/cpl_ME_M', train_loss_cpl_mem, epoch)
        tb_writer.add_scalar('Validation/cpl_ME_M', val_loss_cpl_mem, epoch)
        tb_writer.add_scalar('Train/cpl_ME_E', train_loss_cpl_mee, epoch)
        tb_writer.add_scalar('Validation/cpl_ME_E', val_loss_cpl_mee, epoch)
        tb_writer.add_scalar('Train/cpl_ME_T', train_loss_cpl_met, epoch)
        tb_writer.add_scalar('Validation/cpl_ME_T', val_loss_cpl_met, epoch)
        tb_writer.add_scalar('Train/cpl_T_M', train_loss_cpl_tm, epoch)
        tb_writer.add_scalar('Validation/cpl_T_M', val_loss_cpl_tm, epoch)
        tb_writer.add_scalar('Train/cpl_T_E', train_loss_cpl_te, epoch)
        tb_writer.add_scalar('Validation/cpl_T_E', val_loss_cpl_te, epoch)


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

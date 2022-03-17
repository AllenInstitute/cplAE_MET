import torch
import argparse
from pathlib import Path
from functools import partial

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from cplAE_MET.utils.utils import savepkl
from cplAE_MET.utils.log_helpers import Log_model_weights_histogram
from cplAE_MET.utils.load_config import load_config
from cplAE_MET.models.pytorch_models import Model_TE
from cplAE_MET.models.classification_functions import *
from cplAE_MET.models.torch_helpers import astensor, tonumpy
from cplAE_MET.utils.dataset import TE_Dataset, load_MET_dataset, partitions


parser = argparse.ArgumentParser()
parser.add_argument('--alpha_T',         default=0.0,          type=float, help='T reconstruction loss weight')
parser.add_argument('--alpha_E',         default=0.0,          type=float, help='E reconstruction loss weight')
parser.add_argument('--lambda_TE',       default=0.0,          type=float, help='coupling loss weight between T and E')
parser.add_argument('--latent_dim',      default=5,            type=int,   help='Number of latent dims')
parser.add_argument('--E_noise',         default=0.05,         type=float, help='std of the gaussian noise added to E data')
parser.add_argument('--n_epochs',        default=50000,        type=int,   help='Number of epochs to train')
parser.add_argument('--n_fold',          default=0,            type=int,   help='kth fold in 10-fold CV splits')
parser.add_argument('--run_iter',        default=0,            type=int,   help='Run-specific id')
parser.add_argument('--config_file',     default='config.toml',type=str,   help='config file with data paths')
parser.add_argument('--model_id',        default='TE',         type=str,   help='Model-specific id')
parser.add_argument('--exp_name',        default='DEBUG',      type=str,   help='Experiment set')
parser.add_argument('--log_weights',     default=False,        type=bool,  help='To log the model w')


def set_paths(config_file=None, exp_name='TEMP'):
    paths = load_config(config_file=config_file, verbose=False)
    paths['result'] = f'{str(paths["package_dir"] / "data/results")}/{exp_name}/'
    paths['tb_logs'] = f'{paths["result"]}tb_logs/{exp_name}'
    Path(paths['tb_logs']).mkdir(parents=True, exist_ok=True)
    return paths


def main(alpha_T=1.0,
         alpha_E=1.0,
         lambda_TE=1.0,
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

    fileid = (model_id + f'_aT_{str(alpha_T)}_aE_{str(alpha_E)}_lambda_TE_{str(lambda_TE)}_Enoise_{str(E_noise)}' +
              f'_ld_{latent_dim:d}_ne_{n_epochs:d}_ri_{run_iter:d}_fold_{n_fold:d}').replace('.', '-')


    def save_results(model, data, fname, n_fold, splits, tb_writer, epoch):
        # Run the model in the evaluation mode
        model.eval()
        with torch.no_grad():
            loss_dict, z_dict, xr_dict, mask_dict = model((astensor_(data['XT']),
                                                           astensor_(data['XE'])))
            # convert model output tensors to numpy
            for dict in [z_dict, xr_dict, mask_dict]:
                for k, v in dict.items():
                    dict[k] = tonumpy(v)


            # Run classification task
            classification_acc = {}
            n_class = {}
            for (z, mask, key) in zip([z_dict['zt'], z_dict['ze']],
                                      [mask_dict['valid_T'], mask_dict['valid_E']],
                                      ["zt", "ze"]):

                classification_acc[key], n_class[key] = run_QDA(X=z,
                                                                y=data['cluster_label'][mask],
                                                                test_size=0.1,
                                                                min_label_size=7)

                # Logging
                out_key = "Classification_acc_" + key
                tb_writer.add_scalar(out_key, classification_acc[key], epoch)
                print(f'epoch {epoch:04d} ----- {out_key} {classification_acc[key]:.2f} ----- Number of types {n_class[key]}')


        model.train()

        savedict = {'XE': data['XE'],
                    'specimen_id': data['specimen_id'],
                    'cluster_label': data['cluster_label'],
                    'cluster_color': data['cluster_color'],
                    'cluster_id': data['cluster_id'],
                    'gene_ids': data['gene_ids'],
                    'classification_acc_zt': classification_acc["zt"],
                    'classification_acc_ze': classification_acc["ze"],
                    'T_class': n_class['zt'],
                    'E_class': n_class['ze']}

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
    splits = partitions(celltype=D['cluster_label'], n_partitions=10, seed=0)

    # Helpers ==========================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    astensor_ = partial(astensor, device=device)

    # Dataset and dataloaders
    train_ind = splits[n_fold]['train']
    val_ind = splits[n_fold]['val']
    batchsize = len(train_ind)

    train_dataset = TE_Dataset(XT=D['XT'][train_ind, ...], XE=D['XE'][train_ind, ...])

    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

    # Model ============================
    model = Model_TE(alpha_T=alpha_T,
                     alpha_E=alpha_E,
                     lambda_TE=lambda_TE,
                     E_noise=E_noise * np.nanstd(train_dataset.XE, axis=0),
                     latent_dim=latent_dim)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    # Train ============================
    for epoch in range(n_epochs):
        for step, batch in enumerate(iter(train_dataloader)):
            # zero + forward + backward + update
            optimizer.zero_grad()
            loss_dict, *_ = model((astensor_(batch['XT']), astensor_(batch['XE'])))

            loss = model.alpha_T * loss_dict['recon_T'] + \
                   model.alpha_E * loss_dict['recon_E'] + \
                   model.lambda_TE * loss_dict['cpl_TE']

            loss.backward()
            optimizer.step()

            if step == 0:
                train_loss, val_loss = init_losses(loss_dict)

            # track loss over batches:
            for k, v in loss_dict.items():
                train_loss[k] += loss_dict[k]

        if log_weights:
            Log_model_weights_histogram(model=model, tensorb_writer=tb_writer, epoch=epoch)

        # validation
        model.eval()
        print("model is validation mode")
        with torch.no_grad():
            loss_dict, *_ = model((astensor_(D['XT'][val_ind, ...]), astensor_(D['XE'][val_ind])))

        for k, v in loss_dict.items():
            val_loss[k] += loss_dict[k]

        model.train()
        print("model is training mode")

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
        tb_writer.add_scalar('Train/MSE_XE', train_loss['recon_E'], epoch)
        tb_writer.add_scalar('Validation/MSE_XE', val_loss['recon_E'], epoch)
        tb_writer.add_scalar('Train/cpl_TE', train_loss['cpl_TE'], epoch)
        tb_writer.add_scalar('Validation/cpl_TE', val_loss['cpl_TE'], epoch)


        #Save checkpoint
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

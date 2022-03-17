import argparse
from functools import partial
from pathlib import Path

import torch
from cplAE_MET.models.pytorch_models import Model_E_AE
from cplAE_MET.models.classification_functions import *
from cplAE_MET.models.torch_helpers import astensor, tonumpy
from cplAE_MET.utils.dataset import E_AE_Dataset, load_MET_dataset, partitions
from cplAE_MET.utils.load_config import load_config
from cplAE_MET.utils.utils import savepkl
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--alpha_E',         default=1.0,           type=float, help='T reconstruction loss weight')
parser.add_argument('--latent_dim',      default=3,             type=int,   help='Number of latent dims')
parser.add_argument('--n_epochs',        default=50000,           type=int,   help='Number of epochs to train')
parser.add_argument('--E_noise',          default=0.05,            type=float, help='std of the gaussian noise added to E data')
parser.add_argument('--config_file',     default='config.toml', type=str,   help='config file with data paths')
parser.add_argument('--n_fold',          default=0,             type=int,   help='kth fold in 10-fold CV splits')
parser.add_argument('--run_iter',        default=0,             type=int,   help='Run-specific id')
parser.add_argument('--model_id',        default='E_AE',         type=str,   help='Model-specific id')
parser.add_argument('--exp_name',        default='DEBUG',       type=str,   help='Experiment set')



def set_paths(config_file=None, exp_name='TEMP'):
    paths = load_config(config_file=config_file, verbose=False)
    paths['result'] = f'{str(paths["package_dir"] / "data/results")}/{exp_name}/'
    paths['tb_logs'] = f'{paths["result"]}tb_logs/{exp_name}'
    Path(paths['tb_logs']).mkdir(parents=True, exist_ok=True)
    return paths


def main(alpha_E=1.0,
         latent_dim=3,
         n_epochs=500,
         E_noise=0.05,
         config_file='config.toml',
         n_fold=0,
         run_iter=0,
         model_id='MAE',
         exp_name='DEBUG'):



    dir_pth = set_paths(config_file=config_file, exp_name=exp_name)
    tb_writer = SummaryWriter(log_dir=dir_pth['tb_logs'])

    fileid = (model_id + f'_ld_{latent_dim:d}_ne_{n_epochs:d}_' +
              f'ri_{run_iter:d}_fold_{n_fold:d}').replace('.', '-')

    def save_results(model, data, fname, n_fold, splits, tb_writer, epoch):
        # Run the model in the evaluation mode
        model.eval()
        with torch.no_grad():
            XrE, _, z = model((astensor_(data['XE'])))


            # Run classification task
            small_types_mask = get_small_types_mask(data['cluster_label'], 7)
            X = tonumpy(z[small_types_mask])
            n_classes, y = np.unique(data['cluster_label'][small_types_mask], return_inverse=True)
            classification_acc = run_LogisticRegression(X, y, y, 0.1)
            tb_writer.add_scalar('Classification_acc', classification_acc, epoch)
            print(f'epoch {epoch:04d} ----- Classification_acc {classification_acc:.2f} ----- Number of types {len(n_classes)}')

        model.train()

        savedict = {'z': tonumpy(z),
                    'XrE': tonumpy(XrE),
                    'XE': data['XE'],
                    'specimen_id': data['specimen_id'],
                    'cluster_label': data['cluster_label'],
                    'cluster_color': data['cluster_color'],
                    'cluster_id': data['cluster_id'],
                    'gene_ids': data['gene_ids'],
                    'classification_acc': classification_acc}
        savedict.update(splits[n_fold])
        savepkl(savedict, fname)
        return

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

    train_dataset = E_AE_Dataset(XE=D['XE'][train_ind, ...])

    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

    # Model ============================
    model = Model_E_AE(alpha_E=alpha_E, latent_dim=latent_dim,  E_noise=E_noise * np.nanstd(train_dataset.XE, axis=0))


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    # Train ============================
    for epoch in range(n_epochs):
        train_loss_xe = 0.
        val_loss_xe = 0.

        for batch in iter(train_dataloader):
            # zero + forward + backward + udpate
            optimizer.zero_grad()
            _, loss_dict, _ = model((astensor_(batch['XE'])))

            loss = model.alpha_E * loss_dict['recon_E']
            loss.backward()
            optimizer.step()

            # track loss over batches:
            train_loss_xe += loss_dict["recon_E"]

        # validation
        model.eval()
        with torch.no_grad():
            _, loss_dict, _ = model((astensor_(D['XE'][val_ind, ...])))

        val_loss_xe += loss_dict["recon_E"]
        model.train()

        # Average losses over batches
        train_loss_xe = train_loss_xe / len(train_dataloader)
        val_loss_xe = val_loss_xe

        print(f'epoch {epoch:04d},  Train xe {train_loss_xe:.5f}')
        print(f'epoch {epoch:04d} ----- Val xe {val_loss_xe:.5f}')
        # Logging ==============
        tb_writer.add_scalar('Train/MSE_XE', train_loss_xe, epoch)
        tb_writer.add_scalar('Validation/MSE_XE', val_loss_xe, epoch)

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

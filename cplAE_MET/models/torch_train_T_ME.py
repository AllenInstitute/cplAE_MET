import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from cplAE_MET.models.torch_cplAE import Model_MET, Model_T_EM
from cplAE_MET.utils.dataset import partitions
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--batchsize",          default=500,           type=int,   help="Batch size")
parser.add_argument("--alpha_T",            default=1.0,           type=float, help="Transcriptomic reconstruction loss weight")
parser.add_argument("--alpha_E",            default=1.0,           type=float, help="Electrophisiology reconstruction loss weight")
parser.add_argument("--alpha_M",            default=1.0,           type=float, help="Morphology reconstruction loss weight")
parser.add_argument("--alpha_soma_depth",   default=1.0,           type=float, help="Soma depth reconstruction loss weight")
parser.add_argument("--lambda_T_EM",        default=1.0,           type=float, help="T and EM coupling loss weight")
parser.add_argument("--augment_decoders",   default=1,             type=int,   help="0 or 1 : Train with cross modal reconstruction")
parser.add_argument("--latent_dim",         default=3,             type=int,   help="Number of latent dims")
parser.add_argument("--n_epochs",           default=5,            type=int,   help="Number of epochs to train")
parser.add_argument("--n_fold",             default=0,             type=int,   help="Fold number in the kfold cross validation training")
parser.add_argument("--config_file",        default='config_exc_MET.toml',type=str, help="config file with data paths")
parser.add_argument("--run_iter",           default=0,             type=int,   help="Run-specific id")
parser.add_argument("--model_id",           default='T_EM',        type=str,   help="Model-specific id")
parser.add_argument("--exp_name",           default='T_EM_torch',  type=str,   help="Experiment set")
parser.add_argument("--input_mat_filename", default='inh_MET_model_input_mat.mat', type=str,
                    help="name of the .mat file of input")


def set_paths(config_file=None, exp_name='TEMP', input_mat_filename='ALSO_TEMP'):
    from cplAE_MET.utils.load_config import load_config
    paths = load_config(config_file=config_file, verbose=False)
    paths['input_mat'] = f'{str(paths["package_dir"] / "data/proc")}/{input_mat_filename}'
    paths['result'] = f'{str(paths["package_dir"] / "data/results")}/{exp_name}/'
    paths['logs'] = f'{paths["result"]}logs/'
    Path(paths['logs']).mkdir(parents=True, exist_ok=True)
    return paths


class MET_Dataset(torch.utils.data.Dataset):
    def __init__(self, T_dat, E_dat, M_dat, soma_depth):
        super(MET_Dataset).__init__()
        self.T_dat = T_dat
        self.E_dat = E_dat
        self.M_dat = M_dat
        self.soma_depth = soma_depth
        self.n_samples = np.shape(self.T_dat)[0]

    def __getitem__(self, idx):
        sample = {"XT": self.T_dat[idx, :],
                  "XE": self.E_dat[idx, :],
                  "XM": self.M_dat[idx, :],
                  "X_soma_depth": self.soma_depth[idx]}
        return sample

    def __len__(self):
        return self.n_samples


def main(alpha_T=1.0, alpha_E=1.0, alpha_M=1.0, alpha_soma_depth=1.0, lambda_T_EM=1.0,
         augment_decoders=1.0, batchsize=500, latent_dim=3,
         n_epochs=5000, n_fold=0, run_iter=0, config_file='config_exc_MET.toml', model_id='T_EM',
         exp_name='T_EM_torch', input_mat_filename="inh_MET_model_input_mat.mat"):
    dir_pth = set_paths(config_file=config_file, exp_name=exp_name, input_mat_filename=input_mat_filename)

    fileid = (model_id + f'_aT_{str(alpha_T)}_aE_{str(alpha_E)}_aM_{str(alpha_M)}_asd_{str(alpha_soma_depth)}_' +
              f'csT_EM_{str(lambda_T_EM)}_' +
              f'ad_{str(augment_decoders)}_ld_{latent_dim:d}_bs_{batchsize:d}_ne_{n_epochs:d}_' +
              f'ri_{run_iter:d}').replace('.', '-')

    # Convert int to boolean
    augment_decoders = augment_decoders > 0

    # Data selection===================
    data = sio.loadmat(dir_pth['input_mat'], squeeze_me=True)

    D = {}
    D["XT"] = data['T_dat']
    D["XE"] = data['E_dat']
    D["XM"] = data['M_dat']
    D["X_soma_depth"] = data['soma_depth']
    D['cluster'] = data['cluster']
    D['cluster_id'] = data['cluster_id'].astype(int)
    D['cluster_color'] = data['cluster_color']
    D['sample_id'] = data['sample_id']

    # assert np.all(np.equal(D['mask_soma_depth'], D['mask_M']))
    n_genes = D["XT"].shape[1]
    n_E_features = D["XE"].shape[1]

    splits = partitions(celltype=D['cluster'], n_partitions=10, seed=0)

    # Helpers ==========================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def tensor(x):
        return torch.tensor(x).to(dtype=torch.float32).to(device)

    def tensor_(x):
        return torch.as_tensor(x).to(dtype=torch.float32).to(device)

    def booltensor_(x):
        return torch.as_tensor(x).to(dtype=torch.bool).to(device)

    def tonumpy(x):
        return x.cpu().detach().numpy()

    def report_losses(loss_dict, partition):
        for key in loss_dict.keys():
            if key != 'steps':
                loss_dict[key] = loss_dict[key] / loss_dict['steps']
        return '   '.join([f'{partition}_{key} : {loss_dict[key]:0.5f}' for key in loss_dict])

    def collect_losses(model_loss, tracked_loss):
        # Init if keys dont exist
        if not tracked_loss:
            tracked_loss = {key: 0 for key in model_loss}
            tracked_loss['steps'] = 0.0

        # Add info
        tracked_loss['steps'] += 1
        for key in model_loss:
            tracked_loss[key] += tonumpy(model_loss[key])
        return tracked_loss

    def save_results(model, data, fname, n_fold, splits=splits):
        # Mask the data for the nan values
        XT = data['XT']
        XE = data['XE']
        XM = data['XM']
        X_soma_depth = data['X_soma_depth']

        # Run the model in the evaluation mode
        model.eval()
        zT, zEM, XrT, XrE, XrM, Xr_soma_depth = model((tensor_(XT),
                                                       tensor_(XE),
                                                       tensor_(XM),
                                                       tensor_(X_soma_depth)))

        # Get the crossmodal reconstructions
        XrT_from_zEM = model.dT(zEM)
        XrE_from_zT, XrM_from_zT, Xr_soma_depth_from_zT = model.dEM(zT)

        # Save into a mat file
        savemat = {'zT': tonumpy(zT),
                   'zEM': tonumpy(zEM),
                   'XrT': tonumpy(XrT),
                   'XrE': tonumpy(XrE),
                   'XrM': tonumpy(XrM),
                   'Xr_soma_depth': tonumpy(Xr_soma_depth),
                   'XrT_from_zEM': tonumpy(XrT_from_zEM),
                   'XrE_from_zT': tonumpy(XrE_from_zT),
                   'XrM_from_zT': tonumpy(XrM_from_zT),
                   'Xr_soma_depth_from_zT': tonumpy(Xr_soma_depth_from_zT),
                   'sample_id': data['sample_id'],
                   'cluster_label': data['cluster'],
                   'cluster_id': data['cluster_id'],
                   'cluster_color': data['cluster_color']}

        # Save the train and validation indices
        savemat.update(splits[n_fold])
        sio.savemat(fname, savemat, do_compression=True)
        return

    # Training data
    fold_fileid = fileid + "_fold_" + str(n_fold)
    train_ind = splits[n_fold]['train']
    val_ind = splits[n_fold]['val']
    train_dataset = MET_Dataset(T_dat=D['XT'][train_ind, :],
                                E_dat=D['XE'][train_ind, :],
                                M_dat=D['XM'][train_ind, :],
                                soma_depth=D['X_soma_depth'][train_ind])
    train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False,
                                  sampler=train_sampler, drop_last=True, pin_memory=True)

    # Model ============================
    model = Model_T_EM(T_dim=n_genes, T_int_dim=50, T_dropout=0.2,
                       E_dim=n_E_features, E_int_dim=50, E_dropout=0.2,
                       E_noise_sd=0.1 * np.nanstd(train_dataset.E_dat, axis=0),
                       latent_dim=latent_dim, alpha_T=alpha_T, alpha_E=alpha_E, alpha_M=alpha_M,
                       alpha_soma_depth=alpha_soma_depth, lambda_T_EM=lambda_T_EM, augment_decoders=augment_decoders)
    optimizer = torch.optim.Adam(model.parameters())

    model.to(device)

    # Training loop ====================
    best_loss = np.inf
    monitor_loss = []

    for epoch in range(n_epochs):
        train_loss_dict = {}
        train_datagen = iter(train_dataloader)
        for _ in range(len(train_dataloader)):
            batch = next(train_datagen)

            # T, E and M for each batch
            XT = batch['XT']
            XE = batch['XE']
            XM = batch['XM']
            X_soma_depth = batch['X_soma_depth']

            # zero + forward + backward + udpate
            optimizer.zero_grad()
            _ = model((tensor_(XT),
                       tensor_(XE),
                       tensor_(XM),
                       tensor_(X_soma_depth)))

            model.loss.backward()
            optimizer.step()

            # track loss over batches:
            train_loss_dict = collect_losses(model_loss=model.loss_dict, tracked_loss=train_loss_dict)

        # Validation: train mode -> eval mode + no_grad + eval mode -> train mode
        XT = D['XT'][val_ind, :]
        XE = D['XE'][val_ind, :]
        XM = D['XM'][val_ind, :]
        X_soma_depth = D['X_soma_depth'][val_ind]

        model.eval()
        with torch.no_grad():
            _ = model((tensor_(XT),
                       tensor_(XE),
                       tensor_(XM),
                       tensor_(X_soma_depth)))

        val_loss_dict = collect_losses(model_loss=model.loss_dict, tracked_loss={})
        model.train()

        train_loss = report_losses(loss_dict=train_loss_dict, partition="train")
        val_loss = report_losses(loss_dict=val_loss_dict, partition="val")
        print(f'epoch {epoch:04d} Train {train_loss}')
        print(f'epoch {epoch:04d} ----- Val {val_loss}')

        # Logging ==============
        with open(dir_pth['logs'] + f'{fold_fileid}.csv', "a") as f:
            writer = csv.writer(f, delimiter=',')
            if epoch == 0:
                writer.writerow(['epoch'] + ['train_'+ k for k in train_loss_dict] + ['val_'+ k for k in val_loss_dict])
            writer.writerow([epoch + 1, *train_loss_dict.values(), *val_loss_dict.values()])

        monitor_loss.append(val_loss_dict['recon_T'] +
                            val_loss_dict['recon_E'] +
                            val_loss_dict['recon_M'] +
                            val_loss_dict['recon_soma_depth'] +
                            val_loss_dict['cpl_T_EM'])

        # Checkpoint ===========
        if (monitor_loss[-1] < best_loss) and (epoch > 2500):
            best_loss = monitor_loss[-1]
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss, }, dir_pth['result'] + f'{fold_fileid}-best_loss_weights.pt')
    print('\nTraining completed for fold:', n_fold)
    print()

    # Save model weights on exit
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss}, dir_pth['result'] + f'{fold_fileid}-best_loss_weights.pt')
    save_fname = dir_pth['result'] + f'{fold_fileid}_exit'
    save_results(model=model, data=D.copy(), fname=f'{save_fname}-summary.mat', n_fold=n_fold)
    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))

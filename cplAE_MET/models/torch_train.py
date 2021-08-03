import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from cplAE_MET.models.torch_cplAE import Model_MET
from cplAE_MET.utils.dataset import partitions
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--batchsize",         default=100,             type=int,    help="Batch size")
parser.add_argument("--alpha_T",           default=1.0,             type=float,  help="Transcriptomic reconstruction loss weight")
parser.add_argument("--alpha_E",           default=1.0,             type=float,  help="Electrophisiology reconstruction loss weight")
parser.add_argument("--alpha_M",           default=1.0,             type=float,  help="Morphology reconstruction loss weight")
parser.add_argument("--lambda_TE",         default=1.0,             type=float,  help="T and E coupling loss weight")
parser.add_argument("--lambda_ME",         default=1.0,             type=float,  help="M and E coupling loss weight")
parser.add_argument("--lambda_MT",         default=1.0,             type=float,  help="M and T coupling loss weight")
parser.add_argument("--augment_decoders",  default=1,               type=int,    help="0 or 1 : Train with cross modal reconstruction")
parser.add_argument("--latent_dim",        default=3,               type=int,    help="Number of latent dims")
parser.add_argument("--n_epochs",          default=1000,            type=int,    help="Number of epochs to train")
parser.add_argument("--config_file",       default='config_exc_MET.toml',type=str,help="config file with data paths")
parser.add_argument("--run_iter",          default=0,               type=int,    help="Run-specific id")
parser.add_argument("--model_id",          default='MET',           type=str,    help="Model-specific id")
parser.add_argument("--exp_name",          default='MET_torch',     type=str,    help="Experiment set")


def set_paths(config_file=None, exp_name='TEMP'):
    
    from cplAE_MET.utils.load_config import load_config
    paths = load_config(config_file=config_file, verbose=False)
    paths['result'] = f'{str(paths["package_dir"] / "data/results")}/{exp_name}/'
    paths['logs'] = f'{paths["result"]}logs/'
    Path(paths['logs']).mkdir(parents=True, exist_ok=True)
    return paths


class MET_Dataset(torch.utils.data.Dataset):
    def __init__(self, T_dat, E_dat, M_dat):
        super(MET_Dataset).__init__()
        self.T_dat = T_dat
        self.E_dat = E_dat
        self.M_dat = M_dat
        self.n_samples = np.shape(self.T_dat)[0]

    def __getitem__(self, idx):
        sample = {"XT": self.T_dat[idx, :],
                  "XE": self.E_dat[idx, :],
                  "XM": self.M_dat[idx, :]}
        return sample

    def __len__(self):
        return self.n_samples


def main(alpha_T=1.0, alpha_E=1.0, alpha_M=1.0, lambda_TE=1.0, lambda_ME=1.0,
         lambda_MT=1.0, augment_decoders=1.0, batchsize=500, latent_dim=3,
         n_epochs=5000, run_iter=0, config_file='config_exc_MET.toml', model_id='MET', exp_name='MET_torch'):
    
    
    dir_pth = set_paths(config_file=config_file, exp_name=exp_name)

    fileid = (model_id +
              f'_aT_{str(alpha_T)}_aE_{str(alpha_E)}_aM_{str(alpha_M)}_' +
              f'csTE_{str(lambda_TE)}_csME_{str(lambda_ME)}_csMT_{str(lambda_MT)}_' +
              f'ad_{str(augment_decoders)}_ld_{latent_dim:d}_bs_{batchsize:d}_ne_{n_epochs:d}_' +
              f'ri_{run_iter:d}').replace('.', '-')

    #Convert int to boolean
    augment_decoders = augment_decoders > 0

    #Data selection===================
    n_genes = 1250
    n_E_features = 120
    D={}
    D["XT"] = np.random.randn(1000, 1250)
    D["XE"] = np.random.randn(1000, 120)
    D["XM"] = np.random.randn(1000, 2, 120, 4)
    
    train_ind = [i for i in range(0, 800)]
    val_ind = [i for i in range(800, 1000)]
    #train_ind, val_ind = partitions(celltype=D['cluster'], n_partitions=10, seed=0)
    splits = {'train_ind': train_ind, 'val_ind': val_ind}


    #Helpers ==========================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def tensor(x): return torch.tensor(x).to(dtype=torch.float32).to(device)
    def tensor_(x): return torch.as_tensor(x).to(dtype=torch.float32).to(device)
    def tonumpy(x): return x.cpu().detach().numpy()

    def report_losses(loss_dict, partition):
        for key in loss_dict.keys():
            if key!='steps':
                loss_dict[key] = loss_dict[key]/loss_dict['steps']
        return '   '.join([f'{partition}_{key} : {loss_dict[key]:0.5f}' for key in loss_dict])

    def collect_losses(model_loss, tracked_loss):
        #Init if keys dont exist
        if not tracked_loss:
            tracked_loss = {key:0 for key in model_loss}
            tracked_loss['steps'] = 0.0
        
        #Add info
        tracked_loss['steps'] += 1
        for key in model_loss:
            tracked_loss[key] += tonumpy(model_loss[key])
        return tracked_loss

    def save_results(model, data, fname, splits=splits):
        model.eval()
        zT, zE, zM, XrT, XrE, XrM = model(
            (tensor_(data['XT']),
             tensor_(data['XE']),
             (tensor_(data['XM']))))
        savemat = {'zT': tonumpy(zT), 'XrT': tonumpy(XrT), 
                   'zE': tonumpy(zE), 'XrE': tonumpy(XrE),
                   'zM': tonumpy(zM), 'XrM': tonumpy(XrM)}
        savemat.update(splits)
        sio.savemat(fname, savemat, do_compression=True)
        return

    #Training data
    train_dataset = MET_Dataset(T_dat=D['XT'][train_ind, :],
                                E_dat=D['XE'][train_ind, :],
                                M_dat=D['XM'][train_ind, :])
    train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False,
                                  sampler=train_sampler, drop_last=True, pin_memory=True)

    #Model ============================
    model = Model_MET(T_dim=n_genes, T_int_dim=50, T_dropout=0.2,
                      E_dim=n_E_features, E_int_dim=50, E_dropout=0.2,
                      E_noise_sd=0.1*np.std(train_dataset.E_dat,axis=0),
                      latent_dim=latent_dim, alpha_T=alpha_T, alpha_E=alpha_E, alpha_M=alpha_M,
                      lambda_TE=lambda_TE, lambda_ME=lambda_ME, lambda_MT=lambda_MT,
                      augment_decoders=augment_decoders)
    optimizer = torch.optim.Adam(model.parameters())

    model.to(device)


    #Training loop ====================
    best_loss = np.inf
    monitor_loss = []
    
    for epoch in range(n_epochs):
        train_loss_dict = {}
        train_datagen = iter(train_dataloader)
        for _ in range(len(train_dataloader)):
            batch = next(train_datagen)
            
            #zero + forward + backward + udpate
            optimizer.zero_grad()
            _ = model((tensor_(batch['XT']),
                       tensor_(batch['XE']),
                       tensor_(batch['XM'])))
            model.loss.backward()
            optimizer.step()
            
            #track loss over batches:
            train_loss_dict = collect_losses(model_loss=model.loss_dict, tracked_loss=train_loss_dict)

        #Validation: train mode -> eval mode + no_grad + eval mode -> train mode
        model.eval()
        with torch.no_grad():
            breakpoint()
            _ = model((tensor_(D['XT'][val_ind, :]), tensor_(D['XE'][val_ind, :]), tensor_(D['XM'][val_ind, :])))

        val_loss_dict = collect_losses(model_loss=model.loss_dict, tracked_loss={})
        model.train()
        
        train_loss = report_losses(loss_dict=train_loss_dict, partition="train")
        val_loss = report_losses(loss_dict=val_loss_dict, partition="val")
        print(f'epoch {epoch:04d} Train {train_loss}')
        print(f'epoch {epoch:04d} ----- Val {val_loss}')

        #Logging ==============
        with open(dir_pth['logs'] + f'{fileid}.csv', "a") as f:
            writer = csv.writer(f, delimiter=',')
            if epoch == 0:
                writer.writerow(['epoch', *train_loss_dict.keys(), *val_loss_dict.keys()])
            writer.writerow([epoch+1, *train_loss_dict.values(), *val_loss_dict.values()])
            
        monitor_loss.append(val_loss_dict['recon_T']+val_loss_dict['recon_E']+val_loss_dict['cpl_TE'])

        #Checkpoint ===========
        if (monitor_loss[-1] < best_loss) and (epoch > 2500):
            best_loss = monitor_loss[-1]
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss, }, dir_pth['result'] + f'{fileid}-best_loss_weights.pt')
    print('\nTraining completed.')

    #Save model weights on exit
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss}, dir_pth['result'] + f'{fileid}-best_loss_weights.pt')
    save_fname = dir_pth['result'] + f'{fileid}_exit'
    save_results(model=model, data=D.copy(), fname=f'{save_fname}-summary.mat')
    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))

#Notes: Masks for M data are generated using soma density nans. 

import argparse
import csv
import numpy as np
import scipy.io as sio
import tensorflow as tf
from cplAE_MET.utils.load_helpers import get_MET_dataset, get_paths
from cplAE_MET.models.model import Model_TME


parser = argparse.ArgumentParser()
parser.add_argument("--batchsize",         default=300,                     type=int,     help="Batch size")
parser.add_argument("--alpha_T",           default=1.0,                     type=float,   help="T Reconstruction loss weight")
parser.add_argument("--alpha_E",           default=1.0,                     type=float,   help="E Reconstruction loss weight")
parser.add_argument("--alpha_M",           default=1.0,                     type=float,   help="M Reconstruction loss weight")
parser.add_argument("--lambda_TE",         default=1.0,                     type=float,   help="TE Coupling loss weight")
parser.add_argument("--lambda_ME",         default=0.5,                     type=float,   help="ME Coupling loss weight")
parser.add_argument("--lambda_TM",         default=1.0,                     type=float,   help="TM Coupling loss weight")
parser.add_argument("--augment_decoders",  default=0,                       type=int,     help="0 or 1 : Train with cross modal reconstruction")
parser.add_argument("--latent_dim",        default=3,                       type=int,     help="Number of latent dims")
parser.add_argument("--n_epochs",          default=1500,                    type=int,     help="Number of epochs to train")
parser.add_argument("--n_steps_per_epoch", default=500,                     type=int,     help="Number of model updates per epoch")
parser.add_argument("--run_iter",          default=0,                       type=int,     help="Run-specific id")
parser.add_argument("--model_id",          default='TME',                   type=str,     help="Model-specific id")
parser.add_argument("--exp_name",          default='TME',                   type=str,     help="Experiment set")


def set_paths(exp_name='TEMP'):
    from pathlib import Path
    dir_pth = {}
    paths = get_paths()
    dir_pth['result'] = f'{str(paths["package"] / "data/results")}/{exp_name}/'
    dir_pth['logs'] = f'{dir_pth["result"]}logs/'
    Path(dir_pth['logs']).mkdir(parents=True, exist_ok=True)
    return dir_pth


def TME_get_splits():
    from cplAE_MET.utils.load_helpers import get_paths
    paths = get_paths()
    splits = sio.loadmat(paths['package'] / 'data/proc/train_test_splits.mat', squeeze_me=True)
    return splits['train_ind'], splits['test_ind']


class Datagen():
    """Iterator class to sample the dataset. Tensors T_dat, E_dat, M_dat are provided at runtime.
    """

    def __init__(self, maxsteps, batchsize, T_dat, E_dat, M_dat):
        self.T_dat = T_dat
        self.E_dat = E_dat
        self.M_dat = M_dat
        self.batchsize = batchsize
        self.maxsteps = maxsteps
        self.n_samples = self.T_dat.shape[0]
        self.count = 0
        return

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < self.maxsteps:
            self.count = self.count+1
            ind = np.random.randint(0, self.n_samples, self.batchsize)
            return (tf.constant(self.T_dat[ind, :], dtype=tf.float32),
                    tf.constant(self.E_dat[ind, :], dtype=tf.float32),
                    tf.constant(self.M_dat[ind, :], dtype=tf.float32))
        else:
            raise StopIteration


def main(alpha_T=1.0, alpha_E=1.0, alpha_M=1.0,
         lambda_TE=1.0, lambda_TM=1.0, lambda_ME=.2,
         augment_decoders=0,
         batchsize=200, latent_dim=3, n_epochs=1500, n_steps_per_epoch=500,
         run_iter=0, model_id='NM', exp_name='TE_NM'):

    dir_pth = set_paths(exp_name=exp_name)

    #Augmenting only makes sense when lambda_TE>0
    if lambda_TE == 0.0:
        augment_decoders = 0

    fileid = (model_id +
              f'_alpha_{str(alpha_T)}_{str(alpha_E)}_{str(alpha_M)}_' +
              f'cs_{str(lambda_TE)}-{str(lambda_TM)}-{str(lambda_ME)}_ad_{str(augment_decoders)}' +
              f'_ld_{latent_dim:d}_bs_{batchsize:d}_se_{n_steps_per_epoch:d}_ne_{n_epochs:d}' +
              f'_ri_{run_iter:d}').replace('.', '-')

    #Convert int to boolean
    augment_decoders = augment_decoders > 0

    #Data operations and definitions:
    D = get_MET_dataset()
    train_ind, val_ind = TME_get_splits()
    Partitions = {'train_ind': train_ind, 'val_ind': val_ind}

    train_T_dat = D['XT'][train_ind, :]
    train_M_dat = D['XM'][train_ind, :]
    train_Sd_dat = D['soma_depth'][train_ind]
    train_E_dat = D['XE'][train_ind, :]

    val_T_dat = D['XT'][val_ind, :]
    val_M_dat = D['XM'][val_ind, :]
    val_Sd_dat = D['soma_depth'][val_ind]
    val_E_dat = D['XE'][val_ind, :]

    #Edat_var = np.nanvar(D['XE'],axis=0)
    maxsteps = n_epochs*n_steps_per_epoch

    train_best_loss = 1e10
    val_best_loss = 1e10

    #Model definition
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    train_generator = tf.data.Dataset.from_generator(Datagen,
                                                     output_types=(tf.float32, tf.float32, tf.float32),
                                                     args=(tf.constant(maxsteps), tf.constant(batchsize),
                                                           train_T_dat, train_E_dat, train_M_dat))

    model_TME = Model_TME(T_dim=1252,
                          E_dim=68,
                          M_dim=(120, 4, 2),
                          T_intermediate_dim=50,
                          E_intermediate_dim=40,
                          alpha_T=alpha_T,
                          alpha_E=alpha_E,
                          alpha_M=alpha_M,
                          lambda_TE=lambda_TE,
                          lambda_ME=lambda_ME,
                          lambda_TM=lambda_TM,
                          T_dropout=0.5,
                          E_gauss_noise_wt=1.0,
                          E_gnoise_sd=0.05,
                          E_dropout=0.1,
                          M_gauss_noise_std=0.5,
                          latent_dim=3,
                          train_T=False,
                          train_E=False,
                          train_M=False,
                          augment_decoders=augment_decoders,
                          name='TME')

    #Model training functions
    @tf.function
    def train_fn(model, optimizer, XT, XE, XM):
        """Enclose this with tf.function to create a fast training step. Function can be used for inference as well. 
        Arguments:
            XT: T data for training or validation
            XE: E data for training or validation
            XM: M hist for training or validation
            XSd: Soma depth for training or validation
        """
        model.train_T = True
        model.train_E = True
        model.train_M = True

        with tf.GradientTape() as tape:
            zT, zE, zM, XrT, XrE, XrM = model((XT, XE, XM))
            trainable_weights = [weight for weight in model.trainable_weights]
            loss = sum(model.losses)

        grads = tape.gradient(loss, trainable_weights)
        optimizer.apply_gradients(zip(grads, trainable_weights))
        model.train_T = False
        model.train_E = False
        model.train_M = False
        return zT, zE, zM, XrT, XrE, XrM

    #Logging function
    def report_losses(model, epoch, datatype='train', verbose=False):
        mse_loss_T = model.mse_loss_T.numpy()
        mse_loss_E = model.mse_loss_E.numpy()
        mse_loss_M = model.mse_loss_M.numpy()

        mse_loss_TE = model.mse_loss_TE.numpy()
        mse_loss_ME = model.mse_loss_ME.numpy()
        mse_loss_TM = model.mse_loss_TM.numpy()

        if verbose:
            print(f'Epoch:{epoch:5d}, mse_T: {mse_loss_T:0.3f}, mse_E: {mse_loss_E:0.3f}, mse_M: {mse_loss_M:0.5f} ' +
                  f'mse_TE: {mse_loss_TE:0.5f}, mse_ME: {mse_loss_ME:0.5f}, mse_TM: {mse_loss_TM:0.5f}')

        log_name = [datatype+i for i in ['epoch', 'mse_T', 'mse_E', 'mse_M',
                                         'mse_TE', 'mse_ME', 'mse_TM']]
        log_values = [epoch, mse_loss_T, mse_loss_E,  mse_loss_M,
                      mse_loss_TE, mse_loss_ME, mse_loss_TM]

        total_loss = sum(log_values[1:])
        return log_name, log_values, total_loss
    
    def save_results(model, Data, fname, Inds=Partitions):
        model.train_T = False
        model.train_E = False
        model.train_M = False

        zT, zE, zM, XrT, XrE, XrM = model((tf.constant(Data['XT']),
                                           tf.constant(Data['XE']),
                                           tf.constant(Data['XM'])))

        savemat = {'zT': zT.numpy(),
                   'zE': zE.numpy(),
                   'zM': zM.numpy(),
                   'XrE': XrE.numpy(),
                   'XrM': XrM.numpy(),
                   'XrT': XrT.numpy()}
        savemat.update(Inds)
        sio.savemat(fname, savemat, do_compression=True)
        return

    #Main training loop ----------------------------------------------------------------------
    epoch = 0
    for step, (XT, XE, XM) in enumerate(train_generator):

        train_fn(model=model_TME, optimizer=optimizer, XT=XT, XE=XE, XM=XM)

        if (step+1) % n_steps_per_epoch == 0:
            #Update epoch count
            epoch = epoch+1

            #Collect training metrics
            model_TME((train_T_dat, train_E_dat, train_M_dat, train_Sd_dat))
            train_log_name, train_log_values, train_total_loss = report_losses(model=model_TME ,epoch=epoch, datatype='train_', verbose=True)

            #Collect validation metrics
            model_TME((val_T_dat, val_E_dat, val_M_dat, val_Sd_dat))
            val_log_name, val_log_values, val_total_loss = report_losses(model=model_TME, epoch=epoch, datatype='val_', verbose=True)
            
            with open(dir_pth['logs']+fileid+'.csv', "a") as logfile:
                writer = csv.writer(logfile, delimiter=',')
                #Write headers to the log file
                if epoch == 1:
                    writer.writerow(train_log_name+val_log_name)
                writer.writerow(train_log_values+val_log_values)

            #Save training loss network
            if (epoch > 50) & (train_best_loss > train_total_loss):
                train_best_loss = train_total_loss
                save_fname = dir_pth['result']+fileid+'_train_best_loss'
                model_TME.save_weights(f'{save_fname}-weights.h5')
                save_results(model=model_TME, Data=D.copy(), fname=f'{save_fname}-summary.mat')
                print(f'Model saved with best training loss: {train_best_loss}')

           #Save best validation loss network:
            if (epoch > 50) & (val_best_loss > val_total_loss):
                val_best_loss = val_total_loss
                save_fname = dir_pth['result']+fileid+'_val_best_loss'
                model_TME.save_weights(f'{save_fname}-weights.h5')
                save_results(model=model_TME, Data=D.copy(), fname=f'{save_fname}-summary.mat')
                print(f'Model saved with best validation loss: {val_best_loss}')







    #Save model weights on exit
    save_fname = dir_pth['result']+fileid+'_exit'
    model_TME.save_weights(f'{save_fname}-weights.h5')
    save_results(model=model_TME, Data=D.copy(), fname=f'{save_fname}-summary.mat')
    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))

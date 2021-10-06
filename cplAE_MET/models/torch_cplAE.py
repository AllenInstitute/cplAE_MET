import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
import torch.nn.functional as F
import scipy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def tensor(x): return torch.tensor(x).to(dtype=torch.float32).to(device)
def tensor_(x): return torch.as_tensor(x).to(dtype=torch.float32).to(device)
def tonumpy(x): return x.cpu().detach().numpy()

class Encoder_T(nn.Module):
    """
    Encoder for transcriptomic data

    Args:
        in_dim: input size of data
        int_dim: number of units in hidden layers
        out_dim: set to latent space dim
        dropout_p: dropout probability
    """

    def __init__(self,
                 in_dim=1000,
                 int_dim=50,
                 out_dim=3,
                 dropout_p=0.5):

        super(Encoder_T, self).__init__()
        self.drp = nn.Dropout(p=dropout_p)
        self.fc0 = nn.Linear(in_dim, int_dim)
        self.fc1 = nn.Linear(int_dim, int_dim)
        self.fc2 = nn.Linear(int_dim, int_dim)
        self.fc3 = nn.Linear(int_dim, int_dim)
        self.fc4 = nn.Linear(int_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim, affine=False, eps=1e-05,
                                 momentum=0.1, track_running_stats=True)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        return

    def forward(self, x):
        x = self.drp(x)
        x = self.elu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        z = self.bn(x)
        return z


class Decoder_T(nn.Module):
    """
    Decoder for transcriptomic data

    Args:
        in_dim: set to embedding dim obtained from encoder
        int_dim: number of units in hidden layers
        out_dim: number of outputs
    """

    def __init__(self,
                 in_dim=3,
                 int_dim=50,
                 out_dim=1000):

        super(Decoder_T, self).__init__()
        self.fc0 = nn.Linear(in_dim, int_dim)
        self.fc1 = nn.Linear(int_dim, int_dim)
        self.fc2 = nn.Linear(int_dim, int_dim)
        self.fc3 = nn.Linear(int_dim, int_dim)
        self.Xout = nn.Linear(int_dim, out_dim)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        return

    def forward(self, x):
        x = self.elu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.Xout(x))
        return x


class Encoder_E(nn.Module):
    """
    Encoder for electrophysiology data

    Args:
        per_feature_gaussian_noise_sd: std of gaussian noise injection if training=True
        in_dim: input size of data
        int_dim: number of units in hidden layers
        out_dim: set to latent space dim
        noise_sd: tensor or np.array. with shape (in_dim,) or (in_dim,1) or (1,in_dim)
        dropout_p: dropout probability
    """

    def __init__(self,
                 in_dim=301,
                 int_dim=40,
                 out_dim=3,
                 noise_sd=None,
                 dropout_p=0.1):


        super(Encoder_E, self).__init__()
        if noise_sd is not None:
            self.noise_sd = tensor_(noise_sd)
        else:
            self.noise_sd = None
        self.drp = nn.Dropout(p=dropout_p)
        self.fc0 = nn.Linear(in_dim, int_dim)
        self.fc1 = nn.Linear(int_dim, int_dim)
        self.fc2 = nn.Linear(int_dim, int_dim)
        self.fc3 = nn.Linear(int_dim, int_dim)
        self.fc4 = nn.Linear(int_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim, affine=False, eps=1e-05,
                                 momentum=0.1, track_running_stats=True)

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        return

    def addnoise(self, x):
        if (self.training) and (self.noise_sd is not None):
            # batch dim is inferred from shapes of x and self.noise_sd
            x = torch.normal(mean=x, std=self.noise_sd)
        return x

    def forward(self, x):
        x = self.addnoise(x)
        x = self.drp(x)
        x = self.elu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        z = self.bn(x)
        return z


class Decoder_E(nn.Module):
    """
    Decoder for electrophysiology data

    Args:
        in_dim: set to embedding dim obtained from encoder
        int_dim: number of units in hidden layers
        out_dim: number of outputs
    """

    def __init__(self,
                 in_dim=3,
                 int_dim=40,
                 out_dim=301,
                 dropout_p=0.1):

        super(Decoder_E, self).__init__()
        self.fc0 = nn.Linear(in_dim, int_dim)
        self.fc1 = nn.Linear(int_dim, int_dim)
        self.fc2 = nn.Linear(int_dim, int_dim)
        self.fc3 = nn.Linear(int_dim, int_dim)
        self.drp = nn.Dropout(p=dropout_p)
        self.Xout = nn.Linear(int_dim, out_dim)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        return


    def forward(self, x):
        x = self.elu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.drp(x)
        x = self.Xout(x)
        return x


class Encoder_M(nn.Module):
    """
    Encoder for morphology data. Hard-coded values for architecture. 
    Input is expected to be shape: (batch_size x 2 x 120 x 4)

    Args:
        out_dim: representation dimenionality
        std_dev: gaussian noise std dev
    """

    def __init__(self, std_dev=0.1,  out_dim=3):

        super(Encoder_M, self).__init__()
        self.gaussian_noise_std_dev=std_dev
        self.conv1_ax = nn.Conv2d(1, 10, kernel_size=(4, 3), stride=(4, 1), padding='valid')
        self.conv1_de = nn.Conv2d(1, 10, kernel_size=(4, 3), stride=(4, 1), padding='valid')

        self.conv2_ax = nn.Conv2d(10, 10, kernel_size=(2, 2), stride=(2, 1), padding='valid')
        self.conv2_de = nn.Conv2d(10, 10, kernel_size=(2, 2), stride=(2, 1), padding='valid')

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(301, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, out_dim)
        self.bn = nn.BatchNorm1d(out_dim, affine=False, eps=1e-10,
                                 momentum=0.05, track_running_stats=True)
        self.elu = nn.ELU()

        return

    def forward(self, x, soma_depth):
        if self.training:
            x = x + (torch.randn(x.shape) * self.gaussian_noise_std_dev)

        ax, de = torch.tensor_split(x, 2, dim=1)

        #remove nans from ax and de
        ax = self.elu(self.conv1_ax(ax))
        de = self.elu(self.conv1_de(de))

        ax = self.elu(self.conv2_ax(ax))
        de = self.elu(self.conv2_de(de))
        x = torch.cat(tensors=(self.flat(ax), self.flat(de)), dim=1)

        soma_depth = soma_depth.view(-1, 1)
        x = torch.cat(tensors=(x, soma_depth), dim=1)

        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.fc3(x)
        x = self.bn(x)
        return x


class Decoder_M(nn.Module):
    """
    Decoder for morphology data. Hard-coded architecture.
    Output is expected to be shape: (batch_size x 2 x 120 x 4)

    Args:
        in_dim: representation dimensionality
    """

    def __init__(self, in_dim=3):

        super(Decoder_M, self).__init__()
        self.fc1_dec = nn.Linear(in_dim, 20)
        self.fc2_dec = nn.Linear(20, 20)
        self.fc3_dec = nn.Linear(20, 301)

        self.convT1_ax = nn.ConvTranspose2d(10, 10, kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.convT1_de = nn.ConvTranspose2d(10, 10, kernel_size=(2, 2), stride=(2, 2), padding=0)

        self.convT2_ax = nn.ConvTranspose2d(10, 1, kernel_size=(4, 3), stride=(4, 1), padding=0)
        self.convT2_de = nn.ConvTranspose2d(10, 1, kernel_size=(4, 3), stride=(4, 1), padding=0)

        self.elu = nn.ELU()
        return

    def forward(self, x):
        x = self.elu(self.fc1_dec(x))
        x = self.elu(self.fc2_dec(x))
        x = self.elu(self.fc3_dec(x))

        ax_de = x[:, 0:300]
        soma_depth = x[:, 300]

        ax, de = torch.tensor_split(ax_de, 2, dim=1)
        ax = ax.view(-1, 10, 15, 1)
        de = de.view(-1, 10, 15, 1)

        ax = self.convT1_ax(ax)
        de = self.convT1_de(de)

        ax = self.convT2_ax(ax)
        de = self.convT2_de(de)

        x = torch.cat(tensors=(ax, de), dim=1)
        return x, soma_depth


class Encoder_EM(nn.Module):
    """
    Encoder for EM data.

    Args:
        E_dim: number of E features
        E_int_dim: intermediate linear layer dimension for E
        sd_int_dim: intermediate linear layer dimension for soma depth
        std_dev: gaussian noise std dev
        out_dim: representation dimenionality
        E_noise_sd: per-feature gaussian noise for E data
        E_dropout: dropout probability for E features
    """

    def __init__(self,
                 E_dim=100,
                 E_int_dim=40,
                 EM_int_dim=20,
                 M_noise=0.1,
                 out_dim=3,
                 E_noise=None,
                 E_dropout=0.1):

        super(Encoder_EM, self).__init__()

        if E_noise is not None:
            self.E_noise = tensor_(E_noise)
        else:
            self.E_noise = None

        if M_noise is not None:
            self.M_noise = tensor_(M_noise)
        else:
            self.M_noise = None

        self.drp = nn.Dropout(p=E_dropout)

        self.conv1_ax = nn.Conv2d(1, 5, kernel_size=(4, 3), stride=(4, 1), padding='valid')
        self.conv1_de = nn.Conv2d(1, 5, kernel_size=(4, 3), stride=(4, 1), padding='valid')

        self.conv2_ax = nn.Conv2d(5, 5, kernel_size=(2, 2), stride=(2, 1), padding='valid')
        self.conv2_de = nn.Conv2d(5, 5, kernel_size=(2, 2), stride=(2, 1), padding='valid')

        self.flat = nn.Flatten()
        self.fcsd = nn.Linear(1, 1)
        self.fce0 = nn.Linear(E_dim, E_int_dim)
        self.fce1 = nn.Linear(E_int_dim, E_int_dim)
        self.fce2 = nn.Linear(E_int_dim, E_int_dim)
        self.fce3 = nn.Linear(E_int_dim, EM_int_dim)
        self.bne = nn.BatchNorm1d(EM_int_dim, affine=False, eps=1e-10,
                                  momentum=0.05, track_running_stats=True)
        self.fcm1 = nn.Linear(150+1, EM_int_dim)
        self.fcm2 = nn.Linear(EM_int_dim, EM_int_dim)
        self.bnm = nn.BatchNorm1d(EM_int_dim, affine=False, eps=1e-10,
                                 momentum=0.05, track_running_stats=True)
        self.fc1 = nn.Linear(EM_int_dim + EM_int_dim, EM_int_dim)
        self.fc2 = nn.Linear(EM_int_dim, out_dim)
        # self.fc = nn.Linear(EM_int_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim, affine=False, eps=1e-10,
                                 momentum=0.05, track_running_stats=True)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        return

    def add_noise_E(self, x):
        if (self.training) and (self.E_noise is not None):
            # batch dim is inferred from shapes of x and self.E_noise
            x = torch.normal(mean=x, std=self.E_noise)
        return x

    def fix_negative_noise_M(self, x):
        shape = x.shape
        x[x < 0] = 0
        x = x.reshape(shape[0], -1)
        pixel_sums = torch.sum(x, 1)
        pixel_sums[pixel_sums == 0] = 1.
        x = torch.div(x * 1e2, pixel_sums.view(-1, 1))
        x = x.reshape(shape)
        return x

    def add_noise_M(self, x, dilated_mask_M):
        if self.training:
            x = torch.where(dilated_mask_M, x + (torch.randn(x.shape) * self.M_noise), x)
            ax, de = torch.tensor_split(x, 2, dim=1)
            ax = self.fix_negative_noise_M(ax)
            de = self.fix_negative_noise_M(de)
            x = torch.cat(tensors=(ax, de), dim=1)
        return x

    def forward(self, xe, xm, soma_depth, mask1D_e, mask1D_m, dilated_mask_M):

        #Passing xe through some layers
        xe = self.add_noise_E(xe)
        xe = self.drp(xe)
        xe = self.relu(self.fce0(xe))
        xe = self.relu(self.fce1(xe))
        xe = self.relu(self.fce2(xe))
        xe = self.relu(self.fce3(xe))
        xe = self.sigmoid(xe)

        #Passing xm through some layers
        xm = self.add_noise_M(xm, dilated_mask_M)
        ax, de = torch.tensor_split(xm, 2, dim=1)
        ax = self.elu(self.conv1_ax(ax))
        de = self.elu(self.conv1_de(de))
        ax = self.elu(self.conv2_ax(ax))
        de = self.elu(self.conv2_de(de))
        xm = torch.cat(tensors=(self.flat(ax), self.flat(de)), dim=1)

        #passing soma depth through some layers
        soma_depth = soma_depth.view(-1, 1)
        soma_depth = self.relu(self.fcsd(soma_depth))

        #concat soma depth with M
        xm = torch.cat(tensors=(xm, soma_depth), dim=1)
        xm = self.elu(self.fcm1(xm))
        xm = self.elu(self.fcm2(xm))
        xm = self.sigmoid(xm)

        x = torch.cat(tensors=(xm, xe), dim=1)
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.bn(x)

        # mask1D_only_e = torch.logical_and(mask1D_e, ~mask1D_m) #True if only e is True AND m is False
        # mask1D_only_m = torch.logical_and(mask1D_m, ~mask1D_e) #True if only m is True AND e is False
        # mask1D_both_e_and_m = torch.logical_and(mask1D_e, mask1D_m) #True if only both e and m are True
        # mask1D_both_e_or_m = torch.logical_or(mask1D_e, mask1D_m)
        #
        # y = torch.zeros_like(xm)
        # y = torch.where(mask1D_only_e.view(-1, 1), xe, y)
        # y = torch.where(mask1D_only_m.view(-1, 1), xm, y)
        # #<--------- Instead of averaging, potentially select E or M stochastically
        # y = torch.where(mask1D_both_e_and_m.view(-1, 1), torch.mean(torch.stack((xm, xe)), dim=0), y)
        # #run the final representation through more layers
        # x = self.fc(y)
        # x = torch.where(mask1D_both_e_or_m.view(-1, 1), self.bn(x), x)
        return x


class Decoder_EM(nn.Module):
    """
    Decoder for EM data. M dimensions are hard coded. 

    Args:
        in_dim: representation dimensionality
        EM_int_dim: joint E and M representation dimensionality
        E_int_dim: intermediate layer dims for E
        E_dim: output dim for E
    """

    def __init__(self,
                 in_dim=3,
                 EM_int_dim=20,
                 E_int_dim=40,
                 E_dim=100,
                 ):

        super(Decoder_EM, self).__init__()
        self.fc0_dec = nn.Linear(in_dim, EM_int_dim)
        self.fc1_dec = nn.Linear(EM_int_dim, EM_int_dim)
        self.fcm0_dec = nn.Linear(EM_int_dim, EM_int_dim)
        self.fcm1_dec = nn.Linear(EM_int_dim, 150+1)
        self.fcsd_dec = nn.Linear(1, 1)
        self.fce0_dec = nn.Linear(EM_int_dim, E_int_dim)
        self.fce1_dec = nn.Linear(E_int_dim, E_int_dim)
        self.fce2_dec = nn.Linear(E_int_dim, E_int_dim)
        self.fce3_dec = nn.Linear(E_int_dim, E_dim)

        self.convT1_ax = nn.ConvTranspose2d(5, 5, kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.convT1_de = nn.ConvTranspose2d(5, 5, kernel_size=(2, 2), stride=(2, 2), padding=0)

        self.convT2_ax = nn.ConvTranspose2d(5, 1, kernel_size=(4, 3), stride=(4, 1), padding=0)
        self.convT2_de = nn.ConvTranspose2d(5, 1, kernel_size=(4, 3), stride=(4, 1), padding=0)

        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        return

    def forward(self, x):

        x = self.elu(self.fc0_dec(x))
        x = self.elu(self.fc1_dec(x))

        #separating xm and xe
        xm = self.relu(self.fcm0_dec(x))
        xe = self.relu(self.fce0_dec(x))

        #passing xm through some layers
        xm = self.relu(self.fcm1_dec(xm))

        #separating soma_depth
        ax_de = xm[:, 0:150]
        soma_depth = xm[:, 150:]
        soma_depth = self.fcsd_dec(soma_depth)

        #separating ax and de and passing them through conv layers
        ax, de = torch.tensor_split(ax_de, 2, dim=1)
        ax = ax.view(-1, 5, 15, 1)
        de = de.view(-1, 5, 15, 1)
        ax = self.convT1_ax(ax)
        de = self.convT1_de(de)
        ax = self.convT2_ax(ax)
        de = self.convT2_de(de)
        xm = torch.cat(tensors=(ax, de), dim=1)

        #passing xe through some layers
        xe = self.relu(self.fce1_dec(xe))
        xe = self.relu(self.fce2_dec(xe))
        xe = self.fce3_dec(xe)

        return xe, xm, soma_depth

class Model_T_EM(nn.Module):
    """Coupled autoencoder model for Transcriptomics(T) and EM(Electrophisiology and Morphology)

    Args:
        T_dim: Number of genes in T data
        E_dim: Number of features in E data
        T_int_dim: hidden layer dims for T model
        E_int_dim: hidden layer dims for E model
        EM_int_dim: common dim for E and M
        T_dropout: dropout for T data
        E_dropout: dropout for E data
        E_noise: per-feature gaussian noise
        M_noise: gaussian noise with same variance on all pixels
        latent_dim: dim for representations
        alpha_T: loss weight for T reconstruction
        alpha_E: loss weight for E reconstruction
        alpha_M: loss weight for M reconstruction
        alpha_sd: loss weight for soma depth reconstruction
        lambda_T_EM: loss weight coupling loss between T and EM
        augment_decoders (bool): augment decoder with cross modal representation if True
    """

    def __init__(self,
                 T_dim=1000, T_int_dim=50, T_dropout=0.5,
                 E_dim=100,  E_int_dim=40, EM_int_dim=20,
                 E_dropout=0.5, E_noise=None, M_noise=1.0,
                 latent_dim=3,
                 alpha_T=1.0, alpha_E=1.0, alpha_M=1.0, alpha_sd=1.0,
                 lambda_T_EM=1.0, augment_decoders=True):

        super(Model_T_EM, self).__init__()

        self.T_dim = T_dim
        self.T_int_dim = T_int_dim
        self.T_dropout = T_dropout
        self.E_dim = E_dim
        self.E_int_dim = E_int_dim
        self.EM_int_dim = EM_int_dim
        self.latent_dim = latent_dim

        self.E_dropout = E_dropout
        self.E_noise = E_noise
        self.M_noise = M_noise
        self.augment_decoders = augment_decoders

        self.alpha_T = alpha_T
        self.alpha_E = alpha_E
        self.alpha_M = alpha_M
        self.alpha_sd = alpha_sd
        self.lambda_T_EM = lambda_T_EM

        self.eT = Encoder_T(dropout_p=T_dropout,
                            in_dim=T_dim,
                            out_dim=latent_dim,
                            int_dim=T_int_dim)

        self.eEM = Encoder_EM(E_dim=E_dim,
                              E_int_dim=E_int_dim,
                              EM_int_dim=EM_int_dim,
                              out_dim=latent_dim,
                              E_noise=E_noise,
                              E_dropout=E_dropout,
                              M_noise=M_noise,
                              )

        self.dT = Decoder_T(in_dim=latent_dim,
                            out_dim=T_dim,
                            int_dim=T_int_dim)
        
        self.dEM = Decoder_EM(in_dim=latent_dim,
                              EM_int_dim=EM_int_dim,
                              E_int_dim=E_int_dim,
                              E_dim=E_dim)
        return

    def get_hparams(self):
        hparam_dict = {}
        hparam_dict['T_dim'] = self.T_dim
        hparam_dict['T_int_dim'] = self.T_int_dim
        hparam_dict['T_dropout'] = self.T_dropout
        hparam_dict['E_dim'] = self.E_dim
        hparam_dict['E_int_dim'] = self.E_int_dim
        hparam_dict['EM_int_dim'] = self.EM_int_dim
        hparam_dict['E_dropout'] = self.E_dropout
        hparam_dict['E_noise'] = self.E_noise
        hparam_dict['M_noise'] = self.M_noise
        hparam_dict['latent_dim'] = self.latent_dim
        hparam_dict['alpha_T'] = self.alpha_T
        hparam_dict['alpha_E'] = self.alpha_E
        hparam_dict['alpha_M'] = self.alpha_M
        hparam_dict['alpha_sd'] = self.alpha_sd
        hparam_dict['lambda_T_EM'] = self.lambda_T_EM
        hparam_dict['augment_decoders'] = self.augment_decoders

    def min_var_loss(self, zi, zj, valid_zi, valid_zj):
        if self.training:
            #SVD calculated over all entries in the batch
            zj_masked = zj[valid_zj]
            zj_masked_size = zj_masked.shape[0]
            zj_masked_centered = zj_masked - torch.mean(zj_masked, 0, True)
            min_eig = torch.min(torch.linalg.svdvals(zj_masked_centered))
            min_var_zj = torch.square(min_eig)/(zj_masked_size-1)

            zi_masked = zi[valid_zi]
            zi_masked_size = zi_masked.shape[0]
            zi_masked_centered = zi_masked - torch.mean(zi_masked, 0, True)
            min_eig = torch.min(torch.linalg.svdvals(zi_masked_centered))
            min_var_zi = torch.square(min_eig)/(zi_masked_size-1)

            #Wij_paired is the weight of matched pairs
            both_valid = torch.logical_and(valid_zi, valid_zj)
            zi_zj_sq_dist = torch.sum(torch.square((zi-zj)[both_valid]), axis=1)
            loss_ij = torch.mean(zi_zj_sq_dist/torch.squeeze(torch.minimum(min_var_zi, min_var_zj)))
        else:
            both_valid = torch.logical_and(valid_zi, valid_zj)
            zi_zj_sq_dist = torch.mean(torch.sum(torch.square((zi-zj)[both_valid]), axis=1))
            loss_ij = torch.mean(zi_zj_sq_dist)
        return loss_ij

    @staticmethod
    def mean_sq_diff(x, y):
        # return torch.mean(torch.square(x-y))
        return F.mse_loss(y, x, reduction='mean') if (x.numel() != 0) & (y.numel() != 0) else tensor(0.)

    @staticmethod
    def get_1D_mask(mask):
        mask = mask.reshape(mask.shape[0], -1)
        return torch.any(mask, dim=1)

    def get_pairs(self, mask1, mask2):
        return torch.logical_and(mask1, mask2)

    @staticmethod
    def get_dilation_mask(x):
        ax, de = torch.tensor_split(x, 2, dim=1)
        kernel_tensor = torch.tensor(
            [[[[1., 1., 1.],
               [1., 1., 1.],
               [1., 1., 1.]]]])
        dilated_mask_ax = torch.clamp(torch.nn.functional.conv2d(ax, kernel_tensor, padding=(1, 1)), 0, 1)
        dilated_mask_de = torch.clamp(torch.nn.functional.conv2d(de, kernel_tensor, padding=(1, 1)), 0, 1)
        dilated_mask_M = torch.cat(tensors=(dilated_mask_ax, dilated_mask_de), dim=1).bool()
        return dilated_mask_M

    def forward(self, inputs):
        #inputs
        XT = inputs[0]
        XE = inputs[1]
        XM = inputs[2]
        X_sd = inputs[3]

        #define element-wise nan masks: 0 if nan
        masks={}
        masks['T'] = ~torch.isnan(XT)
        masks['E'] = ~torch.isnan(XE)
        masks['M'] = ~torch.isnan(XM)
        masks['sd'] = ~torch.isnan(X_sd)

        valid_E = self.get_1D_mask(masks['E'])
        valid_M = self.get_1D_mask(masks['M'])
        valid_T = self.get_1D_mask(masks['T'])
        valid_EM = torch.logical_or(valid_E, valid_M)

        #replacing nans in input with zeros
        XT = torch.nan_to_num(XT, nan=0.)
        XE = torch.nan_to_num(XE, nan=0.)
        XM = torch.nan_to_num(XM, nan=0.)
        X_sd = torch.nan_to_num(X_sd, nan=0.)

        #create the dilation mask for M data
        dilated_mask_M = self.get_dilation_mask((XM>0).float())

        #T arm forward pass
        zT = self.eT(XT)
        XrT = self.dT(zT)

        #EM arm forward pass
        zEM = self.eEM(XE, XM, X_sd, valid_E, valid_M, dilated_mask_M)
        XrE, XrM, Xr_sd = self.dEM(zEM)

        #Loss calculations
        loss_dict = {}
        loss_dict['recon_T'] = self.alpha_T * self.mean_sq_diff(XT[masks['T']], XrT[masks['T']])
        loss_dict['recon_E'] = self.alpha_E * self.mean_sq_diff(XE[masks['E']], XrE[masks['E']])
        loss_dict['recon_M'] = self.alpha_M * self.mean_sq_diff(XM[masks['M']], XrM[masks['M']])
        loss_dict['recon_sd'] = self.alpha_sd * self.mean_sq_diff(X_sd[masks['sd']], Xr_sd[masks['sd']])

        loss_dict['cpl_T_EM'] = self.lambda_T_EM * self.min_var_loss(zT, zEM, valid_T, valid_EM)

        if self.augment_decoders:
            XrT_aug = self.dT(zT.detach())
            XrE_aug, XrM_aug, Xr_sd_aug = self.dEM(zEM.detach())
            loss_dict['recon_T_aug'] = self.alpha_T * self.mean_sq_diff(XT[masks['T']], XrT_aug[masks['T']])
            loss_dict['recon_E_aug'] = self.alpha_E * self.mean_sq_diff(XE[masks['E']], XrE_aug[masks['E']])
            loss_dict['recon_M_aug'] = self.alpha_M * self.mean_sq_diff(XM[masks['M']], XrM_aug[masks['M']])
            loss_dict['recon_M_sd_aug'] = self.alpha_sd * self.mean_sq_diff(X_sd[masks['sd']],Xr_sd_aug[masks['sd']])

        self.loss = sum(loss_dict.values())
        return zT, zEM, XrT, XrE, XrM, Xr_sd, loss_dict


class Model_TE(nn.Module):
    """Coupled autoencoder model

    Args:
        T_dim: Number of genes in T data
        E_dim: Number of genes in E data
        T_int_dim: hidden layer dims for T model
        E_int_dim: hidden layer dims for E model
        T_dropout: dropout for T data
        E_dropout: dropout for E data
        E_noise_sd: per-feature Gaussian noise standard deviation
        latent_dim: dim for representations
        alpha_T: loss weight for T reconstruction
        alpha_E: loss weight for E reconstruction
        lambda_TE: loss weight coupling loss
        augment_decoders (bool): augment decoder with cross modal representation if True
        name: TE
    """

    def __init__(self,
                 T_dim=1000, T_int_dim=50, T_dropout=0.5,
                 E_dim=1000, E_int_dim=50, E_dropout=0.5, E_noise_sd=None, 
                 latent_dim=3, alpha_T=1.0, alpha_E=1.0, lambda_TE=1.0,
                 augment_decoders=True):

        super(Model_TE, self).__init__()
        self.alpha_T = alpha_T
        self.alpha_E = alpha_E
        self.lambda_TE = lambda_TE
        self.augment_decoders = augment_decoders

        self.eT = Encoder_T(dropout_p=T_dropout, in_dim=T_dim, out_dim=latent_dim, int_dim=T_int_dim)
        self.eE = Encoder_E(dropout_p=E_dropout, noise_sd=E_noise_sd, in_dim=E_dim, out_dim=latent_dim, int_dim=E_int_dim)
        self.dT = Decoder_T(in_dim=latent_dim, out_dim=T_dim, int_dim=T_int_dim)
        self.dE = Decoder_E(in_dim=latent_dim, out_dim=E_dim, int_dim=E_int_dim)
        return

    @staticmethod
    def min_var_loss(zi, zj):
        #SVD calculated over all entries in the batch
        batch_size = zj.shape[0]
        zj_centered = zj - torch.mean(zj, 0, True)
        min_eig = torch.min(torch.linalg.svdvals(zj_centered))
        min_var_zj = torch.square(min_eig)/(batch_size-1)

        zi_centered = zi - torch.mean(zi, 0, True)
        min_eig = torch.min(torch.linalg.svdvals(zi_centered))
        min_var_zi = torch.square(min_eig)/(batch_size-1)

        #Wij_paired is the weight of matched pairs
        zi_zj_mse = torch.mean(torch.sum(torch.square(zi-zj), 1))
        loss_ij = zi_zj_mse/torch.squeeze(torch.minimum(min_var_zi, min_var_zj))
        return loss_ij

    @staticmethod
    def mean_sq_diff(x, y):
        return torch.mean(torch.square(x-y))

    def forward(self, inputs):
        #T arm forward pass
        XT = inputs[0]
        zT = self.eT(XT)
        XrT = self.dT(zT)

        #E arm forward pass
        XE = inputs[1]
        zE = self.eE(XE)
        XrE = self.dE(zE)

        #Loss calculations
        self.loss_dict = {}
        self.loss_dict['recon_T'] = self.alpha_T * self.mean_sq_diff(XT, XrT)
        self.loss_dict['recon_E'] = self.alpha_E * self.mean_sq_diff(XE, XrE)
        self.loss_dict['cpl_TE'] = self.lambda_TE * self.min_var_loss(zT, zE)

        if self.augment_decoders:
            XrT_aug = self.dT(zE.detach())
            XrE_aug = self.dE(zT.detach())
            self.loss_dict['recon_T_aug'] = self.alpha_T * self.mean_sq_diff(XT, XrT_aug)
            self.loss_dict['recon_E_aug'] = self.alpha_E * self.mean_sq_diff(XE, XrE_aug)

        self.loss = sum(self.loss_dict.values())
        return zT, zE, XrT, XrE


class Model_MET(nn.Module):
    """Coupled autoencoder model for morphology(M), electrophysiology(E) and Transcriptomics(T)

    Args:
        T_dim: Number of genes in T data
        E_dim: Number of genes in E data
        T_int_dim: hidden layer dims for T model
        E_int_dim: hidden layer dims for E model
        T_dropout: dropout for T data
        E_dropout: dropout for E data
        E_noise_sd: per-feature gaussian noise
        latent_dim: dim for representations
        alpha_T: loss weight for T reconstruction
        alpha_E: loss weight for E reconstruction
        alpha_M: loss weight for M reconstruction
        lambda_TE: loss weight coupling loss between T and E
        lambda_MT: loss weight coupling loss between M and T
        lambda_ME: loss weight coupling loss between M and E
        std_dev: gaussian noise std dev
        augment_decoders (bool): augment decoder with cross modal representation if True
    """

    def __init__(self,
                 T_dim=1000, T_int_dim=50, T_dropout=0.5,
                 E_dim=1000, E_int_dim=50, E_dropout=0.5, E_noise_sd=None,
                 latent_dim=3, alpha_T=1.0, alpha_E=1.0, alpha_soma_depth=1.0,
                 alpha_M=1.0, lambda_TE=1.0, lambda_MT=1.0,
                 lambda_ME=1.0, std_dev=1.0, augment_decoders=True):

        super(Model_MET, self).__init__()
        self.alpha_T = alpha_T
        self.alpha_E = alpha_E
        self.alpha_M = alpha_M
        self.alpha_soma_depth = alpha_soma_depth
        self.lambda_TE = lambda_TE
        self.lambda_ME = lambda_ME
        self.lambda_MT = lambda_MT
        self.augment_decoders = augment_decoders

        self.eT = Encoder_T(dropout_p=T_dropout, in_dim=T_dim, out_dim=latent_dim, int_dim=T_int_dim)
        self.eE = Encoder_E(dropout_p=E_dropout, noise_sd=E_noise_sd, in_dim=E_dim, out_dim=latent_dim, int_dim=E_int_dim)
        self.eM = Encoder_M(std_dev=std_dev, out_dim=latent_dim)

        self.dT = Decoder_T(in_dim=latent_dim, out_dim=T_dim, int_dim=T_int_dim)
        self.dE = Decoder_E(in_dim=latent_dim, out_dim=E_dim, int_dim=E_int_dim)
        self.dM = Decoder_M(in_dim=latent_dim)
        return

    @staticmethod
    def min_var_loss(zi, zj):
        #SVD calculated over all entries in the batch
        batch_size = zj.shape[0]
        zj_centered = zj - torch.mean(zj, 0, True)
        min_eig = torch.min(torch.linalg.svdvals(zj_centered))
        min_var_zj = torch.square(min_eig)/(batch_size-1)

        zi_centered = zi - torch.mean(zi, 0, True)
        min_eig = torch.min(torch.linalg.svdvals(zi_centered))
        min_var_zi = torch.square(min_eig)/(batch_size-1)

        #Wij_paired is the weight of matched pairs
        zi_zj_mse = torch.mean(torch.sum(torch.square(zi-zj), 1))
        loss_ij = zi_zj_mse/torch.squeeze(torch.minimum(min_var_zi, min_var_zj))
        return loss_ij

    @staticmethod
    def mean_sq_diff(x, y):
        return torch.mean(torch.square(x-y))

    @staticmethod
    def get_pairs(mask1, mask2):
        mask1 = mask1.reshape(mask1.shape[0], -1)
        mask2 = mask2.reshape(mask2.shape[0], -1)
        mask1 = torch.all(mask1, dim=1)
        mask2 = torch.all(mask2, dim=1)
        return torch.logical_and(mask1, mask2)

    def forward(self, inputs):
        #inputs
        XT = inputs[0]
        XE = inputs[1]
        XM = inputs[2]
        X_soma_depth = inputs[3]

        #Saving the masks for nans
        masks={}
        masks['T'] = ~torch.isnan(XT)
        masks['E'] = ~torch.isnan(XE)
        masks['M'] = ~torch.isnan(XM)
        masks['soma_depth'] = ~torch.isnan(X_soma_depth)

        #replacing nans with zeros
        XT = torch.nan_to_num(XT, nan=0.)
        XE = torch.nan_to_num(XE, nan=0.)
        XM = torch.nan_to_num(XM, nan=0.)
        X_soma_depth = torch.nan_to_num(X_soma_depth, nan=0.)

        #T arm forward pass
        zT = self.eT(XT)
        XrT = self.dT(zT)

        # E arm forward pass
        zE = self.eE(XE)
        XrE = self.dE(zE)

        # M arm forward pass
        zM_z_soma_depth = self.eM(XM, X_soma_depth)
        XrM, Xr_soma_depth = self.dM(zM_z_soma_depth)

        pairs = {}
        pairs['ET'] = self.get_pairs(masks['T'], masks['E'])
        pairs['TM'] = self.get_pairs(masks['T'], masks['M'])
        pairs['EM'] = self.get_pairs(masks['E'], masks['M'])

        #Loss calculations
        self.loss_dict = {}
        self.loss_dict['recon_T'] = self.alpha_T * self.mean_sq_diff(XT[masks['T']], XrT[masks['T']])
        self.loss_dict['recon_E'] = self.alpha_E * self.mean_sq_diff(XE[masks['E']], XrE[masks['E']])
        self.loss_dict['recon_M'] = self.alpha_M * self.mean_sq_diff(XM[masks['M']], XrM[masks['M']])
        self.loss_dict['recon_soma_depth'] = self.alpha_soma_depth * self.mean_sq_diff(X_soma_depth[masks['soma_depth']],
                                                                                       Xr_soma_depth[masks['soma_depth']])

        self.loss_dict['cpl_TE'] = self.lambda_TE * self.min_var_loss(zT[pairs['ET']], zE[pairs['ET']])
        self.loss_dict['cpl_ME'] = self.lambda_ME * self.min_var_loss(zM_z_soma_depth[pairs['EM']], zE[pairs['EM']])
        self.loss_dict['cpl_MT'] = self.lambda_MT * self.min_var_loss(zM_z_soma_depth[pairs['TM']], zT[pairs['TM']])

        if self.augment_decoders:
            XrT_aug = self.dT(zT.detach())
            XrE_aug = self.dE(zE.detach())
            XrM_aug, Xr_soma_depth_aug = self.dM(zM_z_soma_depth.detach())
            self.loss_dict['recon_T_aug'] = self.alpha_T * self.mean_sq_diff(XT[masks['T']], XrT_aug[masks['T']])
            self.loss_dict['recon_E_aug'] = self.alpha_E * self.mean_sq_diff(XE[masks['E']], XrE_aug[masks['E']])
            self.loss_dict['recon_M_aug'] = self.alpha_M * self.mean_sq_diff(XM[masks['M']], XrM_aug[masks['M']])
            self.loss_dict['recon_sd_aug'] = self.alpha_soma_depth * self.mean_sq_diff(
                X_soma_depth[masks['sd']],
                Xr_soma_depth_aug[masks['sd']])

        self.loss = sum(self.loss_dict.values())
        return zT, zE, zM_z_soma_depth, XrT, XrE, XrM, Xr_sd

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Encoder_M(nn.Module):
    """
    supervised classification.

    Args:
    """
    def __init__(self, latent_dim=100, M_noise=0.):
        super(Encoder_M, self).__init__()
        self.drp = nn.Dropout(p=0.5)
        self.flat = nn.Flatten()
        self.fcm1 = nn.Linear(80, 10)
        self.fc = nn.Linear(11, latent_dim)
        self.M_noise = M_noise

        self.conv3d_1 = nn.Conv3d(1, 1, kernel_size=(7, 3, 1), padding=(3, 1, 0))
        self.conv3d_2 = nn.Conv3d(1, 1, kernel_size=(7, 3, 1), padding=(3, 1, 0))

        self.pool3d_1 = nn.MaxPool3d((4, 1, 1), return_indices=True)
        self.pool3d_2 = nn.MaxPool3d((3, 1, 1), return_indices=True)

        self.bn = nn.BatchNorm1d(latent_dim, affine=False, eps=1e-05,
                                 momentum=0.1, track_running_stats=True)

        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        return

    def shift3d(self, arr, num, fill_value=0):
        result = torch.empty_like(arr)
        if num > 0:
            result[:num, :, :] = fill_value
            result[num:, :, :] = arr[:-num, :, :]
        elif num < 0:
            result[num:, :, :] = fill_value
            result[:num, :, :] = arr[-num:, :, :]
        else:
            result = arr
        return result

    def aug_shift_range(self, h, s):
        if self.training:
            zmax = h.shape[2]  # 120
            rand_shifts = torch.zeros((h.shape[0],), dtype=torch.int)
            for i in range(h.shape[0]):
                shift_low = s[i, 0].item()
                shift_high = s[i, 1].item()
                select = torch.nonzero(h[i, 0, :, :, :])
                zrange = torch.min(select[:, 0]).item(), torch.max(select[:, 0]).item()
                low = -np.minimum(zrange[0], abs(shift_low))
                high = np.minimum(zmax - zrange[1], shift_high)
                rand_shifts[i] = np.random.randint(low, high + 1)
                if rand_shifts[i] > 0: break
                h[i, 0, :, :, :] = self.shift3d(h[i, 0, :, :, :], rand_shifts[i])
        else:
            rand_shifts = None
        return (h, rand_shifts)

    def aug_shift_select(self, h):
        if self.training:
            shift_low = -5
            shift_high = 5
            rand_shifts = torch.zeros((h.shape[0],), dtype=torch.int)
            for i in range(h.shape[0]):
                if torch.max(h[i, 0, :, :, 0:2]) > 0:  # apply shift only to aspiny cells
                    select = torch.nonzero(h[i, 0, :, :, :])
                    zrange = torch.min(select[:, 0]).item(), torch.max(select[:, 0]).item()
                    low = -np.minimum(zrange[0], abs(shift_low))
                    high = np.minimum(120 - zrange[1], shift_high)
                    rand_shifts[i] = np.random.randint(low, high + 1)
                    if rand_shifts[i]>0: break
                    h[i, 0, :, :, :] = self.shift3d(h[i, 0, :, :, :], rand_shifts[i])
        else:
            rand_shifts = None
        return (h, rand_shifts)


    def aug_noise(self, h, nonzero_mask_xm):
        if self.training:
            return torch.where(nonzero_mask_xm, h + (torch.randn(h.shape) * self.M_noise), h)
        else:
            return h

    def aug_fnoise(self, f):
        if self.training:
            return f + (torch.randn(f.shape) * self.M_noise)
        else:
            return f

    def aug_soma_depth(self, f, s):
        if self.training:
            return f + torch.unsqueeze(s, 1)
        else:
            return f


    def forward(self, x1, x2, shifts, nonzero_mask_xm):

        #augmentation
        aug_xm = self.aug_noise(x1, nonzero_mask_xm)
        aug_xm, rand_shift = self.aug_shift_range(aug_xm, shifts)

        xm, pool1_indices = self.pool3d_1(self.relu(self.conv3d_1(aug_xm)))
        xm, pool2_indices = self.pool3d_2(self.relu(self.conv3d_2(xm)))
        xm = xm.view(xm.shape[0], -1)
        xm = self.elu(self.fcm1(xm))

        aug_x_sd = self.aug_fnoise(x2)
        aug_x_sd = self.aug_soma_depth(aug_x_sd, rand_shift)
        x_sd = self.sigmoid(aug_x_sd)

        #concat soma depth with M
        xm = torch.cat(tensors=(xm, x_sd), dim=1)
        z = self.bn(self.fc(xm))
        return z, aug_xm, aug_x_sd, pool1_indices, pool2_indices




class Decoder_M(nn.Module):
    """
    Decoder for morphology data. Hard-coded architecture.
    Output is expected to be shape: (batch_size x 2 x 120 x 4)

    Args:
        in_dim: representation dimensionality
    """

    def __init__(self, in_dim=3):

        super(Decoder_M, self).__init__()
        self.fc1_dec = nn.Linear(in_dim, 11)
        self.fcm_dec = nn.Linear(10, 80)



        self.convT1_1 = nn.ConvTranspose3d(1, 1, kernel_size=(7, 3, 1), padding=(3, 1, 0))
        self.convT1_2 = nn.ConvTranspose3d(1, 1, kernel_size=(7, 3, 1), padding=(3, 1, 0))

        self.unpool3d_1 = nn.MaxUnpool3d((4, 1, 1))
        self.unpool3d_2 = nn.MaxUnpool3d((3, 1, 1))

        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        return

    def forward(self, x, p1_ind, p2_ind):
        x = self.fc1_dec(x)

        xm = x[:, 0:10]
        soma_depth = x[:, 10]
        soma_depth = soma_depth.view(-1, 1)
        soma_depth = self.sigmoid(soma_depth)

        xm = self.elu(self.fcm_dec(xm))
        xm = xm.view(-1, 1, 10, 4, 2)
        xm = self.elu(self.unpool3d_2(xm, p2_ind))
        xm = self.convT1_1(xm)
        xm = self.elu(self.unpool3d_1(xm, p1_ind))
        xm = self.convT1_2(xm)
        # more layers

        return xm, soma_depth

class Model_M_AE(nn.Module):
    """Coupled autoencoder model for Transcriptomics(T) and EM(Electrophisiology and Morphology)

    Args:
        M_noise: gaussian noise with same variance on all pixels
        latent_dim: dim for representations
        alpha_M: loss weight for M reconstruction
        alpha_sd: loss weight for soma depth reconstruction
        augment_decoders (bool): augment decoder with cross modal representation if True
    """

    def __init__(self,
                 M_noise=0.,
                 latent_dim=3,
                 alpha_M=1.0,
                 alpha_sd=1.0,
                 augment_decoders=True):

        super(Model_M_AE, self).__init__()
        self.M_noise = M_noise
        self.latent_dim = latent_dim
        self.augment_decoders = augment_decoders
        self.alpha_M = alpha_M
        self.alpha_sd = alpha_sd

        self.eM = Encoder_M(latent_dim=self.latent_dim, M_noise=self.M_noise)

        self.dM = Decoder_M(in_dim=self.latent_dim)

        return

    def get_hparams(self):
        hparam_dict = {}
        hparam_dict['M_noise'] = self.M_noise
        hparam_dict['latent_dim'] = self.latent_dim
        hparam_dict['alpha_M'] = self.alpha_M
        hparam_dict['alpha_sd'] = self.alpha_sd


    @staticmethod
    def mean_sq_diff(x, y):
        return torch.mean(torch.square(x - y))

    @staticmethod
    def get_1D_mask(mask):
        mask = mask.reshape(mask.shape[0], -1)
        return torch.any(mask, dim=1)

    @staticmethod
    def get_dilation_mask(x):
        ax, de = torch.tensor_split(x, 2, dim=1)
        kernel_tensor = torch.tensor(
            [[[[1., 1., 1.],
               [1., 1., 1.],
               [1., 1., 1.]]]])

        dilated_mask_ax = torch.clamp(nn.conv2d(ax, kernel_tensor, padding=(1, 1)), 0, 1)
        dilated_mask_de = torch.clamp(nn.conv2d(de, kernel_tensor, padding=(1, 1)), 0, 1)
        dilated_mask_M = torch.cat(tensors=(dilated_mask_ax, dilated_mask_de), dim=1).bool()
        return dilated_mask_M

    def forward(self, inputs):
        # inputs
        XM = inputs[0]
        X_sd = inputs[1]
        shifts = inputs[2]

        # define element-wise nan masks: 0 if nan
        masks = {}
        masks['M'] = ~torch.isnan(XM)
        masks['sd'] = ~torch.isnan(X_sd)

        nonzero_mask_xm = XM != 0.

        valid_M = self.get_1D_mask(masks['M'])

        # replacing nans in input with zeros
        XM = torch.nan_to_num(XM, nan=0.)
        X_sd = torch.nan_to_num(X_sd, nan=0.)

        # create the dilation mask for M data
        # dilated_mask_M = self.get_dilation_mask((XM > 0).float())

        # EM arm forward pass
        z, aug_xm, aug_x_sd, p1_ind, p2_ind = self.eM(XM, X_sd, shifts, nonzero_mask_xm)
        XrM, Xr_sd = self.dM(z, p1_ind, p2_ind)

        # Loss calculations
        loss_dict = {}
        loss_dict['recon_M'] = self.alpha_M * self.mean_sq_diff(aug_xm[valid_M], XrM[valid_M])
        loss_dict['recon_sd'] = self.alpha_sd * self.mean_sq_diff(aug_x_sd[valid_M], Xr_sd[valid_M])


        self.loss = sum(loss_dict.values())
        return XrM, Xr_sd, loss_dict
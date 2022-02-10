import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Encoder_M(nn.Module):
    """
    Encoder for morphology arbor density data.
    Args:
        latent_dim: dimension of the latent representation
        M_noise: a number that is used to generate noise for the arbor density images
        scale_factor: scale factor to stretch or squeeze the images along the H axis
    """
    def __init__(self, latent_dim=100, M_noise=0., scale_factor=0.):
        super(Encoder_M, self).__init__()

        self.do_aug_scale = True
        self.do_aug_noise = True
        self.conv3d_1 = nn.Conv3d(1, 1, kernel_size=(7, 3, 1), padding=(3, 1, 0))
        self.pool3d_1 = nn.MaxPool3d((4, 1, 1), return_indices=True)
        self.conv3d_2 = nn.Conv3d(1, 1, kernel_size=(7, 3, 1), padding=(3, 1, 0))
        self.pool3d_2 = nn.MaxPool3d((4, 1, 1), return_indices=True)



        self.fcm1 = nn.Linear(240, 10)
        self.fc1 = nn.Linear(11, 11)
        self.fc2 = nn.Linear(11, latent_dim)
        self.M_noise = M_noise
        self.scale_factor = scale_factor
        self.bn = nn.BatchNorm1d(latent_dim, affine=False, eps=1e-05,
                                 momentum=0.1, track_running_stats=True)

        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        return


    def aug_noise(self, h, nonzero_mask_xm):
        if self.training:
            return torch.where(nonzero_mask_xm, h + (torch.randn(h.shape, device=device) * self.M_noise), h)
        else:
            return h


    def aug_fnoise(self, f):
        if self.training:
            return f + (torch.randn(f.shape, device=device) * self.M_noise)
        else:
            return f


    def aug_scale_im(self, im, scaling_by):
        '''
        Scaling the image by interpolation. The original images are padded beforehand to make sure they dont go out
        of the frame during the scaling. We padded them with H/2 along H
        Args:
            im: original images with the size of (N, 1, H, W, C)
            scaling_by: scaling factor for interpolation

        '''
        # scaling the cells
        min_rand = 1. - scaling_by
        max_rand = 1. + scaling_by
        depth_scaling_factor = (torch.rand(1) * (
                    max_rand - min_rand) + min_rand).item()  # generate random numbers between min and max
        print(depth_scaling_factor)
        out = F.interpolate(im.float(), scale_factor=(depth_scaling_factor, 1, 1))  # scale the image
        return out


    def pad_or_crop_im(self, scaled_im, im):
        '''
        Takes the scaled image and the original image and either crop or pad the scaled image to get the original size.
        Args:
            scaled_im: scaled image with the size of (N, 1, h, W, C), h is the scaled image height
            im: original image with the size of (N, 1, H, W, C)

        Returns:
            padded_or_copped_im: padded or cropped images with the size of (M, 1, H, W, C)

        '''
        out_depth = scaled_im.shape[2]
        in_depth = im.shape[2]

        # cropping or padding the image to get to the original size
        depth_diff = out_depth - in_depth
        patch = int(depth_diff / 2)
        patch_correction = depth_diff - patch * 2

        if depth_diff < 0:
            pad = (0, 0, 0, 0, -(patch + patch_correction), -patch)
            paded_or_croped_im = F.pad(scaled_im, pad, "constant", 0)

        elif depth_diff == 1:
            paded_or_croped_im = scaled_im[:, :, 1:, :, :]

        elif depth_diff == 0:
            paded_or_croped_im = scaled_im

        else:
            paded_or_croped_im = scaled_im[:, :, (patch + patch_correction): -patch, :, :]

        return paded_or_croped_im


    def aug_scale(self, im, scaling_by=0.1):
        '''
        Scaling the image and then getting back to the original size by cropping or padding
        Args:
            im: soma aligned images with the size of (N, 1, H, W, C)
            scaling_by: scaling factor, a float between 0 and 1

        Returns:
            scaled image
        '''

        if self.training:
            scaled_im = self.aug_scale_im(im, scaling_by)
            scaled_im = self.pad_or_crop_im(scaled_im, im)
        else:
            scaled_im = im
        return scaled_im


    def forward(self, xm, x_sd, nonzero_mask_xm):

        if self.do_aug_noise:
            xm = self.aug_noise(xm, nonzero_mask_xm)
            x_sd = self.aug_fnoise(x_sd)
        if self.do_aug_scale:
            xm = self.aug_scale(xm, scaling_by=self.scale_factor)

        x, pool1_ind = self.pool3d_1(self.relu(self.conv3d_1(xm)))
        x, pool2_ind = self.pool3d_2(self.relu(self.conv3d_2(x)))
        x = x.view(x.shape[0], -1)
        x = self.elu(self.fcm1(x))

        x = torch.cat(tensors=(x, x_sd), dim=1)
        x = self.relu(self.fc1(x))
        z = self.bn(self.fc2(x))

        return z, xm, x_sd, pool1_ind, pool2_ind



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
        self.fc2_dec = nn.Linear(11, 11)
        self.fcm_dec = nn.Linear(10, 240)

        self.convT1_1 = nn.ConvTranspose3d(1, 1, kernel_size=(7, 3, 1), padding=(3, 1, 0))
        self.convT1_2 = nn.ConvTranspose3d(1, 1, kernel_size=(7, 3, 1), padding=(3, 1, 0))

        self.unpool3d_1 = nn.MaxUnpool3d((4, 1, 1))
        self.unpool3d_2 = nn.MaxUnpool3d((4, 1, 1))

        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        return

    def forward(self, x, p1_ind, p2_ind):
        x = self.elu(self.fc1_dec(x))
        x = self.elu(self.fc2_dec(x))

        xrm = x[:, 0:10]
        x_rsd = x[:, 10]
        x_rsd = torch.clamp(x_rsd.view(-1, 1), min=0, max=1)

        xrm = self.elu(self.fcm_dec(xrm))
        xrm = xrm.view(-1, 1, 15, 4, 4)
        xrm = self.elu(self.unpool3d_1(xrm, p2_ind))
        xrm = self.convT1_1(xrm)
        xrm = self.elu(self.unpool3d_2(xrm, p1_ind))
        xrm = self.convT1_2(xrm)

        return xrm, x_rsd

class Model_M_AE(nn.Module):
    """M autoencoder

    Args:
        M_noise: standard deviation of additive gaussian noise
        latent_dim: dim for representations
        alpha_M: loss weight for M reconstruction
        alpha_sd: loss weight for soma depth reconstruction
    """

    def __init__(self,
                 M_noise=0.,
                 scale_factor=0.,
                 latent_dim=3,
                 alpha_M=1.0,
                 alpha_sd=1.0):

        super(Model_M_AE, self).__init__()
        self.M_noise = M_noise
        self.scale_factor = scale_factor
        self.latent_dim = latent_dim
        self.alpha_M = alpha_M
        self.alpha_sd = alpha_sd

        self.eM = Encoder_M(latent_dim=self.latent_dim,
                            M_noise=self.M_noise,
                            scale_factor=self.scale_factor)

        self.dM = Decoder_M(in_dim=self.latent_dim)
        return

    def get_hparams(self):
        hparam_dict = {}
        hparam_dict['M_noise'] = self.M_noise
        hparam_dict['scale_factor'] = self.scale_factor
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

    def forward(self, inputs):
        # inputs
        XM = inputs[0]
        X_sd = inputs[1]
        mask_XM_nans = ~torch.isnan(XM)
        mask_XM_nonzero = XM != 0.
        valid_M = self.get_1D_mask(mask_XM_nans)

        # replacing nans in input with zeros
        XM = torch.nan_to_num(XM, nan=0.)
        X_sd = torch.nan_to_num(X_sd, nan=0.)

        # EM arm forward pass
        z, xm, x_sd, p1_ind, p2_ind = self.eM(XM, X_sd, mask_XM_nonzero)
        XrM, Xr_sd = self.dM(z, p1_ind, p2_ind)

        # Loss calculations
        loss_dict = {}
        loss_dict['recon_M'] = self.mean_sq_diff(xm[valid_M, :], XrM[valid_M, :])
        loss_dict['recon_sd'] = self.mean_sq_diff(x_sd[valid_M], Xr_sd[valid_M])
        return XrM, Xr_sd, loss_dict, z
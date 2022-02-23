import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from cplAE_MET.models.torch_helpers import astensor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
astensor_ = partial(astensor, device=device)

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


    def forward(self, xm, xsd, nonzero_mask_xm):

        if self.do_aug_noise:
            xm = self.aug_noise(xm, nonzero_mask_xm)
            xsd = self.aug_fnoise(xsd)
        if self.do_aug_scale:
            xm = self.aug_scale(xm, scaling_by=self.scale_factor)

        x, pool1_ind = self.pool3d_1(self.relu(self.conv3d_1(xm)))
        x, pool2_ind = self.pool3d_2(self.relu(self.conv3d_2(x)))
        x = x.view(x.shape[0], -1)
        x = self.elu(self.fcm1(x))

        x = torch.cat(tensors=(x, xsd), dim=1)
        x = self.relu(self.fc1(x))
        z = self.bn(self.fc2(x))

        return z, xm, xsd, pool1_ind, pool2_ind, x



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

    def forward(self, x, p1_ind, p2_ind, common_decoder):
        if not common_decoder:
            x = self.elu(self.fc1_dec(x))
            x = self.elu(self.fc2_dec(x))

        xrm = x[:, 0:10]
        xrsd = x[:, 10]
        xrsd = torch.clamp(xrsd.view(-1, 1), min=0, max=1)

        xrm = self.elu(self.fcm_dec(xrm))
        xrm = xrm.view(-1, 1, 15, 4, 4)
        xrm = self.elu(self.unpool3d_1(xrm, p2_ind))
        xrm = self.convT1_1(xrm)
        xrm = self.elu(self.unpool3d_2(xrm, p1_ind))
        xrm = self.convT1_2(xrm)

        return xrm, xrsd



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
                 in_dim=1252,
                 int_dim=50,
                 latent_dim=3,
                 dropout_p=0.5):

        super(Encoder_T, self).__init__()
        self.drp = nn.Dropout(p=dropout_p)
        self.fc0 = nn.Linear(in_dim, int_dim)
        self.fc1 = nn.Linear(int_dim, int_dim)
        self.fc2 = nn.Linear(int_dim, int_dim)
        self.fc3 = nn.Linear(int_dim, int_dim)
        self.fc4 = nn.Linear(int_dim, latent_dim)
        self.bn = nn.BatchNorm1d(latent_dim, affine=False, eps=1e-05,
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
                 latent_dim=3,
                 int_dim=50,
                 out_dim=1252):

        super(Decoder_T, self).__init__()
        self.fc0 = nn.Linear(latent_dim, int_dim)
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
        E_noise: tensor or np.array. with shape (in_dim,) or (in_dim,1) or (1,in_dim)
        dropout_p: dropout probability
    """

    def __init__(self,
                 in_dim=134,
                 int_dim=40,
                 latent_dim=3,
                 E_noise=None,
                 dropout_p=0.1):


        super(Encoder_E, self).__init__()
        if E_noise is not None:
            self.E_noise = astensor_(E_noise)
        else:
            self.E_noise = None
        self.drp = nn.Dropout(p=dropout_p)
        self.fc0 = nn.Linear(in_dim, int_dim)
        self.fc1 = nn.Linear(int_dim, int_dim)
        self.fc2 = nn.Linear(int_dim, int_dim)
        self.fc3 = nn.Linear(int_dim, int_dim)
        self.fc4 = nn.Linear(int_dim, latent_dim)
        self.bn = nn.BatchNorm1d(latent_dim, affine=False, eps=1e-05,
                                 momentum=0.1, track_running_stats=True)

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        return

    def addnoise(self, x):
        if (self.training) and (self.E_noise is not None):
            # batch dim is inferred from shapes of x and self.noise_sd
            x = torch.normal(mean=x, std=self.E_noise)
        return x

    def forward(self, x):
        x = self.addnoise(x)
        x = self.drp(x)
        x = self.elu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        z = self.bn(self.fc4(x))
        return z, x


class Decoder_E(nn.Module):
    """
    Decoder for electrophysiology data

    Args:
        in_dim: set to embedding dim obtained from encoder
        int_dim: number of units in hidden layers
        out_dim: number of outputs
    """

    def __init__(self,
                 latent_dim=3,
                 int_dim=40,
                 out_dim=134,
                 dropout_p=0.1):

        super(Decoder_E, self).__init__()
        self.fc0 = nn.Linear(latent_dim, int_dim)
        self.fc1 = nn.Linear(int_dim, int_dim)
        self.fc2 = nn.Linear(int_dim, int_dim)
        self.fc3 = nn.Linear(int_dim, int_dim)
        # self.drp = nn.Dropout(p=dropout_p)
        self.Xout = nn.Linear(int_dim, out_dim)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        return


    def forward(self, x, common_decoder):
        if not common_decoder:
            x = self.elu(self.fc0(x))
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3(x))
            # x = self.drp(x)
        x = self.Xout(x)
        return x



class Encoder_ME(nn.Module):
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
                 ME_in_dim=51,
                 ME_int_dim=20,
                 latent_dim=5):

        super(Encoder_ME, self).__init__()

        self.fc_me1 = nn.Linear(ME_in_dim, ME_int_dim)
        self.fc_me2 = nn.Linear(ME_int_dim, ME_int_dim)
        self.fc_me3 = nn.Linear(ME_int_dim, latent_dim)
        self.bn = nn.BatchNorm1d(latent_dim, affine=False, eps=1e-10,
                                 momentum=0.05, track_running_stats=True)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        return


    def forward(self, x):

        x = self.elu(self.fc_me1(x))
        x = self.elu(self.fc_me2(x))
        x = self.elu(self.fc_me3(x))
        zme = self.bn(x)

        return zme


class Decoder_ME(nn.Module):
    """
    Decoder for EM data. M dimensions are hard coded.

    Args:
        in_dim: representation dimensionality
        EM_int_dim: joint E and M representation dimensionality
        E_int_dim: intermediate layer dims for E
        E_dim: output dim for E
    """

    def __init__(self,
                 latent_dim=3,
                 ME_int_dim=20,
                 ME_in_dim=51):

        super(Decoder_ME, self).__init__()
        self.fc_me1 = nn.Linear(latent_dim, ME_int_dim)
        self.fc_me2 = nn.Linear(ME_int_dim, ME_int_dim)
        self.fc_me3 = nn.Linear(ME_int_dim, ME_in_dim)

        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        return

    def forward(self, x):
        x = self.relu(self.fc_me1(x))
        x = self.relu(self.fc_me2(x))
        x = self.relu(self.fc_me3(x))

        return x


class Model_T_AE(nn.Module):
    """T autoencoder

    Args:
        latent_dim: dim for representations
        alpha_T: loss weight for soma depth reconstruction
    """

    def __init__(self, alpha_T=1.0, latent_dim=3):

        super(Model_T_AE, self).__init__()
        self.latent_dim = latent_dim
        self.alpha_T = alpha_T

        self.eT = Encoder_T(latent_dim=self.latent_dim)
        self.dT = Decoder_T(latent_dim=self.latent_dim)

        return

    def get_hparams(self):
        hparam_dict = {}
        hparam_dict['latent_dim'] = self.latent_dim
        hparam_dict['alpha_T'] = self.alpha_T

    @staticmethod
    def mean_sq_diff(x, y):
        return torch.mean(torch.square(x - y))

    @staticmethod
    def get_1D_mask(mask):
        mask = mask.reshape(mask.shape[0], -1)
        return torch.any(mask, dim=1)

    def forward(self, inputs):
        # inputs
        XT = inputs
        mask_XT_nans = ~torch.isnan(XT)  #returns a boolian which is true when the value is not nan
        valid_T = self.get_1D_mask(mask_XT_nans) # if ALL the values for one cell is nan, then that cell is not being used in loss calculation

        # replacing nans in input with zeros
        XT = torch.nan_to_num(XT, nan=0.)

        # EM arm forward pass
        z = self.eT(XT)
        XrT = self.dT(z)

        # Loss calculations
        loss_dict = {}
        loss_dict['recon_T'] = self.mean_sq_diff(XT[valid_T, :], XrT[valid_T, :])
        return XrT, loss_dict, z


class Model_E_AE(nn.Module):
    """E autoencoder

    Args:
        E_noise: standard deviation of additive gaussian noise
        latent_dim: dim for representations
        alpha_E: loss weight for E reconstruction
    """

    def __init__(self, alpha_E=1.0, latent_dim=3, E_noise=None):

        super(Model_E_AE, self).__init__()
        self.latent_dim = latent_dim
        self.alpha_E = alpha_E
        self.E_noise = E_noise

        self.eE = Encoder_E(latent_dim=self.latent_dim, E_noise=self.E_noise)
        self.dE = Decoder_E(latent_dim=self.latent_dim)

        return

    def get_hparams(self):
        hparam_dict = {}
        hparam_dict['latent_dim'] = self.latent_dim
        hparam_dict['alpha_E'] = self.alpha_E
        hparam_dict['E_noise'] = self.E_noise

    @staticmethod
    def mean_sq_diff(x, y):
        return torch.mean(torch.square(x - y))

    @staticmethod
    def get_1D_mask(mask):
        mask = mask.reshape(mask.shape[0], -1)
        return torch.any(mask, dim=1)

    def forward(self, inputs):
        # inputs
        XE = inputs
        mask_XE_nans = ~torch.isnan(XE)  #returns a boolian which is true when the value is not nan
        valid_E = self.get_1D_mask(mask_XE_nans) # if ALL the values for one cell is nan, then that cell is not being used in loss calculation

        # replacing nans in input with zeros
        XE = torch.nan_to_num(XE, nan=0.)

        # EM arm forward pass
        z = self.eE(XE)
        XrE = self.dE(z)

        # Loss calculations
        loss_dict = {}
        loss_dict['recon_E'] = self.mean_sq_diff(XE[valid_E, :], XrE[valid_E, :])
        return XrE, loss_dict, z


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
        Xsd = inputs[1]
        mask_XM_nans = ~torch.isnan(XM)
        mask_XM_nonzero = XM != 0.
        valid_M = self.get_1D_mask(mask_XM_nans)

        # replacing nans in input with zeros
        XM = torch.nan_to_num(XM, nan=0.)
        Xsd = torch.nan_to_num(Xsd, nan=0.)

        # EM arm forward pass
        z, xm, xsd, p1_ind, p2_ind = self.eM(XM, Xsd, mask_XM_nonzero)
        XrM, Xr_sd = self.dM(z, p1_ind, p2_ind)

        # Loss calculations
        loss_dict = {}
        loss_dict['recon_M'] = self.mean_sq_diff(xm[valid_M, :], XrM[valid_M, :])
        loss_dict['recon_sd'] = self.mean_sq_diff(xsd[valid_M], Xr_sd[valid_M])
        return XrM, Xr_sd, loss_dict, z


class Model_T_ME(nn.Module):
    """E autoencoder

    Args:
        E_noise: standard deviation of additive gaussian noise
        latent_dim: dim for representations
        alpha_E: loss weight for E reconstruction
    """

    def __init__(self,
                 alpha_T=1.0,
                 alpha_M=1.0,
                 alpha_sd=1.0,
                 alpha_E=1.0,
                 scale_factor=0.,
                 E_noise=None,
                 M_noise=0.,
                 latent_dim=5):

        super(Model_T_ME, self).__init__()
        self.alpha_T = alpha_T
        self.alpha_M = alpha_M
        self.alpha_sd = alpha_sd
        self.alpha_E = alpha_E
        self.scale_factor = scale_factor
        self.M_noise = M_noise
        self.E_noise = E_noise
        self.latent_dim = latent_dim

        self.eT = Encoder_T(latent_dim=self.latent_dim)
        self.dT = Decoder_T(latent_dim=self.latent_dim)

        self.eM = Encoder_M(latent_dim=self.latent_dim,
                            M_noise=self.M_noise,
                            scale_factor=self.scale_factor)
        self.dM = Decoder_M(in_dim=self.latent_dim)

        self.eE = Encoder_E(latent_dim=self.latent_dim, E_noise=self.E_noise)
        self.dE = Decoder_E(latent_dim=self.latent_dim)

        self.eME = Encoder_ME(ME_in_dim=51, ME_int_dim=20, latent_dim=self.latent_dim)
        self.dME = Decoder_ME(ME_in_dim=51, ME_int_dim=20, latent_dim=self.latent_dim)

        return

    def get_hparams(self):
        hparam_dict = {}
        hparam_dict['alpha_T'] = self.alpha_T
        hparam_dict['alpha_M'] = self.alpha_M
        hparam_dict['alpha_sd'] = self.alpha_sd
        hparam_dict['alpha_E'] = self.alpha_E
        hparam_dict['scale_factor'] = self.scale_factor
        hparam_dict['M_noise'] = self.M_noise
        hparam_dict['E_noise'] = self.E_noise
        hparam_dict['latent_dim'] = self.latent_dim

    @staticmethod
    def min_var_loss(zi, zj):
        # SVD calculated over all entries in the batch
        batch_size = zj.shape[0]
        zj_centered = zj - torch.mean(zj, 0, True)
        min_eig = torch.min(torch.linalg.svdvals(zj_centered))
        min_var_zj = torch.square(min_eig) / (batch_size - 1)

        zi_centered = zi - torch.mean(zi, 0, True)
        min_eig = torch.min(torch.linalg.svdvals(zi_centered))
        min_var_zi = torch.square(min_eig) / (batch_size - 1)

        # Wij_paired is the weight of matched pairs
        zi_zj_mse = torch.mean(torch.sum(torch.square(zi - zj), 1))
        loss_ij = zi_zj_mse / torch.squeeze(torch.minimum(min_var_zi, min_var_zj))
        return loss_ij

    @staticmethod
    def mean_sq_diff(x, y):
        return torch.mean(torch.square(x - y))

    @staticmethod
    def get_1D_mask(x):
        mask = ~torch.isnan((x))
        mask = mask.reshape(mask.shape[0], -1)
        return torch.all(mask, dim=1)

    def forward(self, inputs):
        # inputs
        XT = inputs[0]
        XM = inputs[1]
        Xsd = inputs[2]
        XE = inputs[3]

        # input data masks for encoder M and E
        valid_T = self.get_1D_mask(XT)
        valid_M = self.get_1D_mask(XM)
        valid_sd = self.get_1D_mask(Xsd)
        valid_E = self.get_1D_mask(XE) # if ALL the values for one cell is nan, then that cell is not being used in loss calculation
        valid_ME = torch.where((valid_E) & (valid_M), True, False)
        valid_TM = torch.where((valid_T) & (valid_M), True, False)
        valid_TE = torch.where((valid_T) & (valid_E), True, False)

        assert (valid_sd == valid_M).all()

        # input data mask for encoder ME
        ME_M_pairs = valid_ME[valid_M]
        ME_E_pairs = valid_ME[valid_E]
        ME_T_pairs = valid_ME[valid_T]
        T_M_pairs = valid_TM
        T_E_pairs = valid_TE

        # removing nans
        XT = XT[valid_T]
        XM = XM[valid_M]
        Xsd = Xsd[valid_sd]
        XE = XE[valid_E]

        # get the nanzero mask for M for adding noise
        mask_XM_zero = XM == 0.

        # encoder
        zt = self.eT(XT)
        zmsd, xm_aug, _, pool_ind1, pool_ind2, xmsd_inter = self.eM(XM, Xsd, mask_XM_zero)
        ze, xe_inter = self.eE(XE)
        XME_inter = torch.cat(tensors=(xmsd_inter[ME_M_pairs], xe_inter[ME_E_pairs]), dim=1)
        zme = self.eME(XME_inter)

        # decoder
        XrT = self.dT(zt)
        XrM, Xrsd = self.dM(zmsd, pool_ind1, pool_ind2, common_decoder=False)
        XrE = self.dE(ze, common_decoder=False)
        XrME_inter = self.dME(zme)
        XrM_from_zme, Xrsd_from_zme = self.dM(XrME_inter[:, :11], pool_ind1[ME_M_pairs], pool_ind2[ME_M_pairs], common_decoder=True)
        XrE_from_zme = self.dE(XrME_inter[:, 11:], common_decoder=True)

        # Loss calculations
        loss_dict = {}
        loss_dict['recon_T'] = self.mean_sq_diff(XT, XrT)
        loss_dict['recon_M'] = self.mean_sq_diff(xm_aug, XrM)
        loss_dict['recon_sd'] = self.mean_sq_diff(Xsd, Xrsd)
        loss_dict['recon_E'] = self.mean_sq_diff(XE, XrE)
        loss_dict['recon_ME'] = self.mean_sq_diff(XM[ME_M_pairs], XrM_from_zme) +\
                                self.mean_sq_diff(Xsd[ME_M_pairs], Xrsd_from_zme) +\
                                self.mean_sq_diff(XE[ME_E_pairs], XrE_from_zme)

        loss_dict['cpl_ME_M'] = self.min_var_loss(zmsd[ME_M_pairs], zme)
        loss_dict['cpl_ME_E'] = self.min_var_loss(ze[ME_E_pairs], zme)
        loss_dict['cpl_ME_T'] = self.min_var_loss(zt[ME_T_pairs], zme)

        loss_dict['cpl_TM'] = self.min_var_loss(zt[T_M_pairs], zmsd)
        loss_dict['cpl_TE'] = self.min_var_loss(zt[T_E_pairs], ze)


        return loss_dict, zt , zmsd, ze, zme, XrT, XrM, Xrsd, XrE, valid_T, valid_M, valid_E, valid_ME, valid_TM, valid_TE
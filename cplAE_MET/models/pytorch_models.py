import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from cplAE_MET.models.torch_helpers import astensor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
astensor_ = partial(astensor, device=device)

class Encoder_M_shared(nn.Module):
    """
    Encoder for morphology arbor density data.
    Args:
        latent_dim: (int) dimension of the latent representation
        M_noise: (float) std of gaussian noise injection to non zero pixels
        scale_factor: (float) scale factor to stretch or squeeze the images along the H axis
    """
    def __init__(self,
                 M_noise=0.,
                 scale_factor=0.):
        super(Encoder_M_shared, self).__init__()

        self.M_noise = M_noise
        self.scale_factor = scale_factor

        self.do_aug_scale = True
        self.do_aug_noise = True

        self.conv3d_1 = nn.Conv3d(1, 1, kernel_size=(7, 3, 1), padding=(3, 1, 0))
        self.pool3d_1 = nn.MaxPool3d((4, 1, 1), return_indices=True)
        self.conv3d_2 = nn.Conv3d(1, 1, kernel_size=(7, 3, 1), padding=(3, 1, 0))
        self.pool3d_2 = nn.MaxPool3d((4, 1, 1), return_indices=True)
        self.fcm0 = nn.Linear(240, 10)
        self.fc0 = nn.Linear(11, 11)

        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        return


    def aug_noise(self, im):
        '''
        Get the image and mask for the nonzero pixels and add gaussian noise to the nonzero pixels if training
        Args:
            im: array with the shape of (batch_size x 1 x 240 x 4 x 4)
        '''
        # get the nanzero mask for M for adding noise
        mask = im != 0.
        if self.training:
            noise = torch.randn(im.shape, device=device) * self.M_noise
            return torch.where(mask, im + noise, im)
        else:
            return im


    def aug_fnoise(self, sd):
        '''
        Gets the soma depth and return the soma depth augmented with gaussian noise if training
        Args:
            sd: soma depth location array with the shape of (batch_size x 1)
        '''
        if self.training:
            noise = torch.randn(sd.shape, device=device) * self.M_noise
            return sd + noise
        else:
            return sd


    def aug_scale_im(self, im, scaling_by):
        '''
        Scaling the image by interpolation. The original images are padded beforehand to make sure they dont go out
        of the frame during the scaling. We padded them with H/2 along H and they are soma aligned beforehand
        Args:
            im: padded arbor density images with the size of (batch_size x 1 x 240 x 4 x 4)
            scaling_by: (float) scaling factor for interpolation
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
            scaled_im: scaled image with the size of (batch_size x 1 x 240 x 4 x 4), h is the scaled image height
            im: original image with the size of (batch_size x 1 x 240 x 4 x 4)

        Returns:
            padded_or_copped_im: padded or cropped images with the size of (batch_size x 1 x 240 x 4 x 4)

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
            im: soma aligned images with the size of (batch_size x 1 x 240 x 4 x 4)
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


    def forward(self, xm, xsd):

        if self.do_aug_noise:
            xm = self.aug_noise(xm)
            xsd = self.aug_fnoise(xsd)
        if self.do_aug_scale:
            xm = self.aug_scale(xm, scaling_by=self.scale_factor)

        # xm is the augmented image, if we need to reconstruct the augmented image then we need to output this xm
        x, pool1_ind = self.pool3d_1(self.relu(self.conv3d_1(xm)))
        x, pool2_ind = self.pool3d_2(self.relu(self.conv3d_2(x)))
        x = x.view(x.shape[0], -1)
        # x = x.view((x.shape[0], x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4]))
        x = self.elu(self.fcm0(x))

        x = torch.cat(tensors=(x, xsd), dim=1)
        x = self.relu(self.fc0(x))

        return xm, xsd, pool1_ind, pool2_ind, x


class Encoder_M_specific(nn.Module):
    """
    Encoder for morphology arbor density data.
    Args:
        latent_dim: (int) dimension of the latent representation
        M_noise: (float) std of gaussian noise injection to non zero pixels
        scale_factor: (float) scale factor to stretch or squeeze the images along the H axis
    """
    def __init__(self,
                 latent_dim=100):
        super(Encoder_M_specific, self).__init__()

        self.fc0 = nn.Linear(11, latent_dim)
        self.bn = nn.BatchNorm1d(latent_dim, affine=False, eps=1e-05,
                                 momentum=0.1, track_running_stats=True)

        return


    def forward(self, x):

        z = self.bn(self.fc0(x))

        return z



class Decoder_M_specific(nn.Module):
    """
    NOTE: Decoder M has two modules called "Decoder_M_specific" and "Decoder_M_shared"

    Specific module of Decoder for morphology data is used only when we are reconstructing XM and
     Xsd from zmsd. The architecture is Hard-coded.
    Input is expected to be shape: (batch_size x latent_dim) and the output: (batch_size x 11)

    Args:
        in_dim: set to the representation dimensionality
    """

    def __init__(self,
                 in_dim=3):

        super(Decoder_M_specific, self).__init__()
        self.fc0 = nn.Linear(in_dim, 11)
        self.fc1 = nn.Linear(11, 11)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        return

    def forward(self, x):
        x = self.elu(self.fc0(x))
        x = self.elu(self.fc1(x))
        return x


class Decoder_M_shared(nn.Module):
    """
    Shared module of decoder M is used when we are reconstructing XrM, Xrsd or cross modal reconstruction
    such as XrM_from_zme of Xrsd_from_zme.
    The architecture is hard-coded.
    The input is of the size: (batch_size x 11) and the output dimension is
    XM_from_zme with: (batch_size x 1 x 240 x 4 x 4) and Xrsd_from_zme with the size (batch_size, 1)
    """

    def __init__(self):

        super(Decoder_M_shared, self).__init__()
        self.fcm0 = nn.Linear(10, 240)

        self.convT1_1 = nn.ConvTranspose3d(1, 1, kernel_size=(7, 3, 1), padding=(3, 1, 0))
        self.convT1_2 = nn.ConvTranspose3d(1, 1, kernel_size=(7, 3, 1), padding=(3, 1, 0))

        self.unpool3d_1 = nn.MaxUnpool3d((4, 1, 1))
        self.unpool3d_2 = nn.MaxUnpool3d((4, 1, 1))

        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        return

    def forward(self, x, p1_ind, p2_ind):
        xrm = x[:, 0:10]
        xrsd = x[:, 10]
        xrsd = torch.clamp(xrsd.view(-1, 1), min=0, max=1)

        xrm = self.elu(self.fcm0(xrm))
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
        in_dim: input size of data, hard-coded
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
                 in_dim=3,
                 int_dim=50,
                 out_dim=1252):

        super(Decoder_T, self).__init__()
        self.fc0 = nn.Linear(in_dim, int_dim)
        self.fc1 = nn.Linear(int_dim, int_dim)
        self.fc2 = nn.Linear(int_dim, int_dim)
        self.fc3 = nn.Linear(int_dim, int_dim)
        self.fc4 = nn.Linear(int_dim, out_dim)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        return

    def forward(self, x):
        x = self.elu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return x


class Encoder_E_shared(nn.Module):
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
                 E_noise=None,
                 dropout_p=0.1):


        super(Encoder_E_shared, self).__init__()
        if E_noise is not None:
            self.E_noise = astensor_(E_noise)
        else:
            self.E_noise = None
        self.drp = nn.Dropout(p=dropout_p)
        self.fc0 = nn.Linear(in_dim, int_dim)
        self.fc1 = nn.Linear(int_dim, int_dim)
        self.fc2 = nn.Linear(int_dim, int_dim)
        self.fc3 = nn.Linear(int_dim, int_dim)

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
        return x


class Encoder_E_specific(nn.Module):
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
                 int_dim=40,
                 latent_dim=3):


        super(Encoder_E_specific, self).__init__()
        self.fc0 = nn.Linear(int_dim, latent_dim)
        self.bn = nn.BatchNorm1d(latent_dim, affine=False, eps=1e-05,
                                 momentum=0.1, track_running_stats=True)
        return

    def forward(self, x):
        z = self.bn(self.fc0(x))
        return z


class Decoder_E_specific(nn.Module):
    """
    NOTE: Decoder E has two modules called "Decoder_E_specific" and "Decoder_E_shared"

    Specific module of Decoder for electrophysiology data is used only when we are reconstructing XE from ze.
    The architecture is Hard-coded.
    Input is expected to be shape: (batch_size x latent_dim) and the output: (batch_size x 40)
    Args:
        latent_dim: set to embedding dim obtained from encoder
        out_dim: number of units in hidden layers
    """

    def __init__(self,
                 in_dim=3,
                 out_dim=40,
                 dropout_p=0.1):

        super(Decoder_E_specific, self).__init__()
        self.fc0 = nn.Linear(in_dim, out_dim)
        self.drp = nn.Dropout(p=dropout_p)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        return


    def forward(self, x):
        x = self.elu(self.fc0(x))
        return x



class Decoder_E_shared(nn.Module):
    """
    Shared module of decoder E is used when we are reconstructing XrE or XrE_from_zme(cross modal reconstruction of XE).
    The architecture is hard-coded.
    The input is of the size: (batch_size x 40) and the output is XrE_from_zme with dimension: (batch_size x 134)
    Args:
        int_dim: number of units in hidden layers
        out_dim: number of outputs
    """

    def __init__(self,
                 int_dim=40,
                 out_dim=134):

        super(Decoder_E_shared, self).__init__()
        self.fc0 = nn.Linear(int_dim, int_dim)
        self.fc1 = nn.Linear(int_dim, int_dim)
        self.fc2 = nn.Linear(int_dim, int_dim)
        self.fc3 = nn.Linear(int_dim, out_dim)
        self.relu = nn.ReLU()

        return

    def forward(self, x):
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.drp(x)
        x = self.fc3(x)
        return x


class Encoder_ME(nn.Module):
    """
    Encoder for EM data.

    Args:
        in_dim: number of ME features
        int_dim: intermediate linear layer dimension for ME
        latent_dim: representation dimenionality
    """

    def __init__(self,
                 in_dim=51,
                 int_dim=20,
                 latent_dim=5):

        super(Encoder_ME, self).__init__()

        self.fc0 = nn.Linear(in_dim, int_dim)
        self.fc1 = nn.Linear(int_dim, int_dim)
        self.fc2 = nn.Linear(int_dim, latent_dim)
        self.bn = nn.BatchNorm1d(latent_dim, affine=False, eps=1e-10,
                                 momentum=0.05, track_running_stats=True)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        return


    def forward(self, x):

        x = self.elu(self.fc0(x))
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        zme = self.bn(x)

        return zme


class Decoder_ME(nn.Module):
    """
    Decoder for ME data.

    Args:
        in_dim: representation dimensionality
        int_dim: intermediate layer dims for ME
        out_dim: ME output dimension
    """

    def __init__(self,
                 in_dim=3,
                 int_dim=20,
                 out_dim=51):

        super(Decoder_ME, self).__init__()
        self.fc0 = nn.Linear(in_dim, int_dim)
        self.fc1 = nn.Linear(int_dim, int_dim)
        self.fc2 = nn.Linear(int_dim, out_dim)

        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        return

    def forward(self, x):
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        return x



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
                 alpha_E=1.0,
                 alpha_ME=1.0,
                 lambda_ME_T=1.0,
                 lambda_ME_M=1.0,
                 lambda_ME_E=1.0,
                 scale_factor=0.,
                 E_noise=None,
                 M_noise=0.,
                 latent_dim=5):

        super(Model_T_ME, self).__init__()
        self.alpha_T = alpha_T
        self.alpha_M = alpha_M
        self.alpha_E = alpha_E
        self.alpha_ME = alpha_ME
        self.lambda_ME_T = lambda_ME_T
        self.lambda_ME_M = lambda_ME_M
        self.lambda_ME_E = lambda_ME_E
        self.scale_factor = scale_factor
        self.M_noise = M_noise
        self.E_noise = E_noise
        self.latent_dim = latent_dim

        # T
        self.eT = Encoder_T(latent_dim=self.latent_dim)
        self.dT = Decoder_T(in_dim=self.latent_dim)

        # M
        self.eM_shared = Encoder_M_shared(M_noise=self.M_noise, scale_factor=self.scale_factor)
        self.eM_specific = Encoder_M_specific(latent_dim=self.latent_dim)
        self.dM_specific = Decoder_M_specific(in_dim=self.latent_dim)
        self.dM_shared = Decoder_M_shared()
        self.dM_shared_copy = Decoder_M_shared()


        # E
        self.eE_shared = Encoder_E_shared(E_noise=self.E_noise)
        self.eE_specific = Encoder_E_specific(latent_dim=self.latent_dim)
        self.dE_specific = Decoder_E_specific(in_dim=self.latent_dim)
        self.dE_shared = Decoder_E_shared()
        self.dE_shared_copy = Decoder_E_shared()


        # ME
        self.eME = Encoder_ME(in_dim=51, int_dim=20, latent_dim=self.latent_dim)
        self.dME = Decoder_ME(in_dim=self.latent_dim, int_dim=20, out_dim=51)

        return

    def get_hparams(self):
        hparam_dict = {}
        hparam_dict['alpha_T'] = self.alpha_T
        hparam_dict['alpha_M'] = self.alpha_M
        hparam_dict['alpha_E'] = self.alpha_E
        hparam_dict['alpha_ME'] = self.alpha_ME
        hparam_dict['lambda_ME_T'] = self.lambda_ME_T
        hparam_dict['lambda_ME_M'] = self.lambda_ME_M
        hparam_dict['lambda_ME_E'] = self.lambda_ME_E
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

    @staticmethod
    def get_output_dict(out_list, key_list):
        out_dict = {}
        for (val, key) in zip(out_list, key_list):
            out_dict[key] = val
        return out_dict


    def forward(self, inputs):
        # inputs
        XT = inputs[0]
        XM = inputs[1]
        Xsd = inputs[2]
        XE = inputs[3]

        ############################## masks
        ## masks with the size of the whole batch
        valid_T = self.get_1D_mask(XT)
        valid_M = self.get_1D_mask(XM)
        valid_sd = self.get_1D_mask(Xsd)
        valid_E = self.get_1D_mask(XE) # if ALL the values for one cell is nan, then that cell is not being used in loss calculation
        valid_ME = torch.where((valid_E) & (valid_M), True, False)

        assert (valid_sd == valid_M).all()

        # masks with the size of each modality
        ME_cells_in_Mdata = valid_ME[valid_M] #size of this mask is the same as M
        ME_cells_in_Edata = valid_ME[valid_E] #size of this mask is the same as E
        ME_cells_in_Tdata = valid_ME[valid_T] #size of this mask is the same as T

        T_cells_in_MEdata = valid_T[valid_ME]
        M_cells_in_MEdata = valid_M[valid_ME]
        E_cells_in_MEdata = valid_E[valid_ME]

        ## removing nans
        XT = XT[valid_T]
        XM = XM[valid_M]
        Xsd = Xsd[valid_sd]
        XE = XE[valid_E]

        ############################## ENCODERS
        ## T
        zt = self.eT(XT)

        ## M
        xm_aug, _, pool_ind1, pool_ind2, xmsd_inter = self.eM_shared(XM, Xsd)
        zmsd = self.eM_specific(xmsd_inter.detach()) #by detaching we make sure that eM_shared is not going
        # to be updated in the backward pass

        ## E
        xe_inter = self.eE_shared(XE)
        ze = self.eE_specific(xe_inter.detach()) #Also detaching the xe_inter

        ## ME
        XME_inter = torch.cat(tensors=(xmsd_inter[ME_cells_in_Mdata],
                                       xe_inter[ME_cells_in_Edata]), dim=1)
        zme = self.eME(XME_inter)

        ############################## DECODERS
        ## T
        XrT = self.dT(zt)

        ## M
        Xrmsd_inter = self.dM_specific(zmsd)
        XrM, Xrsd = self.dM_shared_copy(Xrmsd_inter, pool_ind1, pool_ind2)

        ## E
        XrE_inter = self.dE_specific(ze)
        XrE = self.dE_shared_copy(XrE_inter)

        ## ME
        XrME_inter = self.dME(zme)
        ## decoder M for me inputs
        XrM_from_zme, Xrsd_from_zme = self.dM_shared(XrME_inter[:, :11],
                                                     pool_ind1[ME_cells_in_Mdata],
                                                     pool_ind2[ME_cells_in_Mdata])
        ## E for me inputs
        XrE_from_zme = self.dE_shared(XrME_inter[:, 11:])

        ############################## Loss calculations
        loss_dict = {}
        loss_dict['recon_T'] = self.mean_sq_diff(XT, XrT)
        loss_dict['recon_E'] = self.mean_sq_diff(XE, XrE)
        loss_dict['recon_M'] = self.mean_sq_diff(xm_aug, XrM)
        loss_dict['recon_sd'] = self.mean_sq_diff(Xsd, Xrsd)
        loss_dict['recon_ME'] = self.mean_sq_diff(xm_aug[ME_cells_in_Mdata], XrM_from_zme) + \
                                self.mean_sq_diff(Xsd[ME_cells_in_Mdata], Xrsd_from_zme) + \
                                self.mean_sq_diff(XE[ME_cells_in_Edata], XrE_from_zme)

        loss_dict['cpl_ME_T'] = self.min_var_loss(zt.detach()[ME_cells_in_Tdata], zme[T_cells_in_MEdata])
        loss_dict['cpl_ME_M'] = self.min_var_loss(zmsd[ME_cells_in_Mdata], zme[M_cells_in_MEdata].detach())
        loss_dict['cpl_ME_E'] = self.min_var_loss(ze[ME_cells_in_Edata], zme[E_cells_in_MEdata].detach())

        ############################## get output dicts
        z_dict = self.get_output_dict([zt, zmsd, ze, zme], ["zt", "zm", "ze", "zme"])

        xr_dict = self.get_output_dict([XrT, XrM, Xrsd, XrE, XrM_from_zme, Xrsd_from_zme, XrE_from_zme],
                                       ["XrT", "XrM", "Xrsd", "XrE", "XrM_from_zme", "Xrsd_from_zme", "XrE_from_zme"])

        mask_dict = self.get_output_dict([valid_T, valid_M, valid_E, valid_ME],
                                         ["valid_T", "valid_M", "valid_E", "valid_ME"])

        return loss_dict, z_dict, xr_dict, mask_dict



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
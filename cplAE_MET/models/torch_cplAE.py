import torch
import torch.nn as nn
import torch.nn.functional as F

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
        noise_sd: per-feature gaussian noise for E data
        dropout_p: dropout probability for E features
    """

    def __init__(self,
                 E_dim=0,
                 E_int_dim=40,
                 sd_int_dim=10,
                 std_dev=0.1,
                 out_dim=3,
                 noise_sd=None,
                 dropout_p=0.1):

        super(Encoder_EM, self).__init__()

        if noise_sd is not None:
            self.noise_sd = tensor_(noise_sd)
        else:
            self.noise_sd = None

        self.drp = nn.Dropout(p=dropout_p)

        self.gaussian_noise_std_dev = std_dev
        self.conv1_ax = nn.Conv2d(1, 10, kernel_size=(4, 3), stride=(4, 1), padding='valid')
        self.conv1_de = nn.Conv2d(1, 10, kernel_size=(4, 3), stride=(4, 1), padding='valid')

        self.conv2_ax = nn.Conv2d(10, 10, kernel_size=(2, 2), stride=(2, 1), padding='valid')
        self.conv2_de = nn.Conv2d(10, 10, kernel_size=(2, 2), stride=(2, 1), padding='valid')

        self.flat = nn.Flatten()
        self.fcsd = nn.Linear(1, sd_int_dim)
        self.fce0 = nn.Linear(E_dim, E_int_dim)
        self.fce1 = nn.Linear(E_int_dim, E_int_dim)
        self.fce2 = nn.Linear(E_int_dim, E_int_dim)
        self.fce3 = nn.Linear(E_int_dim, 20)
        self.fcm1 = nn.Linear(300+sd_int_dim, 20)
        self.fcm2 = nn.Linear(20, 20)
        self.fc = nn.Linear(20, out_dim)
        self.bn = nn.BatchNorm1d(out_dim, affine=False, eps=1e-10,
                                 momentum=0.05, track_running_stats=True)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        return

    def addnoise(self, x):
        if (self.training) and (self.noise_sd is not None):
            # batch dim is inferred from shapes of x and self.noise_sd
            x = torch.normal(mean=x, std=self.noise_sd)
        return x

    def forward(self, xe, xm, soma_depth, mask1D_e, mask1D_m):

        #Passing xe through some layers
        xe = self.addnoise(xe)
        xe = self.drp(xe)
        xe = self.elu(self.fce0(xe))
        xe = self.relu(self.fce1(xe))
        xe = self.relu(self.fce2(xe))
        xe = self.relu(self.fce3(xe))

        #Passing xm through some layers
        if self.training:
            xm = xm + (torch.randn(xm.shape) * self.gaussian_noise_std_dev)
        ax, de = torch.tensor_split(xm, 2, dim=1)
        ax = self.elu(self.conv1_ax(ax))
        de = self.elu(self.conv1_de(de))
        ax = self.elu(self.conv2_ax(ax))
        de = self.elu(self.conv2_de(de))
        xm = torch.cat(tensors=(self.flat(ax), self.flat(de)), dim=1)

        #passing soma depth through some layers
        soma_depth = soma_depth.view(-1, 1)
        soma_depth = self.elu(self.fcsd(soma_depth))

        #concat soma depth with morpho
        xm = torch.cat(tensors=(xm, soma_depth), dim=1)
        xm = self.elu(self.fcm1(xm))
        xm = self.elu(self.fcm2(xm))

        mask1D_only_e = torch.logical_and(mask1D_e, ~mask1D_m) #True if only e is True AND m is False
        mask1D_only_m = torch.logical_and(mask1D_m, ~mask1D_e) #True if only m is True AND e is False
        mask1D_both_e_and_m = torch.logical_and(mask1D_e, mask1D_m) #True if only both e and m are True

        y = torch.zeros_like(xm)
        y = torch.where(mask1D_only_e.view(-1, 1), xe, y)
        y = torch.where(mask1D_only_m.view(-1, 1), xm, y)
        y = torch.where(mask1D_both_e_and_m.view(-1, 1), torch.mean(torch.stack((xm, xe)), dim=0), y)

        #run the final representation through more layers
        x = self.fc(y)
        x = self.bn(x)
        return x


class Decoder_EM(nn.Module):
    """
    Decoder for EM data.

    Args:
        in_dim: representation dimensionality
    """

    def __init__(self,
                 E_dim=0,
                 E_int_dim=40,
                 in_dim=3,
                 sd_int_dim=10):

        super(Decoder_EM, self).__init__()
        self.fc_dec = nn.Linear(in_dim, 20)
        self.fcm0_dec = nn.Linear(20, 20)
        self.fcm1_dec = nn.Linear(20, 300+sd_int_dim)
        self.fcsd_dec = nn.Linear(sd_int_dim, 1)
        self.fce0_dec = nn.Linear(20, E_int_dim)
        self.fce1_dec = nn.Linear(E_int_dim, E_int_dim)
        self.fce2_dec = nn.Linear(E_int_dim, E_dim)

        self.convT1_ax = nn.ConvTranspose2d(10, 10, kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.convT1_de = nn.ConvTranspose2d(10, 10, kernel_size=(2, 2), stride=(2, 2), padding=0)

        self.convT2_ax = nn.ConvTranspose2d(10, 1, kernel_size=(4, 3), stride=(4, 1), padding=0)
        self.convT2_de = nn.ConvTranspose2d(10, 1, kernel_size=(4, 3), stride=(4, 1), padding=0)

        self.elu = nn.ELU()
        return

    def forward(self, x):

        x = self.elu(self.fc_dec(x))

        #separating xm and xe
        xm = self.elu(self.fcm0_dec(x))
        xe = self.elu(self.fce0_dec(x))

        #passing xm through some layers
        xm = self.elu(self.fcm1_dec(xm))
        #separating soma_depth
        ax_de = xm[:, 0:300]
        soma_depth = xm[:, 300:]
        soma_depth = self.fcsd_dec(soma_depth)
        #separating ax and de and passing them through conv layers
        ax, de = torch.tensor_split(ax_de, 2, dim=1)
        ax = ax.view(-1, 10, 15, 1)
        de = de.view(-1, 10, 15, 1)
        ax = self.convT1_ax(ax)
        de = self.convT1_de(de)
        ax = self.convT2_ax(ax)
        de = self.convT2_de(de)
        xm = torch.cat(tensors=(ax, de), dim=1)

        #passing xe through some layers
        xe = self.elu(self.fce1_dec(xe))
        xe = self.elu(self.fce2_dec(xe))

        return xe, xm, soma_depth

class Model_T_EM(nn.Module):
    """Coupled autoencoder model for Transcriptomics(T) and EM(Electrophisiology and Morphology)

    Args:
        T_dim: Number of genes in T data
        E_dim: Number of features in E data
        T_int_dim: hidden layer dims for T model
        E_int_dim: hidden layer dims for E model
        T_dropout: dropout for T data
        E_dropout: dropout for E data
        E_noise_sd: per-feature gaussian noise
        latent_dim: dim for representations
        alpha_T: loss weight for T reconstruction
        alpha_E: loss weight for E reconstruction
        alpha_M: loss weight for M reconstruction
        lambda_T_EM: loss weight coupling loss between T and EM
        std_dev: gaussian noise std dev
        augment_decoders (bool): augment decoder with cross modal representation if True
    """

    def __init__(self,
                 T_dim=1000, T_int_dim=50, T_dropout=0.5,
                 E_dim=1000,  E_int_dim=40, E_dropout=0.5, E_noise_sd=None,
                 latent_dim=3, alpha_T=1.0, alpha_E=1.0, alpha_soma_depth=1.0,
                 alpha_M=1.0, lambda_T_EM=1.0, std_dev=1.0, augment_decoders=True):

        super(Model_T_EM, self).__init__()
        self.alpha_T = alpha_T
        self.alpha_E = alpha_E
        self.alpha_M = alpha_M
        self.alpha_soma_depth = alpha_soma_depth
        self.lambda_T_EM = lambda_T_EM
        self.augment_decoders = augment_decoders

        self.eT = Encoder_T(dropout_p=T_dropout, in_dim=T_dim, out_dim=latent_dim, int_dim=T_int_dim)
        self.eEM = Encoder_EM(E_dim=E_dim, E_int_dim=E_int_dim, std_dev=std_dev,  out_dim=latent_dim,
                              dropout_p=E_dropout, noise_sd=E_noise_sd)

        self.dT = Decoder_T(in_dim=latent_dim, out_dim=T_dim, int_dim=T_int_dim)
        self.dEM = Decoder_EM(E_dim=E_dim, in_dim=latent_dim, E_int_dim=E_int_dim)
        return

    @staticmethod
    def min_var_loss(zi, zj, mask_1D_zi, mask_1D_zj):
        #SVD calculated over all entries in the batch
        zj_masked = zj[mask_1D_zj]
        zj_masked_size = zj_masked.shape[0]
        zj_masked_centered = zj_masked - torch.mean(zj_masked, 0, True)
        min_eig = torch.min(torch.linalg.svdvals(zj_masked_centered))
        min_var_zj = torch.square(min_eig)/(zj_masked_size-1)

        zi_masked = zi[mask_1D_zi]
        zi_masked_size = zi_masked.shape[0]
        zi_masked_centered = zi_masked - torch.mean(zi_masked, 0, True)
        min_eig = torch.min(torch.linalg.svdvals(zi_masked_centered))
        min_var_zi = torch.square(min_eig)/(zi_masked_size-1)

        #Wij_paired is the weight of matched pairs
        pairs = torch.logical_and(mask_1D_zi, mask_1D_zj)
        paired_dist = (zi-zj)[pairs]
        zi_zj_mse = torch.mean(torch.sum(torch.square(paired_dist), 1))
        loss_ij = zi_zj_mse/torch.squeeze(torch.minimum(min_var_zi, min_var_zj))
        return loss_ij

    @staticmethod
    def mean_sq_diff(x, y):
        # return torch.mean(torch.square(x-y))
        return F.mse_loss(y, x, reduction='mean') if (x.numel() != 0) & (y.numel() != 0) else tensor(0.)

    @staticmethod
    def get_1D_mask(mask):
        mask = mask.reshape(mask.shape[0], -1)
        return torch.all(mask, dim=1)

    def get_pairs(self, mask1, mask2):
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

        #EM arm forward pass
        zEM = self.eEM(XE, XM, X_soma_depth, self.get_1D_mask(masks['E']), self.get_1D_mask(masks['M']))
        XrE, XrM, Xr_soma_depth = self.dEM(zEM)

        mask_1D_T = self.get_1D_mask(masks['T'])
        mask_1D_E = self.get_1D_mask(masks['E'])
        mask_1D_M = self.get_1D_mask(masks['M'])
        mask_1D_EM = torch.logical_or(mask_1D_E, mask_1D_M)

        #pairs_T_EM = torch.logical_and(torch.logical_or(mask_1D_E, mask_1D_M), mask_1D_T)

        #Loss calculations
        self.loss_dict = {}
        self.loss_dict['recon_T'] = self.alpha_T * self.mean_sq_diff(XT[masks['T']], XrT[masks['T']])
        self.loss_dict['recon_E'] = self.alpha_E * self.mean_sq_diff(XE[masks['E']], XrE[masks['E']])
        self.loss_dict['recon_M'] = self.alpha_M * self.mean_sq_diff(XM[masks['M']], XrM[masks['M']])
        self.loss_dict['recon_soma_depth'] = self.alpha_soma_depth * self.mean_sq_diff(X_soma_depth[masks['soma_depth']],
                                                                                       Xr_soma_depth[masks['soma_depth']])

        self.loss_dict['cpl_T_EM'] = self.lambda_T_EM * self.min_var_loss(zT, zEM, mask_1D_T, mask_1D_EM)

        if self.augment_decoders:
            XrT_aug = self.dT(zT.detach())
            XrE_aug, XrM_aug, Xr_soma_depth_aug = self.dEM(zEM.detach())
            self.loss_dict['recon_T_aug'] = self.alpha_T * self.mean_sq_diff(XT[masks['T']], XrT_aug[masks['T']])
            self.loss_dict['recon_E_aug'] = self.alpha_E * self.mean_sq_diff(XE[masks['E']], XrE_aug[masks['E']])
            self.loss_dict['recon_M_aug'] = self.alpha_M * self.mean_sq_diff(XM[masks['M']], XrM_aug[masks['M']])
            self.loss_dict['recon_M_soma_depth_aug'] = self.alpha_soma_depth * self.mean_sq_diff(
                X_soma_depth[masks['soma_depth']],
                Xr_soma_depth_aug[masks['soma_depth']])

        self.loss = sum(self.loss_dict.values())
        return zT, zEM, XrT, XrE, XrM, Xr_soma_depth


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
            self.loss_dict['recon_M_soma_depth_aug'] = self.alpha_soma_depth * self.mean_sq_diff(
                X_soma_depth[masks['soma_depth']],
                Xr_soma_depth_aug[masks['soma_depth']])

        self.loss = sum(self.loss_dict.values())
        return zT, zE, zM_z_soma_depth, XrT, XrE, XrM, Xr_soma_depth

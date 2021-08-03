import torch
import torch.nn as nn

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
                 in_dim=300,
                 int_dim=40,
                 out_dim=3,
                 noise_sd=0,
                 dropout_p=0.1):


        super(Encoder_E, self).__init__()
        self.noise_sd = tensor_(noise_sd)
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

    def addnoise(self,x):
        if self.training:
            # batch dim is inferred from shapes of x and self.noise_sd
            x = torch.normal(mean=x, std=self.noise_sd)
        return x

    def forward(self, x):
        x = self.addnoise(self,x)
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
                 out_dim=300,
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
        self.fc1 = nn.Linear(300, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, out_dim)
        self.bn = nn.BatchNorm1d(out_dim, affine=False, eps=1e-10,
                                 momentum=0.05, track_running_stats=True)
        self.elu = nn.ELU()

        return

    def forward(self, x):
        if self.training:
            x = x + (torch.randn(x.shape) * self.gaussian_noise_std_dev)

        ax, de = torch.tensor_split(x, 2, dim=1)

        ax = self.elu(self.conv1_ax(ax))
        de = self.elu(self.conv1_de(de))

        ax = self.elu(self.conv2_ax(ax))
        de = self.elu(self.conv2_de(de))
        x = torch.cat(tensors=(self.flat(ax), self.flat(de)), dim=1)

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
        self.fc3_dec = nn.Linear(20, 300)

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

        ax, de = torch.tensor_split(x, 2, dim=1)
        ax = ax.view(-1, 10, 15, 1)
        de = de.view(-1, 10, 15, 1)

        ax = self.convT1_ax(ax)
        de = self.convT1_de(de)

        ax = self.convT2_ax(ax)
        de = self.convT2_de(de)

        x = torch.cat(tensors=(ax, de), dim=1)
        return x


class Model_TE(nn.Module):
    """Coupled autoencoder model

    Args:
        T_dim: Number of genes in T data
        E_dim: Number of genes in E data
        T_int_dim: hidden layer dims for T model
        E_int_dim: hidden layer dims for E model
        T_dropout: dropout for T data
        E_dropout: dropout for E data
        latent_dim: dim for representations
        alpha_T: loss weight for T reconstruction
        alpha_E: loss weight for E reconstruction
        lambda_TE: loss weight coupling loss
        augment_decoders (bool): augment decoder with cross modal representation if True
        name: TE
    """

    def __init__(self,
                 T_dim=1000, T_int_dim=50, T_dropout=0.5,
                 E_dim=1000, E_int_dim=50, E_dropout=0.5,
                 latent_dim=3, alpha_T=1.0, alpha_E=1.0, lambda_TE=1.0,
                 augment_decoders=True):

        super(Model_TE, self).__init__()
        self.alpha_T = alpha_T
        self.alpha_E = alpha_E
        self.lambda_TE = lambda_TE
        self.augment_decoders = augment_decoders

        self.eT = Encoder_T(dropout_p=T_dropout, in_dim=T_dim, out_dim=latent_dim, int_dim=T_int_dim)
        self.eE = Encoder_E(dropout_p=E_dropout, in_dim=E_dim, out_dim=latent_dim, int_dim=E_int_dim)
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
                 E_dim=1000, E_int_dim=50, E_dropout=0.5,
                 latent_dim=3, alpha_T=1.0, alpha_E=1.0,
                 alpha_M=1.0, lambda_TE=1.0, lambda_MT=1.0,
                 lambda_ME=1.0, std_dev=1.0, augment_decoders=True):

        super(Model_MET, self).__init__()
        self.alpha_T = alpha_T
        self.alpha_E = alpha_E
        self.alpha_M = alpha_M
        self.lambda_TE = lambda_TE
        self.lambda_ME = lambda_ME
        self.lambda_MT = lambda_MT
        self.augment_decoders = augment_decoders

        self.eT = Encoder_T(dropout_p=T_dropout, in_dim=T_dim, out_dim=latent_dim, int_dim=T_int_dim)
        self.eE = Encoder_E(dropout_p=E_dropout, in_dim=E_dim, out_dim=latent_dim, int_dim=E_int_dim)
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

    def forward(self, inputs):
        #T arm forward pass
        XT = inputs[0]
        zT = self.eT(XT)
        XrT = self.dT(zT)

        #E arm forward pass
        XE = inputs[1]
        zE = self.eE(XE)
        XrE = self.dE(zE)

        # M arm forward pass
        XM = inputs[2]
        zM = self.eM(XM)
        XrM = self.dM(zM)

        #Loss calculations
        self.loss_dict = {}
        self.loss_dict['recon_T'] = self.alpha_T * self.mean_sq_diff(XT, XrT)
        self.loss_dict['recon_E'] = self.alpha_E * self.mean_sq_diff(XE, XrE)
        self.loss_dict['recon_M'] = self.alpha_M * self.mean_sq_diff(XM, XrM)
        self.loss_dict['cpl_TE'] = self.lambda_TE * self.min_var_loss(zT, zE)
        self.loss_dict['cpl_ME'] = self.lambda_ME * self.min_var_loss(zM, zE)
        self.loss_dict['cpl_MT'] = self.lambda_MT * self.min_var_loss(zM, zT)

        if self.augment_decoders:
            XrT_aug = self.dT(zE.detach())
            XrE_aug = self.dE(zT.detach())
            XrM_aug = self.dM(zM.detach()) #Do we have aug for M?
            self.loss_dict['recon_T_aug'] = self.alpha_T * self.mean_sq_diff(XT, XrT_aug)
            self.loss_dict['recon_E_aug'] = self.alpha_E * self.mean_sq_diff(XE, XrE_aug)
            self.loss_dict['recon_M_aug'] = self.alpha_M * self.mean_sq_diff(XM, XrM_aug)

        self.loss = sum(self.loss_dict.values())
        return  zT, zE, zM, XrT, XrE, XrM

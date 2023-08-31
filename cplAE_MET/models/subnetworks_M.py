from torch import nn
import torch

class Enc_xm_to_zm_int(nn.Module):
    """Common encoding network for (only M) and (M and E paired) cases.
    - `xm` expected in [N, C=1, D=240, H=1, W=4] format, with C = 1, D=240, H=1, W=4
     - Missing data is encoded as nans.
     - Output is an intermediate representation, `zm_int`
    """

    def __init__(self, 
                 out_dim=10, 
                 gnoise_std=None,
                 gnoise_std_frac=0.05):
        
        super(Enc_xm_to_zm_int, self).__init__()
        if gnoise_std is not None:
            self.gnoise_std = gnoise_std * gnoise_std_frac
        self.conv_0 = nn.Conv3d(1, 10, kernel_size=(5, 1, 1), padding=(2, 1, 0))
        self.pool_0 = nn.MaxPool3d((2, 1, 1), return_indices=True)
        self.conv_1 = nn.Conv3d(10, 10, kernel_size=(5, 1, 1), padding=(2, 1, 0))
        self.pool_1 = nn.MaxPool3d((2, 1, 1), return_indices=True)
        self.fc_0 = nn.Linear(9600, out_dim)
        self.bn = nn.BatchNorm1d(out_dim, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        return
    
    def aug_noise(self, x):
        # get the nanzero mask for M for adding noise
        mask = x != 0.
        if (self.training) and (self.gnoise_std is not None):
            x = torch.where(mask, torch.normal(mean=x, std=self.gnoise_std), x)
            x = torch.clamp(x, min=0)
            return x
        else:
            return x
        

    def forward(self, xm):
        # x = self.aug_noise(xm)
        x, self.pool_0_ind = self.pool_0(self.relu(self.conv_0(xm)))
        x, self.pool_1_ind = self.pool_1(self.relu(self.conv_1(x)))
        x = x.view(x.shape[0], -1)
        zm_int = self.bn(self.relu(self.fc_0(x)))
        
        return zm_int


class Enc_zm_int_to_zm(nn.Module):
    """Intermediate representation `zm_int` is encoded into `zm`
    """
    def __init__(self, in_dim=10, out_dim=3, variational=False):
        super(Enc_zm_int_to_zm, self).__init__()
        self.fc_mu = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_sigma = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim, eps=1e-05, momentum=0.05, affine=False, track_running_stats=True)
        self.variational = variational
        return

    def forward(self, zm_int):
        if self.variational:
            mu = self.fc_mu(zm_int)
            var = torch.sigmoid(self.fc_sigma(zm_int))
            return mu, var
        else:
            return self.bn(self.fc_mu(zm_int))


class Dec_zm_to_zm_int(nn.Module):
    """Decodes `zm` into `zm_int`
    """

    def __init__(self, in_dim=3, out_dim=10):
        super(Dec_zm_to_zm_int, self).__init__()
        self.fc_0 = nn.Linear(in_dim, out_dim)
        self.fc_1 = nn.Linear(out_dim, out_dim)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        return

    def reparametrize(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, zm):
        x = self.elu(self.fc_0(zm))
        zm_int = self.relu(self.fc_1(x))
        return zm_int


class Dec_zm_int_to_xm(nn.Module):
    """Decodes `zm_int` into the reconstruction `xrm` and `xrsd`
    """

    def __init__(self, in_dim=10):
        super(Dec_zm_int_to_xm, self).__init__()
        self.fc_0 = nn.Linear(in_dim, 9600)
        self.convT_0 = nn.ConvTranspose3d(10, 10, kernel_size=(5, 1, 1), padding=(2, 1, 0))
        self.convT_1 = nn.ConvTranspose3d(10, 1, kernel_size=(5, 1, 1), padding=(2, 1, 0))
        self.unpool_0 = nn.MaxUnpool3d((2, 1, 1))
        self.unpool_1 = nn.MaxUnpool3d((2, 1, 1))
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        return

    def forward(self, zm_int, enc_pool_0_ind, enc_pool_1_ind):
        x = zm_int[:, 0:10]
        x = self.elu(self.fc_0(x))
        x = x.view(-1, 10, 30, 8, 4)
        # x = x.view(-1, 10, 15, 4, 4)
        x = self.elu(self.unpool_0(x, enc_pool_1_ind))
        x = self.convT_0(x)
        x = self.elu(self.unpool_1(x, enc_pool_0_ind))
        xrm = self.relu(self.convT_1(x))
        return xrm


class AE_M(nn.Module):
    def __init__(self, config):
        super(AE_M, self).__init__()
        self.enc_xm_to_zm_int = Enc_xm_to_zm_int(gnoise_std=config['M']['gnoise_std'], gnoise_std_frac=config['M']['gnoise_std_frac'])
        self.enc_zm_int_to_zm = Enc_zm_int_to_zm(out_dim=config['latent_dim'], variational=config['variational'])
        self.dec_zm_to_zm_int = Dec_zm_to_zm_int(in_dim=config['latent_dim'])
        self.dec_zm_int_to_xm = Dec_zm_int_to_xm()
        self.variational = config['variational']
        return

    def forward(self, xm):
        zm_int_enc = self.enc_xm_to_zm_int(xm.nan_to_num())
        if self.variational:
            mu, sigma = self.enc_zm_int_to_zm(zm_int_enc)
            log_sigma = (sigma + 1e-6).log()
            zm = self.dec_zm_to_zm_int.reparametrize(mu, sigma)
        else:
            zm = self.enc_zm_int_to_zm(zm_int_enc)
            mu=[]
            log_sigma=[]

        zm_int_dec = self.dec_zm_to_zm_int(zm)
        xrm = self.dec_zm_int_to_xm(zm_int_dec,
                                          self.enc_xm_to_zm_int.pool_0_ind,
                                          self.enc_xm_to_zm_int.pool_1_ind)
        return zm_int_enc, zm, zm_int_dec, xrm, mu, log_sigma

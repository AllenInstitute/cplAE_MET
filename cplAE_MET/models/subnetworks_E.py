from torch import nn
import torch


class Enc_xe_to_ze_int(nn.Module):
    """Common encoding network for (only E) and (M and E paired) cases. 
    Output is an intermediate representation, `ze_int`

    Args:
            gnoise_std (torch.tensor): Vector with shape (1,in_dim)
            gnoise_std_frac (float): Defaults to 0.05
            dropout_p (float): Defaults to 0.2
            out_dim (int): Defaults to 8.
    """

    def __init__(self,
                 gnoise_std=None, 
                 gnoise_std_frac=0.05,
                 dropout_p=0.2, 
                 out_dim=10):
        super(Enc_xe_to_ze_int, self).__init__()
        if gnoise_std is not None:
            self.gnoise_std = gnoise_std * gnoise_std_frac
        self.drp = nn.Dropout(p=dropout_p)
        self.fc_0 = nn.Linear(82, 40)
        self.fc_1 = nn.Linear(40, 40)
        self.bn = nn.BatchNorm1d(40, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
        self.fc_2 = nn.Linear(40, 40)
        self.fc_3 = nn.Linear(40, out_dim)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        return

    def add_gnoise(self, x):
        if (self.training) and (self.gnoise_std is not None):
            # note: batch dim is inferred from shapes of x and self.noise_sd
            x = x + torch.randn_like(x)*self.gnoise_std
        return x

    def forward(self, xe):
        x = self.add_gnoise(xe)
        x = self.drp(x)
        x = self.elu(self.fc_0(x))
        x = self.bn(self.relu(self.fc_1(x)))
        x = self.relu(self.fc_2(x))
        ze_int = self.relu(self.fc_3(x))
        return ze_int


class Enc_ze_int_to_ze(nn.Module):
    """Encodes `ze_int` to `ze`
    """

    def __init__(self, 
                 in_dim=10, 
                 out_dim=3,
                 variational=False):
        super(Enc_ze_int_to_ze, self).__init__()
        self.variational = variational
        # self.fc_0 = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_mu = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_sigma = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim, eps=1e-05, momentum=0.05, affine=False, track_running_stats=True)
        return

    def forward(self, ze_int):
        if self.variational:
            mu = self.fc_mu(ze_int)
            var = torch.sigmoid(self.fc_sigma(ze_int))
            return mu, var
        else:
            return self.bn(self.fc_mu(ze_int))
        

class Dec_ze_to_ze_int(nn.Module):
    """Decodes `ze` into `ze_int`
    """

    def __init__(self, in_dim=3, out_dim=10):
        super(Dec_ze_to_ze_int, self).__init__()
        self.fc_0 = nn.Linear(in_dim, out_dim)
        self.fc_1 = nn.Linear(out_dim, out_dim)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        return

    def reparametrize(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, ze):
        x = self.elu(self.fc_0(ze))
        ze_int = self.relu(self.fc_1(x))
        return ze_int


class Dec_ze_int_to_xe(nn.Module):
    """Decodes `ze_int` into the reconstruction `xe`
    """

    def __init__(self, in_dim=10, out_dim=82):
        super(Dec_ze_int_to_xe, self).__init__()
        self.fc_0 = nn.Linear(in_dim, 40)
        self.fc_1 = nn.Linear(40, 40)
        self.fc_2 = nn.Linear(40, 40)
        self.fc_3 = nn.Linear(40, out_dim)
        self.relu = nn.ReLU()
        return

    def forward(self, ze_int):
        x = self.relu(self.fc_0(ze_int))
        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        xre = self.fc_3(x)
        return xre


class AE_E(nn.Module):
    def __init__(self, config):
        super(AE_E, self).__init__()
        self.enc_xe_to_ze_int = Enc_xe_to_ze_int(
            config["gauss_e_baseline"], config["gauss_var_frac"], config['dropout'], 10)
        self.enc_ze_int_to_ze = Enc_ze_int_to_ze(
            10, config['latent_dim'], variational = False)
        self.dec_ze_to_ze_int = Dec_ze_to_ze_int(config['latent_dim'], 10)
        self.dec_ze_int_to_xe = Dec_ze_int_to_xe()
        self.variational = False
        return

    def forward(self, xe):
        ze_int_enc = self.enc_xe_to_ze_int(xe.nan_to_num())
        if self.variational:
            mu, sigma = self.enc_ze_int_to_ze(ze_int_enc)
            log_sigma = (sigma + 1e-6).log()
            ze = self.dec_ze_to_ze_int.reparametrize(mu, sigma)
        else:
            ze = self.enc_ze_int_to_ze(ze_int_enc)
            mu=[]
            log_sigma=[]
        
        ze_int_dec = self.dec_ze_to_ze_int(ze)
        xre = self.dec_ze_int_to_xe(ze_int_dec)
        return ze_int_enc, ze, ze_int_dec, xre, mu, log_sigma


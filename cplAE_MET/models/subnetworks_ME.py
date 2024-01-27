import torch
from torch import nn

from cplAE_MET.models.subnetworks_E import Dec_ze_to_xe
from cplAE_MET.models.subnetworks_M import Dec_zm_to_xm

class Enc_xme_to_zme(nn.Module):
    def __init__(self, gnoise_std=None, gnoise_std_frac=0.05, dropout_p=0.2, out_dim=3, variational=False):
        super().__init__()
        self.variational = variational
        if gnoise_std is not None:
            self.gnoise_std = torch.nn.Parameter(torch.from_numpy(gnoise_std*gnoise_std_frac))
        self.fc_0_e = nn.Linear(82, 40)
        self.fc_1_e = nn.Linear(40, 40)
        self.fc_2_e = nn.Linear(40, 40)
        self.fc_3_e = nn.Linear(40, 10)
        self.bn_e = nn.BatchNorm1d(40, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
        
        self.conv_0 = nn.Conv1d(4*4, 10, kernel_size=6, stride = 2)
        self.conv_1 = nn.Conv1d(10, 10, kernel_size=6, stride = 2)
        self.fc_0_m = nn.Linear(270, 10)
        self.bn_1_m = nn.BatchNorm1d(10, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)

        self.fc_0_me = nn.Linear(20, 20)
        self.fc_1_me = nn.Linear(20, 20)
        self.fc_mu = nn.Linear(20, out_dim, bias=False)
        self.fc_sigma = nn.Linear(20, out_dim, bias=False)
        self.drp = nn.Dropout(p=dropout_p)
        self.bn = nn.BatchNorm1d(out_dim,  eps=1e-5, momentum=0.05, affine=False, track_running_stats=True)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()

    def add_gnoise(self, x):
        if (self.training) and (self.gnoise_std is not None):
            x = x + torch.randn_like(x)*self.gnoise_std
        return x

    def forward(self, xe, xm):
        x = self.add_gnoise(xe)
        x = self.drp(x)
        x = self.elu(self.fc_0_e(x))
        x = self.bn_e(self.relu(self.fc_1_e(x)))
        x = self.relu(self.fc_2_e(x))
        ze_int = self.relu(self.fc_3_e(x))

        x = torch.flatten(xm, 2).transpose(1, 2)
        x = self.relu(self.conv_0(x))
        x = self.relu(self.conv_1(x))
        x = x.view(x.shape[0], -1)
        zm_int = self.bn_1_m(self.relu(self.fc_0_m(x)))

        x = torch.cat((ze_int, zm_int), 1)
        x = self.elu(self.fc_0_me(x))
        x = self.elu(self.fc_1_me(x))
        if self.variational:
            mu = self.fc_mu(x)
            var = torch.sigmoid(self.fc_sigma(x))
            return mu, var
        else:
            return self.bn(self.fc_mu(x))

class AE_ME(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Enc_xme_to_zme(
            config["gauss_e_baseline"], config["gauss_var_frac"],
            config['dropout'], config['latent_dim'], variational = False)
        self.decoder_e = Dec_ze_to_xe(config["latent_dim"])
        self.decoder_m = Dec_zm_to_xm(config["latent_dim"])
        self.variational = False
        return

    def decoder(self, z):
        xre = self.decoder_e(z)
        xrm = self.decoder_m(z)
        return (xre, xrm)

    def forward(self, xe, xm):
        if self.variational:
            mu, sigma = self.encoder(xe, xm)
            log_sigma = (sigma + 1e-6).log()
        else:
            zme = self.encoder(xe, xm)
            mu=[]
            log_sigma=[]
        (xre, xrm) = self.decoder(zme)
        return (zme, (xre, xrm))


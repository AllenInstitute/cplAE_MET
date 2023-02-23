import torch
from torch import nn


class Enc_zme_int_to_zme(nn.Module):
    def __init__(self, in_dim=22, out_dim=3, variational=False):
        super(Enc_zme_int_to_zme, self).__init__()
        self.variational = variational
        self.fc_0 = nn.Linear(in_dim, 22)
        self.fc_1 = nn.Linear(22, 22)
        # self.fc_2 = nn.Linear(22, out_dim, bias=False)
        self.fc_mu = nn.Linear(22, out_dim, bias=False)
        self.fc_sigma = nn.Linear(22, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim,  eps=1e-5, momentum=0.05, affine=False, track_running_stats=True)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        return

    def forward(self, xm_int, xe_int):
        x = torch.cat((xm_int, xe_int), 1)
        x = self.elu(self.fc_0(x))
        x = self.elu(self.fc_1(x))
        if self.variational:
            mu = self.fc_mu(x)
            var = torch.sigmoid(self.fc_sigma(x))
            return mu, var
        else:
            return self.bn(self.fc_mu(x))


class Dec_zme_to_zme_int(nn.Module):
    def __init__(self, in_dim=3, out_dim=22):
        super(Dec_zme_to_zme_int, self).__init__()
        self.fc_0 = nn.Linear(in_dim, 22)
        self.fc_1 = nn.Linear(22, 22)
        self.fc_2 = nn.Linear(22, out_dim)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        return
    
    def reparametrize(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, zme):
        x = self.elu(self.fc_0(zme))
        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        xm_int, xe_int = torch.split(x,[11,11], dim = 1)
        return xm_int, xe_int


class AE_ME_int(nn.Module):
    def __init__(self, config):
        super(AE_ME_int, self).__init__()
        self.enc_zme_int_to_zme = Enc_zme_int_to_zme(out_dim=config['latent_dim'], variational=config['variational'])
        self.dec_zme_to_zme_int = Dec_zme_to_zme_int(in_dim=config['latent_dim'])
        self.variational = config['variational']
        return

    def forward(self, xm_int, xe_int):
        if self.variational:
            mu, sigma = self.enc_zme_int_to_zme(xm_int, xe_int)
            log_sigma = (sigma + 1e-6).log()
            zme = self.dec_zme_to_zme_int.reparametrize(mu, sigma)
        else:
            zme = self.enc_zme_int_to_zme(xm_int, xe_int)
            mu=[]
            log_sigma=[]

        xm_int, xe_int = self.dec_zme_to_zme_int(zme)
        return zme, xm_int, xe_int, mu, log_sigma


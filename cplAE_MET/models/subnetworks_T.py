import torch
from torch import nn

class Enc_xt_to_zt(nn.Module):
    def __init__(self, 
                 dropout_p=0.2, 
                 in_dim=1252, 
                 out_dim=3, 
                 variational=False):
        super(Enc_xt_to_zt, self).__init__()
        self.variational = variational
        self.drp = nn.Dropout(p=dropout_p)
        self.fc_0 = nn.Linear(in_dim, 20)
        self.fc_1 = nn.Linear(20, 20)
        self.fc_2 = nn.Linear(20, 20)
        self.fc_3 = nn.Linear(20, 20)
        # self.fc_4 = nn.Linear(20, out_dim, bias=False)
        self.fc_mu = nn.Linear(20, out_dim, bias=False)
        self.fc_sigma = nn.Linear(20, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim, eps=1e-05, momentum=0.05, affine=False, track_running_stats=True)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        return

    def forward(self, xt):
        x = self.drp(xt)
        x = self.elu(self.fc_0(x))
        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        x = self.relu(self.fc_3(x))
        if self.variational:
            mu = self.fc_mu(x)
            var = torch.sigmoid(self.fc_sigma(x))
            return mu, var
        else:
            return self.bn(self.fc_mu(x))

class Dec_zt_to_xt(nn.Module):
    """Encodes `ze_int` to `ze`
    """

    def __init__(self, in_dim=2, out_dim=1252):
        super(Dec_zt_to_xt, self).__init__()
        self.fc_0 = nn.Linear(in_dim, 20)
        self.fc_1 = nn.Linear(20, 20)
        self.fc_2 = nn.Linear(20, 20)
        self.fc_3 = nn.Linear(20, 20)
        self.fc_4 = nn.Linear(20, out_dim)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        return

    def reparametrize(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, zt):
        x = self.elu(self.fc_0(zt))
        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        x = self.relu(self.fc_3(x))
        xrt = self.relu(self.fc_4(x))
        return xrt

class AE_T(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Enc_xt_to_zt(
            config['dropout'], 1252, config['latent_dim'], variational = False)
        self.decoder = Dec_zt_to_xt(config['latent_dim'], 1252)
        self.variational = False

    def forward(self, xt):
        if self.variational:
            mu, sigma = self.encoder(xt)
            log_sigma = (sigma + 1e-6).log()
            zt = self.decoder.reparametrize(mu, sigma)
        else:
            zt =  self.encoder(xt)
            mu = []
            log_sigma = []
        xrt = self.decoder(zt)
        return (zt, xrt)

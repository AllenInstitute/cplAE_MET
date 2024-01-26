from torch import nn
import torch

class Enc_xe_to_ze(nn.Module):
    def __init__(self,
                 gnoise_std=None, 
                 gnoise_std_frac=0.05,
                 dropout_p=0.2, 
                 out_dim=3,
                 variational=False):
        super().__init__()
        self.variational = variational
        if gnoise_std is not None:
            self.gnoise_std = gnoise_std * gnoise_std_frac
        self.drp = nn.Dropout(p=dropout_p)
        self.fc_0 = nn.Linear(82, 40)
        self.fc_1 = nn.Linear(40, 40)
        self.bn_1 = nn.BatchNorm1d(40, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
        self.fc_2 = nn.Linear(40, 40)
        self.fc_3 = nn.Linear(40, 10)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()

        self.fc_mu = nn.Linear(10, out_dim, bias=False)
        self.fc_sigma = nn.Linear(10, out_dim, bias=False)
        self.bn_2 = nn.BatchNorm1d(out_dim, eps=1e-05, momentum=0.05, affine=False, track_running_stats=True)

    def add_gnoise(self, x):
        if (self.training) and (self.gnoise_std is not None):
            x = x + torch.randn_like(x)*self.gnoise_std
        return x

    def forward(self, xe):
        x = self.add_gnoise(xe)
        x = self.drp(x)
        x = self.elu(self.fc_0(x))
        x = self.bn_1(self.relu(self.fc_1(x)))
        x = self.relu(self.fc_2(x))
        ze_int = self.relu(self.fc_3(x))

        if self.variational:
            mu = self.fc_mu(ze_int)
            var = torch.sigmoid(self.fc_sigma(ze_int))
            return mu, var
        else:
            return self.bn_2(self.fc_mu(ze_int))   

class Dec_ze_to_xe(nn.Module):
    def __init__(self, in_dim=3, out_dim=82):
        super().__init__()
        self.fc_0_1 = nn.Linear(in_dim, 10)
        self.fc_1_1 = nn.Linear(10, 10)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()

        self.fc_0_2 = nn.Linear(10, 40)
        self.fc_1_2 = nn.Linear(40, 40)
        self.fc_2_2 = nn.Linear(40, 40)
        self.fc_3_2 = nn.Linear(40, out_dim)

    def forward(self, ze):
        x = self.elu(self.fc_0_1(ze))
        ze_int = self.relu(self.fc_1_1(x))
        
        x = self.relu(self.fc_0_2(ze_int))
        x = self.relu(self.fc_1_2(x))
        x = self.relu(self.fc_2_2(x))
        xre = self.fc_3_2(x)
        return xre

class AE_E(nn.Module):
    def __init__(self, config):
        super(AE_E, self).__init__()
        self.encoder = Enc_xe_to_ze(
            config["gauss_e_baseline"], config["gauss_var_frac"], 
            config['dropout'], config['latent_dim'], variational = False)
        self.decoder = Dec_ze_to_xe(config['latent_dim'])
        self.variational = False

    def forward(self, xe):
        if self.variational:
            mu, sigma = self.encoder(xe)
            log_sigma = (sigma + 1e-6).log()
            ze = self.decoder.reparametrize(mu, sigma)
        else:
            ze = self.encoder(xe)
            mu=[]
            log_sigma=[]
        xre = self.decoder(ze)
        return (ze, xre)


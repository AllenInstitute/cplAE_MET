from torch import nn
import torch

class Enc_xm_to_zm(nn.Module):
    def __init__(self,
                 out_dim=3, 
                 gnoise_std=None,
                 gnoise_std_frac=0.05,
                 variational = False):
        super().__init__()
        self.variational = variational
        if gnoise_std is not None:
            self.gnoise_std = torch.nn.Parameter(torch.from_numpy(gnoise_std*gnoise_std_frac))
        
        self.conv_0 = nn.Conv1d(4*4, 10, kernel_size=6, stride = 2)
        self.conv_1 = nn.Conv1d(10, 10, kernel_size=6, stride = 2)
        self.fc_0 = nn.Linear(270, 10)
        self.bn_1 = nn.BatchNorm1d(10, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        
        self.fc_mu = nn.Linear(10, out_dim, bias=False)
        self.fc_sigma = nn.Linear(10, out_dim, bias=False)
        self.bn_2 = nn.BatchNorm1d(out_dim, eps=1e-05, momentum=0.05, affine=False, track_running_stats=True)

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
        x = torch.flatten(xm, 2).transpose(1, 2)
        x = self.relu(self.conv_0(x))
        x = self.relu(self.conv_1(x))
        x = x.view(x.shape[0], -1)
        zm_int = self.bn_1(self.relu(self.fc_0(x)))
        
        if self.variational:
            mu = self.fc_mu(zm_int)
            var = torch.sigmoid(self.fc_sigma(zm_int))
            return mu, var
        else:
            return self.bn_2(self.fc_mu(zm_int))

class Dec_zm_to_xm(nn.Module):
    def __init__(self, in_dim=3):
        super().__init__()
        self.fc_0_1 = nn.Linear(in_dim, 10)
        self.fc_1_1 = nn.Linear(10, 10)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()

        self.fc_0_2 = nn.Linear(10, 270)
        self.convT_0 = nn.ConvTranspose1d(10, 10, kernel_size=6, stride=2)
        self.convT_1 = nn.ConvTranspose1d(10, 4*4, kernel_size=6, stride=2)

    def forward(self, zm):
        x = self.elu(self.fc_0_1(zm))
        zm_int = self.relu(self.fc_1_1(x))

        x = self.elu(self.fc_0_2(zm_int))
        x = x.view(-1, 10, 27)
        x = self.elu(self.convT_0(x))
        x = self.relu(self.convT_1(x))
        xrm = x.transpose(1, 2).reshape([-1, 120, 4, 4])
        return xrm

class AE_M(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.variational = False
        self.encoder = Enc_xm_to_zm(config['latent_dim'], config["gauss_m_baseline"], 
                                    config["gauss_var_frac"], self.variational)
        self.decoder = Dec_zm_to_xm(config['latent_dim'])

    def forward(self, xm):
        if self.variational:
            mu, sigma = self.encoder(xm)
            log_sigma = (sigma + 1e-6).log()
            zm = self.decoder.reparametrize(mu, sigma)
        else:
            zm = self.encoder(xm)
            mu=[]
            log_sigma=[]
        xrm = self.decoder(zm)
        return (zm, xrm)

from torch import nn
import torch

class Enc_xe_to_ze(nn.Module):
    def __init__(self,
                 hidden_dims,
                 gnoise_std=None, 
                 gnoise_std_frac=0.05,
                 dropout_p=0.2, 
                 out_dim=3,
                 variational=False):
        super().__init__()
        self.variational = variational
        if gnoise_std is not None:
            self.gnoise_std = torch.nn.Parameter(torch.from_numpy(gnoise_std*gnoise_std_frac))
        self.drp = nn.Dropout(p=dropout_p)

        self.layer_names = []
        all_dims = [82] + list(hidden_dims) + [out_dim]
        for (i, (i_dim, o_dim)) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            bias = (False if i == len(hidden_dims) else True)
            name = ("fc_mu" if i == len(hidden_dims) else f"fc_{i}")
            setattr(self, name, torch.nn.Linear(i_dim, o_dim, bias = bias))
            self.layer_names.append(name)
        self.fc_sigma = nn.Linear(all_dims[-2], out_dim, bias=False)

        self.bn = nn.BatchNorm1d(out_dim, eps=1e-05, momentum=0.05, affine=False, track_running_stats=True)
        self.relu = nn.ReLU()

    def add_gnoise(self, x):
        if (self.training) and (self.gnoise_std is not None):
            x = x + torch.randn_like(x)*self.gnoise_std
        return x

    def forward(self, xe):
        x = self.add_gnoise(xe)
        x = self.drp(x)
        for layer_name in self.layer_names[:-1]:
            layer = getattr(self, layer_name)
            x = self.relu(layer(x))
        final_layer = getattr(self, self.layer_names[-1])
        output = self.bn(final_layer(x))
        return output

class Dec_ze_to_xe(nn.Module):
    def __init__(self, hidden_dims, in_dim=3, out_dim=82):
        super().__init__()
        self.layer_names = []
        all_dims = [in_dim] + list(hidden_dims) + [out_dim]
        for (i, (i_dim, o_dim)) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            setattr(self, f"fc_{i}", torch.nn.Linear(i_dim, o_dim))
            self.layer_names.append(f"fc_{i}")
        self.relu = nn.ReLU()
        self.drp = nn.Dropout(p = 0.1)

    def forward(self, ze):
        x = ze
        for layer_name in self.layer_names[:-1]:
            layer = getattr(self, layer_name)
            x = self.relu(layer(x))
        final_layer = getattr(self, self.layer_names[-1])
        x = final_layer(self.drp(x))
        return x

class AE_E(nn.Module):
    def __init__(self, config):
        super(AE_E, self).__init__()
        self.encoder = Enc_xe_to_ze(config["E_hidden"],
            config["gauss_e_baseline"], config["gauss_var_frac"], 
            config['E_dropout'], config['latent_dim'], variational = False)
        self.decoder = Dec_ze_to_xe(reversed(config["E_hidden"]), config['latent_dim'])
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


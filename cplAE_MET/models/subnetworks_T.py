import torch
from torch import nn

class Enc_xt_to_zt(nn.Module):
    def __init__(self, hidden_dims, dropout_p=0.2, in_dim=1252, out_dim=3, variational=False):
        super().__init__()
        self.variational = variational
        self.drp = nn.Dropout(p=dropout_p)

        self.layer_names = []
        all_dims = [in_dim] + list(hidden_dims) + [out_dim]
        for (i, (i_dim, o_dim)) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            bias = (False if i == len(hidden_dims) else True)
            name = ("fc_mu" if i == len(hidden_dims) else f"fc_{i}")
            setattr(self, name, torch.nn.Linear(i_dim, o_dim, bias = bias))
            self.layer_names.append(name)
        self.fc_sigma = nn.Linear(all_dims[-2], out_dim, bias=False)

        self.bn = nn.BatchNorm1d(out_dim, eps=1e-05, momentum=0.05, affine=False, track_running_stats=True)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()

    def forward(self, xt):
        x = self.drp(xt)
        for (i, layer_name) in enumerate(self.layer_names[:-1]):
            activation = (self.elu if i == 0 else self.relu)
            layer = getattr(self, layer_name)
            x = activation(layer(x))
        final_layer = getattr(self, self.layer_names[-1])
        output = self.bn(final_layer(x))
        return output

class Dec_zt_to_xt(nn.Module):
    """Encodes `ze_int` to `ze`
    """

    def __init__(self, hidden_dims, in_dim=2, out_dim=1252):
        super().__init__()
        self.layer_names = []
        all_dims = [in_dim] + list(hidden_dims) + [out_dim]
        for (i, (i_dim, o_dim)) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            setattr(self, f"fc_{i}", torch.nn.Linear(i_dim, o_dim))
            self.layer_names.append(f"fc_{i}")
        self.relu = nn.ReLU()
        self.elu = nn.ELU()

    def reparametrize(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, zt):
        x = zt
        for (i, layer_name) in enumerate(self.layer_names):
            activation = (self.elu if i == 0 else self.relu)
            layer = getattr(self, layer_name)
            x = activation(layer(x))
        return x

class AE_T(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Enc_xt_to_zt(config["T_hidden"],
            config['dropout'], 1252, config['latent_dim'], variational = False)
        self.decoder = Dec_zt_to_xt(reversed(config["T_hidden"]), config['latent_dim'], 1252)
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

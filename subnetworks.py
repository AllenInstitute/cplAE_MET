import torch
from torch import nn

activations = {
    "linear": torch.nn.Identity(),
    "relu": torch.nn.functional.relu,
    "sigmoid": torch.nn.functional.sigmoid
}

def get_model(config):
    model = {}
    for modal in config["modalities"]:
        arm = {}
        if modal == "T":
            activation = activations[config["T_output"]]
            arm["enc"] = Enc_xt_to_zt(config["T_hidden"],
                config['T_dropout'], 1252, config['latent_dim'], variational = False)
            arm["dec"] = Dec_zt_to_xt(reversed(config["T_hidden"]), activation, config['latent_dim'], 1252)
        elif modal == "E":
            activation = activations[config["E_output"]]
            arm["enc"] = Enc_xe_to_ze(config["E_hidden"],
                config["gauss_e_baseline"], config["gauss_var_frac"], 
                config['E_dropout'], config['latent_dim'], variational = False)
            arm["dec"] = Dec_ze_to_xe(reversed(config["E_hidden"]), activation, config['latent_dim'])
        elif modal == "M":
            activation = activations[config["M_output"]]
            arm["enc"] = Enc_xm_to_zm(config["M_conv"], config["M_hidden"], config['latent_dim'], 
                config["gauss_m_baseline"], config["gauss_var_frac"], config["M_dropout"],
                variational = False)
            arm["dec"] = Dec_zm_to_xm(config["M_conv"][::-1], config["M_hidden"][::-1], activation, config['latent_dim'])
        model[modal] = arm
    return model

def get_conv_out_size(conv_dims, initial_length):
    output_length = initial_length
    for (kernel, stride, _) in conv_dims:
        output_length = 1 + (output_length - kernel) / stride
    return int(output_length)

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

    def forward(self, xt):
        x = self.drp(xt)
        for layer_name in self.layer_names[:-1]:
            layer = getattr(self, layer_name)
            x = self.relu(layer(x))
        final_layer = getattr(self, self.layer_names[-1])
        output = self.bn(final_layer(x))
        return output

class Dec_zt_to_xt(nn.Module):
    """Encodes `ze_int` to `ze`
    """

    def __init__(self, hidden_dims, out_activation, in_dim=2, out_dim=1252):
        super().__init__()
        self.layer_names = []
        all_dims = [in_dim] + list(hidden_dims) + [out_dim]
        for (i, (i_dim, o_dim)) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            setattr(self, f"fc_{i}", torch.nn.Linear(i_dim, o_dim))
            self.layer_names.append(f"fc_{i}")
        relu = nn.ReLU()
        self.actvs = [relu]*(len(self.layer_names) - 1) + [out_activation]

    def reparametrize(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, zt):
        x = zt
        for (layer_name, actv) in zip(self.layer_names, self.actvs):
            layer = getattr(self, layer_name)
            x = actv(layer(x))
        return x

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
    def __init__(self, hidden_dims, out_activation, in_dim=3, out_dim=82):
        super().__init__()
        self.layer_names = []
        all_dims = [in_dim] + list(hidden_dims) + [out_dim]
        for (i, (i_dim, o_dim)) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            setattr(self, f"fc_{i}", torch.nn.Linear(i_dim, o_dim))
            self.layer_names.append(f"fc_{i}")
        self.drp = nn.Dropout(p = 0.1)
        relu = nn.ReLU()
        self.actvs = [relu]*(len(self.layer_names) - 1) + [out_activation]

    def forward(self, ze):
        x = ze
        for (layer_name, actv) in zip(self.layer_names[:-1], self.actvs):
            layer = getattr(self, layer_name)
            x = actv(layer(x))
        final_layer = getattr(self, self.layer_names[-1])
        x = final_layer(self.drp(x))
        x = self.actvs[-1](x)
        return x

class Enc_xm_to_zm(nn.Module):
    def __init__(self,
                 conv_params,
                 hidden_dims,
                 out_dim=3, 
                 gnoise_std=None,
                 gnoise_std_frac=0.05,
                 dropout = 0.0,
                 variational = False):
        super().__init__()
        self.variational = variational
        if gnoise_std is not None:
            self.gnoise_std = torch.nn.Parameter(torch.from_numpy(gnoise_std*gnoise_std_frac))
        self.conv_names = []
        self.dense_names = []
        conv_dims = [4*4] + [tupl[-1] for tupl in conv_params]
        dense_dims = [conv_dims[-1]*get_conv_out_size(conv_params, 120)] + list(hidden_dims) + [out_dim]
        for (i, (kernel, stride, _)) in enumerate(conv_params):
            (input_dim, output_dim) = conv_dims[i:i + 2]
            conv_layer = torch.nn.Conv1d(input_dim, output_dim, kernel_size = kernel, stride = stride)
            setattr(self, f"conv_{i}", conv_layer)
            self.conv_names.append(f"conv_{i}")
        for (i, (input_dim, output_dim)) in enumerate(zip(dense_dims[:-1], dense_dims[1:])):
            bias = (False if i == len(hidden_dims) else True)
            name = ("fc_mu" if i == len(hidden_dims) else f"fc_{i}")
            setattr(self, name, torch.nn.Linear(input_dim, output_dim, bias = bias))
            self.dense_names.append(name)
        self.fc_sigma = nn.Linear(dense_dims[-2], out_dim, bias=False)

        self.bn = nn.BatchNorm1d(out_dim, eps=1e-05, momentum=0.05, affine=False, track_running_stats=True)
        self.relu = nn.ReLU()
        self.drop = torch.nn.Dropout(dropout)

    def add_noise(self, x):
        if (self.training) and (self.gnoise_std is not None):
            x = x + torch.randn_like(x)*self.gnoise_std
            x = torch.clamp(x, min=0)
        return x

    def forward(self, xm):
        x = torch.flatten(xm, 2).transpose(1, 2)
        x = self.drop(x)
        for conv_name in self.conv_names:
            conv_layer = getattr(self, conv_name)
            x = self.relu(conv_layer(x))
        x = torch.flatten(x, 1)
        for dense_name in self.dense_names[:-1]:
            dense_layer = getattr(self, dense_name)
            x = self.relu(dense_layer(x))
        final_layer = getattr(self, self.dense_names[-1])
        output = self.bn(final_layer(x))
        return output

class Dec_zm_to_xm(nn.Module):
    def __init__(self, conv_params, hidden_dims, out_activation, in_dim=3):
        super().__init__()
        self.conv_names = []
        self.dense_names = []
        conv_dims = [tupl[-1] for tupl in conv_params] + [4*4]
        self.initial_channels = conv_dims[0]
        dense_dims = [in_dim] + list(hidden_dims) + [conv_dims[0]*get_conv_out_size(conv_params[::-1], 120)]
        for (i, (kernel, stride, _)) in enumerate(conv_params):
            (input_dim, output_dim) = conv_dims[i:i + 2]
            conv_layer = torch.nn.ConvTranspose1d(input_dim, output_dim, kernel_size = kernel, stride = stride)
            setattr(self, f"conv_{i}", conv_layer)
            self.conv_names.append(f"conv_{i}")
        for (i, (input_dim, output_dim)) in enumerate(zip(dense_dims[:-1], dense_dims[1:])):
            setattr(self, f"fc_{i}", torch.nn.Linear(input_dim, output_dim))
            self.dense_names.append(f"fc_{i}")

        relu = nn.ReLU()
        (num_dense, num_conv) = (len(self.dense_names), len(self.conv_names))
        actvs = [relu]*(num_dense + num_conv - 1) + [out_activation]
        (self.dense_actvs, self.conv_actvs) = (actvs[:num_dense], actvs[num_dense:])

    def forward(self, zm):
        x = zm
        for (dense_name, actv) in zip(self.dense_names, self.dense_actvs):
            dense_layer = getattr(self, dense_name)
            x = actv(dense_layer(x))
        x = x.view(x.shape[0], self.initial_channels, -1)
        for (conv_name, actv) in zip(self.conv_names, self.conv_actvs):
            conv_T_layer = getattr(self, conv_name)
            x = actv(conv_T_layer(x))
        xrm = x.transpose(1, 2).reshape([-1, 120, 4, 4])
        return xrm

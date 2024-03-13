import torch
from torch import nn
import numpy as np

from data import get_transformation_function

activations = {
    "linear": nn.Identity,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid
}

def get_conv_out_size(conv_params, initial_length):
    output_length = initial_length
    for (kernel, stride, _) in conv_params:
        output_length = 1 + (output_length - kernel) / stride
    return int(output_length)

def get_gauss_baselines(dataset, form):
    data = dataset.MET.query(dataset.allowed_specimen_ids)[form]
    std = np.nanstd(data, 0, keepdims = True)
    return std

def add_conv_segment(model, conv_params, input_length, input_channels, prefix, transpose):
    layer_class = torch.nn.ConvTranspose1d if transpose else torch.nn.Conv1d
    layer_names = []
    if transpose:
        arbor_dims = [tupl[-1] for tupl in conv_params] + [input_channels]
    else:
        arbor_dims = [input_channels] + [tupl[-1] for tupl in conv_params]
    out_channels = arbor_dims[0] if transpose else arbor_dims[-1]
    for (i, (kernel, stride, _)) in enumerate(conv_params):
        (input_dim, output_dim) = arbor_dims[i:i + 2]
        conv_layer = layer_class(input_dim, output_dim, kernel_size = kernel, stride = stride)
        setattr(model, f"{prefix}_{i}", conv_layer)
        layer_names.append(f"{prefix}_{i}")
    final_shape = (get_conv_out_size(conv_params, input_length), out_channels)
    return (layer_names, final_shape)

def add_dense_segment2(model, hidden_dims, input_size, final_bias, prefix):
    layer_names = []
    layer_sizes = [input_size] + hidden_dims
    for (i, (input_dim, output_dim)) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        bias = (final_bias if i == len(hidden_dims) else True)
        setattr(model, f"{prefix}_{i}", torch.nn.Linear(input_dim, output_dim, bias = bias))
        layer_names.append(f"{prefix}_{i}")
    return layer_names

def get_conv(conv_params, input_channels, actvs, transpose):
    layer_class = torch.nn.ConvTranspose1d if transpose else torch.nn.Conv1d
    if transpose:
        channel_dims = [tupl[-1] for tupl in conv_params] + [input_channels]
    else:
        channel_dims = [input_channels] + [tupl[-1] for tupl in conv_params]
    if type(actvs) != list:
        actvs = [actvs]*len(conv_params)
    layers = []
    for (i, ((kernel, stride, _), actv)) in enumerate(zip(conv_params, actvs)):
        (input_dim, output_dim) = channel_dims[i:i + 2]
        conv_layer = layer_class(input_dim, output_dim, kernel_size = kernel, stride = stride)
        layers.append(conv_layer)
        if actv is not None:
            layers.append(actv())
    return layers

def get_dense(input_size, output_size, hidden_dims, actvs = None, final_bias = True):
    layer_sizes = [input_size] + hidden_dims + [output_size]
    if type(actvs) != list:
        actvs = [actvs]*len(layer_sizes[1:])
    layers = []
    for (i, (input_dim, output_dim, actv)) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:], actvs)):
        bias = (final_bias if i == len(hidden_dims) else True)
        layers.append(torch.nn.Linear(input_dim, output_dim, bias = bias))
        if actv is not None:
            layers.append(actv())
    return layers

class Enc_logcpm(nn.Module):
    def __init__(self, architecture, latent_dim, dataset, variational):
        super().__init__()
        input_dim = architecture["data_size"][0]
        (init_dims, mean_dims, transf_dims) = (architecture["init"], architecture["mean"], architecture["cov"])
        init_out = init_dims[-1] if init_dims else input_dim
        initial_layers = get_dense(input_dim, init_out, init_dims[:-1], nn.ReLU) if init_dims else []
        mean_actvs = [nn.ReLU]*len(mean_dims) + [None]
        transf_actvs = [nn.ReLU]*len(transf_dims) + [None]

        self.initial_segment = nn.Sequential(*initial_layers)
        self.mean_layer = nn.Sequential(*get_dense(init_out, latent_dim, mean_dims, mean_actvs, False))
        if variational:
            self.transf_layer = nn.Sequential(*get_dense(init_out, latent_dim**2, transf_dims, transf_actvs))
        
        self.softplus = nn.Softplus()
        self.drp = nn.Dropout(architecture["dropout"])
        self.bn = nn.BatchNorm1d(latent_dim, momentum = 0.05, affine = False)
        self.latent_dim = latent_dim
        self.variational = variational
    
    def forward(self, x_forms):
        x = x_forms["logcpm"]
        x = self.drp(x)
        x = self.initial_segment(x)
        mean = self.bn(self.mean_layer(x))
        if self.variational:
            transf_raw = self.transf_layer(x).reshape(-1, self.latent_dim, self.latent_dim)
            diagonals = self.softplus(torch.diagonal(transf_raw, 0, -2, -1)) + 1e-4
            transf = torch.diag_embed(diagonals) + torch.tril(transf_raw, -1)
        else:
            transf = mean[:, None] * torch.zeros_like(mean)[..., None]
        return (mean, transf)
    
class Dec_logcpm(nn.Module):
    def __init__(self, architecture, latent_dim, dataset):
        super().__init__()
        output_dim = architecture["data_size"][0]
        hidden_dims = (architecture["init"] + architecture["mean"])[::-1]
        actvs = [nn.ReLU]*len(hidden_dims) + [activations[architecture["out_activation"]]]
        self.network = nn.Sequential(*get_dense(latent_dim, output_dim, hidden_dims, actvs))

    def forward(self, x):
        x = self.network(x)
        x_forms = {"logcpm": x}
        return x_forms

class Enc_pca_ipfx(nn.Module):
    def __init__(self, architecture, latent_dim, dataset, variational):
        super().__init__()
        gauss_frac = architecture["std_frac"]
        gauss_std = get_gauss_baselines(dataset, "pca-ipfx").astype("float32")
        self.gauss_std = torch.nn.Parameter(torch.from_numpy(gauss_std*gauss_frac), False)

        input_dim = architecture["data_size"][0]
        (init_dims, mean_dims, transf_dims) = (architecture["init"], architecture["mean"], architecture["cov"])
        init_out = init_dims[-1] if init_dims else input_dim
        initial_layers = get_dense(input_dim, init_out, init_dims[:-1], nn.ReLU) if init_dims else []
        mean_actvs = [nn.ReLU]*len(mean_dims) + [None]
        transf_actvs = [nn.ReLU]*len(transf_dims) + [None]

        self.initial_segment = nn.Sequential(*initial_layers)
        self.mean_layer = nn.Sequential(*get_dense(init_out, latent_dim, mean_dims, mean_actvs, False))
        if variational:
            self.transf_layer = nn.Sequential(*get_dense(init_out, latent_dim**2, transf_dims, transf_actvs))
        
        self.softplus = nn.Softplus()
        self.drop = nn.Dropout(architecture["dropout"])
        self.bn = nn.BatchNorm1d(latent_dim, momentum = 0.05, affine = False)
        self.latent_dim = latent_dim
        self.variational = variational

    def add_gnoise(self, x):
        if self.training:
            x = x + torch.randn_like(x)*self.gauss_std
        return x

    def forward(self, x_forms):
        x = x_forms["pca-ipfx"]
        x = self.add_gnoise(x)
        x = self.drop(x)
        x = self.initial_segment(x)
        mean = self.bn(self.mean_layer(x))
        if self.variational:
            transf_raw = self.transf_layer(x).reshape(-1, self.latent_dim, self.latent_dim)
            diagonals = self.softplus(torch.diagonal(transf_raw, 0, -2, -1))
            transf = torch.diag_embed(diagonals) + torch.tril(transf_raw, -1)
        else:
            transf = mean[:, None] * torch.zeros_like(mean)[..., None]
        return (mean, transf)

class Dec_pca_ipfx(nn.Module):
    def __init__(self, architecture, latent_dim, dataset):
        super().__init__()
        output_dim = architecture["data_size"][0]
        hidden_dims = (architecture["init"] + architecture["mean"])[::-1]
        actvs = [nn.ReLU]*len(hidden_dims) + [activations[architecture["out_activation"]]]
        self.network = nn.Sequential(*get_dense(latent_dim, output_dim, hidden_dims, actvs))

    def forward(self, x):
        x = self.network(x)
        x_forms = {"pca-ipfx": x}
        return x_forms

class Enc_arbors(nn.Module):
    def __init__(self, architecture, latent_dim, dataset, variational):
        super().__init__()
        (height, radius, process) = architecture["data_size"]
        (conv_params, init_dims) = (architecture["conv_params"], architecture["init"])
        (mean_dims, transf_dims) = (architecture["mean"], architecture["cov"])
        conv_out = get_conv_out_size(conv_params, height)*(conv_params[-1][2] if conv_params else radius*process)
        init_out = init_dims[-1] if init_dims else conv_out
        conv_layers = get_conv(conv_params, radius*process, nn.ReLU, False) if conv_params else []
        initial_layers = get_dense(conv_out, init_out, init_dims[:-1], nn.ReLU) if init_dims else []
        if initial_layers or conv_layers:
            initial_layers.append(nn.BatchNorm1d(init_out, momentum=0.05))
        initial_layers.insert(0, nn.Flatten())
        mean_actvs = [nn.ReLU]*len(mean_dims) + [None]
        transf_actvs = [nn.ReLU]*len(transf_dims) + [None]

        self.conv_segment = nn.Sequential(*conv_layers)
        self.initial_segment = nn.Sequential(*initial_layers)
        self.mean_layer = nn.Sequential(*get_dense(init_out, latent_dim, mean_dims, mean_actvs, False))
        if variational:
            self.transf_layer = nn.Sequential(*get_dense(init_out, latent_dim**2, transf_dims, transf_actvs))
        
        self.softplus = nn.Softplus()
        self.drop = nn.Dropout(architecture["dropout"])
        self.bn = nn.BatchNorm1d(latent_dim, momentum = 0.05, affine = False)
        self.latent_dim = latent_dim
        self.variational = variational

    def forward(self, x_forms):
        x = x_forms["arbors"]
        x = torch.flatten(x, 2).transpose(1, 2)
        x = self.drop(x)
        x = self.conv_segment(x)
        x = self.initial_segment(x)
        mean = self.bn(self.mean_layer(x))
        if self.variational:
            transf_raw = self.transf_layer(x).reshape(-1, self.latent_dim, self.latent_dim)
            diagonals = self.softplus(torch.diagonal(transf_raw, 0, -2, -1))
            transf = torch.diag_embed(diagonals) + torch.tril(transf_raw, -1)
        else:
            transf = mean[:, None] * torch.zeros_like(mean)[..., None]
        return (mean, transf)

class Dec_arbors(nn.Module):
    def __init__(self, architecture, latent_dim, dataset):
        super().__init__()
        (height, radius, process) = architecture["data_size"]
        hidden_dims = (architecture["init"] + architecture["mean"])[::-1]
        conv_params = architecture["conv_params"][::-1]
        output_actv = activations[architecture["out_activation"]]
        dense_actvs = [nn.ReLU]*len(hidden_dims) + [None]
        conv_actvs = [nn.ReLU]*len(conv_params[:-1]) + [output_actv]
        conv_T_layers = get_conv(conv_params, radius*process, conv_actvs, True) if conv_params else []
        unflat_length = get_conv_out_size(conv_params[::-1], height)
        unflat_channels = conv_params[0][2] if conv_params else radius*process
        dense_layers = get_dense(latent_dim, unflat_length*unflat_channels, hidden_dims, dense_actvs)
        dense_layers.append(nn.Unflatten(1, (unflat_channels, unflat_length)))
        if not conv_T_layers:
            dense_layers.append(output_actv())

        self.dense_segment = nn.Sequential(*dense_layers)
        self.conv_T_segment = nn.Sequential(*conv_T_layers)
        self.reshape_to_arbors = nn.Unflatten(2, (radius, process))

    def forward(self, x):
        x = self.dense_segment(x)
        x = self.conv_T_segment(x)
        x = self.reshape_to_arbors(x.transpose(1, 2))
        x_forms = {"arbors": x}
        return x_forms

class Enc_ivscc(nn.Module):
    def __init__(self, architecture, latent_dim, dataset, variational):
        super().__init__()
        gauss_frac = architecture["std_frac"]
        gauss_std = get_gauss_baselines(dataset, "ivscc").astype("float32")
        self.gauss_std = torch.nn.Parameter(torch.from_numpy(gauss_std*gauss_frac), False)

        input_dim = architecture["data_size"][0]
        (init_dims, mean_dims, transf_dims) = (architecture["init"], architecture["mean"], architecture["cov"])
        init_out = init_dims[-1] if init_dims else input_dim
        initial_layers = get_dense(input_dim, init_out, init_dims[:-1], nn.ReLU) if init_dims else []
        mean_actvs = [nn.ReLU]*len(mean_dims) + [None]
        transf_actvs = [nn.ReLU]*len(transf_dims) + [None]

        self.initial_segment = nn.Sequential(*initial_layers)
        self.mean_layer = nn.Sequential(*get_dense(init_out, latent_dim, mean_dims, mean_actvs, True))
        if variational:
            self.transf_layer = nn.Sequential(*get_dense(init_out, latent_dim**2, transf_dims, transf_actvs))
        
        self.softplus = nn.Softplus()
        self.drp = nn.Dropout(architecture["dropout"])
        self.bn = nn.BatchNorm1d(latent_dim, momentum = 0.05, affine = False)
        self.latent_dim = latent_dim
        self.variational = variational

    def add_gnoise(self, x):
        if self.training:
            x = x + torch.randn_like(x)*self.gauss_std
        return x

    def forward(self, x_forms):    
        x = x_forms["ivscc"]
        x = self.add_gnoise(x)
        x = torch.nan_to_num(x)
        x = self.drp(x)
        x = self.initial_segment(x)
        mean = self.bn(self.mean_layer(x))
        if self.variational:
            transf_raw = self.transf_layer(x).reshape(-1, self.latent_dim, self.latent_dim)
            diagonals = self.softplus(torch.diagonal(transf_raw, 0, -2, -1))
            transf = torch.diag_embed(diagonals) + torch.tril(transf_raw, -1)
        else:
            transf = mean[:, None] * torch.zeros_like(mean)[..., None]
        return (mean, transf)

class Dec_ivscc(nn.Module):
    def __init__(self, architecture, latent_dim, dataset):
        super().__init__()
        super().__init__()
        output_dim = architecture["data_size"][0]
        hidden_dims = (architecture["init"] + architecture["mean"])[::-1]
        actvs = [nn.ReLU]*len(hidden_dims) + [activations[architecture["out_activation"]]]
        self.network = nn.Sequential(*get_dense(latent_dim, output_dim, hidden_dims, actvs))

    def forward(self, x):
        x = self.network(x)
        x_forms = {"ivscc": x}
        return x_forms

class Enc_arbors_ivscc(nn.Module):
    def __init__(self, architecture, latent_dim, dataset, variational):
        super().__init__()
        ((height, radius, process), conv_params) = (architecture["arbors_size"], architecture["conv_params"])
        conv_out = get_conv_out_size(conv_params, height)*(conv_params[-1][2] if conv_params else radius*process)
        conv_layers = get_conv(conv_params, radius*process, nn.ReLU, False) if conv_params else []
        conv_layers.append(nn.Flatten())

        (ivscc_size, ivscc_dims) = (architecture["ivscc_size"][0], architecture["ivscc"])
        ivscc_out = ivscc_dims[-1] if ivscc_dims else ivscc_size
        ivscc_layers = get_dense(ivscc_size, ivscc_dims[-1], ivscc_dims[:-1], nn.ReLU) if ivscc_dims else []

        shared_dims = architecture["shared"]
        shared_in = ivscc_out + conv_out
        shared_layers = get_dense(shared_in, shared_dims[-1], shared_dims[:-1], nn.ReLU, True) if shared_dims else []
        shared_out = shared_dims[-1] if shared_dims else shared_in
        
        (mean_dims, transf_dims) = (architecture["mean"], architecture["cov"])
        mean_actvs = [nn.ReLU]*len(mean_dims) + [None]
        transf_actvs = [nn.ReLU]*len(transf_dims) + [None]

        self.conv_segment = nn.Sequential(*conv_layers)
        self.ivscc_segment = nn.Sequential(*ivscc_layers)
        self.shared_segment = nn.Sequential(*shared_layers)
        self.mean_layer = nn.Sequential(*get_dense(shared_out, latent_dim, mean_dims, mean_actvs))
        if variational:
            self.transf_layer = nn.Sequential(*get_dense(shared_out, latent_dim**2, transf_dims, transf_actvs))
        
        self.softplus = nn.Softplus()
        self.arbor_drop = torch.nn.Dropout(architecture["arbors_dropout"])
        self.ivscc_drop = torch.nn.Dropout(architecture["ivscc_dropout"])
        self.bn = nn.BatchNorm1d(latent_dim, momentum = 0.05, affine = False)
        self.latent_dim = latent_dim
        self.variational = variational

    def forward(self, x_forms):
        # Arbor sub-output:
        x = x_forms["arbors"]
        x = torch.flatten(x, 2).transpose(1, 2)
        x = self.arbor_drop(x)
        arbor_x = self.conv_segment(x)
        # IVSCC sub-output:
        x = x_forms["ivscc"]
        x = torch.nan_to_num(x)
        x = self.ivscc_drop(x)
        ivscc_x = self.ivscc_segment(x)
        # Dense output:
        x = torch.concat([arbor_x, ivscc_x], 1)
        x = self.shared_segment(x)
        mean = self.bn(self.mean_layer(x))
        if self.variational:
            transf_raw = self.transf_layer(x).reshape(-1, self.latent_dim, self.latent_dim)
            diagonals = self.softplus(torch.diagonal(transf_raw, 0, -2, -1))
            transf = torch.diag_embed(diagonals) + torch.tril(transf_raw, -1)
        else:
            transf = mean[:, None] * torch.zeros_like(mean)[..., None]
        return (mean, transf)

class Dec_arbors_ivscc(nn.Module):
    def __init__(self, architecture, latent_dim, dataset):
        super().__init__()
        ((height, radius, process), conv_params) = (architecture["arbors_size"], architecture["conv_params"][::-1])
        conv_input = get_conv_out_size(conv_params, height)*(conv_params[0][2] if conv_params else radius*process)
        unflat_length = get_conv_out_size(conv_params[::-1], height)
        unflat_channels = conv_params[0][2] if conv_params else radius*process
        arbors_out_actv = activations[architecture["arbors_activation"]]
        conv_actvs = [nn.ReLU]*len(conv_params[:-1]) + [None]
        conv_T_layers = get_conv(conv_params, radius*process, conv_actvs, True) if conv_params else []
        conv_T_layers.insert(0, nn.Unflatten(1, (unflat_channels, unflat_length)))
        if conv_T_layers:
            conv_T_layers.insert(1, nn.ReLU())
        conv_T_layers.append(arbors_out_actv())

        (ivscc_size, ivscc_dims) = (architecture["ivscc_size"][0], architecture["ivscc"])
        ivscc_input = ivscc_dims[0] if ivscc_dims else ivscc_size
        ivscc_out_actv = activations[architecture["ivscc_activation"]]
        ivscc_actvs = [nn.ReLU]*len(ivscc_dims) + [None]
        ivscc_layers = get_dense(ivscc_input, ivscc_size, ivscc_dims[1:], ivscc_actvs) if ivscc_dims else []
        if ivscc_layers:
            ivscc_layers.insert(0, nn.ReLU())
        ivscc_layers.append(ivscc_out_actv())

        shared_dims = (architecture["shared"] + architecture["mean"])[::-1]
        shared_out = ivscc_input + conv_input
        shared_actvs = [nn.ReLU]*len(shared_dims) + [None]
        shared_layers = get_dense(latent_dim, shared_out, shared_dims, shared_actvs, True)

        self.shared_segment = nn.Sequential(*shared_layers)
        self.conv_T_segment = nn.Sequential(*conv_T_layers)
        self.ivscc_segment = nn.Sequential(*ivscc_layers)
        self.reshape_to_arbors = nn.Unflatten(2, (radius, process))
        self.ivscc_input = ivscc_input

    def forward(self, x):
        # Shared intermediate:
        x = self.shared_segment(x)
        ivscc_x = x[:, :self.ivscc_input]
        arbors_x = x[:, self.ivscc_input:]
        # Arbor output:
        arbors_x = self.conv_T_segment(arbors_x)
        arbors_x = self.reshape_to_arbors(arbors_x.transpose(1, 2))
        # IVSCC output:
        ivscc_x = self.ivscc_segment(ivscc_x)
        return {"arbors": arbors_x, "ivscc": ivscc_x}

class Enc_Dummy(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.dummy_latent = torch.nn.Parameter(torch.zeros([1, latent_dim]))
    
    def forward(self, x_forms):
        x_exc = next(iter(x_forms.values()))
        mean = 0*self.dummy_latent + torch.ones_like(self.dummy_latent.tile((x_exc.shape[0], 1)))
        transf = torch.diag_embed(torch.ones_like(mean))
        return (mean, transf)

class Dec_Dummy(nn.Module):
    def __init__(self, forms, dataset, trans_funcs):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.as_tensor(0.0))
        (self.means, self.axis_tuples) = ({}, {})
        for form in forms:
            data = dataset.MET.query(dataset.allowed_specimen_ids, formats = [(form,)])[form]
            transformed = trans_funcs.get(form, lambda x: x)(data)
            cleaned = np.nan_to_num(transformed)
            self.means[form] = torch.from_numpy(np.mean(cleaned, 0, keepdims = True))
            self.axis_tuples[form] = (data.ndim - 1)*[1]

    def forward(self, x):
        x_forms = {}
        for (form, mean) in self.means.items():
            xr = 0*self.dummy_param + mean.tile([x.shape[0]] + self.axis_tuples[form])
            x_forms[form] = xr
        return x_forms

class Coupler(nn.Module):
    def __init__(self, layer_dims, dataset):
        super().__init__()
        actvs = [nn.ReLU]*len(layer_dims[1:-1]) + [None]
        layers = get_dense(layer_dims[0], layer_dims[-1], layer_dims[1:-1], actvs, True)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        return x
    
class Mapper(nn.Module):
    def __init__(self, init_hidden, mean_hidden, transf_hidden, latent_dim, fixed_mean):
        super().__init__()
        init_out = init_hidden[-1] if init_hidden else latent_dim
        initial_layers = get_dense(latent_dim, init_out, init_hidden[:-1], nn.ReLU) if init_hidden else []
        mean_actvs = [nn.ReLU]*len(mean_hidden) + [None]
        mean_layers = get_dense(init_out, latent_dim, mean_hidden, mean_actvs) if not fixed_mean else []
        transf_actvs = [nn.ReLU]*len(transf_hidden) + [None]
        self.initial_segment = nn.Sequential(*initial_layers)
        self.mean_layer = nn.Sequential(*mean_layers)
        self.transf_layer = nn.Sequential(*get_dense(init_out, latent_dim**2, transf_hidden, transf_actvs))
        self.softplus = nn.Softplus()
        self.latent_dim = latent_dim

    def forward(self, z):
        z_init = self.initial_segment(z)
        mean = self.mean_layer(z_init)
        transf_raw = self.transf_layer(z_init).reshape(-1, self.latent_dim, self.latent_dim)
        diagonals = self.softplus(torch.diagonal(transf_raw, 0, -2, -1)) + 1e-4
        transf = torch.diag_embed(diagonals) + torch.tril(transf_raw, -1)
        return (mean, transf)

def get_coupler(config, train_dataset):
    model = {}
    for (in_modal, in_specs) in config["modal_specs"].items():
        for (out_modal, out_specs) in config["modal_specs"].items():
            if in_modal == out_modal:
                continue
            layer_dims = [in_specs["latent_dim"]] + config["hidden_dims"] + [out_specs["latent_dim"]]
            model[f"{in_modal}-{out_modal}"] = Coupler(layer_dims, train_dataset)
    return model

def get_mapper(config, train_dataset):
    model = torch.nn.ModuleDict()
    specs = config["variational"]["mapper"]
    for in_modal in config["modalities"]:
        for out_modal in config["modalities"]:
            if in_modal != out_modal:
                mapper = Mapper(specs["init"], specs["mean"], specs["transf"], config["latent_dim"], specs["fixed_mean"])
                model[f"{in_modal}-{out_modal}"] = mapper
    return model

def get_model(config, train_dataset):
    architectures = {frozenset(forms.split("_")): params for (forms, params) in config["architecture"].items()}
    model = {}
    variational = config["inference"]
    for modal in config["modalities"]:
        arm = {}
        forms = frozenset(config["formats"][modal])
        architecture = architectures[forms]
        if architecture.get("dummy"):
            if config["transform"]:
                trans_funcs = {form: get_transformation_function(transform_dict)
                    for (form, transform_dict) in config["transform"].items()}
            else:
                trans_funcs = {}
            arm["enc"] = Enc_Dummy(config["latent_dim"])
            arm["dec"] = Dec_Dummy(forms, train_dataset, trans_funcs)
        else:
            arm["enc"] = modules[forms]["enc"](architecture, config["latent_dim"], train_dataset, variational)
            arm["dec"] = modules[forms]["dec"](architecture, config["latent_dim"], train_dataset)
        model[modal] = arm
    return model

modules = {
    frozenset(["logcpm"]): {
        "enc": Enc_logcpm,
        "dec": Dec_logcpm
        },
    frozenset(["pca-ipfx"]): {
        "enc": Enc_pca_ipfx,
        "dec": Dec_pca_ipfx
        },
    frozenset(["arbors"]): {
        "enc": Enc_arbors,
        "dec": Dec_arbors
        },
    frozenset(["ivscc"]): {
        "enc": Enc_ivscc,
        "dec": Dec_ivscc
    },
    frozenset(["ivscc", "arbors"]): {
        "enc": Enc_arbors_ivscc,
        "dec": Dec_arbors_ivscc
    }
}
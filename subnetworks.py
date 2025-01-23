import torch
from torch import nn
import numpy as np

from data import get_transformation_function

activations = {
    "linear": nn.Identity,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus
}

conv_classes = {
    1: (torch.nn.Conv1d, torch.nn.ConvTranspose1d),
    2: (torch.nn.Conv2d, torch.nn.ConvTranspose2d),
    3: (torch.nn.Conv3d, torch.nn.ConvTranspose3d)
}

def get_conv_out_size(conv_params, *initial_dims):
    output_padding = []
    outputs = initial_dims
    for (kernels, strides, _) in conv_params:
        padding = [int((out_len - kernel_len) % stride_len)
                   for (kernel_len, stride_len, out_len) in zip(kernels, strides, outputs)]
        outputs = [1 + (out_len - kernel_len) / stride_len
                   for (kernel_len, stride_len, out_len) in zip(kernels, strides, outputs)]
        output_padding.append(tuple(padding))
    outputs = [int(out_len) for out_len in outputs]
    return (outputs, output_padding)

def get_gauss_baselines(dataset, form):
    data = dataset.MET.query(dataset.allowed_specimen_ids)[form]
    std = np.nanstd(data, 0, keepdims = True)
    return std

def get_conv(conv_params, input_channels, actvs, transpose, output_padding = None):
    dim = len(conv_params[0][0])
    layer_class = conv_classes[dim][1 if transpose else 0]
    if transpose:
        channel_dims = [tupl[-1] for tupl in conv_params] + [input_channels]
    else:
        channel_dims = [input_channels] + [tupl[-1] for tupl in conv_params]
    if type(actvs) != list:
        actvs = [actvs]*len(conv_params)
    layers = []
    for (i, ((kernel, stride, _), actv)) in enumerate(zip(conv_params, actvs)):
        (input_dim, output_dim) = channel_dims[i:i + 2]
        if transpose:
            conv_layer = layer_class(input_dim, output_dim, kernel_size = kernel, stride = stride, output_padding = output_padding[i])
        else:
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
    def __init__(self, forms, architecture, latent_dim, dataset, variational):
        super().__init__()
        self.form = next(iter(forms))
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
        x = x_forms[self.form]
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
    def __init__(self, forms, architecture, latent_dim, dataset):
        super().__init__()
        self.form = next(iter(forms))
        output_dim = architecture["data_size"][0]
        hidden_dims = (architecture["init"] + architecture["mean"])[::-1]
        actvs = [nn.ReLU]*len(hidden_dims) + [activations[architecture["out_activation"]]]
        self.network = nn.Sequential(*get_dense(latent_dim, output_dim, hidden_dims, actvs))

    def forward(self, x):
        x = self.network(x)
        x_forms = {self.form: x}
        return x_forms

class Enc_pca_ipfx(nn.Module):
    def __init__(self, forms, architecture, latent_dim, dataset, variational):
        super().__init__()
        self.form = next(iter(forms))
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
        x = x_forms[self.form]
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
    def __init__(self, forms, architecture, latent_dim, dataset):
        super().__init__()
        self.form = next(iter(forms))
        output_dim = architecture["data_size"][0]
        hidden_dims = (architecture["init"] + architecture["mean"])[::-1]
        actvs = [nn.ReLU]*len(hidden_dims) + [activations[architecture["out_activation"]]]
        self.network = nn.Sequential(*get_dense(latent_dim, output_dim, hidden_dims, actvs))

    def forward(self, x):
        x = self.network(x)
        x_forms = {self.form: x}
        return x_forms

class Enc_arbors(nn.Module):
    def __init__(self, forms, architecture, latent_dim, dataset, variational):
        super().__init__()
        self.form = next(iter(forms))
        (data_dims, process) = (architecture["data_size"][:-1], architecture["data_size"][-1])
        (conv_params, init_dims) = (architecture["conv_params"], architecture["init"])
        (mean_dims, transf_dims) = (architecture["mean"], architecture["cov"])
        int_actv = activations[architecture["int_activation"]]
        output_dims = get_conv_out_size(conv_params, *data_dims)[0]
        conv_out = np.prod(output_dims)*(conv_params[-1][2] if conv_params else process)
        init_out = init_dims[-1] if init_dims else conv_out
        conv_layers = get_conv(conv_params, process, int_actv, False) if conv_params else []
        initial_layers = get_dense(conv_out, init_out, init_dims[:-1], int_actv) if init_dims else []
        if initial_layers or conv_layers:
            initial_layers.append(nn.BatchNorm1d(init_out, momentum=0.05))
        initial_layers.insert(0, nn.Flatten())
        mean_actvs = [int_actv]*len(mean_dims) + [None]
        transf_actvs = [int_actv]*len(transf_dims) + [None]

        dim = len(data_dims)
        self.permutation = (0, dim + 1, *range(1, dim + 1))
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
        x = x_forms[self.form]
        x = torch.permute(x, self.permutation)
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
    def __init__(self, forms, architecture, latent_dim, dataset):
        super().__init__()
        self.form = next(iter(forms))
        (data_dims, process) = (architecture["data_size"][:-1], architecture["data_size"][-1])
        hidden_dims = (architecture["init"] + architecture["mean"])[::-1]
        conv_params = architecture["conv_params"][::-1]
        output_actv = activations[architecture["out_activation"]]
        int_actv = activations[architecture["int_activation"]]
        dense_actvs = [int_actv]*len(hidden_dims) + [None]
        conv_actvs = [int_actv]*len(conv_params[:-1]) + [output_actv]
        (unflat_dims, out_padding) = get_conv_out_size(conv_params[::-1], *data_dims)
        conv_T_layers = get_conv(conv_params, process, conv_actvs, True, out_padding[::-1]) if conv_params else []
        unflat_channels = conv_params[0][2] if conv_params else process
        dense_layers = get_dense(latent_dim, np.prod(unflat_dims)*unflat_channels, hidden_dims, dense_actvs)
        dense_layers.append(nn.Unflatten(1, (unflat_channels, *unflat_dims)))
        if not conv_T_layers:
            dense_layers.append(output_actv())

        dim = len(data_dims)
        self.permutation = (0, *range(2, dim + 2), 1)
        self.dense_segment = nn.Sequential(*dense_layers)
        self.conv_T_segment = nn.Sequential(*conv_T_layers)

    def forward(self, x):
        x = self.dense_segment(x)
        x = self.conv_T_segment(x)
        x = torch.permute(x, self.permutation)
        x_forms = {self.form: x}
        return x_forms

class Enc_morphometric(nn.Module):
    def __init__(self, forms, architecture, latent_dim, dataset, variational):
        super().__init__()
        self.form = next(iter(forms))
        gauss_frac = architecture["std_frac"]
        gauss_std = get_gauss_baselines(dataset, self.form).astype("float32")
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
        x = x_forms[self.form]
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

class Dec_morphometric(nn.Module):
    def __init__(self, forms, architecture, latent_dim, dataset):
        super().__init__()
        self.form = next(iter(forms))
        output_dim = architecture["data_size"][0]
        hidden_dims = (architecture["init"] + architecture["mean"])[::-1]
        actvs = [nn.ReLU]*len(hidden_dims) + [activations[architecture["out_activation"]]]
        self.network = nn.Sequential(*get_dense(latent_dim, output_dim, hidden_dims, actvs))

    def forward(self, x):
        x = self.network(x)
        x_forms = {self.form: x}
        return x_forms

class Enc_arbors_features(nn.Module):
    def __init__(self, forms, architecture, latent_dim, dataset, variational):
        super().__init__()
        self.feat_name = [form for form in forms if form != "arbors"][0]

        (data_dims, process) = (architecture["arbors_size"][:-1], architecture["arbors_size"][-1])
        conv_params = architecture["conv_params"]
        conv_output_dims = get_conv_out_size(conv_params, *data_dims)[0]
        conv_out = np.prod(conv_output_dims)*(conv_params[-1][2] if conv_params else process)
        conv_layers = get_conv(conv_params, process, activations["relu"], False) if conv_params else []
        conv_layers.append(nn.Flatten())

        (feat_input_dim, feat_params) = (architecture["feat_size"][0], architecture["feat_params"])
        feat_out = feat_params[-1] if feat_params else feat_input_dim
        feat_layers = get_dense(feat_input_dim, feat_out, feat_params[:-1], nn.ReLU) if feat_params else []

        shared_dims = architecture["shared"]
        shared_in = feat_out + conv_out
        shared_out = shared_dims[-1] if shared_dims else shared_in
        shared_layers = get_dense(shared_in, shared_out, shared_dims[:-1], nn.ReLU, True) if shared_dims else []
        
        (mean_dims, transf_dims) = (architecture["mean"], architecture["cov"])
        mean_actvs = [nn.ReLU]*len(mean_dims) + [None]
        transf_actvs = [nn.ReLU]*len(transf_dims) + [None]

        dim = len(data_dims)
        self.permutation = (0, dim + 1, *range(1, dim + 1))
        self.conv_segment = nn.Sequential(*conv_layers)
        self.feat_segment = nn.Sequential(*feat_layers)
        self.shared_segment = nn.Sequential(*shared_layers)
        self.mean_layer = nn.Sequential(*get_dense(shared_out, latent_dim, mean_dims, mean_actvs, False))
        if variational:
            self.transf_layer = nn.Sequential(*get_dense(shared_out, latent_dim**2, transf_dims, transf_actvs))
        
        self.softplus = nn.Softplus()
        self.arbor_drop = torch.nn.Dropout(architecture["arbors_dropout"])
        self.feat_drop = torch.nn.Dropout(architecture["feat_dropout"])
        self.bn = nn.BatchNorm1d(latent_dim, momentum = 0.05, affine = False)
        self.latent_dim = latent_dim
        self.variational = variational

    def forward(self, x_forms):
        # Arbor sub-output:
        x = x_forms["arbors"]
        x = torch.permute(x, self.permutation)
        x = self.arbor_drop(x)
        arbor_x = self.conv_segment(x)
        # Feature sub-output:
        x = x_forms[self.feat_name]
        x = torch.nan_to_num(x)
        x = self.feat_drop(x)
        feat_x = self.feat_segment(x)
        # Dense output:
        x = torch.concat([arbor_x, feat_x], 1)
        x = self.shared_segment(x)
        mean = self.bn(self.mean_layer(x))
        if self.variational:
            transf_raw = self.transf_layer(x).reshape(-1, self.latent_dim, self.latent_dim)
            diagonals = self.softplus(torch.diagonal(transf_raw, 0, -2, -1))
            transf = torch.diag_embed(diagonals) + torch.tril(transf_raw, -1)
        else:
            transf = mean[:, None] * torch.zeros_like(mean)[..., None]
        return (mean, transf)

class Dec_arbors_features(nn.Module):
    def __init__(self, forms, architecture, latent_dim, dataset):
        super().__init__()
        self.feat_name = [form for form in forms if form != "arbors"][0]

        (conv_dims, process) = (architecture["arbors_size"][:-1], architecture["arbors_size"][-1])
        conv_params = architecture["conv_params"][::-1]
        arbors_out_actv = activations[architecture["arbors_activation"]]
        conv_actvs = [nn.ReLU]*len(conv_params[:-1]) + [arbors_out_actv]
        (unflat_dims, out_padding) = get_conv_out_size(conv_params[::-1], *conv_dims)
        unflat_channels = conv_params[0][2] if conv_params else process
        conv_T_layers = get_conv(conv_params, process, conv_actvs, True, out_padding[::-1]) if conv_params else []
        conv_T_layers.insert(0, nn.Unflatten(1, (unflat_channels, *unflat_dims)))

        (feat_size, feat_dims) = (architecture["feat_size"][0], architecture["feat_params"])
        feat_input = feat_dims[0] if feat_dims else feat_size
        feat_out_actv = activations[architecture["feat_activation"]]
        feat_actvs = [nn.ReLU]*len(feat_dims) + [None]
        feat_layers = get_dense(feat_input, feat_size, feat_dims[1:], feat_actvs) if feat_dims else []
        if feat_layers:
            feat_layers.insert(0, nn.ReLU())
        feat_layers.append(feat_out_actv())

        conv_input = np.prod(unflat_dims)*unflat_channels
        shared_dims = (architecture["shared"] + architecture["mean"])[::-1]
        shared_out = feat_input + conv_input
        shared_actvs = [nn.ReLU]*len(shared_dims) + [None]
        shared_layers = get_dense(latent_dim, shared_out, shared_dims, shared_actvs, True)

        dim = len(conv_dims)
        self.permutation = (0, *range(2, dim + 2), 1)
        self.shared_segment = nn.Sequential(*shared_layers)
        self.conv_T_segment = nn.Sequential(*conv_T_layers)
        self.feat_segment = nn.Sequential(*feat_layers)
        self.feat_input = feat_input

    def forward(self, x):
        # Shared intermediate:
        x = self.shared_segment(x)
        feat_x = x[:, :self.feat_input]
        arbors_x = x[:, self.feat_input:]
        # Arbor output:
        arbors_x = self.conv_T_segment(arbors_x)
        arbors_x = torch.permute(arbors_x, self.permutation)
        # Feature output:
        feat_x = self.feat_segment(feat_x)
        return {"arbors": arbors_x, self.feat_name: feat_x}

class Enc_arbors_sholl(nn.Module):
    def __init__(self, forms, architecture, latent_dim, dataset, variational):
        super().__init__()

        (arbor_dims, arbor_process) = (architecture["arbors_size"][:-1], architecture["arbors_size"][-1])
        arbor_params = architecture["arbor_params"]
        arbor_output_dims = get_conv_out_size(arbor_params, *arbor_dims)[0]
        arbor_out = np.prod(arbor_output_dims)*(arbor_params[-1][2] if arbor_params else arbor_process)
        arbor_conv_layers = get_conv(arbor_params, arbor_process, activations["relu"], False) if arbor_params else []
        arbor_conv_layers.append(nn.Flatten())

        (sholl_dims, sholl_process) = (architecture["sholl_size"][:-1], architecture["sholl_size"][-1])
        sholl_params = architecture["sholl_params"]
        sholl_output_dims = get_conv_out_size(sholl_params, *sholl_dims)[0]
        sholl_out = np.prod(sholl_output_dims)*(sholl_params[-1][2] if sholl_params else sholl_process)
        sholl_conv_layers = get_conv(sholl_params, sholl_process, activations["relu"], False) if sholl_params else []
        sholl_conv_layers.append(nn.Flatten())

        shared_dims = architecture["shared"]
        shared_in = sholl_out + arbor_out
        shared_out = shared_dims[-1] if shared_dims else shared_in
        shared_layers = get_dense(shared_in, shared_out, shared_dims[:-1], nn.ReLU, True) if shared_dims else []
        
        (mean_dims, transf_dims) = (architecture["mean"], architecture["cov"])
        mean_actvs = [nn.ReLU]*len(mean_dims) + [None]
        transf_actvs = [nn.ReLU]*len(transf_dims) + [None]

        (arbor_dim, sholl_dim) = (len(arbor_dims), len(sholl_dims))
        self.arbor_permutation = (0, arbor_dim + 1, *range(1, arbor_dim + 1))
        self.sholl_permutation = (0, sholl_dim + 1, *range(1, sholl_dim + 1))
        self.arbor_segment = nn.Sequential(*arbor_conv_layers)
        self.sholl_segment = nn.Sequential(*sholl_conv_layers)
        self.shared_segment = nn.Sequential(*shared_layers)
        self.mean_layer = nn.Sequential(*get_dense(shared_out, latent_dim, mean_dims, mean_actvs, False))
        if variational:
            self.transf_layer = nn.Sequential(*get_dense(shared_out, latent_dim**2, transf_dims, transf_actvs))
        
        self.softplus = nn.Softplus()
        self.arbor_drop = torch.nn.Dropout(architecture["arbors_dropout"])
        self.sholl_drop = torch.nn.Dropout(architecture["sholl_dropout"])
        self.bn = nn.BatchNorm1d(latent_dim, momentum = 0.05, affine = False)
        self.latent_dim = latent_dim
        self.variational = variational

    def forward(self, x_forms):
        # Arbor sub-output:
        x = x_forms["arbors"]
        x = torch.permute(x, self.arbor_permutation)
        x = self.arbor_drop(x)
        arbor_x = self.arbor_segment(x)
        # Sholl sub-output:
        x = x_forms["sholl"]
        x = torch.permute(x, self.sholl_permutation)
        x = self.sholl_drop(x)
        sholl_x = self.sholl_segment(x)
        # Dense output:
        x = torch.concat([arbor_x, sholl_x], 1)
        x = self.shared_segment(x)
        mean = self.bn(self.mean_layer(x))
        if self.variational:
            transf_raw = self.transf_layer(x).reshape(-1, self.latent_dim, self.latent_dim)
            diagonals = self.softplus(torch.diagonal(transf_raw, 0, -2, -1))
            transf = torch.diag_embed(diagonals) + torch.tril(transf_raw, -1)
        else:
            transf = mean[:, None] * torch.zeros_like(mean)[..., None]
        return (mean, transf)

class Dec_arbors_sholl(nn.Module):
    def __init__(self, forms, architecture, latent_dim, dataset):
        super().__init__()

        (arbor_dims, arbor_process) = (architecture["arbors_size"][:-1], architecture["arbors_size"][-1])
        arbor_params = architecture["arbor_params"][::-1]
        arbors_out_actv = activations[architecture["arbors_activation"]]
        arbor_actvs = [nn.ReLU]*len(arbor_params[:-1]) + [arbors_out_actv]
        (arbor_unflat_dims, arbor_out_padding) = get_conv_out_size(arbor_params[::-1], *arbor_dims)
        arbor_unflat_channels = arbor_params[0][2] if arbor_params else arbor_process
        arbor_T_layers = get_conv(arbor_params, arbor_process, arbor_actvs, True, arbor_out_padding[::-1]) if arbor_params else []
        arbor_T_layers.insert(0, nn.Unflatten(1, (arbor_unflat_channels, *arbor_unflat_dims)))

        (sholl_dims, sholl_process) = (architecture["sholl_size"][:-1], architecture["sholl_size"][-1])
        sholl_params = architecture["sholl_params"][::-1]
        sholl_out_actv = activations[architecture["sholl_activation"]]
        sholl_actvs = [nn.ReLU]*len(sholl_params[:-1]) + [sholl_out_actv]
        (sholl_unflat_dims, sholl_out_padding) = get_conv_out_size(sholl_params[::-1], *sholl_dims)
        sholl_unflat_channels = sholl_params[0][2] if sholl_params else sholl_process
        sholl_T_layers = get_conv(sholl_params, sholl_process, sholl_actvs, True, sholl_out_padding[::-1]) if sholl_params else []
        sholl_T_layers.insert(0, nn.Unflatten(1, (sholl_unflat_channels, *sholl_unflat_dims)))

        arbor_input = np.prod(arbor_unflat_dims)*arbor_unflat_channels
        sholl_input = np.prod(sholl_unflat_dims)*sholl_unflat_channels
        shared_out = arbor_input + sholl_input
        shared_dims = (architecture["shared"] + architecture["mean"])[::-1]
        shared_actvs = [nn.ReLU]*len(shared_dims) + [None]
        shared_layers = get_dense(latent_dim, shared_out, shared_dims, shared_actvs, True)

        self.arbor_permutation = (0, *range(2, len(arbor_dims) + 2), 1)
        self.sholl_permutation = (0, *range(2, len(sholl_dims) + 2), 1)
        self.shared_segment = nn.Sequential(*shared_layers)
        self.arbor_T_segment = nn.Sequential(*arbor_T_layers)
        self.sholl_T_segment = nn.Sequential(*sholl_T_layers)
        self.arbor_input = arbor_input

    def forward(self, x):
        # Shared intermediate:
        x = self.shared_segment(x)
        arbors_x = x[:, :self.arbor_input]
        sholl_x = x[:, self.arbor_input:]
        # Arbor output:
        arbors_x = self.arbor_T_segment(arbors_x)
        arbors_x = torch.permute(arbors_x, self.arbor_permutation)
        # Sholl output:
        sholl_x = self.sholl_T_segment(sholl_x)
        sholl_x = torch.permute(sholl_x, self.sholl_permutation)
        return {"arbors": arbors_x, "sholl": sholl_x}

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

class Classifier(nn.Module):
    def __init__(self, latent_dim, hidden_dims, num_classes):
        super().__init__()
        actvs = len(hidden_dims)*[nn.ReLU] + [None]
        dense = get_dense(latent_dim, num_classes, hidden_dims, actvs)
        self.layers = nn.Sequential(*dense)

    def forward(self, z):
        log_probs = self.layers(z)
        return log_probs

def get_mapper(config, train_dataset):
    model = torch.nn.ModuleDict()
    specs = config["variational"]["mapper"]
    for in_modal in config["modalities"]:
        for out_modal in config["modalities"]:
            if in_modal != out_modal:
                mapper = Mapper(specs["init"], specs["mean"], specs["transf"], config["latent_dim"], specs["fixed_mean"])
                model[f"{in_modal}-{out_modal}"] = mapper
    return model

def get_classifiers(config, train_dataset):
    model = torch.nn.ModuleDict()
    specs = config["variational"]["classifier"]
    for (i, label_type) in enumerate(config["variational"]["classifier"]["label"]):
        num_classes = np.unique(train_dataset.MET.labels[:, i]).max() + 1
        model[label_type] = Classifier(config["latent_dim"], specs["hidden"], num_classes)
    return model

def get_model(config, train_dataset):
    architectures = {frozenset(forms.split("_")): params for (forms, params) in config["architecture"].items()}
    model = torch.nn.ModuleDict()
    variational = config["inference"]
    for modal in config["modalities"]:
        arm = torch.nn.ModuleDict()
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
            arm["enc"] = modules[forms]["enc"](forms, architecture, config["latent_dim"], train_dataset, variational)
            arm["dec"] = modules[forms]["dec"](forms, architecture, config["latent_dim"], train_dataset)
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
    frozenset(["morphometric"]): {
        "enc": Enc_morphometric,
        "dec": Dec_morphometric
    },
    frozenset(["sholl"]): {
        "enc": Enc_arbors,
        "dec": Dec_arbors
    },
    frozenset(["morphometric", "arbors"]): {
        "enc": Enc_arbors_features,
        "dec": Dec_arbors_features
    },
    frozenset(["sholl", "arbors"]): {
        "enc": Enc_arbors_sholl,
        "dec": Dec_arbors_sholl
    }
}

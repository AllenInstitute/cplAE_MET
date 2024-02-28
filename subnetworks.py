import torch
from torch import nn
import numpy as np

from data import get_transformation_function

activations = {
    "linear": torch.nn.Identity(),
    "relu": torch.nn.functional.relu,
    "sigmoid": torch.nn.functional.sigmoid
}

def get_conv_out_size(conv_dims, initial_length):
    output_length = initial_length
    for (kernel, stride, _) in conv_dims:
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

def add_dense_segment(model, hidden_dims, input_size, final_bias, prefix):
    layer_names = []
    layer_sizes = [input_size] + hidden_dims
    for (i, (input_dim, output_dim)) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        bias = (final_bias if i == len(hidden_dims) else True)
        setattr(model, f"{prefix}_{i}", torch.nn.Linear(input_dim, output_dim, bias = bias))
        layer_names.append(f"{prefix}_{i}")
    return layer_names

class Enc_logcpm(nn.Module):
    def __init__(self, architecture, latent_dim, dataset):
        super().__init__()
        self.layer_names = []
        hidden_dims = architecture["hidden"]
        all_dims = architecture["data_size"] + list(hidden_dims) + [latent_dim]
        for (i, (i_dim, o_dim)) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            bias = (False if i == len(hidden_dims) else True)
            name = ("fc_mu" if i == len(hidden_dims) else f"fc_{i}")
            setattr(self, name, torch.nn.Linear(i_dim, o_dim, bias = bias))
            self.layer_names.append(name)
        self.fc_sigma = nn.Linear(all_dims[-2], latent_dim, bias = False)

        self.drp = nn.Dropout(architecture["dropout"])
        self.bn = nn.BatchNorm1d(latent_dim, momentum = 0.05, affine = False)
        relu = nn.ReLU()
        self.actvs = [relu]*(len(self.layer_names) - 1) + [torch.nn.Identity()]

    def forward(self, x_forms):
        x = x_forms["logcpm"]
        x = self.drp(x)
        for (layer_name, actv) in zip(self.layer_names, self.actvs):
            layer = getattr(self, layer_name)
            x = actv(layer(x))
        z = self.bn(x)
        return z

class Dec_logcpm(nn.Module):
    def __init__(self, architecture, latent_dim, dataset):
        super().__init__()
        self.layer_names = []
        hidden_dims = architecture["hidden"][::-1]
        all_dims = [latent_dim] + hidden_dims + architecture["data_size"]
        for (i, (i_dim, o_dim)) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            setattr(self, f"fc_{i}", torch.nn.Linear(i_dim, o_dim))
            self.layer_names.append(f"fc_{i}")
        relu = nn.ReLU()
        output_activation = activations[architecture["out_activation"]]
        self.actvs = [relu]*(len(self.layer_names) - 1) + [output_activation]

    def forward(self, x):
        for (layer_name, actv) in zip(self.layer_names, self.actvs):
            layer = getattr(self, layer_name)
            x = actv(layer(x))
        x_forms = {"logcpm": x}
        return x_forms

class Enc_pca_ipfx(nn.Module):
    def __init__(self, architecture, latent_dim, dataset):
        super().__init__()
        gauss_frac = architecture["std_frac"]
        gauss_std = get_gauss_baselines(dataset, "pca-ipfx").astype("float32")
        self.gauss_std = torch.nn.Parameter(torch.from_numpy(gauss_std*gauss_frac))

        self.layer_names = []
        hidden_dims = architecture["hidden"]
        all_dims = architecture["data_size"] + list(hidden_dims) + [latent_dim]
        for (i, (i_dim, o_dim)) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            bias = (False if i == len(hidden_dims) else True)
            name = ("fc_mu" if i == len(hidden_dims) else f"fc_{i}")
            setattr(self, name, torch.nn.Linear(i_dim, o_dim, bias = bias))
            self.layer_names.append(name)
        self.fc_sigma = nn.Linear(all_dims[-2], latent_dim, bias = False)

        self.drp = nn.Dropout(architecture["dropout"])
        self.bn = nn.BatchNorm1d(latent_dim, momentum = 0.05, affine = False)
        relu = nn.ReLU()
        self.actvs = [relu]*(len(self.layer_names) - 1) + [torch.nn.Identity()]

    def add_gnoise(self, x):
        if self.training:
            x = x + torch.randn_like(x)*self.gauss_std
        return x

    def forward(self, x_forms):
        x = x_forms["pca-ipfx"]
        x = self.add_gnoise(x)
        x = self.drp(x)
        for (layer_name, actv) in zip(self.layer_names, self.actvs):
            layer = getattr(self, layer_name)
            x = actv(layer(x))
        z = self.bn(x)
        return z

class Dec_pca_ipfx(nn.Module):
    def __init__(self, architecture, latent_dim, dataset):
        super().__init__()
        self.layer_names = []
        hidden_dims = architecture["hidden"][::-1]
        all_dims = [latent_dim] + hidden_dims + architecture["data_size"]
        for (i, (i_dim, o_dim)) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            setattr(self, f"fc_{i}", torch.nn.Linear(i_dim, o_dim))
            self.layer_names.append(f"fc_{i}")
        self.drp = nn.Dropout(p = 0.1)
        relu = nn.ReLU()
        output_activation = activations[architecture["out_activation"]]
        self.actvs = [relu]*(len(self.layer_names) - 1) + [output_activation]

    def forward(self, x):
        for (layer_name, actv) in zip(self.layer_names, self.actvs):
            layer = getattr(self, layer_name)
            x = actv(layer(x))
        x_forms = {"pca-ipfx": x}
        return x_forms

class Enc_arbors(nn.Module):
    def __init__(self, architecture, latent_dim, dataset):
        super().__init__()
        (self.conv_names, self.dense_names) = ([], [])
        (height, radius, process) = architecture["data_size"]
        (conv_params, hidden_dims) = (architecture["conv_params"], architecture["hidden"])
        conv_dims = [radius*process] + [tupl[-1] for tupl in conv_params]
        dense_dims = [conv_dims[-1]*get_conv_out_size(conv_params, height)] + list(hidden_dims) + [latent_dim]
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
        self.fc_sigma = nn.Linear(dense_dims[-2], latent_dim, bias = False)

        self.bn = nn.BatchNorm1d(latent_dim, momentum = 0.05, affine = False)
        self.drop = torch.nn.Dropout(architecture["dropout"])
        relu = nn.ReLU()
        (num_dense, num_conv) = (len(self.dense_names), len(self.conv_names))
        actvs = [relu]*(num_dense + num_conv - 1) + [torch.nn.Identity()]
        (self.conv_actvs, self.dense_actvs) = (actvs[:num_conv], actvs[num_conv:])

    def forward(self, x_forms):
        x = x_forms["arbors"]
        x = torch.flatten(x, 2).transpose(1, 2)
        x = self.drop(x)
        for (conv_name, actv) in zip(self.conv_names, self.conv_actvs):
            conv_layer = getattr(self, conv_name)
            x = actv(conv_layer(x))
        x = torch.flatten(x, 1)
        for (dense_name, actv) in zip(self.dense_names, self.dense_actvs):
            dense_layer = getattr(self, dense_name)
            x = actv(dense_layer(x))
        z = self.bn(x)
        return z

class Dec_arbors(nn.Module):
    def __init__(self, architecture, latent_dim, dataset):
        super().__init__()
        (self.height, self.radius, self.process) = architecture["data_size"]
        (self.conv_names, self.dense_names) = ([], [])
        (conv_params, hidden_dims) = (architecture["conv_params"][::-1], architecture["hidden"][::-1])
        conv_dims = [tupl[-1] for tupl in conv_params] + [self.radius*self.process]
        self.initial_channels = conv_dims[0]
        dense_dims = [latent_dim] + hidden_dims + [conv_dims[0]*get_conv_out_size(conv_params, self.height)]
        for (i, (kernel, stride, _)) in enumerate(conv_params):
            (input_dim, output_dim) = conv_dims[i:i + 2]
            conv_layer = torch.nn.ConvTranspose1d(input_dim, output_dim, kernel_size = kernel, stride = stride)
            setattr(self, f"conv_{i}", conv_layer)
            self.conv_names.append(f"conv_{i}")
        for (i, (input_dim, output_dim)) in enumerate(zip(dense_dims[:-1], dense_dims[1:])):
            setattr(self, f"fc_{i}", torch.nn.Linear(input_dim, output_dim))
            self.dense_names.append(f"fc_{i}")

        relu = nn.ReLU()
        output_activation = activations[architecture["out_activation"]]
        (num_dense, num_conv) = (len(self.dense_names), len(self.conv_names))
        actvs = [relu]*(num_dense + num_conv - 1) + [output_activation]
        (self.dense_actvs, self.conv_actvs) = (actvs[:num_dense], actvs[num_dense:])

    def forward(self, x):
        for (dense_name, actv) in zip(self.dense_names, self.dense_actvs):
            dense_layer = getattr(self, dense_name)
            x = actv(dense_layer(x))
        x = x.view(x.shape[0], self.initial_channels, -1)
        for (conv_name, actv) in zip(self.conv_names, self.conv_actvs):
            conv_T_layer = getattr(self, conv_name)
            x = actv(conv_T_layer(x))
        x = x.transpose(1, 2).reshape([-1, self.height, self.radius, self.process])
        x_forms = {"arbors": x}
        return x_forms

class Enc_ivscc(nn.Module):
    def __init__(self, architecture, latent_dim, dataset):
        super().__init__()
        gauss_frac = architecture["std_frac"]
        gauss_std = get_gauss_baselines(dataset, "ivscc").astype("float32")
        self.gauss_std = torch.nn.Parameter(torch.from_numpy(gauss_std*gauss_frac))

        self.layer_names = []
        hidden_dims = architecture["hidden"]
        all_dims = architecture["data_size"] + list(hidden_dims) + [latent_dim]
        for (i, (i_dim, o_dim)) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            bias = (False if i == len(hidden_dims) else True)
            name = ("fc_mu" if i == len(hidden_dims) else f"fc_{i}")
            setattr(self, name, torch.nn.Linear(i_dim, o_dim, bias = bias))
            self.layer_names.append(name)
        self.fc_sigma = nn.Linear(all_dims[-2], latent_dim, bias = False)

        self.drp = nn.Dropout(architecture["dropout"])
        self.bn = nn.BatchNorm1d(latent_dim, momentum = 0.05, affine = False)
        relu = nn.ReLU()
        self.actvs = [relu]*(len(self.layer_names) - 1) + [torch.nn.Identity()]

    def add_gnoise(self, x):
        if self.training:
            x = x + torch.randn_like(x)*self.gauss_std
        return x

    def forward(self, x_forms):
        x = self.add_gnoise(x_forms["ivscc"])
        x = torch.nan_to_num(x)
        x = self.drp(x)
        for (layer_name, actv) in zip(self.layer_names, self.actvs):
            layer = getattr(self, layer_name)
            x = actv(layer(x))
        z = self.bn(x)
        return z

class Dec_ivscc(nn.Module):
    def __init__(self, architecture, latent_dim, dataset):
        super().__init__()
        self.layer_names = []
        hidden_dims = architecture["hidden"][::-1]
        all_dims = [latent_dim] + hidden_dims + architecture["data_size"]
        for (i, (i_dim, o_dim)) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            setattr(self, f"fc_{i}", torch.nn.Linear(i_dim, o_dim))
            self.layer_names.append(f"fc_{i}")
        relu = nn.ReLU()
        output_activation = activations[architecture["out_activation"]]
        self.actvs = [relu]*(len(self.layer_names) - 1) + [output_activation]

    def forward(self, x):
        for (layer_name, actv) in zip(self.layer_names, self.actvs):
            layer = getattr(self, layer_name)
            x = actv(layer(x))
        x_forms = {"ivscc": x}
        return x_forms

class Enc_arbors_ivscc(nn.Module):
    def __init__(self, architecture, latent_dim, dataset):
        super().__init__()
        # Build optional arbor and ivscc subnetworks:
        (self.arbor_names, self.ivscc_names) = ([], [])
        (height, radius, process) = architecture["arbors_size"]
        (arbor_params, ivscc_hidden) = (architecture["conv_params"], architecture["ivscc_hidden"])
        arbor_out_shape = (height, radius*process)
        if arbor_params:
            (self.arbor_names, arbor_out_shape) = add_conv_segment(self, arbor_params, height, radius*process, "arbor", False)
        if ivscc_hidden:
            self.ivscc_names = add_dense_segment(self, ivscc_hidden, architecture["ivscc_size"][0], True, "ivscc")
        # Build final dense layers:
        dense_input_size = np.prod(arbor_out_shape) + (ivscc_hidden[-1] if ivscc_hidden else architecture["ivscc_size"][0])
        dense_dims = architecture["final_hidden"] + [latent_dim]
        self.dense_names = add_dense_segment(self, dense_dims, dense_input_size, False, "dense")
        # Get activation list:
        relu = nn.ReLU()
        (self.arbor_actvs, self.ivscc_actvs) = ([relu]*len(self.arbor_names), [relu]*len(self.ivscc_names))
        self.dense_actvs = [relu]*(len(self.dense_names) - 1) + [torch.nn.Identity()]
        # Create augmentation/normalization layers
        self.bn = nn.BatchNorm1d(latent_dim, momentum = 0.05, affine = False)
        self.arbor_drop = torch.nn.Dropout(architecture["arbors_dropout"])
        self.ivscc_drop = torch.nn.Dropout(architecture["ivscc_dropout"])

    def forward(self, x_forms):
        # Arbor sub-output:
        x = x_forms["arbors"]
        x = torch.flatten(x, 2).transpose(1, 2)
        x = self.arbor_drop(x)
        for (layer_name, actv) in zip(self.arbor_names, self.arbor_actvs):
            conv_layer = getattr(self, layer_name)
            x = actv(conv_layer(x))
        arbor_x = torch.flatten(x, 1)
        # IVSCC sub-output:
        x = x_forms["ivscc"]
        x = torch.nan_to_num(x)
        x = self.ivscc_drop(x)
        for (layer_name, actv) in zip(self.ivscc_names, self.ivscc_actvs):
            dense_layer = getattr(self, layer_name)
            x = actv(dense_layer(x))
        x = torch.concat([arbor_x, x], 1)
        # Dense output:
        for (layer_name, actv) in zip(self.dense_names, self.dense_actvs):
            dense_layer = getattr(self, layer_name)
            x = actv(dense_layer(x))
        z = self.bn(x)
        return z

class Dec_arbors_ivscc(nn.Module):
    def __init__(self, architecture, latent_dim, dataset):
        super().__init__()
        # Build optional arbor and ivscc subnetworks:
        (self.arbor_names, self.ivscc_names) = ([], [])
        (self.height, self.radius, self.process) = architecture["arbors_size"]
        (arbor_params, ivscc_hidden) = (architecture["conv_params"][::-1], architecture["ivscc_hidden"][::-1])
        arbor_input_shape = (self.height, self.radius*self.process)
        if arbor_params:
            (self.arbor_names, arbor_input_shape) = add_conv_segment(
                self, arbor_params, self.height, self.radius*self.process, "arbor", True)
        if ivscc_hidden:
            ivscc_dims = ivscc_hidden[1:] + architecture["ivscc_size"]
            self.ivscc_names = add_dense_segment(self, ivscc_dims, ivscc_hidden[0], True, "ivscc")
        # Build final dense layers:
        self.initial_channels = arbor_input_shape[1]
        self.arbor_flat_size = np.prod(arbor_input_shape)
        dense_output_size = self.arbor_flat_size + (ivscc_hidden[0] if ivscc_hidden else architecture["ivscc_size"][0])
        dense_dims = architecture["final_hidden"][::-1] + [dense_output_size]
        self.dense_names = add_dense_segment(self, dense_dims, latent_dim, True, "dense")
        # Get activation list:
        relu = nn.ReLU()
        identity = torch.nn.Identity()
        self.arbor_actvs = [relu]*(len(self.arbor_names) - 1)  + [identity]
        self.ivscc_actvs = [relu]*(len(self.arbor_names) - 1) + [identity]
        self.dense_actvs = [relu]*(len(self.dense_names) - 1) + [identity]
        self.dense_arbor_actv = relu if self.arbor_names else identity
        self.dense_ivscc_actv = relu if self.ivscc_names else identity
        self.arbors_out_actv = activations[architecture["arbors_activation"]]
        self.ivscc_out_actv = activations[architecture["ivscc_activation"]]

    def forward(self, x):
        # Dense intermediate:
        for (dense_name, actv) in zip(self.dense_names, self.dense_actvs):
            dense_layer = getattr(self, dense_name)
            x = actv(dense_layer(x))
        arbors_x = self.dense_arbor_actv(x[:, :self.arbor_flat_size])
        ivscc_x = self.dense_ivscc_actv(x[:, self.arbor_flat_size:])
        # Arbor output:
        arbors_x = arbors_x.view(arbors_x.shape[0], self.initial_channels, -1)
        for (layer_name, actv) in zip(self.arbor_names, self.arbor_actvs):
            conv_T_layer = getattr(self, layer_name)
            arbors_x = actv(conv_T_layer(arbors_x))
        arbors_x = arbors_x.transpose(1, 2).reshape([-1, self.height, self.radius, self.process])
        arbors_x = self.arbors_out_actv(arbors_x)
        # IVSCC sub-output:
        for (layer_name, actv) in zip(self.ivscc_names, self.ivscc_actvs):
            dense_layer = getattr(self, layer_name)
            ivscc_x = actv(dense_layer(ivscc_x))
        ivscc_x = self.ivscc_out_actv(ivscc_x)
        return {"arbors": arbors_x, "ivscc": ivscc_x}

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

class Enc_Dummy(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.dummy_latent = torch.nn.Parameter(torch.zeros([1, latent_dim]))
    
    def forward(self, x_forms):
        x_exc = next(iter(x_forms.values()))
        z = 0*self.dummy_latent + torch.randn_like(self.dummy_latent.tile((x_exc.shape[0], 1)))
        return z

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

def get_model(config, train_dataset):
    architectures = {frozenset(forms.split("_")): params for (forms, params) in config["architecture"].items()}
    model = {}
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
            arm["enc"] = modules[forms]["enc"](architecture, config["latent_dim"], train_dataset)
            arm["dec"] = modules[forms]["dec"](architecture, config["latent_dim"], train_dataset)
        model[modal] = arm
    return model
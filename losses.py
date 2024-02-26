import torch
import numpy as np
from data import get_transformation_function

def get_variances(met_data, specimens, formats, transformations, device, dtype):
    transformations = {} if transformations is None else transformations
    variances = {}
    for modal_forms in formats.values():
        for form in modal_forms:
            data = met_data.query(specimens, formats = [(form,)])[form]
            if len(data):
                if form in transformations:
                    transf_func = get_transformation_function(transformations[form])
                    data = transf_func(data)
                variances[form] = torch.from_numpy(np.nanvar(data, 0)).to(device, dtype)
    return variances

def min_var_loss(zi, zj):
    # This function computes a loss which penalizes differences
    # between the passed latent vectors (from different modalities).
    # The value is computed by taking the L2 distance between 
    # the vectors, and then dividing this value by the smallest 
    # singular value of the latent space covariance matrices 
    # (approximated using the passed batch of latent vectors). This
    # scaling helps prevent the latent spaces from collpasing into
    # the origin or into a low-dimensional subspace. 

    batch_size = zj.shape[0]
    zj_centered = zj - torch.mean(zj, 0, True)
    try: # If SVD fails, do not scale L2 distance
        min_eig = torch.min(torch.linalg.svdvals(zj_centered))
    except torch._C._LinAlgError:
        print("SVD failed.")
        min_eig = torch.as_tensor((batch_size - 1)**0.5).to(zj)
    min_var_zj = torch.square(min_eig)/(batch_size-1)
    zi_centered = zi - torch.mean(zi, 0, True)
    try: # If SVD fails, do not scale L2 distance
        min_eig = torch.min(torch.linalg.svdvals(zi_centered))
    except torch._C._LinAlgError:
        print("SVD failed.")
        min_eig = torch.as_tensor((batch_size - 1)**0.5).to(zi)
    min_var_zi = torch.square(min_eig)/(batch_size-1)
    zi_zj_mse = torch.mean(torch.sum(torch.square(zi-zj), 1))
    loss_ij = zi_zj_mse/torch.squeeze(torch.minimum(min_var_zi, min_var_zj))
    return loss_ij


class ReconstructionLoss():
    def __init__(self, config, met_data, specimens):
        self.enc_grad = config["encoder_cross_grad"]
        self.losses = {form: loss_classes[loss](config, met_data, specimens) 
                       for (form, loss) in config["losses"].items()}

    def loss(self, x_forms, xr_forms):
        loss = sum([torch.numel(x[0])*self.losses[form](x, xr_forms[form], form)
                  for (form, x) in x_forms.items()])
        loss_normed = loss / sum([torch.numel(x[0]) for x in x_forms.values()])
        return loss_normed

    def cross(self, model, x_forms, z, out_modal):
        z = (z.detach() if not self.enc_grad else z)
        xr_forms = model[out_modal]["dec"](z)
        loss = self.loss(x_forms, xr_forms)
        return loss

class MSE():
    def __init__(self, config, met_data, specimens):
        pass

    def __call__(self, x, xr, form):
        mask = ~torch.isnan(xr)
        (x_flat, xr_flat) = (torch.masked_select(x, mask), torch.masked_select(xr, mask))
        loss = torch.nn.functional.mse_loss(x_flat, xr_flat)
        return loss

class SampleR2():
    def __init__(self, config, met_data, specimens):
        variances = get_variances(met_data, specimens, config["formats"], 
                               config["transform"], config["device"], torch.float32)
        self.var_means = {form: torch.nanmean(var) for (form, var) in variances.items()}

    def __call__(self, x, xr, form):
        (x_flat, xr_flat) = (torch.flatten(x, 1), torch.flatten(xr, 1))
        mask = ~torch.isnan(x_flat)
        (x_flat, xr_flat) = (torch.masked_select(x_flat, mask), torch.masked_select(xr_flat, mask))
        mse = torch.square(x_flat - xr_flat).mean()
        loss_ratio = mse / self.var_means[form]
        return loss_ratio

class FeatureR2():
    def __init__(self, config, met_data, specimens):
        self.variances = get_variances(met_data, specimens, config["formats"], 
                                       config["transform"], config["device"], torch.float32)
        
    def __call__(self, x, xr, form):
        sample_counts = torch.count_nonzero(~torch.isnan(x), 0)
        feature_mask = sample_counts > 0
        squares_unnorm = torch.square(torch.nan_to_num(x) - xr).sum(0)
        mean_squares = squares_unnorm[feature_mask] / sample_counts[feature_mask]
        r2_error = torch.mean(mean_squares / self.variances[form][feature_mask])
        return r2_error
    
class CrossEntropy():
    def __init__(self, config, met_data, specimens):
        pass

    def __call__(self, x, xr, form):
        mask = ~torch.isnan(xr)
        (x_flat, xr_flat) = (torch.masked_select(x, mask), torch.masked_select(xr, mask))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xr_flat, x_flat)
        return loss

loss_classes = {"mse": MSE, "feature_r2": FeatureR2, "sample_r2": SampleR2, "bce": CrossEntropy}
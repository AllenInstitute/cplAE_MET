import math

import torch
import numpy as np
from data import get_transformation_function

def apply_mask(dct, mask):
    masked = {key: value[mask] for (key, value) in dct.items()}
    return masked

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
    min_eig = torch.min(torch.linalg.svdvals(zj_centered))
    min_var_zj = torch.square(min_eig)/(batch_size-1)
    zi_centered = zi - torch.mean(zi, 0, True)
    min_eig = torch.min(torch.linalg.svdvals(zi_centered))
    min_var_zi = torch.square(min_eig)/(batch_size-1)
    zi_zj_mse = torch.mean(torch.sum(torch.square(zi-zj), 1))
    loss_ij = zi_zj_mse/torch.squeeze(torch.minimum(min_var_zi, min_var_zj))
    return loss_ij

class VariationalLoss():
    def __init__(self, config, met_data, specimens):
        self.config = config
    
    def process_batch(self, model, X_dict, mask_dict):
        (latent_dict, mapper_dict, loss_dict, coupling_dict) = ({}, {}, {}, {})
        for modal in self.config["modalities"]:
            (arm, x_forms, mask) = (model[modal], X_dict[modal], mask_dict[modal])
            x_masked = apply_mask(x_forms, mask)
            (z_mean, z_transf) = arm["enc"](x_masked)
            latent_dict[modal] = (z_mean, z_transf)
            loss_dict[modal] = self.get_within_loss(model, modal, x_masked, z_mean, z_transf, self.config["samples"])
            for (prev_modal, (prev_mean, prev_transf)) in list(latent_dict.items())[:-1]:
                (_, cross_sample, cross_mean, cross_transf) = model.cross_z_sample(modal, prev_modal, z_mean, z_transf, self.config["samples"])
                (_, prev_cross_sample, prev_cross_mean, prev_cross_transf) = model.cross_z_sample(prev_modal, modal, prev_mean, prev_transf, self.config["samples"])
                mapper_dict[f"{prev_modal}={modal}"] = (prev_cross_mean, prev_cross_transf)
                mapper_dict[f"{modal}={prev_modal}"] = (cross_mean, cross_transf)
                (prev_x_forms, prev_mask) = (X_dict[prev_modal], mask_dict[prev_modal])
                if torch.any(prev_mask[mask]):
                    (cross_masked, prev_cross_masked) = (cross_sample[prev_mask[mask]], prev_cross_sample[mask[prev_mask]])
                    (x_dbl_masked, prev_x_masked) = (apply_mask(x_forms, mask & prev_mask), apply_mask(prev_x_forms, mask & prev_mask))
                    cross_loss = self.get_cross_loss(model, prev_modal, prev_x_masked, cross_masked)
                    prev_cross_loss = self.get_cross_loss(model, modal, x_dbl_masked, prev_cross_masked)
                    loss_dict[f"{modal}={prev_modal}"] = cross_loss
                    loss_dict[f"{prev_modal}={modal}"] = prev_cross_loss
                    coupling_dict[f"{modal}={prev_modal}"] = torch.square(z_mean[prev_mask[mask]] - prev_mean[mask[prev_mask]].detach()).mean(0).sum()
                    coupling_dict[f"{prev_modal}={modal}"] = torch.square(z_mean[prev_mask[mask]].detach() - prev_mean[mask[prev_mask]]).mean(0).sum()
        total_loss = self.combine_losses(loss_dict, latent_dict, mapper_dict, coupling_dict)
        return (loss_dict, total_loss)

    def reconstruction_loss(self, x_forms, xr_forms):
        loss = 0
        for (form, x) in x_forms.items():
            mask = ~torch.isnan(x)
            x = torch.nan_to_num(x)
            x_recon = xr_forms[form]
            squared_diff = torch.square(x[:, None] - x_recon)
            mse = torch.masked_select(squared_diff, mask[:, None]).sum() / (x_recon.shape[0]*x_recon.shape[1])
            loss = loss + mse
        return loss
    
    def get_within_loss(self, model, modal, x_forms, z_mean, z_transf, num_samples):
        z_sample = model.z_sample(z_mean, z_transf, num_samples)
        xr_forms_flat = model[modal]["dec"](z_sample.flatten(0, 1))
        xr_forms = {form: tensor.unflatten(0, z_sample.shape[:2]) 
                    for (form, tensor) in xr_forms_flat.items()}
        loss = self.reconstruction_loss(x_forms, xr_forms)
        return loss
    
    def get_cross_loss(self, model, out_modal, x_forms, cross_sample):
        xr_forms_flat = model[out_modal]["dec"](cross_sample.flatten(0, 1))
        xr_forms = {form: tensor.unflatten(0, cross_sample.shape[:2]) 
                    for (form, tensor) in xr_forms_flat.items()}
        loss = self.reconstruction_loss(x_forms, xr_forms)
        return loss

    def combine_losses(self, loss_dict, latent_dict, mapper_dict, coupling_dict):
        var_config = self.config["var_weights"]
        total_loss = 0
        for (i, modal_1) in enumerate(self.config["modalities"]):
            (mean_1, transf_1) = latent_dict[modal_1]
            losses = {
                "within": var_config["recon_scale"]*var_config["duplicate"]*loss_dict[modal_1],
                "mean_reg": var_config["duplicate"]*torch.square(mean_1).sum(1).mean(),
                "trace_reg": transf_1.square().mean(0).sum(),
                "det_reg": var_config["duplicate"]*-2*torch.log(torch.det(transf_1)).mean()}
            total_loss += sum([var_config[modal_1][key]*loss 
                               for (key, loss) in losses.items()])
            for modal_2 in self.config["modalities"][i + 1:]:
                for (first, second) in [(modal_1, modal_2), (modal_2, modal_1)]:
                    (map_mean, map_transf) = mapper_dict[f"{first}={second}"]
                    orig_mean = latent_dict[first][0]
                    losses = {
                        "cross": var_config["recon_scale"]*loss_dict[f"{first}={second}"],
                        "mean_diff_reg": torch.square(map_mean - orig_mean).sum(1).mean(),
                        "map_trace_reg": map_transf.square().mean(0).sum(),
                        "map_det_reg": -2*torch.log(torch.det(map_transf)).mean(),
                        "coupling": coupling_dict[f"{first}={second}"]}
                    total_loss += sum([var_config[first][second][key]*loss for (key, loss) in losses.items()])
        return total_loss

class ReconstructionLoss():
    def __init__(self, config, met_data, specimens):
        self.config = config
        self.enc_grad = config["encoder_cross_grad"]
        self.losses = {form: loss_classes[loss](config, met_data, specimens) 
                       for (form, loss) in config["losses"].items()}
        
    def process_batch(self, model, X_dict, mask_dict):
        # This function processes a single batch during model optimization. It takes as
        # argument the target model, a dictionary of data from different modalities, a
        # dictionary of masks specifying which samples hold valid data for each modality,
        # and the experiment configuration dictionary. For each modality, the latent space
        # and reconstruction are calculated, along with the the self-modal R2 loss. The function
        # then iterates through any previous modalities and computes the latent space coupling loss
        # and the cross-modal R2 loss. The modality masks are combined in order to select data for
        # pairs of modalities.

        (latent_dict, recon_dict, loss_dict) = ({}, {}, {})
        for modal in self.config["modalities"]:
            (arm, x_forms, mask) = (model[modal], X_dict[modal], mask_dict[modal])
            x_masked = apply_mask(x_forms, mask)
            z = arm["enc"](x_masked)[0]
            xr_forms = arm["dec"](z)
            (latent_dict[modal], recon_dict[modal]) = (z, xr_forms)
            loss_dict[modal] = self.loss(x_masked, xr_forms)
            for (prev_modal, prev_z) in list(latent_dict.items())[:-1]:
                (prev_x_forms, prev_mask) = (X_dict[prev_modal], mask_dict[prev_modal])
                if torch.any(prev_mask[mask]):
                    (z_masked, prev_z_masked) = (z[prev_mask[mask]], prev_z[mask[prev_mask]])
                    (x_dbl_masked, prev_x_masked) = (apply_mask(x_forms, mask & prev_mask), apply_mask(prev_x_forms, mask & prev_mask))
                    if self.config["weights"][f"{prev_modal}-{modal}"] > 0:
                        loss_dict[f"{prev_modal}-{modal}"] = min_var_loss(z_masked, prev_z_masked.detach())
                    if self.config["weights"][f"{modal}-{prev_modal}"] > 0:
                        loss_dict[f"{modal}-{prev_modal}"] = min_var_loss(z_masked.detach(), prev_z_masked)
                    loss_dict[f"{modal}={prev_modal}"] = self.cross(model, prev_x_masked, z_masked, prev_modal)
                    loss_dict[f"{prev_modal}={modal}"] = self.cross(model, x_dbl_masked, prev_z_masked, modal)
        weighted = sum([self.config["weights"][key]*loss_value for (key, loss_value) in loss_dict.items()])
        return (loss_dict, weighted)

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
        mask = ~torch.isnan(x)
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
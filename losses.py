import math
import itertools
import functools

import torch
import numpy as np
from data import get_transformation_function

def powerset(iterable, min_size = 0):
    elements = list(iterable)
    comb_gen = (itertools.combinations(elements, r) for r in range(min_size, len(elements) + 1))
    return itertools.chain.from_iterable(comb_gen)

def apply_mask(dct, mask):
    masked = {key: value[mask] for (key, value) in dct.items()}
    return masked

def get_variances(met_data, specimens, formats, transformations, device, dtype):
    transformations = {} if transformations is None else transformations
    variances = {}
    for modal_forms in formats.values():
        for form in modal_forms:
            spec_ids = met_data.query(specimens, formats = [(form,)], outputs = ["specimen_id"])["specimen_id"]
            data = met_data.get_specimens(spec_ids[:1000], outputs = [form])[form]
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

def get_indices(modalities):
        modal_sets = list(powerset(modalities, min_size = 2))
        modal_indices = {modal: i for (i, modal) in enumerate(modalities)}
        set_indices = {frozenset(modal_set): [modal_indices[modal] for modal in modal_set]
                       for modal_set in modal_sets}
        return (modal_indices, set_indices)

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
        (weighted_loss_dict, total_loss) = self.combine_losses(loss_dict, latent_dict, mapper_dict, coupling_dict)
        return (weighted_loss_dict, total_loss)

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
        weighted_loss_dict = {}
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
            weighted_loss_dict = {
                **weighted_loss_dict, 
                **{f"{modal_1}_{key}": var_config[modal_1][key]*loss for (key, loss) in losses.items()}}
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
                    weighted_loss_dict = {
                        **weighted_loss_dict,
                        **{f"{first}-{second}_{key}": var_config[first][second][key]*loss for (key, loss) in losses.items()}}
        return (weighted_loss_dict, total_loss)
    
    def log(self, tb_writer, train_loss, val_loss, epoch):
        # This function takes the training/validation losses and logs them
        # in Tensoboard. The component losses are reported without any scaling,
        # alongside the weighted sum of the losses.

        tb_writer.add_scalars("Weighted Loss", {"Train": train_loss["total"], "Validation": val_loss["total"]}, epoch)
        tb_writer.add_scalars("MSE/Train", 
            {key: loss for (key, loss) in train_loss.items() if "within" in key}, epoch)
        tb_writer.add_scalars("MSE/Validation", 
            {key: loss for (key, loss) in val_loss.items() if "within" in key}, epoch)
        tb_writer.add_scalars("Cross-MSE/Train", 
            {key: loss for (key, loss) in train_loss.items() if "cross" in key}, epoch)
        tb_writer.add_scalars("Cross-MSE/Validation", 
            {key: loss for (key, loss) in val_loss.items() if "cross" in key}, epoch)
        tb_writer.add_scalars("Within-Reg/Train", 
            {key: loss for (key, loss) in train_loss.items() if not ("cross" in key) and not ("within" in key) and not ("total" in key)}, epoch)
        tb_writer.add_scalars("Within-Reg/Validation",
            {key: loss for (key, loss) in val_loss.items() if not ("cross" in key) and not ("within" in key) and not ("total" in key)}, epoch)

class ELBO_Loss():
    def __init__(self, config, met_data, specimens):
        self.config = config
        (self.modal_indices, self.set_indices) = get_indices(config["modalities"])

    # def get_operators(self):
    #     num_modalities = len(self.joint_cov)
    #     mean_diff = torch.zeros([len(self.indices), num_modalities, num_modalities])
    #     cond_prec = torch.zeros([len(self.indices), num_modalities])
    #     cuml_log_var = 0
    #     for (in_indices, out_indices, dest_indices) in self.indices.values():
    #         inv_cov = torch.linalg.inv(self.joint_cov[in_indices][:, in_indices])
    #         off_block = self.joint_cov[out_indices][:, in_indices]
    #         transf = off_block @ inv_cov
    #         cond_var = self.joint_cov[out_indices, out_indices] - transf @ off_block.T
    #         mean_diff[dest_indices, out_indices, in_indices] = transf / cond_var**0.5
    #         cond_prec[dest_indices, out_indices] = 1 / cond_prec
    #         cuml_log_var = cuml_log_var + torch.log(cond_var).sum()
    #     return (mean_diff, cond_prec, cuml_log_var)

    def process_batch(self, model, X_dict, mask_dict):
        (recon_dict, entropy_dict, reg_dict, alignment_dict) = ({}, {}, {}, {})
        num_samples = len(next(iter(mask_dict.values())))
        num_modalities = len(self.config["modalities"])
        latent_dim = self.config["latent_dim"]
        latent_tensor = torch.zeros([num_samples, num_modalities, latent_dim], device = self.config["device"])
        cov_tensor = torch.zeros([num_samples, num_modalities, latent_dim, latent_dim], device = self.config["device"])
        joint_cov = model.decoder_cov()
        for (modal, modal_index) in self.modal_indices.items():
            (arm, x_forms, mask) = (model[modal], X_dict[modal], mask_dict[modal])
            x_masked = apply_mask(x_forms, mask)
            (z_mean, z_transf) = arm["enc"](x_masked)
            recon_dict[f"{modal}_recon"] = self.get_within_loss(model, modal, x_masked, z_mean, z_transf, self.config["samples"])
            (sign, logdet) = torch.linalg.slogdet(z_transf)
            entropy_dict[f"{modal}_entropy"] = torch.mean(sign*logdet)
            z_covs = torch.einsum("nij,nkj->nik", z_transf, z_transf)
            reg_dict[f"{modal}_reg"] = self.compute_unimodal_reg(z_mean, z_covs, joint_cov[modal_index, :, modal_index])
            latent_tensor[mask, modal_index] = z_mean
            cov_tensor[mask, modal_index] = z_covs
        for (modal_set, set_indices) in self.set_indices.items():
            joint_mask = functools.reduce(torch.logical_and, (mask_dict[modal] for modal in modal_set))
            latent_masked = latent_tensor[joint_mask]
            latent_cov_masked = cov_tensor[joint_mask]
            cuml = 0
            alignment = 0
            for out_index in set_indices:
                in_indices  = torch.as_tensor([i for i in set_indices if i != out_index])
                (mean_transf, off_diag) = self.get_mean_transform(joint_cov, in_indices, out_index)
                inv_cond_cov = self.get_inv_conditional_cov(joint_cov[out_index, :, out_index], mean_transf, off_diag)
                (new_cuml, new_alignment) = self.compute_multimodal_reg(latent_masked[:, out_index], latent_masked[:, in_indices], 
                                                                        latent_cov_masked[:, out_index], mean_transf, inv_cond_cov)
                cuml = cuml + new_cuml
                alignment = alignment + torch.linalg.norm(new_alignment, dim = 1).mean()
            set_size = len(set_indices)
            reg_dict["-".join(modal_set) + "_reg"] = math.factorial(num_modalities  - set_size)*math.factorial(set_size - 1)*cuml
            alignment_dict["-".join(modal_set) + "_align"] = alignment
        comb_loss = self.combine_losses(num_modalities, recon_dict, entropy_dict, reg_dict)
        loss_dict = {**recon_dict, **entropy_dict, **reg_dict, **alignment_dict}
        return (loss_dict, comb_loss)
        
    def compute_unimodal_reg(self, z_means, z_cov, marg_cov):
        inv_marg_cov = torch.linalg.inv(marg_cov)
        (sign, logdet) = torch.linalg.slogdet(marg_cov)
        trace = torch.einsum("ij,nji", inv_marg_cov, z_cov)
        mean_norm = torch.einsum("ni,ij,nj->n", z_means, inv_marg_cov, z_means)
        reg_loss = 0.5*(sign*logdet + trace + mean_norm).mean()
        return reg_loss
    
    def compute_multimodal_reg(self, z_out_means, z_in_means, z_cov, mean_transf, inv_cond_cov):
        (sign, logdet) = torch.linalg.slogdet(inv_cond_cov)
        trace = torch.einsum("ij,nji->n", inv_cond_cov, z_cov)
        mean_diff = z_out_means - torch.einsum("ijk,njk->ni", mean_transf, z_in_means)
        coupling = torch.einsum("ni,ij,nj->n", mean_diff, inv_cond_cov, mean_diff)
        reg_loss = 0.5*(-sign*logdet + trace + coupling).mean()
        return (reg_loss, mean_diff)
    
    def get_inv_conditional_cov(self, out_marg, mean_transf, off_diag):
        cond = out_marg - torch.einsum("ijk,ojk->io", mean_transf, off_diag)
        inv_cond = torch.linalg.inv(cond)
        return inv_cond
    
    def get_mean_transform(self, joint_cov, in_indices, out_index):
        z_size = len(in_indices)*joint_cov.shape[1]
        in_marg = joint_cov[in_indices][:, :, in_indices]
        inv_in_marg = torch.linalg.inv(in_marg.reshape((z_size, z_size))).reshape(in_marg.shape)
        off_diag = joint_cov[out_index, :, in_indices]
        mean_transf = torch.einsum("ijk,jklm->ilm", off_diag, inv_in_marg)
        return (mean_transf, off_diag)

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
    
    def combine_losses(self, num_modalities, recon_dict, entropy_dict, reg_dict):
        weights = self.config["elbo_weights"]
        recon_sum = sum([weights[key]*loss for (key, loss) in recon_dict.items()])
        comb_loss = weights["recon_scale"]*recon_sum - sum(entropy_dict.values()) + sum(reg_dict.values()) / math.factorial(num_modalities)
        return comb_loss

    def log(self, tb_writer, train_loss, val_loss, epoch):
        # This function takes the training/validation losses and logs them
        # in Tensoboard. The component losses are reported without any scaling,
        # alongside the weighted sum of the losses.

        weights = self.config["elbo_weights"]
        tb_writer.add_scalars("Alignment/Train", 
            {key: alignment for (key, alignment) in train_loss.items() if "align" in key}, epoch)
        tb_writer.add_scalars("Alignment/Validation", 
            {key: alignment for (key, alignment) in val_loss.items() if "align" in key}, epoch)
        tb_writer.add_scalars("Weighted Loss", {"Train": train_loss["total"], "Validation": val_loss["total"]}, epoch)
        tb_writer.add_scalars("Reconstruction/Train", 
            {key: loss*weights[key] for (key, loss) in train_loss.items() if "recon" in key}, epoch)
        tb_writer.add_scalars("Reconstruction/Validation", 
            {key: loss*weights[key] for (key, loss) in val_loss.items() if "recon" in key}, epoch)
        tb_writer.add_scalars("Entropy/Train", 
            {key: loss for (key, loss) in train_loss.items() if "entropy" in key}, epoch)
        tb_writer.add_scalars("Entropy/Validation",
            {key: loss for (key, loss) in val_loss.items() if "entropy" in key}, epoch)
        tb_writer.add_scalars("Reg/Train", 
            {key: loss for (key, loss) in train_loss.items() if "reg" in key}, epoch)
        tb_writer.add_scalars("Reg/Validation",
            {key: loss for (key, loss) in val_loss.items() if "reg" in key}, epoch)

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
    
    def log(self, tb_writer, train_loss, val_loss, epoch):
        # This function takes the training/validation losses and logs them
        # in Tensoboard. The component losses are reported without any scaling,
        # alongside the weighted sum of the losses.

        tb_writer.add_scalars("Weighted Loss", {"Train": train_loss["total"], "Validation": val_loss["total"]}, epoch)
        tb_writer.add_scalars("R2/Train", 
            {key: 1 - value for (key, value) in train_loss.items() if key in {"T", "E", "M"}}, epoch)
        tb_writer.add_scalars("R2/Validation", 
            {key: 1 - value for (key, value) in val_loss.items() if key in {"T", "E", "M"}}, epoch)
        tb_writer.add_scalars("Cross-R2/Train", 
            {key: 1 - value for (key, value) in train_loss.items() if "=" in key}, epoch)
        tb_writer.add_scalars("Cross-R2/Validation", 
            {key: 1 - value for (key, value) in val_loss.items() if "=" in key}, epoch)
        tb_writer.add_scalars("Coupling/Train", 
            {key: value for (key, value) in train_loss.items() if "-" in key}, epoch)
        tb_writer.add_scalars("Coupling/Validation",
            {key:value for (key, value) in val_loss.items() if "-" in key}, epoch)

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
    
class BinaryCrossEntropy():
    def __init__(self, config, met_data, specimens):
        pass

    def __call__(self, x, xr, form):
        mask = ~torch.isnan(xr)
        (x_flat, xr_flat) = (torch.masked_select(x, mask), torch.masked_select(xr, mask))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xr_flat, x_flat)
        return loss

class CrossEntropy():
    def __init__(self, config, met_data, specimens):
        pass

    def __call__(self, x, xr, form):
        loss = torch.nn.functional.cross_entropy(xr.flatten(0, -2), x.flatten(0, -1).long())
        return loss

loss_classes = {"mse": MSE, "feature_r2": FeatureR2, "sample_r2": SampleR2, "bce": BinaryCrossEntropy, "ce": CrossEntropy}
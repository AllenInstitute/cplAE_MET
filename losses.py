import itertools

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
    for modal_forms in formats:
        for form in modal_forms:
            spec_ids = met_data.query(specimens, formats = [(form,)], outputs = ["specimen_id"])["specimen_id"]
            np.random.RandomState(42).shuffle(spec_ids)
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

class ContrastiveLoss():
    def __init__(self, config, met_data, specimens):
        self.config = config
        self.temperature = config["contrastive"]["temperature"]
        proj_head = {modal: torch.nn.Sequential(
            torch.nn.Linear(config["latent_dim"], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, config["contrastive"]["loss_dim"])
        ) for modal in config["modalities"]}
        self.proj_head = torch.nn.ModuleDict(proj_head)

    def process_batch(self, model, X_dict, mask_dict):
        (latent_dict, loss_dict) = ({}, {})
        for modal in self.config["modalities"]:
            (arm, x_forms, mask) = (model[modal], X_dict[modal], mask_dict[modal])
            x_masked = apply_mask(x_forms, mask)
            (x_rep, _) = arm["enc"](x_masked)
            z = x_rep #self.proj_head[modal](x_rep)
            latent_dict[modal] = z
            z_mag = torch.linalg.norm(z, dim = 1)
            # overlap = torch.einsum("ni,si->ns", z, z)
            # sim = overlap / (z_mag[:, None]*z_mag[None])
            # unpaired = torch.logsumexp(sim.fill_diagonal_(-1e20) / self.temperature, dim = 1).mean()
            # loss_dict[f"{modal}-{modal}"] = unpaired
            for (prev_modal, prev_z) in list(latent_dict.items())[:-1]:
                prev_z_mag = torch.linalg.norm(prev_z, dim = 1)
                overlap = torch.einsum("ni,si->ns", z, prev_z)
                sim = overlap / (z_mag[:, None]*prev_z_mag[None])
                paired = torch.mean(torch.diagonal(sim) / self.temperature)
                unpaired = torch.logsumexp(sim.fill_diagonal_(-1e20) / self.temperature, dim = 1).mean()
                loss_dict[f"{modal}-{prev_modal}"] = unpaired - paired
        total_loss = sum([self.config["contrastive"][key]*loss_value for (key, loss_value) in loss_dict.items()])
        return (loss_dict, total_loss)

    def log(self, tb_writer, train_loss, val_loss, epoch):
        # This function takes the training/validation losses and logs them
        # in Tensoboard. The component losses are reported without any scaling,
        # alongside the weighted sum of the losses.

        tb_writer.add_scalars("Weighted Loss", {"Train": train_loss["total"], "Validation": val_loss["total"]}, epoch)
        tb_writer.add_scalars("Contrast/Train", 
            {key: loss for (key, loss) in train_loss.items()}, epoch)
        tb_writer.add_scalars("Contrast/Validation", 
            {key: loss for (key, loss) in val_loss.items()}, epoch)

class VariationalLoss():
    def __init__(self, config, met_data, specimens):
        self.config = config
        self.recon_loss_funcs = {form: loss_classes[loss](config, met_data, specimens) 
                                 for (form, loss) in config["losses"].items()}
    
    def process_batch(self, model, X_dict, mask_dict, labels):
        (latent_dict, mapper_dict, loss_dict, coupling_dict) = ({}, {}, {}, {})
        for modal in self.config["modalities"]:
            (arm, x_forms, mask) = (model[modal], X_dict[modal], mask_dict[modal])
            x_masked = apply_mask(x_forms, mask)
            (z_mean, z_transf) = arm["enc"](x_masked)
            latent_dict[modal] = (z_mean, z_transf)
            loss_dict[modal] = self.get_within_loss(model, modal, x_masked, z_mean, z_transf, self.config["samples"])
            loss_dict[f"{modal}_pred"] = self.get_prediction_loss(model, modal, z_mean, labels[mask])
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
            x_recon = xr_forms[form]
            x = x[:, None].expand(x.shape[:1] + (-1,) + x.shape[1:])
            loss_func = self.recon_loss_funcs[form]
            loss = loss + loss_func(x.flatten(0, 1), x_recon.flatten(0, 1), form)
        return loss
    
    def get_prediction_loss(self, model, modal, z_mean, labels):
        is_labeled = (labels >= 0)
        log_probs = model.classifiers[modal](z_mean)
        loss = torch.nn.functional.cross_entropy(log_probs[is_labeled], labels[is_labeled])
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
                "predict": var_config["pred_scale"]*loss_dict[f"{modal_1}_pred"],
                "mean_reg": var_config["reg_scale"]*var_config["duplicate"]*torch.square(mean_1).sum(1).mean(),
                "trace_reg": var_config["reg_scale"]*transf_1.square().mean(0).sum(),
                "det_reg": var_config["reg_scale"]*var_config["duplicate"]*-2*torch.log(torch.diagonal(transf_1, 0, -2, -1)).sum(-1).mean()}
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
                        "mean_diff_reg": var_config["reg_scale"]*torch.square(map_mean - orig_mean).sum(1).mean(),
                        "map_trace_reg": var_config["reg_scale"]*map_transf.square().mean(0).sum(),
                        "map_det_reg": var_config["reg_scale"]*-2*torch.log(torch.diagonal(map_transf, 0, -2, -1)).sum(-1).mean(),
                        "coupling": var_config["couple_scale"]*coupling_dict[f"{first}={second}"]}
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
        tb_writer.add_scalars("Predict/Train", 
            {key: loss for (key, loss) in train_loss.items() if "predict" in key}, epoch)
        tb_writer.add_scalars("Predict/Validation", 
            {key: loss for (key, loss) in val_loss.items() if "predict" in key}, epoch)
        tb_writer.add_scalars("Cross-MSE/Train", 
            {key: loss for (key, loss) in train_loss.items() if "cross" in key}, epoch)
        tb_writer.add_scalars("Cross-MSE/Validation", 
            {key: loss for (key, loss) in val_loss.items() if "cross" in key}, epoch)
        tb_writer.add_scalars("Within-Reg/Train", 
            {key: loss for (key, loss) in train_loss.items() 
             if not ("cross" in key) and not ("within" in key) and not ("total" in key) and not ("predict" in key)}, epoch)
        tb_writer.add_scalars("Within-Reg/Validation",
            {key: loss for (key, loss) in val_loss.items() 
             if not ("cross" in key) and not ("within" in key) and not ("total" in key) and not ("predict" in key)}, epoch)

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
        x = torch.nan_to_num(x)
        squared_diff = torch.square(x - xr)
        mse = torch.masked_select(squared_diff, mask).sum() / xr.shape[0]
        return mse

class SampleR2():
    def __init__(self, config, met_data, specimens):
        active_forms = [config["formats"][modal] for modal in config["modalities"]]
        variances = get_variances(met_data, specimens, active_forms, 
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
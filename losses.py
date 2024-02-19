import torch
import numpy as np
from data import get_transformation_function

def get_variances(met_data, specimens, modalities, transformations, device, dtype):
    transformations = {} if transformations is None else transformations
    variances = {}
    for modal in modalities:
        data = met_data.query(specimens, [modal])[f"{modal}_dat"]
        if modal in transformations:
            transf_func = get_transformation_function(transformations[modal])
            data = transf_func(data)
        variances[modal] = torch.from_numpy(np.var(data, 0)).to(device, dtype)
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

    def loss(self, x, xr, modal):
        pass

    def cross(self, model, x, z, out_modal):
        z = (z.detach() if not self.enc_grad else z)
        xr = model[out_modal]["dec"](z)
        loss = self.loss(x, xr, out_modal)
        return loss
    
class MSE(ReconstructionLoss):
    def __init__(self, config, met_data, specimens):
        super().__init__(config, met_data, specimens)
        self.variances = get_variances(met_data, specimens, config["modalities"], 
                                       config["transform"], config["device"], torch.float32)
    
    def loss(self, x, xr, modal):
        squares = torch.square(x - xr).mean()
        variance = self.variances[modal]
        mean_squared_error = squares / variance.mean()
        return mean_squared_error

class R2(ReconstructionLoss):
    def __init__(self, config, met_data, specimens):
        super().__init__(config, met_data, specimens)
        self.variances = get_variances(met_data, specimens, config["modalities"], 
                                       config["transform"], config["device"], torch.float32)

    def loss(self, x, xr, modal):
        squares = torch.square(x - xr).mean(0)
        variance = self.variances[modal]
        r2_error = (squares / variance).mean()
        return r2_error
    
class CrossEntropy(ReconstructionLoss):
    def __init__(self, config, met_data, specimens):
        super().__init__(config, met_data, specimens)

    def loss(self, x, xr, modal):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xr, x)
        return loss

loss_classes = {"r2": R2, "mse": MSE, "bce": CrossEntropy}
import torch
import numpy as np

def get_variances(dataset, modalities, device, dtype):
    variances = {}
    for modal in modalities:
        data = dataset.MET.query(dataset.allowed_specimen_ids, [modal])[f"{modal}_dat"]
        variances[modal] = torch.from_numpy(np.var(data, 0)).to(device, dtype)
    return variances

class MSE():
    def __init__(self, config, dataset):
        self.enc_grad = config["encoder_cross_grad"]
        self.variances = get_variances(dataset, ["T", "E", "M"], config["device"], torch.float32)
    
    def loss(self, x, xr, modal):
        squares = torch.square(x - xr).mean()
        variance = self.variances[modal]
        mean_squared_error = squares / variance.mean()
        return mean_squared_error
    
    def cross_loss(self, model, x, z, out_modal):
        z = (z.detach() if not self.enc_grad else z)
        xr = model.modal_arms[out_modal].decoder(z)
        loss = self.loss(x, xr, out_modal)
        return loss

class R2():
    def __init__(self, config, dataset):
        self.enc_grad = config["encoder_cross_grad"]
        self.variances = get_variances(dataset, ["T", "E", "M"], config["device"], torch.float32)

    def loss(self, x, xr, modal):
        squares = torch.square(x - xr).mean(0)
        variance = self.variances[modal]
        r2_error = (squares / variance).mean()
        return r2_error
    
    def cross_loss(self, model, x, z, out_modal):
        z = (z.detach() if not self.enc_grad else z)
        xr = model.modal_arms[out_modal].decoder(z)
        loss = self.loss(x, xr, out_modal)
        return loss

loss_classes = {"r2": R2, "mse": MSE}
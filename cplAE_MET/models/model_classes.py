import torch.nn as nn
from cplAE_MET.models.subnetworks_T import AE_T
from cplAE_MET.models.subnetworks_ME import AE_ME
from cplAE_MET.models.subnetworks_E import AE_E
from cplAE_MET.models.subnetworks_M import AE_M

modal_classes = {
    "T": AE_T,
    "E": AE_E,
    "M": AE_M,
    "EM": AE_ME
}

class MultiModal(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.modal_arms = {}
        for modal in model_config["modalities"]:
            arm = modal_classes[modal](model_config)
            self.modal_arms[modal] = arm
            setattr(self, modal, arm)
        self.variational = False

    def forward(self, X, in_modal, out_modals):
        latent = self.modal_arms[in_modal].encoder(X)
        outputs = []
        for modal in out_modals:
            reconstruct = self.modal_arms[modal].decoder(latent)
            outputs.append(reconstruct)
        return (latent, outputs)

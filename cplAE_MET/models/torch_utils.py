import numpy as np
import torch
from torch.utils.data import Dataset


def tensor(x, device):
    return torch.tensor(x).to(dtype=torch.float32).to(device)


def astensor(x, device):
    return torch.as_tensor(x).to(dtype=torch.float32).to(device)


def boolastensor(x, device):
    return torch.as_tensor(x).to(dtype=torch.bool).to(device)


def tonumpy(x):
    return x.cpu().detach().numpy()


def min_var_loss(zi, zj):
    #SVD calculated over all entries in the batch
    batch_size = zj.shape[0]

    zj_centered = zj - torch.mean(zj, 0, True)
    min_eig = torch.min(torch.linalg.svdvals(zj_centered))
    min_var_zj = torch.square(min_eig)/(batch_size-1)

    zi_centered = zi - torch.mean(zi, 0, True)
    min_eig = torch.min(torch.linalg.svdvals(zi_centered))
    min_var_zi = torch.square(min_eig)/(batch_size-1)

    #Wij_paired is the weight of matched pairs
    zi_zj_mse = torch.mean(torch.sum(torch.square(zi-zj), 1))
    loss_ij = zi_zj_mse/torch.squeeze(torch.minimum(min_var_zi, min_var_zj))
    return loss_ij


class MET_dataset(Dataset):
    """Patchseq MET dataset"""

    def __init__(self, met_dataclass_obj, device=None):
        """
        Args:
            met_dataclass_obj (obj): object of `met_dataclass` - see `cplAE_MET.utils.dataclass`
            batch_size (int): number of samples in a batch
            device (torch.device): device to send batch to
        """

        if device is None:
            self.device = 'cpu'
        else:
            self.device = device

        self.xt = met_dataclass_obj.XT
        self.xe = met_dataclass_obj.XE
        self.xm = np.expand_dims(met_dataclass_obj.XM_centered, axis=1)
        self.xsd = met_dataclass_obj.Xsd
        self.astensor = lambda x: torch.as_tensor(x).float().to(self.device)
        self.gnoise_std = torch.var(torch.nan_to_num(self.astensor(self.xe)), dim=0, keepdim=True).sqrt()

        self.is_m_1d = met_dataclass_obj.isM_1d
        self.is_e_1d = met_dataclass_obj.isE_1d
        self.is_t_1d = met_dataclass_obj.isT_1d
        self.is_me_1d = np.logical_and(self.is_m_1d, self.is_e_1d)
        self.is_met_1d = np.logical_and(self.is_me_1d, self.is_t_1d)

    def __len__(self):
        return self.xt.shape[0]

    def __getitem__(self, idx):
        xm = self.astensor(self.xm[idx, ...])
        xsd = self.astensor(self.xsd[idx, ...])
        valid_xm = ~torch.isnan(xm)
        valid_xsd = ~torch.isnan(xsd)
        xm = torch.nan_to_num(xm)
        xsd = torch.nan_to_num(xsd)

        xe = self.astensor(self.xe[idx, ...])
        valid_xe = ~torch.isnan(xe)
        xe = torch.nan_to_num(xe).float()

        xt = self.astensor(self.xt[idx, ...])
        valid_xt = ~torch.isnan(xt)
        xt = torch.nan_to_num(xt).float()

        is_m_1d = self.is_m_1d[idx]
        is_e_1d = self.is_e_1d[idx]
        is_t_1d = self.is_t_1d[idx]
        is_me_1d = self.is_me_1d[idx]
        is_met_1d = self.is_met_1d[idx]

        return dict(xm=xm, xsd=xsd, xe=xe, xt=xt,
                    valid_xm=valid_xm, valid_xsd=valid_xsd,
                    valid_xe=valid_xe, valid_xt=valid_xt,
                    is_m_1d=is_m_1d, is_e_1d=is_e_1d, is_t_1d=is_t_1d,
                    is_me_1d=is_me_1d, is_met_1d=is_met_1d)

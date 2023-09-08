import numpy as np
import torch
import torch.nn.functional as F
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


def add_noise(x, sd=0.1, clamp_min=None, clamp_max=None, scale_by_x=False):
    """
    Args:
        x: torch.tensor 
        sd: standard deviation of additive gaussian noise
        clamp_min: clamps min value of output
        clamp_max: clamps max value of output
        scale_by_x: boolean. Multiplies gaussian noise by value of x (element-wise) if True. 
    """
    noise = (torch.randn(x.shape) * sd).to(device=x.device)
    if scale_by_x:
        noise = torch.mul(noise, x)
    x = torch.where(x > 0, x + noise, x)
    x = torch.clamp(x, min=clamp_min, max=clamp_max)
    return x


def scale_depth(x, scale_by=None, random=False, interpolation_mode="nearest"):
    """
    Takes the 5D input tensors and scale them along dim=2
    Args:
        x: 5D input tensor
        scale_by: scaling is done as (1 +/- scale_by) if random, else used as is.
        random: boolean
        interpolation_mode: torch interpolation algorithm 
    """
    if random:
        scale_by = 1 + (2*(torch.rand(1) - 1) * scale_by).item()
    x_scaled = F.interpolate(x.float(), scale_factor=(
        scale_by, 1, 1), mode=interpolation_mode)
    return scale_by, x_scaled


def center_resize(x, target_size):
    """ 
    Center and resize x
    Args:
        x: 3D tensor
        target_size: tuple with volumetric output size
    """
    target_H = target_size[0]
    H = x.shape[0]
    y_center = target_H // 2
    x_center = H // 2
    y = torch.zeros(target_size)

    if x_center < y_center:
        y[y_center-x_center:y_center, :, :] = x[:x_center, :, :]
        y[y_center: y_center + H - x_center, :, :] = x[x_center:, :, :]
    else:
        y[:y_center, :, :] = x[x_center-y_center:x_center, :, :]
        y[y_center:, :, :] = x[x_center:x_center + target_H - y_center, :, :]
    return y




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
        self.xm = met_dataclass_obj.XM
        self.group = met_dataclass_obj.group
        self.subgroup = met_dataclass_obj.subgroup
        self.class_id = met_dataclass_obj.class_id
        self.platform = met_dataclass_obj.platform
        self.astensor = lambda x: torch.as_tensor(x).float().to(self.device)
        self.gnoise_e_std = torch.var(torch.nan_to_num(self.astensor(self.xe)), dim=0, keepdim=True).sqrt()
        self.gnoise_m_std = torch.var(torch.nan_to_num(self.astensor(self.xm)), dim=0, keepdim=True).sqrt()

        self.is_m_1d = met_dataclass_obj.isM_1d
        self.is_e_1d = met_dataclass_obj.isE_1d
        self.is_t_1d = met_dataclass_obj.isT_1d
        self.is_me_1d = np.logical_and(self.is_m_1d, self.is_e_1d)
        self.is_met_1d = np.logical_and(self.is_me_1d, self.is_t_1d)

    def __len__(self):
        return self.xt.shape[0]

    def __getitem__(self, idx):
        xm = self.astensor(self.xm[idx, ...])
        valid_xm = ~torch.isnan(xm)
        xm = torch.nan_to_num(xm)

        xe = self.astensor(self.xe[idx, ...])
        valid_xe = ~torch.isnan(xe)
        xe = torch.nan_to_num(xe).float()

        xt = self.astensor(self.xt[idx, ...])
        valid_xt = ~torch.isnan(xt)
        xt = torch.nan_to_num(xt).float()

        class_id = self.astensor(self.class_id[idx, ...])
        assert ~torch.isnan(class_id) #make sure all the cells have a class

        group = self.astensor(self.group[idx, ...])
        assert ~torch.isnan(group) #make sure all the cells have a class

        subgroup = self.astensor(self.subgroup[idx, ...])
        assert ~torch.isnan(subgroup) #make sure all the cells have a class

        platform = self.astensor(self.platform[idx, ...])
        assert ~torch.isnan(platform)

        is_m_1d = self.is_m_1d[idx]
        is_e_1d = self.is_e_1d[idx]
        is_t_1d = self.is_t_1d[idx]
        is_me_1d = self.is_me_1d[idx]
        is_met_1d = self.is_met_1d[idx]

        return dict(xm=xm, xe=xe, xt=xt, 
                    platform=platform,
                    group=group, subgroup=subgroup, class_id=class_id,
                    valid_xm=valid_xm, valid_xe=valid_xe, valid_xt=valid_xt,
                    is_m_1d=is_m_1d, is_e_1d=is_e_1d, is_t_1d=is_t_1d,
                    is_me_1d=is_me_1d, is_met_1d=is_met_1d)


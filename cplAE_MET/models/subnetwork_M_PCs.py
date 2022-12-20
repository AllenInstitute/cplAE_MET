from torch import nn
import torch


class Enc_xm_to_zm_int(nn.Module):
    """Common encoding network for (only M) and (M and E paired) cases.
    Output is an intermediate representation, `zm_int`
     - Elements of xm expected in range (0, ~40). Missing data is encoded as nans.
     - `xsd` expected in range (0,1)
     - Output is an intermediate representation, `zm_int`
    """

    def __init__(self,
                 gnoise_std_frac=0.05,
                 dropout_p=0.1, out_dim=11):
        super(Enc_xm_to_zm_int, self).__init__()
        # if gnoise_std is not None:
        #     self.gnoise_std = gnoise_std * gnoise_std_frac
        self.gnoise_std_frac=gnoise_std_frac
        self.drp = nn.Dropout(p=dropout_p)
        self.fc_0 = nn.Linear(103, 103)
        self.fc_1 = nn.Linear(103, 103)
        self.fc_2 = nn.Linear(103, 103)
        self.fc_3 = nn.Linear(103, 103)
        self.bn = nn.BatchNorm1d(103, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
        self.fc_4 = nn.Linear(104, out_dim)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        return


    # def add_noise(self, x):
    #     if (self.training) and (self.gnoise_std is not None):
    #             # note: batch dim is inferred from shapes of x and self.noise_sd
    #             x = torch.normal(mean=x, std=self.gnoise_std)
    #     return x

    def add_noise(self, x, sd=0.1, clamp_min=None, clamp_max=None, scale_by_x=False):
        if (self.training):
            noise = (torch.randn(x.shape) * sd).to(device=x.device)
            if scale_by_x:
                noise = torch.mul(noise, x)
            x = torch.where(x > 0, x + noise, x)
            if (clamp_min or clamp_max):
                x = torch.clamp(x, min=clamp_min, max=clamp_max)
        return x


    def forward(self, xm, xsd):
        # x = self.add_noise(xm, sd=self.gnoise_std_frac, scale_by_x=True)
        # xsd = self.add_noise(xsd, sd=self.gnoise_std_frac, clamp_min=0.)
        # x = self.drp(xm)
        x = self.elu(self.fc_0(xm))
        x = self.elu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        x = self.bn(self.relu(self.fc_3(x)))
        x = torch.cat(tensors=(x, xsd.reshape(-1, 1)), dim=1)
        zm_int = self.relu(self.fc_4(x))
        return zm_int


class Enc_zm_int_to_zm(nn.Module):
    """Encodes `zm_int` to `zm`
    """

    def __init__(self, in_dim=11, out_dim=3):
        super(Enc_zm_int_to_zm, self).__init__()
        self.fc_0 = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim, eps=1e-05, momentum=0.05, affine=False, track_running_stats=True)
        return

    def forward(self, zm_int):
        zm = self.bn(self.fc_0(zm_int))
        return zm

class Dec_zm_to_zm_int(nn.Module):
    """Decodes `ze` into `ze_int`
    """

    def __init__(self, in_dim=3, out_dim=11):
        super(Dec_zm_to_zm_int, self).__init__()
        self.fc_0 = nn.Linear(in_dim, out_dim)
        self.fc_1 = nn.Linear(out_dim, out_dim)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        return

    def forward(self, zm):
        x = self.elu(self.fc_0(zm))
        zm_int = self.relu(self.fc_1(x))
        return zm_int


class Dec_zm_int_to_xm(nn.Module):
    """Decodes `zm_int` into the reconstruction `xm` and `xsd`
    """

    def __init__(self, in_dim=10, out_dim=103):
        super(Dec_zm_int_to_xm, self).__init__()
        self.fc_0 = nn.Linear(in_dim, 10)
        self.fc_1 = nn.Linear(10, 103)
        self.fc_2 = nn.Linear(103, 103)
        self.fc_3 = nn.Linear(103, 103)
        self.fc_4 = nn.Linear(103, out_dim)
        self.relu = nn.ReLU()
        return

    def forward(self, zm_int):
        x = zm_int[:, 0:10]
        xrsd = torch.clamp(zm_int[:, 10].view(-1), min=0, max=1)
        x = self.relu(self.fc_0(x))
        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        x = self.relu(self.fc_3(x))
        xrm = self.fc_4(x)
        return xrm, xrsd


class AE_M(nn.Module):
    def __init__(self, config):
        super(AE_M, self).__init__()
        self.enc_xm_to_zm_int = Enc_xm_to_zm_int(gnoise_std_frac=config['M']['gnoise_std_frac'],
                                                 dropout_p=config['M']['dropout_p'])
        self.enc_zm_int_to_zm = Enc_zm_int_to_zm(out_dim=config['latent_dim'])
        self.dec_zm_to_zm_int = Dec_zm_to_zm_int(in_dim=config['latent_dim'])
        self.dec_zm_int_to_xm = Dec_zm_int_to_xm()
        return

    def forward(self, xm, xsd):
        zm_int_enc = self.enc_xm_to_zm_int(xm, xsd)
        zm = self.enc_zm_int_to_zm(zm_int_enc)
        zm_int_dec = self.dec_zm_to_zm_int(zm)
        xrm, xrsd = self.dec_zm_int_to_xm(zm_int_dec)
        return zm_int_enc, zm, zm_int_dec, xrm, xrsd

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
                 gnoise_std=None, 
                 gnoise_std_frac=0.05,
                 dropout_p=0.2, out_dim=11):
        super(Enc_xm_to_zm_int, self).__init__()
        if gnoise_std is not None:
            self.gnoise_std = gnoise_std * gnoise_std_frac
        self.drp = nn.Dropout(p=dropout_p)
        self.fc_0 = nn.Linear(129, 40)
        self.fc_1 = nn.Linear(40, 40)
        self.bn = nn.BatchNorm1d(40, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
        self.fc_2 = nn.Linear(40, 40)
        self.fc_3 = nn.Linear(40, out_dim)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        return

    def add_gnoise(self, x):
        if (self.training) and (self.gnoise_std is not None):
            # note: batch dim is inferred from shapes of x and self.noise_sd
            x = torch.normal(mean=x, std=self.gnoise_std)
        return x

    def forward(self, xm):
        x = self.add_gnoise(xm)
        x = self.drp(x)
        x = self.elu(self.fc_0(x))
        x = self.bn(self.relu(self.fc_1(x)))
        x = self.relu(self.fc_2(x))
        zm_int = self.relu(self.fc_3(x))
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
    """Decodes `zm` into `zm_int`
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
    """Decodes `zm_int` into the reconstruction `xm`
    """

    def __init__(self, in_dim=11, out_dim=129):
        super(Dec_zm_int_to_xm, self).__init__()
        self.fc_0 = nn.Linear(in_dim, 40)
        self.fc_1 = nn.Linear(40, 40)
        self.fc_2 = nn.Linear(40, 40)
        self.fc_3 = nn.Linear(40, out_dim)
        self.relu = nn.ReLU()
        return

    def forward(self, zm_int):
        x = self.relu(self.fc_0(zm_int))
        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        xrm = self.fc_3(x)
        return xrm


class AE_M(nn.Module):
    def __init__(self, config, gnoise_std):
        super(AE_M, self).__init__()
        self.enc_xm_to_zm_int = Enc_xm_to_zm_int(gnoise_std=gnoise_std,
                                                 gnoise_std_frac=config['M']['gnoise_std_frac'],
                                                 dropout_p=config['M']['dropout_p'])
        self.enc_zm_int_to_zm = Enc_zm_int_to_zm(out_dim=config['latent_dim'])
        self.dec_zm_to_zm_int = Dec_zm_to_zm_int(in_dim=config['latent_dim'])
        self.dec_zm_int_to_xm = Dec_zm_int_to_xm()
        return

    def forward(self, xm):
        zm_int_enc = self.enc_xm_to_zm_int(xm.nan_to_num())
        zm = self.enc_zm_int_to_zm(zm_int_enc)
        zm_int_dec = self.dec_zm_to_zm_int(zm)
        xrm = self.dec_zm_int_to_xm(zm_int_dec)
        return zm_int_enc, zm, zm_int_dec, xrm




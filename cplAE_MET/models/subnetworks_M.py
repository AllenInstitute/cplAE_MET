from torch import nn
import torch

class Enc_xm_to_zm_int(nn.Module):
    """Common encoding network for (only M) and (M and E paired) cases.
    - `xm` expected in [N, C=1, D=240, H=4, W=4] format, with C = 1, D=240, H=4, W=4
     - Elements of xm expected in range (0, ~40). Missing data is encoded as nans.
     - `xsd` expected in range (0,1)
     - Output is an intermediate representation, `zm_int`
    """

    def __init__(self, out_dim=11):
        super(Enc_xm_to_zm_int, self).__init__()
        self.conv_0 = nn.Conv3d(1, 10, kernel_size=(7, 3, 1), padding=(3, 1, 0))
        self.pool_0 = nn.MaxPool3d((4, 1, 1), return_indices=True)
        self.conv_1 = nn.Conv3d(10, 20, kernel_size=(7, 3, 1), padding=(3, 1, 0))
        self.pool_1 = nn.MaxPool3d((4, 1, 1), return_indices=True)
        self.fc_0 = nn.Linear(4800, 100)
        self.fc_1 = nn.Linear(101, out_dim)
        self.bn = nn.BatchNorm1d(out_dim, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        return

    def forward(self, xm, xsd):
        x, self.pool_0_ind = self.pool_0(self.relu(self.conv_0(xm)))
        x, self.pool_1_ind = self.pool_1(self.relu(self.conv_1(x)))
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc_0(x))
        x = torch.cat(tensors=(x, xsd.reshape(-1, 1)), dim=1)
        zm_int = self.bn(self.relu(self.fc_1(x)))
        
        return zm_int


class Enc_zm_int_to_zm(nn.Module):
    """Intermediate representation `zm_int` is encoded into `zm`
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
    """Decodes `zm_int` into the reconstruction `xrm` and `xrsd`
    """

    def __init__(self, in_dim=11):
        super(Dec_zm_int_to_xm, self).__init__()
        self.fc_0 = nn.Linear(10, 4800)
        self.convT_0 = nn.ConvTranspose3d(20, 10, kernel_size=(7, 3, 1), padding=(3, 1, 0))
        self.convT_1 = nn.ConvTranspose3d(10, 1, kernel_size=(7, 3, 1), padding=(3, 1, 0))
        self.unpool_0 = nn.MaxUnpool3d((4, 1, 1))
        self.unpool_1 = nn.MaxUnpool3d((4, 1, 1))
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        return

    def forward(self, zm_int, enc_pool_0_ind, enc_pool_1_ind):
        x = zm_int[:, 0:10]
        xrsd = torch.clamp(zm_int[:, 10].view(-1, 1), min=0, max=1)
        x = self.elu(self.fc_0(x))
        x = x.view(-1, 20, 15, 4, 4)
        x = self.elu(self.unpool_0(x, enc_pool_1_ind))
        x = self.convT_0(x)
        x = self.elu(self.unpool_1(x, enc_pool_0_ind))
        xrm = self.relu(self.convT_1(x))
        return xrm, xrsd


class AE_M(nn.Module):
    def __init__(self, config):
        super(AE_M, self).__init__()
        self.enc_xm_to_zm_int = Enc_xm_to_zm_int()
        self.enc_zm_int_to_zm = Enc_zm_int_to_zm(out_dim=config['latent_dim'])
        self.dec_zm_to_zm_int = Dec_zm_to_zm_int(in_dim=config['latent_dim'])
        self.dec_zm_int_to_xm = Dec_zm_int_to_xm()
        return

    def forward(self, xm, xsd):
        zm_int_enc = self.enc_xm_to_zm_int(xm.nan_to_num(), xsd.nan_to_num())
        zm = self.enc_zm_int_to_zm(zm_int_enc)
        zm_int_dec = self.dec_zm_to_zm_int(zm)
        xrm, xrsd = self.dec_zm_int_to_xm(zm_int_dec,
                                          self.enc_xm_to_zm_int.pool_0_ind,
                                          self.enc_xm_to_zm_int.pool_1_ind)
        return zm_int_enc, zm, zm_int_dec, xrm, xrsd

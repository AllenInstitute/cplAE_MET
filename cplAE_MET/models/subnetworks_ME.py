import torch
from torch import nn


class Enc_zme_int_to_zme(nn.Module):
    def __init__(self, in_dim=22, out_dim=3):
        super(Enc_zme_int_to_zme, self).__init__()
        self.fc_0 = nn.Linear(in_dim, 22)
        self.fc_1 = nn.Linear(22, 22)
        self.fc_2 = nn.Linear(22, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim,  eps=1e-5, momentum=0.05, affine=False, track_running_stats=True)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        return

    def forward(self, xm_int, xe_int):
        x = torch.cat((xm_int, xe_int), 1)
        x = self.elu(self.fc_0(x))
        x = self.elu(self.fc_1(x))
        zme = self.bn(self.fc_2(x))
        return zme


class Dec_zme_to_zme_int(nn.Module):
    def __init__(self, in_dim=3, out_dim=22):
        super(Dec_zme_to_zme_int, self).__init__()
        self.fc_0 = nn.Linear(in_dim, 22)
        self.fc_1 = nn.Linear(22, 22)
        self.fc_2 = nn.Linear(22, out_dim)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        return

    def forward(self, zme):
        x = self.elu(self.fc_0(zme))
        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        xm_int, xe_int = torch.split(x,[11,11], dim = 1)
        return xm_int, xe_int


class AE_ME_int(nn.Module):
    def __init__(self, config):
        super(AE_ME_int, self).__init__()
        self.enc_zme_int_to_zme = Enc_zme_int_to_zme()
        self.dec_zme_to_zme_int = Dec_zme_to_zme_int()
        return

    def forward(self, xm_int, xe_int):
        zme = self.enc_zme_int_to_zme(xm_int, xe_int)
        xm_int, xe_int = self.dec_zme_to_zme_int(zme)
        return zme, xm_int, xe_int


def test():
    enc_zme_int_to_zme = Enc_zme_int_to_zme()
    dec_zme_to_zme_int = Dec_zme_to_zme_int()

    # dummy data
    x = torch.ones(100, 22).float()
    xm_int, xe_int = torch.split(x, [11, 11], dim=1)

    # check model i/o
    zme = enc_zme_int_to_zme(xm_int, xe_int)
    xm_int, xe_int = dec_zme_to_zme_int(zme)
    return

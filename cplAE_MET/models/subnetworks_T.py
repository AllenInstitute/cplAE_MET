from torch import nn

class Enc_xt_to_zt(nn.Module):
    def __init__(self, dropout_p=0.2, in_dim=1252, out_dim=3):
        super(Enc_xt_to_zt, self).__init__()
        self.drp = nn.Dropout(p=dropout_p)
        self.fc_0 = nn.Linear(in_dim, 50)
        self.fc_1 = nn.Linear(20, 20)
        self.fc_2 = nn.Linear(20, 20)
        self.fc_3 = nn.Linear(20, 20)
        self.fc_4 = nn.Linear(20, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim, eps=1e-05, momentum=0.05, affine=False, track_running_stats=True)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        return

    def forward(self, xt):
        x = self.drp(xt)
        x = self.elu(self.fc_0(x))
        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        x = self.relu(self.fc_3(x))
        zt = self.bn(self.fc_4(x))
        return zt


class Dec_zt_to_xt(nn.Module):
    """Encodes `ze_int` to `ze`
    """

    def __init__(self, in_dim=2, out_dim=1252):
        super(Dec_zt_to_xt, self).__init__()
        self.fc_0 = nn.Linear(in_dim, 20)
        self.fc_1 = nn.Linear(20, 20)
        self.fc_2 = nn.Linear(20, 20)
        self.fc_3 = nn.Linear(20, 20)
        self.fc_4 = nn.Linear(50, out_dim)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        return

    def forward(self, zt):
        x = self.elu(self.fc_0(zt))
        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        x = self.relu(self.fc_3(x))
        xrt = self.relu(self.fc_4(x))
        return xrt


class AE_T(nn.Module):
    def __init__(self, config):
        super(AE_T, self).__init__()
        self.enc_xt_to_zt = Enc_xt_to_zt(dropout_p=config['T']['dropout_p'],
                                         out_dim=config['latent_dim'])
        self.dec_zt_to_xt = Dec_zt_to_xt(in_dim=config['latent_dim'])
        return

    def forward(self, xt):
        zt = self.enc_xt_to_zt(xt)
        xrt = self.dec_zt_to_xt(zt)
        return zt, xrt

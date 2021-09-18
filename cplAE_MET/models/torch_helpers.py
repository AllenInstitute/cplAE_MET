import torch
import numpy as np


def tensor(x, device):
    return torch.tensor(x).to(dtype=torch.float32).to(device)


def astensor(x, device):
    return torch.as_tensor(x).to(dtype=torch.float32).to(device)


def boolastensor(x, device):
    return torch.as_tensor(x).to(dtype=torch.bool).to(device)


def tonumpy(x):
    return x.cpu().detach().numpy()

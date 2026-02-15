import numpy as np
import torch


def generate_samix(data, alpha=25.0):
    lamb = np.random.beta(alpha, alpha)
    indices = get_perm(data.size(0))
    data_perm = data[indices]
    mixed_views = data * lamb + (1 - lamb) * data_perm

    return mixed_views, lamb, indices


def get_perm(l):
    perm = torch.randperm(l)
    while torch.all(torch.eq(perm, torch.arange(l))):
        perm = torch.randperm(l)
    return perm

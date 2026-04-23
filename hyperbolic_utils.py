import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MIN_NORM = 1e-15


def artanh(x):
    x = torch.clamp(x, -1 + 1e-7, 1 - 1e-7)
    return (torch.log(1 + x).sub(torch.log(1 - x))).mul(0.5)


def expmap0(v, c=1.0):
    v_norm = v.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    sqrt_c = math.sqrt(c)
    return torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)


def logmap0(y, c=1.0):
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    sqrt_c = math.sqrt(c)
    return artanh(sqrt_c * y_norm) * y / (sqrt_c * y_norm)


class HyperbolicAugmentor(nn.Module):
    def __init__(self, c=1.0, radial_std=0.1, angular_std=0.1):
        super().__init__()
        self.c = c
        self.radial_std = radial_std
        self.angular_std = angular_std

    def forward(self, x_euclidean):
        z_hyp = expmap0(x_euclidean, self.c)
        v = logmap0(z_hyp, self.c)

        v_norm = v.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
        v_dir = v / v_norm

        noise_r = torch.randn_like(v_norm) * self.radial_std
        perturbed_norm = F.relu(v_norm + noise_r)

        noise_a = torch.randn_like(v) * self.angular_std
        noise_a_orthogonal = noise_a - (noise_a * v_dir).sum(dim=-1, keepdim=True) * v_dir
        perturbed_dir = v_dir + noise_a_orthogonal
        perturbed_dir = perturbed_dir / perturbed_dir.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)

        perturbed_v = perturbed_norm * perturbed_dir
        z_hyp_aug = expmap0(perturbed_v, self.c)
        z_euc_aug = logmap0(z_hyp_aug, self.c)

        return x_euclidean, z_euc_aug

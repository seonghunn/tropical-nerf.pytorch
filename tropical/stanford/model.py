# TropicalNeRF
# Copyright (c) 2024-present NAVER Cloud Corp.
# CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)

from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

import tinycudann as tcnn
from tropical import TropicalHashGrid
from tropical.utils.chamfer_distance import sample_surface_from_rays, chamfer_distance


class Net(nn.Module):
    def __init__(self, num_layers: int = 3, num_hidden: int = 16, levels: int = 4,
                 r_min: int = 2, r_max: int = 32, T: int = 19, eps: float = 1e-4):
        super().__init__()

        # Constants
        DIM = 3

        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.eps = eps

        # hashgrid
        L = levels
        F = 2
        N_min = r_min
        self.scale = 1
        N_max = r_max
        self.enc = TropicalHashGrid(1.0, DIM, L, F, T, N_min, N_max, eps)

        # mlp
        num_nodes = [L * F] + [num_hidden] * (num_layers - 1) + [2]
        self.num_nodes = num_nodes

        modules = []
        for i in range(len(num_nodes) - 1):
            modules.append(nn.Linear(num_nodes[i], num_nodes[i + 1]))
            # Deprecated the below due to weight-norm loss. The weight normalization
            # prevents diminishing the magnitude of normal vectors. While, weight decay
            # and eikonal loss prevent exploding that.
            # modules.append(torch.nn.utils.weight_norm(
            #     nn.Linear(num_nodes[i], num_nodes[i + 1])))
        self.fc = nn.ModuleList(modules)

    def forward(self, x, gather: bool = False, group: int = 1):
        if gather:
            inputs = []

        # hashgrid
        x = self.preprocess(x)
        x = self.enc(x).float()  # half to full

        for i in range(len(self.num_nodes) - 1):
            x = self.fc[i](x)
            if i != len(self.num_nodes) - 2:
                if gather:
                    inputs.append(x)
                if 1 == group:
                    x = F.relu(x)
                else:
                    # infer within a common linear space
                    m = (x[::group] > self.eps) | (x[group-1::group] > self.eps)
                    x = x * m.repeat(1, group).view(*x.shape)
            if i == len(self.num_nodes) - 2:
                if gather:
                    inputs.append(x[:, 1:] - x[:, :1])
        if gather:
            return x, inputs
        return x

    def preprocess(self, x):
        return (x + self.scale) / (self.scale * 2)

    def preprocess_inverse(self, x):
        return x * (self.scale * 2) - self.scale

    def sdf(self, x):
        output = self.forward(x)
        # tanh does not impact on the Chamfer Distance evaluation
        return torch.tanh(output[:, 1:] - output[:, :1])
        # return output[:, 1:] - output[:, :1]

    def region(self, vertices: Tensor, output: Tensor = None, eps=None):
        eps = eps if eps is not None else self.eps

        # indicator to the current linear regions
        if output is None:
            output = torch.cat(self(vertices, gather=True)[1], dim=-1)

        m = (output > 0) * 2 - 1
        m[output.abs() <= eps] = 0

        m_, offset = self.enc.region(self.preprocess(vertices), eps)
        m = torch.cat([m_, m], dim=-1)

        return m, offset, output

    def normal(self, vertices: Tensor, l: int = None, h: int = None,
               create_graph=False, return_y=False) -> Tensor:
        # Requires grad
        vertices.requires_grad = True

        # Get surface normals to decide the order of vertices, which determines the
        # inside or outside of faces for a renderer.
        with torch.enable_grad():
            if l is None or h is None or h == net.num_hidden:
                y = self.sdf(vertices)
            else:
                y = torch.cat(
                    self(vertices, gather=True)[1], dim=-1)[:, l * net.num_hidden + h]
            J = torch.autograd.grad(
                y.sum(), vertices, create_graph=create_graph)[0]  # Bx3
        if return_y:
            return J, y

        return J

    @torch.no_grad()
    def check_orthogonality(self):
        for i in range(len(self.fc)):
            w = self.fc[i].weight
            w = w / w.norm(p=2, dim=-1, keepdim=True)
            loss = (w @ w.t() - torch.eye(w.shape[0], device=w.device)).abs().max()
            print(w.shape)
            print(f"{i} layer orthogonality: {loss:.4f}")

    def device(self):
        return next(self.parameters()).device

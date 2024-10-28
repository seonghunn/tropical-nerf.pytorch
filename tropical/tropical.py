# TropicalNeRF
# Copyright (c) 2024-present NAVER Cloud Corp.
# CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)

from typing import *
from torch import Tensor, LongTensor
from torch.nn import Module

import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn


class TropicalHashGrid(Module):
    def __init__(self, scale: float = 1.0, D: int = 3, L: int = 16, F: int = 2,
                 T: int = 19, N_min: int = 16, N_max: int = 2048, eps: float = 1e-4):
        super().__init__()
        self.scale = scale
        self.D = D
        self.L = L
        self.F = F
        self.T = T
        self.N_min = N_min
        self.N_max = N_max
        self.b = np.exp2(np.log2(N_max * scale / N_min) / (L - 1))
        self.module = tcnn.Encoding(D, {
            "otype": "Grid",
            "type": "Hash",
            "n_levels": L,
            "n_features_per_level": F,
            "log2_hashmap_size": T,
            "base_resolution": N_min,
            "per_level_scale": self.b},
            dtype=torch.float)

        # aggregated and sorted marks in multi-resolution grids
        self.eps = eps
        self.marks = self._marks()

    def forward(self, x):
        return self.module(x)

    def _marks(self):
        # vertices for mark candidates
        vertices = []

        for l in range(self.L):  # L-level resolution
            # The # of vertices for each axis
            grid_scale = np.exp2(l * np.log2(self.b)) * self.N_min - 1.0

            # Use one less than resolution to have # of cellsm not # of vertices.
            unit = 1 / grid_scale

            # The offset of 0.5 mentioned in Appendix A of Müller et al. (2022)
            vertices += [torch.arange(0, 1.5, unit, device=self.device()) - 0.5 * unit]

        vertices += [torch.Tensor([0, self.scale]).to(self.device())]  # valid boundary
        marks, _ = torch.cat(vertices).unique().sort()

        m = marks.new_zeros(len(marks)).bool().fill_(True)
        for i in range(len(marks) - 1):
            if self.eps > (marks[i] - marks[i + 1]).abs():
                marks[i + 1] = (marks[i] + marks[i + 1]) / 2
                m[i] = False
                # print(f"tropical warning: {marks[i]:.5f} vs {marks[i + 1]:.5f}")

        marks = marks[m]

        # filter within (-scale, scale)
        marks = marks[marks >= 0]
        marks = marks[marks <= self.scale]

        return marks

    def _skeleton(self, indices: Tensor, future: Tensor, pruning: bool = True) -> Tensor:
        """Find valid edges

        Args:
            indices (Tensor): UxUxUxD
            future (Tensor): UxUxUxR
            pruning (bool): Prune the edges outside of the surfaces
        """
        assert 3 == indices.shape[-1]
        indices = self.p2v(indices)  # UxUxUxD -> UxUxU

        if pruning:
            m = (future[1:, :, :] != future[:-1, :, :]).sum(dim=-1) > 0
            # print(f"{m.sum()}/{m.numel()} valid edges on axis-x")
            edges = [torch.stack(
                [indices[1:, :, :][m], indices[:-1, :, :][m]], dim=-1)]  # Ex2
            m = (future[:, 1:, :] != future[:, :-1, :]).sum(dim=-1) > 0
            # print(f"{m.sum()}/{m.numel()} valid edges on axis-y")
            edges += [torch.stack([indices[:, 1:, :][m], indices[:, :-1, :][m]], dim=-1)]
            m = (future[:, :, 1:] != future[:, :, :-1]).sum(dim=-1) > 0
            # print(f"{m.sum()}/{m.numel()} valid edges on axis-z")
            edges += [torch.stack([indices[:, :, 1:][m], indices[:, :, :-1][m]], dim=-1)]
        else:
            edges = [torch.stack([indices[1:, :, :].reshape(-1),
                                  indices[:-1, :, :].reshape(-1)], dim=-1)]
            edges += [torch.stack([indices[:, 1:, :].reshape(-1),
                                   indices[:, :-1, :].reshape(-1)], dim=-1)]
            edges += [torch.stack([indices[:, :, 1:].reshape(-1),
                                   indices[:, :, :-1].reshape(-1)], dim=-1)]

        return torch.cat(edges, dim=0)

    # serialized vertex index from global position indices
    def p2v(self, indices: LongTensor) -> LongTensor:
        L = len(self.marks)
        indices = indices.clone()
        for i in range(self.D):
            indices[..., -1-i] *= L ** i
        return indices.sum(dim=-1).long()

    # global position indices from serialized vertex index
    def v2p(self, v_idx: LongTensor) -> LongTensor:
        L = len(self.marks)
        p = []
        v_idx = v_idx.clone()
        for i in range(self.D - 1, -1, -1):
            p += [v_idx.div(L ** i).floor().long()]
            v_idx.sub_(p[-1] * L ** i)
        return torch.stack(p, dim=-1)

    def skeleton(self, net: Module, unit: int = 128) -> Tensor:
        """Find valid edges that would be divided by future (folded) hyperplanes using
            the region indicators (signs of hidden representations) of given networks.
            This is critical to prune the edges since the number of edges by hashgrid is
            enormous.

        Args:
            net (Module): Tropical module that has `region()` getting region indicators
            unit (int, optional): The number of vertices on an axis (totally, unit**3)
                to compute in a batch.

        Returns:
            Tensor: Edges tensor in (V x 2). Vertex indices may be sparse.
        """
        L = len(self.marks)
        D = self.D
        edges = []

        for i in range(0, L, unit - 1):
            for j in range(0, L, unit - 1):
                for k in range(0, L, unit - 1):
                    start = [i, j, k]
                    # overlapping the last vertices
                    end = [min(L, x + unit) for x in start]

                    x = [torch.arange(x, y, device=net.device()) for x, y in zip(
                        start, end)]
                    indices = torch.stack(torch.meshgrid(x, indexing="ij"), dim=-1)
                    x = net.preprocess_inverse(self.marks[indices.reshape(-1, D)])

                    GRID_PRUNING = True
                    if GRID_PRUNING:
                        future, offset, _ = net.region(x)
                        future = future[:, offset.shape[1]:]
                        future = future.view(*indices.shape[:-1], -1)
                        edges += [self._skeleton(indices, future)]
                    else:
                        edges += [self._skeleton(indices, None, pruning=GRID_PRUNING)]

        edges = torch.cat(edges, dim=0)

        if 0 == edges.shape[0]:
            return torch.Tensor([]).to(edges), torch.Tensor([]).to(edges)

        def squeeze_edges(edges):
            v_idx = edges.unique().sort()[0]  # V

            # Create a tensor to hold the inverted vector
            v_inv = torch.full((v_idx.max() + 1,), -1, device=edges.device)

            # Fill in the inverted vector with indices from the original vector
            v_inv[v_idx] = torch.arange(len(v_idx), device=edges.device)

            return v_inv[edges], v_idx

        edges, v_idx = squeeze_edges(edges)
        vertices = net.preprocess_inverse(self.marks[self.v2p(v_idx)])  # Vx3

        return vertices, edges

    def region(self, x: Tensor, eps: float = None) -> Tuple[Tensor, Tensor]:
        eps = eps if eps is not None else self.eps
        # efficiently find epsilon-tolerate offsets
        offset = torch.searchsorted(self.marks, x + eps)  # vx3
        offset -= 1  # the index -1 will be tolerate in the logic.

        # mask value 1 indicates inside of grid, 0 indicates on the edge
        mask = ((self.marks[offset] - x).abs() > eps).long()

        return mask, offset

    def device(self):
        return next(self.parameters()).device


class Tropical(Module):
    def __init__(self, module: Module, dim: int = 3, scale: float = 1.0):
        super().__init__()
        self.module = module
        self.dim = dim
        self.scale = scale

    def region(self, x: Tensor) -> Any:
        return NotImplementedError

    def grid(self) -> Tuple[Tensor, Tensor]:
        # hashgrid provides a starting skeleton
        for m in self.module.modules():
            if isinstance(m, TropicalHashGrid):
                return m.skeleton()

        # otherwise, we start with a hypercube
        vertices, edges, _ = self.get_hypercube(self.dim, self.scale / 2)
        return vertices, edges

    def get_hypercube(self, d, size):
        x = torch.Tensor([-size, size])
        grids = torch.stack(torch.meshgrid(x, x, x, indexing='ij'), dim=-1)

        # vertices
        vertices = grids.view(-1, 3)

        # edges
        edges = []
        for i in range(vertices.shape[0]):
            for j in range(i + 1, vertices.shape[0]):
                if 1 == (vertices[i] * vertices[j] < 0).sum():
                    edges.append([i, j])
        edges = torch.LongTensor(edges)

        # faces, counter clock-wise
        faces = [[0, 3, 5, 1], [0, 2, 8, 4], [3, 4, 10, 7],
                 [1, 2, 9, 6], [8, 9, 11, 10], [7, 11, 6, 5]]

        return vertices, edges, faces


def low_precision(x):
    x *= 100000
    x = x.floor()
    x /= 100000
    return x


def analytical_marks(m, f: int = 1, l: int = 1, dx: float = 1e-5):
    with torch.enable_grad():
        x = torch.arange(-1, 1, dx).cuda()
        points = torch.stack([x] * 3, dim=-1)
        points[:, 1:].fill_(m.marks[1])  # TODO
        points.requires_grad = True
        y = m(points)
        J = torch.autograd.grad(y[:, l*f].sum(), points)[0]  # Bx4
        _, r_idx = low_precision(J[:, 0]).unique(dim=0, return_inverse=True)
        for i in range(r_idx.shape[0] - 1):
            if r_idx[i] != r_idx[i+1]:
                print(f"{points[i+1][0]:.5f}, {J[:,0][i+1]:.5f}")


if "__main__" == __name__:
    # random seed
    torch.manual_seed(0)
    random.seed(0)
    torch.backends.cudnn.benchmark = False

    # m = TropicalHashGrid()

    L = 2
    T = 19
    N_min = 2
    N_max = 6
    scale = 1
    DIM = 3
    m = TropicalHashGrid(1.0, DIM, L, 1, T, N_min, N_max)
    for i in range(m.marks.shape[0]):
        print(f"{m.marks[i]:.5f}, ", end="")
    print()

    # example 0
    # x = torch.arange(-2, 2, 1e-2).cuda()
    # x = torch.stack([x] * 3, dim=-1)
    # x[:, 1:].fill_(m.marks[1])

    # example 1
    # x0 = torch.arange(0.1, 0.3, 1e-2).cuda()
    # x2 = torch.arange(0.5, 0.7, 1e-2).cuda()
    # x = torch.stack([x0, x0, x2], dim=-1)
    # x[:,1].fill_(0.5)

    # Example 2: Taylor series for cubic-Bézier curves
    x0 = torch.arange(0, .5, 1e-3).cuda()
    x = torch.stack([x0, x0, x0], dim=-1)
    y = m(x)

    points = x.clone()
    points.requires_grad = True
    y = m(points)
    J0 = torch.autograd.grad(y[:, 1].sum(), points, create_graph=True)[0]

    v = torch.Tensor([[1, 1, 1]]).to(x)
    v.requires_grad = False
    def dot(x, y): return (x * y).sum(dim=-1)

    J1 = torch.autograd.grad(J0.sum(), points, create_graph=True)[0]

    y0 = dot(J0, v)
    y1 = dot(J1, v)

    z = 1e-3
    y2 = (y1[1:] / z - y1[:-1] / z)
    y2 = torch.cat([y2, y2[-1:]], dim=0)

    for i in (y2.abs() > 1).nonzero():
        y2[i] = y2[i-1]

    def cbc(x, idx=200):
        a = idx * 1e-3
        return y[idx][1] + y0[idx] * (x - a) + (y1[idx] / 2) * (x - a)**2 + (y2[idx] / 6) * (x - a)**3

    for i in range(x.shape[0]):
        print(f"{x[i][0]:.4f}, {y[i][1]*1000:.4f}, {y0[i]*1000:.4f}, {y1[i]*1000:.4f}, {y2[i]*1000:.4f}, {cbc(i * 1e-3)*1000:.4f}")

    # analytical_marks(m)

    # for i in range(x.shape[0]):
    #     print(f"{x[i][0]:.4f}\t{y[i][0]*1000:.4f}\t{y[i][1]*1000:.4f}")

    # import pdb; pdb.set_trace()

    # print(m.skeleton())

    # print(len(m.marks))
    # print("initialized")
    # edges = TropicalHashGrid._skeleton(3, 3)
    # print(edges.shape)
    # print("edges found")

    # vertices = m.vertices()
    # print(vertices.shape)
    # print("vertices found")

    # x = torch.Tensor([[0.11, 0.2, 0.5]])

    # mask, offset = m.region(x)
    # print(m.marks)
    # print(mask, offset)

    # import pdb
    # pdb.set_trace()

    # with torch.enable_grad():
    #     def sanity_check(eps, l, m):
    #         vertices = m.skeleton()[l]
    #         points = vertices.unsqueeze(-1).repeat(1, 3)
    #         points.requires_grad = True
    #         with torch.no_grad():
    #             points -= eps  # perturbation within a unit
    #         y = m(points)
    #         J0 = torch.autograd.grad(y[:, l*2:l*2+1].sum(), points)[0]
    #         vertices = m.skeleton()[l]
    #         points = vertices.unsqueeze(-1).repeat(1, 3)
    #         points.requires_grad = True
    #         with torch.no_grad():
    #             points += eps  # perturbation within a unit
    #         y = m(points)
    #         J1 = torch.autograd.grad(y[:, l*2:l*2+1].sum(), points)[0]
    #         return J0, J1

    #     # sanity check wheather we found critical points
    #     for l, eps in zip([0, 7, 15], [1e-2, 1e-3, 1e-4]):
    #         x, y = sanity_check(eps, l, m)
    #         z = (x - y).norm(dim=1).min()
    #         print(l, z)
    #         assert z > 1e-4  # at least gradient should be differ than 1e-4

    # x = torch.cat([torch.arange(2048).unsqueeze(-1) / 2048, torch.zeros(2048, 2)], dim=-1)

    # x -= 0.5 / 15  # to align the first layer grids
    # y = m(x)

    # with open("tmp/hash.txt", "w") as f:
    #     for i in range(y.shape[0]):
    #         for j in range(y.shape[1]):
    #             f.write(f"{y[i,j].item()*1e4:.4f}\t")
    #         f.write("\n")

    # x /= 2047 / 3
    # x -= 0.5 / 2047  # to align the first layer grids
    # y = m(x)

    # with open("tmp/hash.txt", "w") as f:
    #     for i in range(y.shape[0]):
    #         for j in [30, 31]:
    #             f.write(f"{y[i,j].item()*1e4:.4f}\t")
    #         f.write("\n")

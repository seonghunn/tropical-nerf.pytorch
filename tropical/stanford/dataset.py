# TropicalNeRF
# Copyright (c) 2024-present NAVER Cloud Corp.
# CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)

from typing import *
import os

import numpy as np
import time
import random
import numpy as np

import mcubes
import trimesh
import cubvh

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader


class StanfordDataset(Dataset):
    def __init__(self, name: str = "dragon"):
        self.R = .8
        self.name = name
        self.init()
        self.resample()

    def __len__(self):
        return 50000

    def init(self):
        BASE_DIR = os.path.dirname(__file__)
        if "bunny" == self.name.lower():
            print(f"Loading bun_zipper.ply ...")
            self.mesh = trimesh.load(os.path.join(
                BASE_DIR, f"{self.name.lower()}/reconstruction/bun_zipper.ply"))
        elif "bunny_npy" == self.name.lower():
            print(f"Loading bunny.npy ...")
            density_grid = np.load(os.path.join(BASE_DIR, "models/bunny.npy"))
            # build mesh using Trimesh
            vertices, triangles = mcubes.marching_cubes(density_grid, 0)
            vertices = (vertices / 32 - 1).astype(np.float32) * self.R  # range in [-1, 1]
            triangles = triangles.astype(np.int32)
            self.mesh = trimesh.Trimesh(vertices, triangles, process=False)
        elif "armadillo" == self.name.lower():
            print(f"Loading {self.name}.ply ...")
            self.mesh = trimesh.load(os.path.join(
                BASE_DIR, f"{self.name.lower()}/{self.name.capitalize()}.ply"))
        elif "drill" == self.name.lower():
            print(f"Loading {self.name}_shaft_vrip.ply ...")
            PATH_TO_DRILL = f"reconstruction/{self.name}_shaft_vrip.ply"
            self.mesh = trimesh.load(os.path.join(
                BASE_DIR, f"{self.name.lower()}/{PATH_TO_DRILL}"))
        elif "lucy" == self.name.lower():
            print(f"Loading {self.name}_res10.ply ...")
            # Applied the MeshLab simplification filter using clustering decimation
            # to reduce the excessive number of vertices for this task to speed up.
            self.mesh = trimesh.load(os.path.join(
                BASE_DIR, f"{self.name.lower()}/{self.name}_res10.ply"))
        else:
            print(f"Loading {self.name}_vrip_res3.ply ...")
            self.mesh = trimesh.load(os.path.join(
                BASE_DIR, f"{self.name}_recon/{self.name}_vrip_res3.ply"))
        print("Done.", flush=True)
        vertices = torch.Tensor(self.mesh.vertices)

        if "bunny_npy" != self.name.lower():
            scale = (vertices.max(dim=0)[0] - vertices.min(dim=0)[0]).max()
            vertices = vertices / scale * 2
            vertices -= (vertices.max(dim=0)[0] + vertices.min(dim=0)[0]) / 2

        self.mesh.vertices = vertices.numpy()
        self.BVH = cubvh.cuBVH(self.mesh.vertices, self.mesh.faces)
        print("BVH initialized.", flush=True)

    def resample(self):
        # points = torch.rand(len(self), 3) * 2 - 1
        vertices = torch.Tensor(self.mesh.vertices)
        if "lucy" != self.name.lower():  # lucy has too many vertices.
            vertices = vertices.repeat(10, 1)
        d = 0.4
        if vertices.shape[0] < len(self):  # drill has few vertices.
            vertices = torch.Tensor(self.mesh.vertices).repeat(30, 1)
            d = 0.2
        points = vertices[torch.randperm(vertices.shape[0])[:len(self)]] + \
            (torch.rand(len(self), 3) * d - d / 2)

        distances, _, _ = self.BVH.signed_distance(points)  # [N], [N], [N, 3]
        distances = distances.cpu()

        self.X = points  # [:len(self)]
        self.Y = distances  # [:len(self)]  # inside is positive

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

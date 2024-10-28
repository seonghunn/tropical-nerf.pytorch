# TropicalNeRF
# Copyright (c) 2024-present NAVER Cloud Corp.
# CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)

from typing import *
import os
import time
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import glob
from PIL import Image
import argparse

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

import mcubes
import trimesh
import cubvh

from tropical.stanford.dataset import StanfordDataset
from tropical.stanford.model import Net
import tropical.subpoly as sp
from tropical.utils.chamfer_distance import sample_surface_from_rays, chamfer_distance
from tropical.utils.mtet import marching_tetrahedras


parser = argparse.ArgumentParser(
    prog="python -m tropical.stanford.evaluate",
    description="Polyhedral complex derivation from piecewise trilinear networks")

parser.add_argument("-d", "--dataset", default="dragon",
                    choices=["bunny", "dragon", "happy", "armadillo", "drill", "lucy"],
                    help="Stanford 3D scanning model name")
parser.add_argument("-s", "--seed", default=45, type=int, help="Seed")
parser.add_argument("-m", "--model_size", default="small",
                    choices=["small", "medium", "large"], help="Model size")
parser.add_argument("-t", "--method", default="mc",
                    choices=["mc", "mtet"], help="Mesh extraction method")

args = parser.parse_args()
print(args)
seed = args.seed

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Constants
CANVAS_SIZE = 1.2
training_data_R = 0.8
our_t = -1

if "small" == args.model_size:
    r_min = 2
    r_max = 32
elif "medium" == args.model_size:
    r_min = 4
    r_max = 64
elif "large" == args.model_size:
    r_min = 8
    r_max = 128


# Model and mesh paths
model_path = os.path.join(
    os.path.dirname(__file__),
    f"models/{args.dataset}/{args.dataset}_sdf_{args.model_size}_{seed}.pth")
mesh_path = os.path.join(
    f"meshes/{args.dataset}", f'our_mesh_{args.model_size}_{seed}.ply')

# Check if files exist
if not os.path.isfile(model_path):
    print(f"Model path is not found: {model_path}")
    exit()
if not os.path.isfile(mesh_path):
    print(f"Mesh path is not found: {mesh_path}")
    exit()

# Load the model and mesh
net = Net(num_layers=3, num_hidden=16, levels=4, r_min=r_min, r_max=r_max).cuda()
net.load_state_dict(torch.load(model_path, map_location=net.device()))
print(f"The pretrained model is loaded from {model_path}")
our_mesh = trimesh.load(mesh_path)
print(f"The mesh is loaded from {mesh_path}")
print(f"Ours: {our_mesh.vertices.shape}/{our_mesh.faces.shape}")


# Check the number of on-grid vertices
def count_vertices_near_values(vertices, tensor_values, threshold=1e-4):
    # Calculate absolute differences between vertex coordinates and tensor values
    abs_diff = np.abs(vertices[..., None] - tensor_values)

    # Check if any absolute difference is less than the threshold
    near_values_mask = np.any(abs_diff < threshold, axis=-1)
    near_values_mask = np.any(near_values_mask, axis=-1)

    # Count the number of vertices where at least one coordinate is near a tensor value
    num_near_vertices = np.sum(near_values_mask)

    return num_near_vertices


count = count_vertices_near_values(our_mesh.vertices, net.enc.marks.cpu().numpy())
print(f"Number of vertices near the grid marks: {count} "
      f"({count/our_mesh.vertices.shape[0]:.4f})")


@torch.no_grad()
def run_marching_cubes(MC_SAMPLE_SIZE=100):
    # print(f"MC SAMPLE SIZE: {MC_SAMPLE_SIZE}")
    s = torch.linspace(-CANVAS_SIZE, CANVAS_SIZE, MC_SAMPLE_SIZE)
    grid_x, grid_y, grid_z = torch.meshgrid(s, s, s, indexing='ij')
    sdfs = net.sdf(
        torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).cuda())[:, 0]
    sdfs = sdfs.reshape(*grid_x.shape)
    sdfs = sdfs.cpu().numpy()

    vertices, triangles = mcubes.marching_cubes(-sdfs, 0)
    vertices = vertices / (MC_SAMPLE_SIZE - 1.0) * 2 * CANVAS_SIZE - CANVAS_SIZE

    vertices /= training_data_R
    mc_vertices = vertices.astype(np.float32)
    mc_triangles = triangles.astype(np.int32)
    mc_mesh = trimesh.Trimesh(mc_vertices, mc_triangles, process=False)
    # print(f"MC: {mc_mesh.vertices.shape}/{mc_mesh.faces.shape}")
    return mc_mesh


# Generate tetrahedra indices
def generate_tetrahedra_indices(MC_SAMPLE_SIZE):
    # Number of points in one dimension
    n = MC_SAMPLE_SIZE

    # Indices offset for the 3D grid
    def idx(x, y, z):
        return x * n * n + y * n + z

    tetrahedra = []

    for x in range(n-1):
        for y in range(n-1):
            for z in range(n-1):
                # Define the vertices of the current cube
                v0 = idx(x, y, z)
                v1 = idx(x+1, y, z)
                v2 = idx(x, y+1, z)
                v3 = idx(x, y, z+1)
                v4 = idx(x+1, y+1, z)
                v5 = idx(x+1, y, z+1)
                v6 = idx(x, y+1, z+1)
                v7 = idx(x+1, y+1, z+1)

                # Subdivide the cube into 5 tetrahedra
                tetrahedra.append([v0, v1, v2, v6])
                tetrahedra.append([v1, v2, v4, v6])
                tetrahedra.append([v0, v1, v3, v6])
                tetrahedra.append([v1, v3, v5, v6])
                tetrahedra.append([v4, v5, v6, v7])
                tetrahedra.append([v1, v4, v5, v6])

    return torch.tensor(tetrahedra, dtype=torch.long)


@torch.no_grad()
def run_marching_tetrahedras(MC_SAMPLE_SIZE=100):
    s = torch.linspace(-CANVAS_SIZE, CANVAS_SIZE, MC_SAMPLE_SIZE)
    grid_x, grid_y, grid_z = torch.meshgrid(s, s, s, indexing='ij')
    points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).cuda()
    sdfs = net.sdf(points)[:, 0]
    sdfs = sdfs.reshape(-1)

    tets = generate_tetrahedra_indices(MC_SAMPLE_SIZE).cuda()
    vertices, triangles = marching_tetrahedras(points, tets, sdfs, 0)

    vertices = vertices.cpu().numpy()
    triangles = triangles.cpu().numpy()

    vertices /= training_data_R
    mc_vertices = vertices.astype(np.float32)
    mc_triangles = triangles.astype(np.int32)
    mc_mesh = trimesh.Trimesh(mc_vertices, mc_triangles, process=False)
    # print(f"MC: {mc_mesh.vertices.shape}/{mc_mesh.faces.shape}")
    return mc_mesh


def box_align(source, target):
    t_center = np.mean(target.vertices, axis=0)
    s_center = np.mean(source.vertices, axis=0)
    # source.vertices += t_center - s_center
    # print(t_center - s_center)

    t_min = np.min(target.vertices, axis=0)
    t_max = np.max(target.vertices, axis=0)
    s_min = np.min(source.vertices, axis=0)
    s_max = np.max(source.vertices, axis=0)
    # source.vertices *= (t_max - t_min)/(s_max - s_min)
    print((t_max - t_min)/(s_max - s_min))


def get_rays(CD_SAMPLES_N=100000):
    theta = torch.rand(CD_SAMPLES_N) * 2 * torch.pi
    phi = torch.rand(CD_SAMPLES_N) * 2 * torch.pi
    x = 1 * torch.cos(theta) * torch.sin(phi)
    y = 1 * torch.sin(theta) * torch.sin(phi)
    z = 1 * torch.cos(phi)
    rays_d = torch.stack([x, y, z], 1)
    rays_o = torch.zeros_like(rays_d)
    return rays_o, rays_d


def angular_distance(x, y):
    deg = np.degrees(np.arccos(np.clip(np.sum(x * y, axis=-1), -1, 1)))
    mean = np.mean(deg)
    std = np.std(deg)
    return mean, std


rays_o, rays_d = get_rays()

# Additionaly, we could use the samples from inward rays.
# rays_o = torch.cat([rays_o, rays_d * 2], dim=0)
# rays_d = torch.cat([rays_d, -rays_d], dim=0)

our_samples, our_normals, our_mask = sample_surface_from_rays(rays_o, rays_d, our_mesh,
                                                              return_normal=True)

# Compute chamfer distance and angular distance
GT_RES = 256 if "small" == args.model_size else 512
if "mc" == args.method:
    method = "Cubes"
    RESOLS = [GT_RES, 16, 24, 32, 40, 48, 56, 64, 128, 192, 224]
elif "mtet" == args.method:
    method = "Tetrahedra"
    RESOLS = [GT_RES, 16, 32, 48, 64, 96]
    if "large" == args.model_size:
        RESOLS = [GT_RES, 16, 32, 48, 64, 96, 128, 192]

print(f"Marching {method} Results:")
print("#samples, #vertices, CD, AD, time")
for i in RESOLS:
    t = time.time()
    if "mc" == args.method or GT_RES == i:
        mc_mesh = run_marching_cubes(i)
    elif "mtet" == args.method:
        mc_mesh = run_marching_tetrahedras(i)
        # box_align(mc_mesh, gt_mesh)
    else:
        print("Undefined method!")

    t = time.time() - t
    try:
        mc_samples, mc_normals, mc_mask = \
            sample_surface_from_rays(rays_o, rays_d, mc_mesh, return_normal=True)
    except:
        print(f"{i:4d}, {0:5d}, {0:0.6f}, {0:4.1f}, {t:.2f}")
        continue
    if GT_RES == i:
        gt_mesh = mc_mesh
        gt_samples = mc_samples
        gt_normals = mc_normals
        gt_mask = mc_mask

        our_cd = chamfer_distance(our_samples, gt_samples)
        common_mask = our_mask & gt_mask
        our_ad, our_ad_std = angular_distance(
            our_normals[common_mask], gt_normals[common_mask])
        print(f"{'Ours'}, {our_mesh.vertices.shape[0]:5d}, {our_cd:0.6f}, "
              f"{our_ad:4.1f}, {our_t:.2f}")

    mc_cd = chamfer_distance(mc_samples, gt_samples)
    common_mask = mc_mask & gt_mask
    mc_ad, mc_ad_std = angular_distance(mc_normals[common_mask], gt_normals[common_mask])
    print(f"{i:4d}, {mc_mesh.vertices.shape[0]:5d}, {mc_cd:0.6f}, {mc_ad:4.1f}, {t:.2f}")
    mc_mesh.export(os.path.join(
        f"meshes/{args.dataset}",
        f'{args.method}{i:03d}_mesh_{args.model_size}_{seed}.ply'))

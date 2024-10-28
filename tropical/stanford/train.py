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


parser = argparse.ArgumentParser(
    prog="python -m tropical.stanford.train",
    description="Polyhedral complex derivation from piecewise trilinear networks")

parser.add_argument("-d", "--dataset", default="dragon",
                    choices=["bunny", "dragon", "happy", "armadillo", "drill", "lucy",
                             "bunny_npy"],
                    help="Stanford 3D scanning model name")
parser.add_argument("-s", "--seed", default=45, type=int, help="Seed")
parser.add_argument("-c", "--cache", default=True, action='store_false',
                    help="Cache the trained SDF?")
parser.add_argument("-m", "--model_size", default="small",
                    choices=["small", "medium", "large"], help="Model size")
parser.add_argument("-e", "--eval", default=False, action='store_true',
                    help="Run evaluation?")
parser.add_argument("-f", "--force", default=True, action='store_false',
                    help="Force flat assumption to skip curve approximation.")

args = parser.parse_args()
print(args)
seed = args.seed

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Constants
DIM = 3
CANVAS_SIZE = 1.2
BATCH_SIZE = 1000
EPOCH = 6 if "drill" == args.dataset else 10
CACHE = args.cache

if "small" == args.model_size:
    r_min = 2
    r_max = 32
elif "medium" == args.model_size:
    r_min = 4
    r_max = 64
elif "large" == args.model_size:
    r_min = 8
    r_max = 128
    # This option is not significant; but it matchs with the released models.
    T = 21 if "bunny" in args.dataset.lower() else 19

net = Net(num_layers=3, num_hidden=16, levels=4, r_min=r_min, r_max=r_max, T=T).cuda()
training_data = StanfordDataset(args.dataset)
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

T_max = EPOCH * len(training_data) / BATCH_SIZE
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)

# To run GUI event loop
plt.ion()

# Create a figure and axis
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
our_t = -1


def init_canvas():
    # Set the aspect ratio of the plot to be equal,
    # so the circle looks like a circle
    ax.set_aspect('equal', 'box')

    # Set axis labels
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_zlabel('Z-Axis')

    # Set the axis limits
    ax.set_xlim(-CANVAS_SIZE, CANVAS_SIZE)
    ax.set_ylim(-CANVAS_SIZE, CANVAS_SIZE)
    ax.set_zlim(-CANVAS_SIZE, CANVAS_SIZE)


def draw_canvas(it=50):
    # Draw lines
    ax.cla()  # clear the axes
    init_canvas()

    # Record starting time
    t = time.time()

    # Get polygons and draw them
    polygons, vertices, faces_with_indices = \
        sp.subpoly(net, DIM, CANVAS_SIZE, force=args.force)
    our_t = time.time() - t
    print(f" take {our_t:.2f}")

    if 0 < len(polygons):
        plane = Poly3DCollection(polygons, alpha=0.2, linewidths=0.5,
                                 edgecolors='r', facecolors='b')
        ax.add_collection3d(plane)

    # Grid points
    # x = net.preprocess_inverse(net.enc.marks).cpu()
    # grid = torch.meshgrid(x, x, x, indexing="ij")
    # ax.scatter(grid[0].reshape(-1), grid[1].reshape(-1), grid[2].reshape(-1))

    # draw canvas
    # fig.canvas.draw()
    # fig.canvas.flush_events()
    # time.sleep(0.1)

    # save image
    # os.makedirs('frames/', exist_ok=True)
    # plt.savefig(f'frames/iter_{it:03d}.png')  # every 10k

    return polygons, vertices, faces_with_indices, our_t


for epoch in range(EPOCH):  # loop over the dataset multiple times
    model_path = os.path.join(
        os.path.dirname(__file__),
        f"models/{args.dataset}/{args.dataset}_sdf_{args.model_size}_{seed}.pth")
    if CACHE and os.path.isfile(model_path):
        net.load_state_dict(torch.load(model_path, map_location=net.device()))
        print(f"The pretrained model loaded from {model_path}")
        polygons, vertices, faces_with_indices, our_t = draw_canvas()
        break
    elif 0 == epoch:
        print(f"warning: cannot find a pretrained model for seed ({seed})! This training"
              f" code is not guarantee the convergence of training nor a reliable SDF. "
              f"For the reproduction, please use the attached trained SDFs with the "
              f"corresponding seeds. Although the seed produced the same result with "
              f"our machine, it may differ in other circumstances.", flush=True)

    running_loss = 0.0
    train_dataloader.dataset.resample()
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        pred_sdf = net.sdf(inputs)[:, 0]

        # enforce_minmax
        minT = -0.2
        maxT = +0.2
        pred_sdf = torch.clamp(pred_sdf, minT, maxT)
        gt_sdf = torch.clamp(labels, minT, maxT)

        # l1-loss for SDF
        loss = criterion(pred_sdf, gt_sdf)
        l1_loss = loss.clone()

        # Eikonal loss to mimic sdf. otherwise, estimated sdf values have sharp slopes.
        points = inputs.clone()
        points.requires_grad = True
        J = torch.autograd.grad(net.sdf(points).sum(), points, create_graph=True)[0]
        loss += 1e-2 * (J.norm(p=2) - 1).pow(2) / BATCH_SIZE

        # Weight norm loss
        loss += 1e-1 * sum((1 - net.fc[i].weight.norm(p=2, dim=1)).pow(2).mean()
                           for i in range(len(net.fc))) / len(net.fc)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # print statistics
        running_loss += loss.item()

        if i % 10 == 9:    # print every 10000 mini-batches
            print(f"[{epoch + 1}, {i + 1:5d}] lr: {scheduler.get_last_lr()[0]:.4f}, "
                  f"loss: {running_loss / 10:.5f}", f"l1: {l1_loss / 10:.5f}", end="")
            running_loss = 0.0
            it = len(training_data) * epoch // BATCH_SIZE // 10 + (i + 1) // 10
            if 5 * EPOCH > it:
                print(" mesh calculation skipped.")
                continue
            else:
                if False:  # Sanity check on weight norm
                    for l in range(len(net.fc)):
                        for i in range(net.fc[l].weight.shape[0]):
                            print(f"{l}/{i} {net.fc[l].weight[i].norm(p=2)}")
            # torch.save(net.state_dict(), model_path)
            polygons, vertices, faces_with_indices, our_t = draw_canvas(it)

print('Finished training.', flush=True)

# save the model
if CACHE:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(net.state_dict(), model_path)


def make_gif(frame_folder):
    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"))]
    frame_one = frames[0]
    frame_one.save("sphere.gif", format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)


# make_gif("frames")

# Extract our mesh
vertices = vertices.cpu().numpy() / training_data.R
faces = np.array(faces_with_indices, dtype=list)


def extract_triangles(faces):
    triangles = []
    for i in range(len(faces)):
        assert 3 <= len(faces[i])
        if 3 < len(faces[i]):
            triangles += [[faces[i][0], faces[i][j], faces[i][j+1]]
                          for j in range(1, len(faces[i]) - 1)]
        elif 3 == len(faces[i]):
            triangles += [list(faces[i])]
    return triangles


# triangles = extract_triangles(faces)
triangles = faces

our_mesh = trimesh.Trimesh(vertices, triangles, process=False)
print(f"Ours: {our_mesh.vertices.shape}/{our_mesh.faces.shape}")

# Save meshes
os.makedirs(f"meshes/{args.dataset}", exist_ok=True)
our_mesh.export(os.path.join(f"meshes/{args.dataset}",
                             f'our_mesh_{args.model_size}_{seed}.ply'))

if not args.eval:
    exit()


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
    vertices /= training_data.R
    mc_vertices = vertices.astype(np.float32)
    mc_triangles = triangles.astype(np.int32)
    mc_mesh = trimesh.Trimesh(mc_vertices, mc_triangles, process=False)
    # print(f"MC: {mc_mesh.vertices.shape}/{mc_mesh.faces.shape}")
    return mc_mesh


def get_rays(CD_SAMPLES_N=100000):
    theta = torch.rand(CD_SAMPLES_N) * 2 * torch.pi
    phi = torch.rand(CD_SAMPLES_N) * 2 * torch.pi
    x = 1 * torch.cos(theta) * torch.sin(phi)
    y = 1 * torch.sin(theta) * torch.sin(phi)
    z = 1 * torch.cos(phi)
    rays_d = torch.stack([x, y, z], 1)
    rays_o = torch.zeros_like(rays_d)
    return rays_o, rays_d


rays_o, rays_d = get_rays()

# Additionaly, we could use the samples from inward rays.
# rays_o = torch.cat([rays_o, rays_d * 2], dim=0)
# rays_d = torch.cat([rays_d, -rays_d], dim=0)

our_samples, our_normals, our_mask = sample_surface_from_rays(rays_o, rays_d, our_mesh,
                                                              return_normal=True)


def angular_distance(x, y):
    deg = np.degrees(np.arccos(np.clip(np.sum(x * y, axis=-1), -1, 1)))
    mean = np.mean(deg)
    std = np.std(deg)
    return mean, std


# Compute chamfer distance and angular distance
print(f"Marching Cubes Results:")
print("#samples, #vertices, CD, AD, time")
for i in [512, 16, 24, 32, 40, 48, 56, 64, 128, 192, 224, 256]:
    t = time.time()
    mc_mesh = run_marching_cubes(i)
    t = time.time() - t
    try:
        mc_samples, mc_normals, mc_mask = \
            sample_surface_from_rays(rays_o, rays_d, mc_mesh, return_normal=True)
    except:
        print(f"{i:4d}, {0:5d}, {0:0.6f}, {0:4.1f}, {t:.2f}")
        continue
    if 512 == i:
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
    mc_mesh.export(os.path.join(f"meshes/{args.dataset}",
                                f'mc{i:03d}_mesh_{args.model_size}_{seed}.ply'))
print()

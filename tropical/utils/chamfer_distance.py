# TropicalNeRF
# Copyright (c) 2024-present NAVER Cloud Corp.
# CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)

import warnings
import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as Rot
import os, json
import torch
import random
import cubvh
import matplotlib.pyplot as plt

from packaging import version as pver

import logging
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

W = 800
H = 800

def chamfer_distance(x, y):
    nn_x = NearestNeighbors(
        n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(x)
    min_yx = nn_x.kneighbors(y)[0]

    nn_y = NearestNeighbors(
        n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(y)
    min_xy = nn_y.kneighbors(x)[0]

    return (np.mean(min_yx) + np.mean(min_xy)) / 2.


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, patch_size=1, coords=None):
    ''' get rays
    Args:
        poses: [N/1, 4, 4], cam2world
        intrinsics: [N/1, 4] tensor or [4] ndarray
        H, W, N: int
    Returns:
        rays_o, rays_d: [N, 3]
        i, j: [N]
    '''

    device = poses.device

    if isinstance(intrinsics, np.ndarray):
        fx, fy, cx, cy = intrinsics
    else:
        fx, fy, cx, cy = intrinsics[:, 0], intrinsics[:, 1], intrinsics[:, 2], intrinsics[:, 3]

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) # float
    i = i.t().contiguous().view(-1) + 0.5
    j = j.t().contiguous().view(-1) + 0.5

    results = {}

    if N > 0:

        if coords is not None:
            inds = coords[:, 0] * W + coords[:, 1]

        elif patch_size > 1:

            # random sample left-top cores.
            num_patch = N // (patch_size ** 2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1) # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1) # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0) # [np, p^2, 2]
            inds = inds.view(-1, 2) # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1] # [N], flatten


        else: # random sampling
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['i'] = i.long()
        results['j'] = j.long()

    else:
        inds = torch.arange(H*W, device=device)

    zs = -torch.ones_like(i) # z is flipped
    xs = (i - cx) / fx
    ys = -(j - cy) / fy # y is flipped
    directions = torch.stack((xs, ys, zs), dim=-1) # [N, 3]
    # do not normalize to get actual depth, ref: https://github.com/dunbar12138/DSNeRF/issues/29
    # directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = (directions.unsqueeze(1) @ poses[:, :3, :3].transpose(-1, -2)).squeeze(1) # [N, 1, 3] @ [N, 3, 3] --> [N, 1, 3]

    rays_o = poses[:, :3, 3].expand_as(rays_d) # [N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    # visualize_rays(rays_o[0].detach().cpu().numpy(), rays_d[0].detach().cpu().numpy())

    return results


def sample_surface(poses, intrinsics, mesh, N):

    # normalize
    vmin, vmax = mesh.bounds
    center = (vmin + vmax) / 2
    scale = 1 / (vmax - vmin)
    mesh.vertices = (mesh.vertices - center) * scale

    RT = cubvh.cuBVH(mesh.vertices, mesh.faces)

    # need to cast rays ...
    all_positions = []

    per_frame_n = N // len(poses)

    for pose in poses:

        pose = torch.from_numpy(pose).unsqueeze(0).cuda()
        rays = get_rays(pose, intrinsics, H, W, -1)
        rays_o = rays['rays_o'].contiguous().view(-1, 3)
        rays_d = rays['rays_d'].contiguous().view(-1, 3)

        positions, face_id, depth = RT.ray_trace(rays_o, rays_d)

        # depth = depth.detach().cpu().numpy().reshape(H, W, 1)
        # mask = depth >= 10
        # mn = depth[~mask].min()
        # mx = depth[~mask].max()
        # depth = (depth - mn) / (mx - mn + 1e-5)
        # depth[mask] = 0
        # depth = depth.repeat(3, -1)
        # plt.imshow(depth)
        # plt.show()

        mask = face_id >= 0
        positions = positions[mask].detach().cpu().numpy().reshape(-1, 3)

        try:
            indices = np.random.choice(len(positions), per_frame_n, replace=False)
        except:
            warnings.warn(f"Warning: {name<5} temporal error exception")
            indices = np.random.choice(len(positions), per_frame_n, replace=True)
        positions = positions[indices]

        all_positions.append(positions)

    all_positions = np.concatenate(all_positions, axis=0)

    # revert
    all_positions = (all_positions / scale) + center

    # scene = trimesh.Scene([mesh, trimesh.PointCloud(all_positions)])
    # scene.show()

    return all_positions

def sample_surface_from_rays(rays_o, rays_d, mesh, return_normal = False):

    # normalize
    # vmin, vmax = mesh.bounds
    # center = (vmin + vmax) / 2
    # scale = 1 / (vmax - vmin)
    # mesh.vertices = (mesh.vertices - center) * scale

    RT = cubvh.cuBVH(mesh.vertices, mesh.faces)

    # need to cast rays ...
    positions, face_id, depth = RT.ray_trace(rays_o, rays_d)

    mask = face_id >= 0
    positions = positions[mask].detach().cpu().numpy().reshape(-1, 3)

    # revert
    # positions = (positions / scale) + center

    # normal
    if return_normal:
        face_id[~mask] = 0
        faces = mesh.vertices[mesh.faces[face_id.detach().cpu().numpy()]]
        normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        normals /= (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-9)
        mask = mask.detach().cpu().numpy()
        return positions, normals, mask

    return positions

# TropicalNeRF
# Copyright (c) 2024-present NAVER Cloud Corp.
# CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)

from typing import *

import copy
import time
import numpy as np

from torch import Tensor
from torch.nn import Module
import torch
import torch.nn as nn
import torch.nn.functional as F

from tropical import TropicalHashGrid
import tropical.geometry as gm
import tropical.subpoly_debug as debug
from . import deprecated


@torch.no_grad()
def subpoly(net: Module, d: int, size: int, eps: float = 1e-4, force: bool = False) \
        -> List[List]:
    """Subdivision polygons algorithm

    Args:
        net (Module): Piece-wise linear networks
        d (int): Dimension
        size (int): Canvas size representing (-size, size)^d
        eps (flaot): A small number determining zero-level set
        force (bool): Force flat assumption to skip curve approximation

    Returns:
        List[List]: List of polygons
    """
    vertices = torch.Tensor([]).to(net.device())
    edges = torch.Tensor([]).to(net.device())

    # check hashgrid marks in the world-coordinate
    # print(net.enc.marks)
    # print(net.preprocess_inverse(net.enc.marks))

    # or, hashgrid provides a starting skeleton
    for m in net.modules():
        if isinstance(m, TropicalHashGrid):
            vertices, edges = m.skeleton(net)
            break

    if 0 == edges.shape[0]:  # we start with a hypercube
        vertices, edges, _ = get_hypercube(d, size)

    # get a device type
    vertices = vertices.to(net.device())
    edges = edges.to(net.device())

    # intermediate layers
    outputs = None
    for l in range(net.num_layers - 1):
        for h in range(net.num_hidden):
            # Subdivide polygons for the first layer
            vertices, edges, outputs = \
                subpoly_(vertices, edges, net, l, h, eps, outputs, force=force)

    # the final layer
    vertices, edges, outputs = \
        subpoly_(vertices, edges, net, net.num_layers-2, net.num_hidden, eps, outputs,
                 force=force)

    print()
    print(f"# of vertices and edges = {vertices.shape[0]}/{edges.shape[0]} => ", end="")

    vertices, edges, v_idx = extract_skeleton(vertices, edges, net, eps, outputs)
    outputs = outputs[v_idx]

    print(f"{vertices.shape[0]}/{edges.shape[0]}", end=", ")

    # if you wish to check with vertices
    # return vertices[edges].cpu().numpy(), vertices.cpu().numpy(), edges.cpu().numpy()

    faces, faces_with_indices = extract_faces(vertices, edges, net, outputs, eps)

    print(f"{len(faces)} faces", end=", ")

    return faces, vertices, faces_with_indices


# @debug.timeit
def subpoly_(vertices, edges, net, l, h, eps, outputs_=None, pruning=True, strict=True,
             force=False):
    if outputs_ is None:
        outputs_ = torch.cat(net(vertices, gather=True)[1], dim=-1)
    else:
        assert outputs_.shape[0] == vertices.shape[0]

    # Get the current plane distance
    idx = l * net.num_hidden + h
    outputs = outputs_[:, idx]  # V

    # 1. Subdivide edges
    d = outputs[edges]  # distances from a plane

    m = (d[:, 0] * d[:, 1]) < 0  # mask to divide (E)
    m &= (d[:, 0].abs() > eps) & (d[:, 1].abs() > eps)
    # hits = ((d[:, 0].abs() < eps) | (d[:, 1].abs() < eps)).sum()
    # if 0 < hits:
    #     print(f"there are the {hits} cases where a plane hit vertices!")

    if 0 == m.sum():  # stop. there is nothing to do.
        return vertices, edges, outputs_

    d = d[m] / eps
    e = vertices[edges][m]

    # 1-1. Linear cases
    w = d[:, :1].abs() / (d[:, 1:] - d[:, :1]).abs()  # linear weights (E)
    _regions, _offset, _ = net.region(vertices, output=outputs_)

    if not force:
        # 1-2. Bi- and tri-linear corrections
        c = 1 < ((e[:, 1, :] - e[:, 0, :]).abs() > eps).sum(dim=-1)

        # 1-2a-1. Coefficients using corner points
        corners = gm.corner_points(e[c]).view(-1, 3)  # Ex8x3 -> (Ex8)x3
        d = torch.cat(net(corners, gather=True, group=8)[1], dim=-1)
        d = d.view(-1, 8, d.shape[-1])  # Ex8xR

        # 1-2a-2. Two tri-linear regions
        regions = _regions[edges][m][c][:, :, 3:]  # Ex2xR
        r_edges = (regions[:, 0] == 0) & (regions[:, 1] == 0)  # ExR
    else:
        c = torch.ones(e.shape[0], device=e.device).bool()

    if not force and 0 < r_edges.shape[0]:
        debug.check_new_vertices_on_two_planes(edges, _regions, _offset, l, h, m, c, idx)

        # the plane idx which the edge lies
        inds = torch.ext.nonzero_last(r_edges[:, :idx])
        try:
            assert r_edges.shape[0] == inds.shape[0], \
                f"{r_edges.shape[0]} != {inds.shape[0]}"
        except:
            _k = r_edges.sum(-1) == 0
            print(r_edges.int()[_k].cpu())
            print(vertices[edges][m][c][_k].cpu())
            print(outputs_[edges][m][c][_k][:, :idx+1].cpu())
            exit()

        # the (curve) edge region distances
        p = d.gather(-1, inds[:, 1].view(-1, 1, 1).repeat(1, 8, 1)).squeeze(-1)

        # the plane region distances
        q = d[:, :, idx]

        # intersection of two (curved) planes
        ints = gm.intersection_of_two_planes(p, q, plane="xz")

        # TODO: refactor these lines!
        r_ints, _, _outputs = net.region(e[c, 0] * (1 - ints) + e[c, 1] * ints)
        d_new = torch.stack(
            [_outputs.gather(-1, inds[:, 1:]).squeeze(1), _outputs[:, idx]], dim=-1)

        # exclude no intersected edges
        gg = 0 < ((ints < 0) | (ints > 1)).sum(-1)

        # deal with corner cases using gradient descent
        ints, d_new = debug.deal_with_gradient_descent(
            c, d_new, e, eps, gg, idx, inds, ints, net)

        # check if new vertices are not on the current surface
        if 0 < (d_new[~gg].abs() > eps).sum():
            check_new_vertices_on_surface(
                c, d_new, e, eps, gg, h, inds, ints, l, net, p, q)
    else:
        ints = torch.empty(0, 3).to(e)
        d_new = torch.zeros(1, 2).to(e)

    # 1-3. New vertices
    v_new = e[:, 0] * (1 - w) + e[:, 1] * w  # linear part

    if not force:
        v_new[c] = e[c, 0] + ints * (e[c, 1] - e[c, 0])  # bi- and tri-linear part

    m_rgn, offset, outputs_new = net.region(v_new)
    m_idx = offset.shape[1] + idx
    m_rgn, m_rgn_ = m_rgn.split([m_idx, m_rgn.shape[1] - m_idx], dim=1)

    if debug.check_edges_with_new_vertices(d, edges, eps, h, idx, l, m, outputs_new,
                                           _regions, _offset, failover=True):
        # update `outputs_new` and the corresponding region mask to make sure that new
        # vertices are on two planes from edges and the current surface.
        m_rgn, offset, outputs_new = net.region(v_new, outputs_new)
        m_rgn, m_rgn_ = m_rgn.split([m_idx, m_rgn.shape[1] - m_idx], dim=1)

    # sanity check
    assert 0 < m_rgn.shape[0]

    if not strict:
        chk = outputs_new[:, idx]
        if chk.abs().max() > eps:
            g = chk.abs() < eps
            print(f"\n{(~g).sum()}/{g.numel()} > {eps} at {l}/{h} ({chk.abs().max()}).")
    elif not force and strict:
        m, v_new, m_rgn, m_rgn_, offset, outputs_new = debug.strict_check(
            c, d_new, eps, h, idx, ints, l, m, m_rgn, m_rgn_, offset, outputs_new,
            r_edges, v_new)

    # 1-2. Update left edges (new vertices take the second indices)
    temp = edges[m][:, 1].clone()
    edges[:, 1].masked_scatter_(
        m, torch.arange(v_new.shape[0]).to(edges) + vertices.shape[0])

    # 1-3. Add right edges
    e_new = torch.stack([temp, edges[m][:, 1].clone()], dim=-1)

    # 1-4. Add connecting edges
    v_idx = edges[m][:, 1]  # new vertices' indices
    c_new = []

    LEGACY_FIND_EDGES = False
    if LEGACY_FIND_EDGES == True:  # it's inefficient and incorrect so it's updated below.
        for r, vs in regions_to_vertices(m_rgn, offset).items():
            if 3 > len(vs):
                # print(f"number of vertices is {len(vs)}")
                continue
            idx = edges.new_tensor(vs)
            pv = plane_to_vertices(m_rgn[idx], offset[idx], len(net.enc.marks))
            n_idx = edges.new_tensor([v for k, v in pv.items() if 2 == len(v)])
            c_new.append(v_idx[idx][n_idx])
    else:
        # include hit vertices to make new connecting edges
        h_idx = outputs_[:, idx].abs() < eps
        v_rgn = torch.cat([m_rgn, _regions[h_idx, :m_idx]], dim=0)
        v_off = torch.cat([offset, _offset[h_idx]], dim=0)

        o_idx = torch.arange(vertices.shape[0], device=edges.device)[h_idx]
        v_idx = torch.cat([v_idx, o_idx], dim=0)
        c_new.append(v_idx[edge_vertices(vertices, v_rgn, net, v_off)])

    if 0 < len(c_new):
        c_new = torch.cat(c_new, dim=0)
        c_new, _ = c_new.sort(dim=-1)
        c_new = c_new.unique(dim=0)

    vertices_ = vertices
    vertices = torch.cat([vertices, v_new], dim=0)
    edges = torch.cat([edges, e_new], dim=0)
    if 0 < len(c_new):
        edges = torch.cat([edges, c_new], dim=0)

    # 1-5. Prune edges
    if h < net.num_hidden and pruning:
        # future region indicators for all vertices including new ones
        # _, m_prn, _ = region_indicator(net, vertices_, l, h, outputs_)
        m_prn, _, _ = net.region(vertices_, outputs_)
        m_prn = m_prn[:, m_idx:]
        m_prn = torch.cat([m_prn, m_rgn_], dim=0)
        assert m_prn.shape[0] == vertices.shape[0]

        # if two vertices shares future regions, remove the edges
        _, r_idx = m_prn.unique(dim=0, return_inverse=True)
        e_prn = r_idx[edges]
        p_idx = (e_prn[:, 0] != e_prn[:, 1])
        edges = edges[p_idx]

        # prune vertices, accordingly
        v_idx, r_idx = edges.view(-1).unique(return_inverse=True)

        # update the vertex indices in edges
        vertices = vertices[v_idx]
        edges = r_idx.view(-1, 2)

    # update outputs to be cached
    outputs_ = torch.cat([outputs_, outputs_new], dim=0)
    if h < net.num_hidden and pruning:
        outputs_ = outputs_[v_idx]

    return vertices, edges, outputs_


def regions_to_vertices(m: Tensor, offset: Tensor = None, return_inverse=False,
                        debug=False) -> Dict:
    """Calculate the map from regions to vertices. It requires little more memory
        to keep 2^3 variants but it does not need unique keys for each region.

    Args:
        m (Tensor): The mask indicating regions in [-1, 0, +1]^(V x (L x H)), while
                    [0, +1]^(V x D) for grid-based encodings along with `offset`.
        offset (Tensor, optional): A region offset for grid-based encodings.

    Returns:
        Dict: Regions to vertices
    """
    rv = {}

    # integer to left-aligned binary
    def _torbin(x): return [x % 2] + _torbin(x//2) if x > 1 else [x]

    # fill zeros with fixed size
    def torbin(x, place=2):
        y = _torbin(x)
        y += [0] * (place - len(y)) if place > len(y) else []
        return y

    if 0 == m.shape[0]:
        return rv

    k = (m == 0).sum(dim=1, keepdim=True)  # the number of planes on which a point lies
    dim = k.max().item()  # the maximum number of planes, caution: numerical instability
    mm = []  # mask holding variants

    if 0 < dim:  # need to consider variants
        if debug:
            t = time.time()
        for i in range(2 ** dim):
            s = m.new_tensor(torbin(i, dim)).unsqueeze(0).repeat(m.shape[0], 1)
            a = torch.arange(dim, device=m.device).unsqueeze(0).repeat(m.shape[0], 1)
            s = s * 2 - 1  # binary to [-1, +1]
            s = s[a < k]  # some samples may have extra zeros due to a numerical issue
            mm += [m.masked_scatter(m == 0, s)]

        # concat in this way to index
        m = torch.cat(mm, dim=-1).view(-1, m.shape[-1])
        if debug:
            print(f"  - build 8 variants took {t - time.time():.3f}s")

        if offset is not None and 0 < m.shape[0]:
            D = offset.shape[1]
            m[:, :D] = (m[:, :D] - 1) // 2
            m[:, :D] += offset.repeat(1, 2 ** dim).view(-1, D)

    # build a dict to map regions to belonging vertices
    if debug:
        t = time.time()
    _, r_idx = m.unique(dim=0, return_inverse=True)
    if debug:
        print(f"  - unique took {t - time.time():.3f}s")

    if return_inverse:
        return r_idx, dim

    if debug:
        t = time.time()
    for i in range(r_idx.shape[0]):
        k = r_idx[i].item()
        v = i // 2 ** dim
        rv[k] = rv.get(k, []) + ([v] if v not in rv.get(k, []) else [])
    if debug:
        print(f"  - build the rv map took {t - time.time():.3f}s")
    return rv


@deprecated
def edge_vertices_v1(vertices: Tensor, m: Tensor, offset: Tensor = None) -> Tensor:
    """Calculate new valid edges from vertex masks. Each edge should belong to the same
        region and two planes, simultaneously.

    Args:
        m (Tensor): The mask indicating regions in [-1, 0, +1]^(V x (L x H)), while
                    [0, +1]^(V x D) for grid-based encodings along with `offset`.
        offset (Tensor, optional): A region offset for grid-based encodings.

    Returns:
        Tensor: Edges with vertex indices (V'x2)
    """
    # input dimension
    D = offset.shape[1]
    m = m.half()
    offset = offset.half()

    # find vertices sharing the same plane. More-than-two vertices can lie on a plane.
    s_plane = (m == 0).half()
    m_plane = torch.einsum("ap, bp -> ab", s_plane[:, D:], s_plane[:, D:]) >= 1

    # find vertices sharing a grid plane.
    diff = offset.unsqueeze(1) - offset.unsqueeze(0)  # V x V x D
    g_plane = torch.einsum("ap, bp -> abp", m[:, :D] == 0, m[:, :D] == 0)  # V x V x D
    m_plane |= ((diff == 0) & (g_plane == 1)).sum(-1) >= 1

    # find vertices within the same plane region
    if 3 < m.shape[1]:
        r_plane = (m[:, D:].unsqueeze(1).half() - m[:, D:].unsqueeze(0).half()
                   ).abs().max(dim=-1)[0] <= 1
    else:
        r_plane = m_plane.clone().fill_(True)

    # find vertices within the same grid region
    r_grid = diff == 0
    r_grid |= diff == (m[:, :D].unsqueeze(0) == 0).float() * -1
    r_grid |= diff == (m[:, :D].unsqueeze(1) == 0).float()
    r_plane &= r_grid.sum(-1) == D

    # find edges
    return torch.triu(m_plane & r_plane, diagonal=1).nonzero()


@deprecated
def edge_vertices_v2(vertices: Tensor, m: Tensor, net: Module, offset: Tensor = None) \
        -> Tensor:
    """Calculate new valid edges from vertex masks. Each edge should belong to the same
        region and a plane, except the current intersecting plane, simultaneously.

    Args:
        vertices (Tensor): Vertices.
        m (Tensor): The mask indicating regions in [-1, 0, +1]^(V x (L x H)), while
                    [0, +1]^(V x D) for grid-based encodings along with `offset`.
        net (Module): Networks to get the jacobians.
        offset (Tensor, optional): A region offset for grid-based encodings.

    Returns:
        Tensor: Edges with vertex indices (V'x2)
    """
    # Find regions
    r_idx, dim = regions_to_vertices(m, offset, return_inverse=True)
    v_indices = r_idx_as_tensor(r_idx, dim, offset)

    mean_points, points, v_indices = mean_points_with_valid(
        vertices, v_indices, return_points=True)

    mean_points.requires_grad = True
    jacobians = net.normal(mean_points)

    # Connecting neighbors using Jacobians
    faces, indices = gm.sort_polygon_vertices_batch(points, jacobians, return_index=True)
    v_indices = v_indices.gather(1, indices)
    counts = (v_indices != -1).sum(dim=1)

    # Left-align
    v_indices_ = v_indices.new_zeros(v_indices.shape[0], counts.max() + 1).fill_(-1)
    mask = torch.arange(counts.max() + 1, device=offset.device).unsqueeze(0).repeat(
        v_indices.shape[0], 1) < counts.unsqueeze(1)
    v_indices_.masked_scatter_(mask, v_indices[v_indices != -1])

    # Consider loops with the first elements
    mask = torch.arange(counts.max() + 1, device=offset.device).unsqueeze(0).repeat(
        v_indices.shape[0], 1) == counts.unsqueeze(1)
    v_indices_.masked_scatter_(mask, v_indices_[:, 0])

    out = []
    # sorting and connecting edges. TODO: fix missing edges..
    for i in range(3, counts.max() + 1):  # each face has different number of vertices
        for j in range(i):
            out += [v_indices_[counts == i][:, j:j+2]]

    output = torch.cat(out, dim=0)
    output = output.unique(dim=0)

    # Exclude self-edges
    output = output[output[:, 0] != output[:, 1]]

    # Check the number of planes
    D = offset.shape[1]
    chk1 = (m == 0)[output]
    zero_counts = (chk1[:, 0] * chk1[:, 1]).sum(dim=-1)  # zero-match counts

    # Discount the grid plane indices mismatch
    chk2 = offset[output]
    zero_counts -= (
        (chk1[:, 0, :D] * chk1[:, 1, :D]) * (chk2[:, 0] - chk2[:, 1]) != 0).sum(dim=-1)

    output = output[zero_counts >= 1]  # except the current plane, so >= 1.
    return output


def edge_vertices(vertices: Tensor, m: Tensor, net: Module, offset: Tensor = None) \
        -> Tensor:
    """Calculate new valid edges from vertex masks. Each edge should belong to the same
        region and a plane, except the current intersecting plane, simultaneously.

    Args:
        vertices (Tensor): Vertices.
        m (Tensor): The mask indicating regions in [-1, 0, +1]^(V x (L x H)), while
                    [0, +1]^(V x D) for grid-based encodings along with `offset`.
        net (Module): Networks to get the jacobians.
        offset (Tensor, optional): A region offset for grid-based encodings.

    Returns:
        Tensor: Edges with vertex indices (V'x2)
    """
    # Find regions
    r_idx, dim = regions_to_vertices(m, offset, return_inverse=True)
    v_indices = r_idx_as_tensor(r_idx, dim, offset)
    counts = (v_indices != -1).sum(dim=1)

    def extract_every_valid_edge(input_tensor, C):  # remove inner-loop
        out = []
        for i in range(1, C):
            a = v_indices[:, i]
            m = a != -1  # exploit the fact that input is left-aligned
            a = a[m].repeat(i)
            b = v_indices[m, :i].t().flatten()
            out += [torch.stack([a, b], dim=1)]
        return torch.cat(out, dim=0)

    # Extract every edge
    output = extract_every_valid_edge(v_indices, counts.max())
    output = output.unique(dim=0)

    # Exclude self-edges
    output = output[output[:, 0] != output[:, 1]]

    # Check the number of planes
    D = offset.shape[1]
    chk1 = (m == 0)[output]
    zero_counts = (chk1[:, 0] * chk1[:, 1]).sum(dim=-1)  # zero-match counts

    # Discount the grid plane indices mismatch
    chk2 = offset[output]
    zero_counts -= (
        (chk1[:, 0, :D] * chk1[:, 1, :D]) * (chk2[:, 0] - chk2[:, 1]) != 0).sum(dim=-1)

    output = output[zero_counts >= 1]  # except the current plane, so >= 1.
    return output


def plane_to_vertices(m, offset: Tensor = None, L: int = None):
    pv = {}
    d = 0 if offset is None else offset.shape[1]
    base = 0 if offset is None else L * d

    # planes from neural networks
    for x in (m[:, d:] == 0).nonzero().cpu().numpy():
        k = base + x[1]
        pv[k] = pv.get(k, []) + ([x[0]] if x[0] not in pv.get(k, []) else [])

    # planes from a grid-based encoding
    if offset is not None:
        for x in (m[:, :offset.shape[1]] == 0).nonzero().cpu().numpy():
            k = offset[x[0], x[1]].item() + L * x[1]
            pv[k] = pv.get(k, []) + ([x[0]] if x[0] not in pv.get(k, []) else [])
    return pv


def extract_skeleton(vertices, edges, net, eps, outputs=None):
    # vertices on the surfaces
    if outputs is None:
        m = net.sdf(vertices)[:, 0].abs() < eps
    else:
        m = outputs[:, -1].abs() < eps

    # vertices within valid range of [0, 1]^D
    v = net.preprocess(vertices)
    m[(v > 1).sum(dim=-1) > 0] = False
    m[(v < 0).sum(dim=-1) > 0] = False

    if 3 > m.sum():
        return torch.Tensor([]).to(edges), torch.Tensor([]).to(edges), None

    # edges on the surfaces
    edges = edges[m[edges].sum(dim=-1) == 2]

    # prune vertices, accordingly
    v_idx, r_idx = edges.view(-1).unique(return_inverse=True)

    # update the vertex indices in edges
    vertices = vertices[v_idx]
    edges = r_idx.view(-1, 2)

    return vertices, edges, v_idx


def extract_faces(vertices: Tensor, edges: Tensor, net: Module, outputs: Tensor = None,
                  eps: float = None) -> Tuple[List[List[float]], List[List[int]]]:
    """Extract faces using the list of vertices and edges.

    Args:
        vertices (Tensor): A tensor (Vx3)
        edges (Tensor): A tensor (Ex2)
        net (Module):
        outputs (Tensor): Pre-calculated outputs using `vertices`

    Returns:
        Tuple[List[List[float]], List[List[int]]]: faces with positions and faces with
                                                   the indices of vertices
    """
    DEBUG = False
    if 0 == vertices.shape[0]:
        return [], []

    # initialize outputs
    faces = []
    faces_with_indices = []

    m_rgn, offset, _ = net.region(vertices, outputs, eps)
    D = offset.shape[1]

    if DEBUG:
        t = time.time()
    # r2v = regions_to_vertices(m_rgn[:, :-1], offset)
    r_idx, dim = regions_to_vertices(m_rgn[:, :-1], offset, return_inverse=True)
    if DEBUG:
        print(f"- regions_to_vertices took {time.time() - t:.3f}")

    if DEBUG:
        t = time.time()
    # v_indices = regions_to_vertices_as_tensor(r2v).to(edges.device)
    v_indices = r_idx_as_tensor(r_idx, dim, edges)

    # deduplication
    v_indices = v_indices.unique(dim=0)

    if DEBUG:
        print(f"- regions_to_vertices_as_tensor took {time.time() - t:.3f}")

    if DEBUG:
        t = time.time()
    mean_points, points, v_indices = mean_points_with_valid(
        vertices, v_indices, return_points=True)
    if DEBUG:
        print(f"- mean_points_with_valid took {time.time() - t:.3f}")

    if DEBUG:
        t = time.time()
    mean_points.requires_grad = True
    jacobians = net.normal(mean_points)
    if DEBUG:
        print(f"- net.normal() took {time.time() - t:.3f}")

    if DEBUG:
        t = time.time()
    # TODO: `faces` has duplicated faces. Use `faces_with_indices` instead.
    faces, indices = gm.sort_polygon_vertices_batch(points, jacobians, return_index=True)
    if DEBUG:
        print(f"- sort_polygon_vertices_batch took {time.time() - t:.3f}")

    if DEBUG:
        t = time.time()
    faces_with_indices = tensor_to_triangle_faces(v_indices.gather(1, indices))
    if DEBUG:
        print(f"- build tensor_to_triangle_faces took {time.time() - t:.3f}")

    return faces, faces_with_indices


def regions_to_vertices_as_tensor(data_dict, null_value=-1):
    # Extract values from the dictionary
    values = list(data_dict.values())

    # Determine the maximum length of the lists
    max_length = max(len(item) for item in values)

    # Pad the lists with -1
    padded_values = \
        [item + [null_value] * (max_length - len(item)) for item in values]

    # Convert the list of lists into a PyTorch tensor
    return torch.tensor(padded_values)


def r_idx_as_tensor(r_idx: Tensor, dim: int, tensor: Tensor, null_value=-1) \
        -> Tensor:
    """Make a index tensor (Region) x (A left-aligned list of vertex indices)
        using `masked_scatter`. Caution: duplicated elements.

    Args:
        r_idx (Tensor): The region indices for where vertices belongs to
        dim (int): The maximum hyperplane-intersection counts (usually 3)
        tensor (Tensor): To get the current working dtype and device
        null_value (TYPE, optional): Masking value

    Returns:
        Tensor: Left-aligned region x a list of vertex indices
    """
    # Get unique values and counts
    unique_values, counts = torch.unique(r_idx, return_counts=True)

    # Determine the maximum length of the lists (max # of vertices per polygon)
    max_length = counts.max()

    # Create the output tensor with the specified null value
    output = torch.full((r_idx.max() + 1, max_length), null_value,
                        dtype=tensor.dtype, device=tensor.device)

    # Calculate indices and fill the output tensor.
    # Vertex indices are not sorted although it doesn't matter
    # since we will sort the vertices using normal to define a polygon.
    indices = torch.stack(
        [r_idx, torch.arange(r_idx.shape[0], device=r_idx.device)], dim=1)
    indices = indices[indices[:, 0].sort()[1]]

    # Find the original vertex indices
    indices[:, 1] //= 2 ** dim

    # Efficient updating using counts
    mask = torch.arange(max_length, dtype=tensor.dtype, device=tensor.device
                        ).unsqueeze(0).repeat(output.size(0), 1) < counts.unsqueeze(1)
    output.masked_scatter_(mask, indices[:, 1])

    return output


def mean_points_with_valid(vertices, v_indices, dim=1, null_value=-1,
                           return_points=False):
    points = vertices[v_indices + (v_indices == null_value)]
    points[v_indices == null_value] = 0  # mask zero for invalid

    Z = (v_indices != null_value).sum(dim=dim, keepdim=True)
    mean_points = points.sum(dim=dim) / Z
    m = Z.squeeze() >= 3  # exclude less-than-three-vertices cases

    return mean_points[m], points[m], v_indices[m]


@deprecated("Improved time complexity.")
def tensor_to_polygon_faces(tensor, null_value=-1, debug=False):
    output = []
    invalid_cnt = 0
    for row in tensor.cpu().numpy():
        # unique and unsorted indices
        arr = row[row != null_value]
        unique_values, indices = np.unique(arr, return_index=True)
        unique_values_in_order = arr[np.sort(indices)]
        vertices = unique_values_in_order.tolist()
        if 2 < len(vertices):
            output += [vertices]
        else:
            invalid_cnt += 1
    if debug and 0 < invalid_cnt:
        print(f"warning: invalid faces with less than three vertices: {invalid_cnt}")
    return output


def tensor_to_triangle_faces(tensor, null_value=-1, debug=False):
    # Remove duplicated indices
    for i in range(tensor.size(1)):
        m = tensor[:, :i] == tensor[:, i:i+1]
        tensor[0 < m.sum(-1), i] = null_value

    # Extract triangle indices
    mask = tensor != null_value
    counts = mask.sum(-1)
    cumsum = counts.cumsum(0)
    indices = torch.cat([counts.new_zeros(1), cumsum[:-1]], dim=0).long()
    vertices = tensor[mask].view(-1)  # 1d valid vertex indices
    v0_ = vertices[indices]  # the first elements

    faces = []
    indices += 1
    m = indices < cumsum

    for i in range(counts.max() - 2):
        v0 = v0_
        indices += 1
        m &= indices < cumsum
        indices[m]
        v1 = vertices[(indices - 1)[m]]
        v2 = vertices[indices[m]]
        v0 = v0[m]
        faces += [torch.stack([v0, v1, v2], dim=1)]
    faces = torch.cat(faces, dim=0)
    return faces.cpu().numpy()


def get_hypercube(d, size):
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

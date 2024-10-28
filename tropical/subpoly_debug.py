# TropicalNeRF
# Copyright (c) 2024-present NAVER Cloud Corp.
# CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)

import functools
import time

from torch import Tensor
import torch
import torch.nn.functional as F

import tropical.geometry as gm


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Hook before calling the function
        start_time = time.time()

        # Call the original function
        result = func(*args, **kwargs)

        # Hook after calling the function
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds.")

        return result
    return wrapper


def check_edges_with_new_vertices(d, edges, eps, h, idx, l, m, outputs_new, _regions,
                                  _offset, failover=False, silence=True, debug=False):
    m_rgn = _regions[edges][m]
    offset = _offset[edges][m]
    m_chk = (m_rgn[:, 0] == 0) & (m_rgn[:, 1] == 0)
    m_chk[:, :3] &= offset[:, 0] == offset[:, 1]
    if failover:
        b = m_chk[:, 3:]  # 3d off-grid
        b[:, idx:] = False  # fill `False` in out-of-scope
        b[:, idx] = True  # fill `True` at the current idx
        c = outputs_new[b].abs() > eps
        if 0 < c.sum():
            if not silence:
                print(f"🎯 {c.sum()}/{c.numel()} > {eps} at {l}/{h}.")
                print(m_chk[:, :3+idx+1].sum(-1).unique(return_counts=True))
            outputs_new.masked_fill_(b, 0)  # override edge
            return True
    else:
        return False


def check_planary_among_vertices(vertices, v_indices, null_value=-1, eps=1e-4):
    v_indices = torch.ext.batched_unique_consecutive(v_indices, null_value=null_value)

    points = vertices[v_indices + (v_indices == null_value)]
    points[v_indices == null_value] = 0  # mask zero for invalid

    counts = (v_indices != null_value).sum(-1)
    def vdir(x, y): return F.normalize(torch.cross(x, y), dim=1)
    def vdot(x, y): return (x * y).sum(1)

    d = vertices.new_zeros(points.shape[0], counts.max()).fill_(null_value)

    for i in range(2, counts.max()):
        m = (v_indices[:, i] != null_value)
        if 0 < m.sum():
            n = vdir(points[m, 1] - points[m, 0], points[m, 2] - points[m, 0])
            v = vdir(points[m, 1] - points[m, 0], points[m, i] - points[m, 0])
            d[m, i] = vdot(n, v)

    chk = (0 != d) & (null_value != d) & (1 - eps > d.abs())
    if 0 < chk.sum():
        m = 0 < chk.sum(-1)
        import pdb
        pdb.set_trace()


def check_new_vertices_on_two_planes(edges, _regions, _offset, l, h, m, c, idx):
    """
    Check if new vertices are on two planes.

    Args:
        edges (array-like): Indices for edges.
        _regions (dict): Dictionary containing regions of vertices.
        _offset (dict): Dictionary containing offsets for vertices.
        l (int): Layer index
        h (int): Neuron index
        m (int): Valid-edge mask
        c (int): Bi-linear and tri-linear mask
        idx (int): Index modifier for slicing regions.
    """
    m_rgn = _regions[edges][m][c][:, :, :3 + idx]
    offset = _offset[edges][m][c]
    _m_chk = (m_rgn[:, 0] == 0) & (m_rgn[:, 1] == 0)
    _m_chk[:, :3] &= offset[:, 0] == offset[:, 1]
    m_chk = _m_chk.sum(-1)

    if 0 < (m_chk < 2).sum():
        print("warning: two vertices of an edge must be on at least two planes!")
        print(f"{(m_chk < 2).sum()} / {m_chk.size} {l}/{h}")
        print(_m_chk[m_chk < 2].astype(int))
        print(edges[c][m_chk < 2])


# check if new vertices on three planes?
# usage: isInvalid, m_chk = check_new_vertices(h, m_chk, l)
def check_new_vertices(h, m_chk, l, silence=False, debug=False):
    m_chk = m_chk.sum(-1)  # including the current idx
    if 0 < (m_chk < 3).sum():
        if not silence:
            print("warning: new vertices must on at least three planes!")
            print(f"{(m_chk < 3).sum()} / {m_chk.numel()} {l}/{h} (idx={idx})")
            print(m_chk.unique(return_counts=True))
        return True, m_chk
    else:
        return False, m_chk


def deal_with_gradient_descent(c, d_new, e, eps, gg, idx, inds, ints, net):
    """
    Deal with corner cases using gradient descent.

    Args:
        c (int): Index for the current edge.
        d_new (Tensor): Tensor containing new distances.
        e (Tensor): Tensor containing edge points.
        eps (float): Epsilon for convergence criteria. Default is 1e-6.
        gg (Tensor): Mask for excluding no intersected edges
        idx (int): Index for slicing outputs.
        inds (Tensor): Tensor containing indices.
        ints (Tensor): Tensor containing intersection points.
        net (torch.nn.Module): Neural network model.
    """
    gd = ~gg & (0 < (d_new.abs() > eps).sum(dim=-1))

    if 0 < gd.sum():
        with torch.enable_grad():
            x = ints[gd]  # copied by indexing
            x.requires_grad = True

            i = 0
            d0 = d1 = torch.tensor(1.0)
            # Though, we do not expect this optimizing process is needed.
            while ((d0.abs().max() > eps) | (d1.abs().max() > eps)) and i < 500:
                x_ = e[c][gd, 0] + x * (e[c][gd, 1] - e[c][gd, 0])
                outputs = torch.cat(net(x_, gather=True)[1], dim=-1)
                d0 = outputs.gather(-1, inds[gd, 1:]).squeeze(1)
                d1 = outputs[:, idx]
                y = (d0.pow(2) + d1.pow(2)).sum()
                x.data -= 1e-2 * F.normalize(torch.autograd.grad(y, x)[0])
                x.data.clamp_(0, 1)
                if 50 == i + 1:
                    print(" ", end="")
                if (i + 1) % 50 == 0:
                    # print(i + 1, d0.abs().max().item(), d1.abs().max().item(),
                    #       x[0].detach().cpu().numpy())
                    print("*", end="")
                i += 1
        ints[gd] = x
        d_new[gd, 0] = d0
        d_new[gd, 1] = d1

    return ints, d_new


def check_new_vertices_on_surface(c, d_new, e, eps, gg, h, inds, ints, l, net, p, q):
    """
    Check if new vertices are not on the current surface.

    Args:
        c (int): Index for the current edge.
        d_new (torch.Tensor): Tensor containing new distances.
        e (torch.Tensor): Tensor containing edge points.
        eps (float): Epsilon for convergence criteria.
        gg (torch.Tensor): Tensor used in the gradient descent logic.
        h (int): Some parameter related to the edges or planes.
        inds (torch.Tensor): Tensor containing indices.
        ints (torch.Tensor): Tensor containing intersection points.
        l (int): Some parameter related to the edges or planes.
        net (torch.nn.Module): Neural network model.
        p (torch.Tensor): Tensor p for interpolation.
        q (torch.Tensor): Tensor q for interpolation.
    """
    if 0 < (d_new[~gg].abs() > eps).sum():
        print(f"check if the below ints. d to be near-zeros "
              f"({d_new[~gg].abs().max()} > {eps}) at {l}/{h}")
        print(f"see the indices of "
              f"({d_new[~gg, 0].abs().max(0)[1]}, {d_new[~gg, 1].abs().max(0)[1]})")
        print(torch.cat([
            torch.arange(d_new[~gg].shape[0], device=d_new.device).unsqueeze(-1),
            d_new[~gg]], dim=-1))
        test_idx = d_new[~gg, 1].abs().max(0)[1]
        debug_test_idx(test_idx, e, c, net, gg, p, q, ints, debug, inds)


def debug_test_idx(test_idx, gg, ints, e, c, net, p, q, inds):
    print(f"-------------------------------------------")
    print(f"test_idx: {test_idx}")
    print(e[c][~gg][test_idx].cpu())
    print(ints[~gg][test_idx].cpu())
    print("check if the below inters. d to be near-zeros:")
    print(gm.trilinear_interpolation(p, ints)[~gg][test_idx].cpu())
    print(gm.trilinear_interpolation(q, ints)[~gg][test_idx].cpu())
    print("p and q:")
    print(p[~gg][test_idx].cpu())
    print(q[~gg][test_idx].cpu())

    def quad_y(q, r, s, x):
        X = torch.stack([(1 - x)**2, x * (1 - x), x * (1 - x), x**2], dim=-1)
        AX = (q[:, r] * X).sum(-1)
        BX = (q[:, s] * X).sum(-1)
        print(f"\tAX: {AX[0]}, BX: {BX[0]}")
        return AX / (AX - BX)
    r = p.new_tensor([0, 1, 4, 5]).long()  # lower grid
    s = p.new_tensor([2, 3, 6, 7]).long()  # upper grid

    print("y:")
    print(quad_y(q[~gg][test_idx].unsqueeze(0), r, s,
                 ints[~gg][test_idx, 0].unsqueeze(0)))
    print(quad_y(p[~gg][test_idx].unsqueeze(0), r, s,
                 ints[~gg][test_idx, 0].unsqueeze(0)))

    print("vertices recheck")
    _, _, _outputs_v0 = net.region(e[c, 0])
    _, _, _outputs_v1 = net.region(e[c, 1])
    print(_outputs_v0.gather(-1, inds[:, 1:]).squeeze(1)[test_idx].cpu())
    print(_outputs_v1.gather(-1, inds[:, 1:]).squeeze(1)[test_idx].cpu())
    print(f"-------------------------------------------")
    exit()


def strict_check(c, d_new, eps, h,
                 idx, ints, l, m, m_rgn, m_rgn_, offset, outputs_new,
                 r_edges, v_new):
    chk = outputs_new[:, idx]  # V
    if (chk.abs().max() >= eps) | (d_new[:, 0].abs().max() >= eps) | \
            (0 < r_edges.shape[0]):
        g = chk.abs() < eps

        if 0 < r_edges.shape[0]:
            # exclude no trilinear-intersected edges
            gg = 0 < ((ints < 0) | (ints > 1)).sum(-1)
            g[c] |= gg  # permit now for passing filter
            d_new[:, 0][gg] = 0

        if 0 < (~g).sum():
            print(f"\n{(~g).sum()}/{g.numel()} new vertices are filtered at "
                  f"{l}/{h} ({chk[~g].abs().max()}).")

        if eps < d_new[:, 0].abs().max():
            g1 = (d_new[:, 0].abs() < eps)  # & (d_new[:,1].abs() < eps)
            print(f"\n{(~g1).sum()}/{g1.numel()} old vertices are filtered at "
                  f"{l}/{h} ({d_new[~g1].abs().max()}).")

        if 0 < r_edges.shape[0]:
            g[c] = (chk[c].abs() < eps) & ~gg
            # print(f"{gg.sum()}/{gg.numel()} have no roots at {l}/{h}.")
            if eps < d_new[:, 0].abs().max():
                g[c] &= g1

        # failover
        m.masked_scatter_(m, g)
        v_new = v_new[g]
        m_rgn = m_rgn[g]
        m_rgn_ = m_rgn_[g]
        offset = offset[g]
        outputs_new = outputs_new[g]

    return m, v_new, m_rgn, m_rgn_, offset, outputs_new

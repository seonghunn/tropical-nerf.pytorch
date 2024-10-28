# TropicalNeRF
# Copyright (c) 2024-present NAVER Cloud Corp.
# CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)

import warnings
from typing import *
from deprecation import deprecated

import time
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import tropical.torch_ext as torch_ext
torch.ext = torch_ext

warnings.simplefilter('always', DeprecationWarning)


def intersection_of_two_planes(p: Tensor, q: Tensor, plane="xz", eps=1e-6) -> Tensor:
    """Intersection of two planes and a given plane in a trilinear cube.

    Args:
        p (Tensor): Eight corner distances from the `p` plane in Bx8.
        q (Tensor): Eight corner distances from the `q` plane in Bx8.
        plane (str, optional): A given plane in a cube which the intersections lie.
    """
    # groups from the trilinear interpolation and the xz-plane assumption
    assert "xz" == plane

    # Tx' = x, x' = [a,b,c] where a(1-x)^2 + bx(1-x) + cx^2
    T = p.new_tensor([[[1, -2, 1], [-1, 1, 0], [1, 0, 0]]])  # 1x3x3

    # outputs
    x = p.new_tensor(p.shape[0])
    y = p.new_tensor(p.shape[0])

    r = p.new_tensor([0, 1, 4, 5]).long()  # lower grid
    s = p.new_tensor([2, 3, 6, 7]).long()  # upper grid

    def z(x): return torch.stack([x[:, 0], x[:, 1] + x[:, 2], x[:, 3]], dim=-1)

    # x^TAx = 0; x = [(1-x)^2, x(1-x), x^2]
    A0 = z(q[:, r]).unsqueeze(2) * z(p[:, s]).unsqueeze(1)
    A1 = z(q[:, s]).unsqueeze(2) * z(p[:, r]).unsqueeze(1)
    A = A0 - A1
    B = T.transpose(1, 2) @ A @ T
    coeffs = torch.stack([B[:, 0, 0],
                          B[:, 1, 0] + B[:, 0, 1],
                          B[:, 2, 0] + B[:, 1, 1] + B[:, 0, 2],
                          B[:, 1, 2] + B[:, 2, 1],
                          B[:, 2, 2]], dim=-1)

    x = batched_polynomial_roots(coeffs)
    z = x.clone()

    def quad_y(q, r, s, x):
        X = torch.stack([(1 - x)**2, x * (1 - x), x * (1 - x), x**2], dim=-1)
        AX = (q[:, r] * X).sum(-1)
        BX = (q[:, s] * X).sum(-1)
        return AX / (AX - BX)

    y = quad_y(q, r, s, x)

    # xz-bilinear case
    def intersection_of_bilinear(p, q, m, t, u, x, y, z):
        T = p.new_tensor([[[-1, 1], [1, 0]]])  # 1x2x2
        A0 = q[m][:, t].unsqueeze(2) * p[m][:, u].unsqueeze(1)
        A1 = q[m][:, u].unsqueeze(2) * p[m][:, t].unsqueeze(1)
        A = A0 - A1
        B = T.transpose(1, 2) @ A @ T
        coeffs = torch.stack([B[:, 0, 0],
                              B[:, 1, 0] + B[:, 0, 1],
                              B[:, 1, 1]], dim=-1)
        x[m] = batched_polynomial_roots(coeffs)

        def linear_y(p, t, u, x):
            X = torch.stack([(1 - x), x], dim=-1)
            AX = (p[:, t] * X).sum(-1)
            BX = (p[:, u] * X).sum(-1)
            return AX / (AX - BX)

        z[m] = linear_y(p[m], t, u, x[m])
        y[m] = .5

        failover = False
        if failover:
            # failover using derivatives to find the closest if there is no root.
            assert coeffs.shape[1] == 3  # quadratic equation
            d_coeffs = torch.stack([2 * coeffs[:, 0], coeffs[:, 1]], dim=-1)
            x0 = batched_polynomial_roots(d_coeffs).clamp(0, 1)
            chk = torch.stack([
                coeffs[:, 0] * x0**2 + coeffs[:, 1] * x0 + coeffs[:, 2],
                coeffs[:, 2], coeffs.sum(-1)], dim=-1)
            sel = (chk.abs().min(-1)[1]).float()
            sel[sel == 0] = x0[sel == 0]
            sel[sel == 1] = 0
            sel[sel == 2] = 1
            m__ = (x[m] < 0) | (z[m] < 0) | (z[m] > 1)
            m_ = m.masked_scatter(m, m__)
            x[m_] = sel[m__]
            z[m_] = linear_y(q[m_], t, u, x[m_]).clamp(0, 1)
        else:
            x[m] = y[m] = z[m] = -1

    planes = p.new_zeros(p.shape[0])
    for plane in ["xz", "xy", "yz"]:
        if "xz" == plane:
            # m = A.abs().sum(-1).sum(-1) < eps
            t = p.new_tensor([0, 1, 4, 5]).long()  # lower grid
            u = p.new_tensor([2, 3, 6, 7]).long()  # upper grid
            m = ((p[:, t] == p[:, u]) & (q[:, t] == q[:, u])).sum(-1) == 4
            planes[m] = 1
            t = p.new_tensor([0, 1]).long()  # lower grid
            u = p.new_tensor([6, 7]).long()  # upper grid
            intersection_of_bilinear(p, q, m, t, u, x, y, z)
        elif "xy" == plane:
            t = p.new_tensor([0, 1, 2, 3]).long()  # lower grid
            u = p.new_tensor([4, 5, 6, 7]).long()  # upper grid
            m = ((p[:, t] == p[:, u]) & (q[:, t] == q[:, u])).sum(-1) == 4
            planes[m] = 2
            t = p.new_tensor([0, 1]).long()  # lower grid
            u = p.new_tensor([6, 7]).long()  # upper grid
            intersection_of_bilinear(p, q, m, t, u, x, z, y)
        else:
            t = p.new_tensor([0, 4, 2, 6]).long()  # lower grid
            u = p.new_tensor([1, 5, 3, 7]).long()  # upper grid
            m = ((p[:, t] == p[:, u]) & (q[:, t] == q[:, u])).sum(-1) == 4
            planes[m] = 3
            t = p.new_tensor([0, 2]).long()  # lower grid
            u = p.new_tensor([5, 7]).long()  # upper grid
            intersection_of_bilinear(p, q, m, t, u, y, x, z)

    return torch.stack([x, y, z], dim=-1)


@deprecated()
def corner_points_coeffs(y: Tensor) -> Tensor:
    """Coefficients of cubic-Bézier curves defined by eight corner points

    Args:
        y (Tensor): The points in Bx8 which defines the cubic Bézier function.

    Returns:
        Tensor: Coefficient of cubic-Bézier curves
    """
    G0 = y[:, 0]
    G1 = y[:, 1] + y[:, 2] + y[:, 4]
    G2 = y[:, 3] + y[:, 5] + y[:, 6]
    G3 = y[:, 7]
    return torch.stack(
        [-G0 + G1 - G2 + G3, 3 * G0 + -2 * G1 + G2, -3 * G0 + G1, G0], dim=-1)


def trilinear_interpolation(p: Tensor, w: Tensor):
    out = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                weight = 1
                weight *= 1 - w[:, 0] if k == 0 else w[:, 0]
                weight *= 1 - w[:, 1] if j == 0 else w[:, 1]
                weight *= 1 - w[:, 2] if i == 0 else w[:, 2]
                idx = 4 * i + 2 * j + k
                out += weight * p[:, idx]
    return out


@deprecated()
def cubic_bezier_roots(coeffs: Tensor, interval: List[float] = [0., 1.],
                       eps: float = 1e-3) -> Tensor:
    """Find the roots of a cubic Bézier function.

    Args:
        coeffs (Tensor): Coefficients of the cubic Bézier function.

    Returns:
        Tensor: The position t where the weights are (1-t)^3, ..., t^3.
    """
    t = []
    for i in range(coeffs.shape[0]):
        roots = []
        if 2 > (coeffs.abs() > eps).sum():
            roots_ = []
        elif coeffs[i][0].abs() < eps:
            if coeffs[i][1].abs() < eps:  # first-order eqn.
                roots_ = [coeffs[i][3] / coeffs[i][2]]
            else:  # second-order eqn.
                roots_ = quad_roots(coeffs[i].cpu().numpy())
        else:
            roots_ = np.roots(coeffs[i].cpu().numpy())
        for r in roots_:
            if "complex" in str(type(r)):
                if abs(r.imag) < eps:
                    r = r.real
                else:
                    continue
            if r <= interval[1] and r >= interval[0]:
                roots += [r]

        if 1 == len(roots):
            t += [roots[0] if type(roots[0]) != complex else roots[0].reale]
        else:
            # if 1 < len(roots):
            #     print("too many valid roots!")
            # elif 0 == len(roots):
            #     print("no valid root!")
            # print(roots, roots_)
            # print(coeffs[i])
            t += [-1]
    t = torch.Tensor(t).to(coeffs).unsqueeze(-1)
    return t


@deprecated()
def polynomial_roots(coeffs: Tensor, interval: List[float] = [0., 1.],
                     eps: float = 1e-6) -> Tensor:
    """Find the roots of a polynomial function.

    Args:
        coeffs (Tensor): Coefficients of a polynomial function.

    Returns:
        Tensor: A valid root of a polynomal function
    """
    coeffs[coeffs.abs() < eps] = 0  # avoid numerical issue

    def _polynomial_roots(coeffs: Tensor, interval: List[float] = [0., 1.],
                          eps: float = 1e-6) -> float:
        roots = []
        roots_ = np.roots(coeffs.cpu().numpy())
        for r in roots_:
            if "complex" in str(type(r)):
                if abs(r.imag) < eps:
                    r = r.real
                else:
                    continue
            if r <= interval[1] and r >= interval[0]:
                roots += [r]

        if 1 <= len(roots):
            x = roots[0] if type(roots[0]) != complex else roots[0].real
        else:
            # print("no valid root!")
            # print(roots, roots_)
            # print(coeffs_[i])
            x = -1
        return x

    x = [_polynomial_roots(coeffs[i], interval, eps) for i in range(coeffs.shape[0])]

    return coeffs.new_tensor(x)


def batched_polynomial_roots(coeffs: Tensor, interval: List = [0, 1], eps=1e-9) -> Tensor:
    assert 1 < coeffs.shape[1]
    coeffs[coeffs.abs() < eps] = 0  # avoid numerical issue
    # print(coeffs)
    roots = coeffs.new_zeros(coeffs.shape[0]).fill_(-1)
    for i in range(coeffs.shape[1] - 1):
        m = (coeffs[:, :i].abs().sum(-1) <= eps) & (coeffs[:, i].abs() > eps)
        roots[m] = _batched_polynomial_roots(coeffs[m][:, i:], interval, eps)

    return roots


def _batched_polynomial_roots(coeffs: Tensor, interval: List = [0, 1],
                              eps=1e-9) -> Tensor:
    # Eigenvalues of companion matrices are polynomial roots
    coeffs = torch.flip(coeffs, [1])
    N = coeffs.shape[1] - 1
    valid = coeffs.abs().mean(-1) > eps
    B = valid.sum()
    C = coeffs.new_zeros(B, N, N)
    for i in range(N - 1):
        C[:, i, i+1] = 1
    inds = torch.ext.nonzero_last(coeffs[valid].abs() > eps)
    assert inds.shape[0] == B
    # print(coeffs)
    # print(coeffs[valid].gather(1, inds[:,1:]))
    C[:, -1] = -coeffs[valid][:, :-1] / coeffs[valid].gather(1, inds[:, 1:])

    _roots = torch.linalg.eigvals(C)
    # print(_roots)
    roots = coeffs.new_zeros(coeffs.shape[0]).fill_(-1)

    # filter
    m = (_roots.imag.abs() <= eps) & (_roots.real >= interval[0]) & \
        (_roots.real <= interval[1])
    inds = torch.ext.nonzero_last(m)
    valid_ = valid.masked_scatter_(valid, m.sum(-1) > 0)
    roots[valid_] = _roots[m.sum(-1) > 0].real.gather(1, inds[:, 1:]).squeeze()
    # print(roots)

    return roots


@deprecated()
def quad_roots(coeffs):
    a, b, c = coeffs[-3], coeffs[-2], coeffs[-1]
    D = b**2 - 4 * a * c
    if D > 0:
        return [(-b - np.sqrt(D))/(2*a), (-b + np.sqrt(D))/(2*a)]
    elif D == 0:
        return [-b / a]
    else:
        return []


@deprecated()
def quad_roots_t(coeffs: Tensor, bilinear_weights: bool = False, eps: float = 1e-6) \
        -> Tensor:
    a, b, c = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]

    if bilinear_weights:  # from a(1-x)^2 + bx(1-x) + cx^2 notation
        p, q, r = (a - b + c), (-2 * a + b), a
        a, b, c = p, q, r

    x = coeffs.new_zeros(coeffs.shape[0]).fill_(-1)

    # linear case
    m = a.abs() < eps
    x[m] = -c[m] / b[m]

    # bilinear case
    D = b**2 - 4 * a * c
    m = (D > 0) * (a.abs() > eps) > 0

    # two solutions
    x[m] = (-b[m] - D[m].sqrt()) / a[m] / 2

    n = ((x[m] < 0) + (x[m] > 1)) > 0
    x[m][n] = (-b[m][n] + D[m][n].sqrt()) / a[m][n] / 2

    if 0 < ((x[m][n] < 0) + (x[m][n] > 1)).sum():
        print("quad_roots_t: out of range solution:")
        print(x)

    # one solution
    m = (D == 0) * (a.abs() > eps) > 0
    x[m] = -b[m] / a[m]

    return x


def corner_points(expanded_edges: Tensor) -> Tensor:
    """Eight corner points on a cube defined by an edge.

    Args:
        expanded_edges (Tensor): Batched 3D edges in B x 2 x 3

    Returns:
        Tensor: Corner points in B x 8 x 3
    """
    e = expanded_edges
    m = e.new_zeros(1, 2, 3).float()
    x = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                m.zero_()
                m[0, k, 0] = 1
                m[0, j, 1] = 1
                m[0, i, 2] = 1
                # Note that you need to transpose(1, 2) for ordering when you want to
                # mask it instead of sum in this line.
                x += [(e * m).sum(dim=1, keepdim=True)]
    return torch.cat(x, dim=1)


def find_polygon(q: Tensor, planes: Tensor, eps=1e-3) -> Tensor:
    """Find a convex polygon (a list of points) intersecting a convex
        polyhedron, defined by the halfspaces of given n `planes`, with a given
        querying plane `q`. The time complexity is O(n^2).

    Args:
        q (Tensor): a plane defined by a*x + b*y + c*z + d = 0 (4 params)
        planes (Tensor): n planes (n x 4 params)
    """
    N = planes.size(0)
    q = q.unsqueeze(0) if 1 == len(q.shape) else q
    w = torch.stack([  # 1x1x4, 1xnx4, nx1x4 -> nxnx3x4
        q.unsqueeze(0).repeat(N, N, 1),
        planes.unsqueeze(0).repeat(N, 1, 1),
        planes.unsqueeze(1).repeat(1, N, 1)], dim=2)

    # pseudo-inverse in double precision since diagnoals have singular cases.
    w = w.double()
    p = (w[..., :3].pinverse() @ -w[..., 3:]).float()  # nxnx3x1
    w = w.float()

    # Find vertices within convex hull
    m = 1 != torch.eye(N, N).to(p.device)  # to exclude signular cases
    p = p[m].view(-1, 3, 1)  # Mx3x1
    p = torch.unique(p, dim=0)

    # evaluate vertices (1xNx3 @ Mx3x1 + 1xNx1 -> MxNx1)
    x = planes[:, :3].unsqueeze(0) @ p + planes[:, 3:].unsqueeze(0)

    # candidate points should be within convex hull.
    p = p[(x > -eps).sum(2).sum(1) == N].view(-1, 3)
    p = p.view(-1, 3)

    # Sort vertices forming a polygon
    p = sort_polygon_vertices(p, q[0, :3])

    return p


def find_polygon_batch(q: Tensor, planes: Tensor, eps=1e-3) -> Tensor:
    """Find a convex polygon (a list of points) intersecting a convex
        polyhedron, defined by the halfspaces of given n `planes`, with a given
        querying plane `q`. The time complexity is O(n^2).

    Args:
        q (Tensor): a plane defined by a*x + b*y + c*z + d = 0 (bx1x4)
        planes (Tensor): n planes (bxnx4)
    """
    B = planes.size(0)
    N = planes.size(1)
    q = q.unsqueeze(1) if 2 == len(q.shape) else q
    w = torch.stack([  # bx1x1x4, bx1xnx4, bxnx1x4 -> bxnxnx3x4
        q.unsqueeze(1).repeat(1, N, N, 1),
        planes.unsqueeze(1).repeat(1, N, 1, 1),
        planes.unsqueeze(2).repeat(1, 1, N, 1)], dim=3)

    # pseudo-inverse in double precision since diagnoals have singular cases.
    w = w.double()
    p = (w[..., :3].pinverse() @ -w[..., 3:]).float()  # bxnxnx3x1
    w = w.float()

    # Find vertices within convex hull
    m = (1 - torch.eye(N, N)).to(p.device)  # to exclude signular cases
    m = m.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    p = (p * m).view(B, -1, 3, 1)  # bxMx3x1
    p = torch.unique(p, dim=1)

    # evaluate vertices (bx1xNx3 @ bxMx3x1 + bx1xNx1 -> bxMxNx1)
    x = planes[..., :3].unsqueeze(1) @ p + planes[..., 3:].unsqueeze(1)

    # candidate points should be within convex hull.
    p[(x > -eps).sum(-1).sum(-1) != N] = 0
    p = p.view(B, -1, 3)

    # Sort vertices forming a polygon
    p = sort_polygon_vertices_batch(p, q[:, 0, :3])

    return p


def sort_polygon_vertices(v: Tensor, n: Tensor = None, idx: int = 0,
                          return_index: bool = False, null_value: int = -1) -> Tensor:
    # null masking to handle variable-length vertices
    v = v[v != null_value].view(-1, 3)

    u = v - v.mean(dim=0, keepdim=True)
    d = torch.cross(u[idx:idx+1], u)  # do not need to be normalized

    if n is None:
        n = d[idx+1] / d[idx+1].norm(p=2)

    # cosine similarity
    c = F.cosine_similarity(u[idx:idx+1], u)

    # to find (counter) clock-wise directions
    d = d @ n

    # make an order by degree using cosines and directions to cover 360 degrees
    s = c * ((d >= 0).float() * 2 - 1) + (d < 0).float() * 2

    _, idx = s.sort(descending=True)

    if return_index:
        return v[idx], idx

    return v[idx]


def sort_polygon_vertices_batch(v: Tensor, n: Tensor, idx: int = 0,
                                return_index: bool = False) -> List[List]:
    """Sort polygon vertices in a batch

    Args:
        v (Tensor): B x M x 3
        n (Tensor): B x 3
        idx (int, optional): index of the base vector

    Returns:
        List[List]: List of the list of vertices

    """
    m = v.norm(dim=-1, keepdim=True) > 0     # bxmx1
    k = m.sum(dim=-2, keepdim=True)          # bx1x1
    k[k == 0] = 1
    u = v - v.sum(dim=-2, keepdim=True) / k  # bxmx3
    d = torch.cross(u[:, idx:idx+1], u)       # bxmx3; do not need to be normalized

    if n is None:
        n = d[:, idx+1] / d[:, idx+1].norm(p=2, dim=-1, keepdim=True)

    # cosine similarity
    c = F.cosine_similarity(u[:, idx:idx+1], u, dim=-1)

    # to find (counter) clock-wise directions
    d = (d @ n.unsqueeze(-1)).squeeze(-1)

    # make an order by degree using cosines and directions to cover 360 degrees
    s = c * ((d >= 0).float() * 2 - 1) + (d < 0).float() * 2

    _, idx = s.sort(dim=-1, descending=True)  # bxm
    p = torch.ext.batched_index_select(v, -2, idx)
    m = torch.ext.batched_index_select(m, -2, idx)
    m = m.squeeze()

    # Extract triangle directly for efficiency compared to v1
    faces = extract_triangles_from_sorted_vertices_and_mask(p, m)

    if return_index:
        return faces, idx

    return faces


def extract_triangles_from_sorted_vertices_and_mask_v1(vertices: Tensor, mask: Tensor):
    mask = mask.cpu().numpy()  # cpu() and numpy() would cost much
    faces = []
    for i, row in enumerate(vertices.cpu().numpy()):
        faces += [row[mask[i]].tolist()]
    return faces


def extract_triangles_from_sorted_vertices_and_mask(vertices: Tensor, mask: Tensor):
    counts = mask.sum(-1)
    cumsum = counts.cumsum(0)
    indices = torch.cat([counts.new_zeros(1), cumsum[:-1]], dim=0).long()
    vertices = vertices[mask].view(-1, 3)  # 1d valid vertices
    v0_ = vertices[indices]  # the first elements

    faces = []
    indices += 1
    m = indices < cumsum

    for i in range(counts.max()):
        v0 = v0_
        indices += 1
        m &= indices < cumsum
        v1 = vertices[(indices - 1)[m]]
        v2 = vertices[indices[m]]
        v0 = v0[m]
        faces += [torch.stack([v0, v1, v2], dim=1)]
    faces = torch.cat(faces, dim=0)
    return faces.cpu().numpy()


if "__main__" == __name__:
    import torch
    # coeffs = torch.Tensor([[3, 1,-2,1], [1, -3/2, 3/4, -1/8]])
    coeffs = torch.rand(10, 4) - 0.7
    coeffs[:, :2] = 0
    print(polynomial_roots(coeffs))
    print(batched_polynomial_roots(coeffs))

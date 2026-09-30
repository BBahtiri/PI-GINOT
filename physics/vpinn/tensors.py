"""Precomputed test-function tensors (the FastVPINNs tensor form), per geometry.

Quadrilaterals: ``quad_tensors`` returns quadrature points x (E*Nq, 2) and
val, gx, gy (E, Nt, Nq) with the weight w_q * det J_q folded in, exactly as
the library's ``mult``.  A residual of the Poisson-type form
    R[e,k] = sum_q ( a(x_q) . grad v_k(x_q) ) w_q det J_q
is then ``einsum('ekq,eq->ek', gx, a_x) + einsum('ekq,eq->ek', gy, a_y)``.

Triangles (Lagrange): ``tri_tensors`` returns quadrature points, the local test
gradients with w_q |T| folded in (E, n_loc, Nq, 2) and the global test index
of each local function (E, n_loc) -- vertex nodes first, then (P2) one index
per unique edge -- so residuals scatter-add into one vector per component.
"""
from dataclasses import dataclass

import numpy as np

from physics.vpinn.basis import edge_functions, lagrange_tri, tensor_2d
from physics.vpinn.mapping import affine_tri, bilinear, to_physical_grad
from physics.vpinn.quadrature import square_rule, triangle_rule


@dataclass
class QuadTensors:
    x: np.ndarray          # (E*Nq, 2)
    val: np.ndarray        # (E, Nt, Nq), w*detJ folded in
    gx: np.ndarray
    gy: np.ndarray
    wdet: np.ndarray       # (E, Nq)
    area: np.ndarray       # (E,)
    n_elem: int
    n_test: int
    n_quad: int


def quad_tensors(cells, K, Q, kind="bubble", rule="gauss-legendre"):
    xi, eta, w = square_rule(Q, rule)
    v, v_xi, v_eta = tensor_2d(K, xi, eta, kind)
    x, J, det = bilinear(cells, xi, eta)
    if not (det > 0).all():
        raise ValueError(f"non-positive det J in {int((det <= 0).any(1).sum())} cells")
    gx, gy = to_physical_grad(J, det, v_xi, v_eta)
    wdet = w[None, :] * det
    return QuadTensors(x=x.reshape(-1, 2), val=v[None] * wdet[:, None, :],
                       gx=gx * wdet[:, None, :], gy=gy * wdet[:, None, :], wdet=wdet,
                       area=wdet.sum(1), n_elem=len(cells), n_test=K * K, n_quad=len(w))


def edge_tensors(cells, edges, K, Q, rule="gauss-legendre"):
    """C1-n edge test functions for the boundary cells.  ``edges`` (M, 2) rows of
    (cell index, reference edge 0..3).  Returns x (M*Nq,2), gx, gy (M, K, Nq)."""
    xi, eta, w = square_rule(Q, rule)
    cells = np.asarray(cells)[edges[:, 0]]
    x, J, det = bilinear(cells, xi, eta)
    gx = np.zeros((len(edges), K, len(w)))
    gy = np.zeros_like(gx)
    val = np.zeros_like(gx)
    for e in range(4):
        m = edges[:, 1] == e
        if not m.any():
            continue
        v, v_xi, v_eta = edge_functions(K, xi, eta, e)
        Jm = tuple(a[m] for a in J)
        a, b = to_physical_grad(Jm, det[m], v_xi, v_eta)
        gx[m], gy[m] = a, b
        val[m] = v[None]
    wdet = w[None, :] * det
    return (x.reshape(-1, 2), gx * wdet[:, None, :], gy * wdet[:, None, :],
            val * wdet[:, None, :])


@dataclass
class TriTensors:
    x: np.ndarray          # (E*Nq, 2)
    grad: np.ndarray       # (E, n_loc, Nq, 2), w*|T| folded in
    dof: np.ndarray        # (E, n_loc) global test index
    n_dof: int
    dof_xy: np.ndarray     # (n_dof, 2) node / edge-midpoint coordinates
    area: np.ndarray
    n_quad: int


def tri_tensors(nodes, tris, degree=1, quad_degree=4):
    r, s, w = triangle_rule(quad_degree)
    txy = np.asarray(nodes)[np.asarray(tris)]
    x, invT, det = affine_tri(txy, r, s)
    if not (det > 0).all():
        raise ValueError("clockwise or degenerate triangle")
    _, Nr, Ns = lagrange_tri(degree, r, s)
    gref = np.stack([Nr, Ns], -1)                        # (n_loc, Nq, 2)
    g = np.einsum("eij,lqj->elqi", invT, gref)           # physical gradients
    area = 0.5 * det
    g = g * (w[None, None, :, None] * area[:, None, None, None])
    n = len(nodes)
    dof = np.asarray(tris, np.int64)
    dof_xy = np.asarray(nodes, np.float64)
    if degree == 2:
        e = np.concatenate([dof[:, [0, 1]], dof[:, [1, 2]], dof[:, [2, 0]]])
        uniq, inv = np.unique(np.sort(e, 1), axis=0, return_inverse=True)
        E = len(dof)
        mid = n + inv.ravel()
        dof = np.concatenate([dof, np.stack([mid[:E], mid[E:2 * E], mid[2 * E:]], 1)], 1)
        dof_xy = np.vstack([dof_xy, 0.5 * (dof_xy[uniq[:, 0]] + dof_xy[uniq[:, 1]])])
    return TriTensors(x=x.reshape(-1, 2), grad=g, dof=dof, n_dof=len(dof_xy), dof_xy=dof_xy,
                      area=area, n_quad=len(w))

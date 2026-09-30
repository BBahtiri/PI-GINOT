"""Test functions on reference elements.

Quadrilaterals, reference square [-1,1]^2 (FastVPINNs / hp-VPINN):
  * ``bubble``: phi_n(t) = P_{n+1}(t) - P_{n-1}(t), n = 1..K -- vanishes at
    t = +-1, so every tensor product vanishes on the element boundary.  This is
    the library's fe_type "jacobi" / "legendre".
  * ``plain``:  phi_n(t) = P_{n-1}(t) -- the library's "jacobi_plain" (does NOT
    vanish on the boundary; used only to cross-check the library).
  Tensor products are ordered index = i*K + j, i the xi-index, j the eta-index,
  as the library orders them.

  ``edge_functions`` (arm C1-n, ours): for a boundary element whose edge e lies
  on a natural boundary, K functions that are nonzero on that edge and vanish on
  the other three: a linear factor (1 -+ t)/2 across the edge times a bubble
  along it.  They test the natural condition on that edge weakly.

Triangles, reference triangle {r, s >= 0, r + s <= 1} (arm C2, ours):
  Lagrange P1 (3 vertex functions) and P2 (3 vertex + 3 edge-midpoint functions,
  midpoints ordered edge 01, 12, 20).
"""
import numpy as np
from numpy.polynomial import legendre as _L


def _P(n, x):
    return _L.legval(x, [0.0] * n + [1.0])


def _dP(n, x):
    return _L.legval(x, _L.legder([0.0] * n + [1.0]))


def basis_1d(K, t, kind="bubble"):
    """Values and derivatives, each (K, len(t))."""
    t = np.asarray(t, np.float64)
    if kind == "bubble":
        v = np.stack([_P(n + 1, t) - _P(n - 1, t) for n in range(1, K + 1)])
        d = np.stack([_dP(n + 1, t) - _dP(n - 1, t) for n in range(1, K + 1)])
    elif kind == "plain":
        v = np.stack([_P(n - 1, t) for n in range(1, K + 1)])
        d = np.stack([_dP(n - 1, t) for n in range(1, K + 1)])
    else:
        raise ValueError(kind)
    return v, d


def tensor_2d(K, xi, eta, kind="bubble"):
    """(v, v_xi, v_eta), each (K*K, Nq); index i*K + j."""
    vx, dx = basis_1d(K, xi, kind)
    vy, dy = basis_1d(K, eta, kind)
    v = (vx[:, None, :] * vy[None, :, :]).reshape(K * K, -1)
    v_xi = (dx[:, None, :] * vy[None, :, :]).reshape(K * K, -1)
    v_eta = (vx[:, None, :] * dy[None, :, :]).reshape(K * K, -1)
    return v, v_xi, v_eta


# reference-square edges: 0 -> eta = -1, 1 -> xi = +1, 2 -> eta = +1, 3 -> xi = -1
def edge_functions(K, xi, eta, edge):
    """(v, v_xi, v_eta), each (K, Nq), nonzero only on the given reference edge."""
    xi, eta = np.asarray(xi, np.float64), np.asarray(eta, np.float64)
    if edge in (0, 2):
        s = -1.0 if edge == 0 else 1.0
        lin, dlin = 0.5 * (1.0 + s * eta), 0.5 * s        # 1 on the edge, 0 opposite
        b, db = basis_1d(K, xi)
        return b * lin, db * lin, b * dlin
    s = 1.0 if edge == 1 else -1.0
    lin, dlin = 0.5 * (1.0 + s * xi), 0.5 * s
    b, db = basis_1d(K, eta)
    return b * lin, b * dlin, db * lin


def lagrange_tri(degree, r, s):
    """(N, N_r, N_s), each (n_loc, Nq) on the reference triangle."""
    r, s = np.asarray(r, np.float64), np.asarray(s, np.float64)
    l0, l1, l2 = 1.0 - r - s, r, s
    if degree == 1:
        N = np.stack([l0, l1, l2])
        Nr = np.stack([-np.ones_like(r), np.ones_like(r), np.zeros_like(r)])
        Ns = np.stack([-np.ones_like(r), np.zeros_like(r), np.ones_like(r)])
        return N, Nr, Ns
    if degree == 2:
        N = np.stack([l0 * (2 * l0 - 1), l1 * (2 * l1 - 1), l2 * (2 * l2 - 1),
                      4 * l0 * l1, 4 * l1 * l2, 4 * l2 * l0])
        Nr = np.stack([-(4 * l0 - 1), 4 * l1 - 1, np.zeros_like(r),
                       4 * (l0 - l1), 4 * l2, -4 * l2])
        Ns = np.stack([-(4 * l0 - 1), np.zeros_like(r), 4 * l2 - 1,
                       -4 * l1, 4 * l1, 4 * (l0 - l2)])
        return N, Nr, Ns
    raise ValueError(degree)

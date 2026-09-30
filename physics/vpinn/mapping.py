"""Element maps.

Bilinear quadrilateral (as FastVPINNs, ``quad_bilinear.py``): vertices
x0..x3 counter-clockwise, x = xc0 + xc1 xi + xc2 eta + xc3 xi eta with
xc0 = (x0+x1+x2+x3)/4, xc1 = (-x0+x1+x2-x3)/4, xc2 = (-x0-x1+x2+x3)/4,
xc3 = (x0-x1+x2-x3)/4 (same for y).  The Jacobian varies inside the cell.
The library takes |det J|; here det J is kept signed and required > 0, so a
folded or clockwise cell is an error rather than silently integrated.

Affine triangle: x = x0 + (x1-x0) r + (x2-x0) s.
"""
import numpy as np


def bilinear(cells, xi, eta):
    """cells (E,4,2); xi, eta (Nq,).  Returns x (E,Nq,2), J entries and det (E,Nq)."""
    c = np.asarray(cells, np.float64)
    X, Y = c[..., 0], c[..., 1]
    xc = [(X.sum(1)) / 4, (-X[:, 0] + X[:, 1] + X[:, 2] - X[:, 3]) / 4,
          (-X[:, 0] - X[:, 1] + X[:, 2] + X[:, 3]) / 4, (X[:, 0] - X[:, 1] + X[:, 2] - X[:, 3]) / 4]
    yc = [(Y.sum(1)) / 4, (-Y[:, 0] + Y[:, 1] + Y[:, 2] - Y[:, 3]) / 4,
          (-Y[:, 0] - Y[:, 1] + Y[:, 2] + Y[:, 3]) / 4, (Y[:, 0] - Y[:, 1] + Y[:, 2] - Y[:, 3]) / 4]
    e = lambda a: a[:, None]
    x = e(xc[0]) + e(xc[1]) * xi + e(xc[2]) * eta + e(xc[3]) * xi * eta
    y = e(yc[0]) + e(yc[1]) * xi + e(yc[2]) * eta + e(yc[3]) * xi * eta
    x_xi, x_eta = e(xc[1]) + e(xc[3]) * eta, e(xc[2]) + e(xc[3]) * xi
    y_xi, y_eta = e(yc[1]) + e(yc[3]) * eta, e(yc[2]) + e(yc[3]) * xi
    det = x_xi * y_eta - x_eta * y_xi
    return np.stack([x, y], -1), (x_xi, x_eta, y_xi, y_eta), det


def to_physical_grad(J, det, v_xi, v_eta):
    """Reference -> physical gradients.  v_* (Nt,Nq); J entries, det (E,Nq).
    Returns (E,Nt,Nq) each."""
    x_xi, x_eta, y_xi, y_eta = (a[:, None, :] for a in J)
    d = det[:, None, :]
    gx = (y_eta * v_xi[None] - y_xi * v_eta[None]) / d
    gy = (-x_eta * v_xi[None] + x_xi * v_eta[None]) / d
    return gx, gy


def affine_tri(tris_xy, r, s):
    """tris_xy (E,3,2).  Returns x (E,Nq,2), inverse-transpose (E,2,2), det (E,)."""
    t = np.asarray(tris_xy, np.float64)
    A = np.stack([t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]], -1)   # columns: d x/dr, d x/ds
    det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
    x = (t[:, None, 0, :] + r[None, :, None] * A[:, None, :, 0]
         + s[None, :, None] * A[:, None, :, 1])
    invT = np.linalg.inv(A).transpose(0, 2, 1)
    return x, invT, det

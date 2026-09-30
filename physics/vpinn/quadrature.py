"""Quadrature rules.

Square [-1,1]^2: tensor Gauss-Legendre (library "gauss-legendre") and
Gauss-Lobatto-Legendre (library "gauss-jacobi"), both ordered as the library
orders them: meshgrid(nodes, nodes) flattened, so point index = r*Q + c with
xi = nodes[c], eta = nodes[r], weight w[r] w[c].

Triangle: Dunavant rules of degree 2, 4 and 6 on the reference triangle,
weights summing to 1 (multiply by the element area).
"""
import numpy as np
from numpy.polynomial import legendre as _L


def gauss_legendre_1d(Q):
    return np.polynomial.legendre.leggauss(Q)


def lobatto_1d(Q):
    """Gauss-Lobatto-Legendre: endpoints plus the roots of P'_{Q-1}."""
    c = [0.0] * (Q - 1) + [1.0]
    inner = np.sort(np.real(_L.legroots(_L.legder(c))))
    x = np.concatenate([[-1.0], inner, [1.0]])
    w = 2.0 / (Q * (Q - 1) * _L.legval(x, c) ** 2)
    return x, w


def square_rule(Q, kind="gauss-legendre"):
    x, w = gauss_legendre_1d(Q) if kind == "gauss-legendre" else lobatto_1d(Q)
    XI, ETA = np.meshgrid(x, x)
    return XI.ravel(), ETA.ravel(), (w[:, None] * w[None, :]).ravel()


def triangle_rule(degree):
    """(r, s, w) on the reference triangle; sum(w) = 1."""
    def sym3(a, w):            # (a, a, 1-2a) and permutations; barycentric
        b = 1.0 - 2.0 * a
        bary = [(b, a, a), (a, b, a), (a, a, b)]
        return [(l1, l2, w) for (_, l1, l2) in bary]

    def sym6(a, b, w):
        c = 1.0 - a - b
        bary = [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)]
        return [(l1, l2, w) for (_, l1, l2) in bary]

    if degree == 2:
        pts = sym3(1.0 / 6.0, 1.0 / 3.0)
    elif degree == 4:
        pts = (sym3(0.445948490915965, 0.223381589678011)
               + sym3(0.091576213509771, 0.109951743655322))
    elif degree == 6:
        pts = (sym3(0.249286745170910, 0.116786275726379)
               + sym3(0.063089014491502, 0.050844906370207)
               + sym6(0.053145049844817, 0.310352451033784, 0.082851075618374))
    else:
        raise ValueError(degree)
    a = np.array(pts)
    return a[:, 0], a[:, 1], a[:, 2]

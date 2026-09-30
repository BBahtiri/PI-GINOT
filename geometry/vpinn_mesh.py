#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 10, S2 -- meshes for the weak-form arms.

Everything starts from the same graded triangulation the energy form
integrates on (``FamilyEnergyLoss._mesh``, or ``EnergyLoss._mesh`` for the
dog-bone), at a chosen element budget.  From it:

* **quadrilaterals for C1** (FastVPINNs needs quads): every triangle
  (a, b, c) is split at its centroid g and edge midpoints into
  (a, m_ab, g, m_ca), (b, m_bc, g, m_ab), (c, m_ca, g, m_bc).  Each is convex and
  counter-clockwise when the triangle is, and the split keeps the triangulation's
  grading and its polygon exactly.  Vertex k of a quad maps to the reference
  corner (-1,-1), (1,-1), (1,1), (-1,1); reference edge 0 is eta = -1 (v0-v1),
  1 is xi = 1, 2 is eta = 1, 3 is xi = -1 (v3-v0).  Boundary half-edges of a split
  triangle are always reference edge 0 or 3 of their quad.
* **the triangulation itself for C2** (Lagrange test functions).
* **boundary bookkeeping for every arm**: each boundary edge carries its
  segment name, its outward unit (chord) normal, and which displacement
  components are Dirichlet on that segment.  The flags are read off the FEM's
  own Dirichlet set (``fam.fem_dirichlet``, or the solver's ``_dirichlet`` for
  the dog-bone), so the weak forms drop exactly the degrees of freedom the FEM
  constrains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from config import LOADING_CONFIG

UD = LOADING_CONFIG["u_max"]


# --------------------------------------------------------------------------
def base_triangulation(fam_name: str, params: dict, n_elem_target: int = 1400):
    """The energy form's quadrature triangulation at a chosen budget."""
    from training.family_trainer import build_loss
    loss = build_loss(fam_name)
    loss.n_elem_target = n_elem_target
    return loss._mesh(params)


def dirichlet_dofs(fam_name: str, mesh) -> np.ndarray:
    """The FEM's constrained DOFs (2*node + component) on this mesh."""
    if fam_name == "dogbone":
        from verification.fem.solver import _dirichlet
        con, _ = _dirichlet(mesh, UD)
        return np.asarray(con, np.int64)
    from geometry.families import FAMILIES
    con, _ = FAMILIES[fam_name].fem_dirichlet(mesh, UD)
    return np.asarray(con, np.int64)


@dataclass
class Boundary:
    edges: np.ndarray        # (M, 2) node indices, oriented with the domain on the left
    seg: np.ndarray          # (M,) segment index
    names: List[str]
    normal: np.ndarray       # (M, 2) outward unit chord normal
    length: np.ndarray       # (M,)
    dirichlet: np.ndarray    # (n_seg, 2) bool: component c Dirichlet on segment s


def boundary(fam_name: str, mesh) -> Boundary:
    tris = np.asarray(mesh.tris, np.int64)
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    key = np.sort(e, 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    edges = e[cnt[inv.ravel()] == 1]                    # oriented as in the CCW triangle
    names = list(mesh.boundary.keys())
    member = {s: set(np.asarray(v).tolist()) for s, v in mesh.boundary.items()}
    seg = np.full(len(edges), -1)
    for k, (a, b) in enumerate(edges):
        hits = [i for i, s in enumerate(names) if a in member[s] and b in member[s]]
        if hits:
            seg[k] = hits[0]
    d = mesh.nodes[edges[:, 1]] - mesh.nodes[edges[:, 0]]
    length = np.hypot(d[:, 0], d[:, 1])
    normal = np.stack([d[:, 1], -d[:, 0]], 1) / length[:, None]
    con = set(dirichlet_dofs(fam_name, mesh).tolist())
    flags = np.zeros((len(names), 2), bool)
    for i, s in enumerate(names):
        nodes = np.asarray(mesh.boundary[s])
        for c in range(2):
            frac = np.mean([2 * n + c in con for n in nodes]) if len(nodes) else 0.0
            flags[i, c] = frac > 0.5
    return Boundary(edges=edges, seg=seg, names=names, normal=normal, length=length,
                    dirichlet=flags)


# --------------------------------------------------------------------------
@dataclass
class QuadMesh:
    nodes: np.ndarray        # (Nn, 2): triangle nodes, then edge midpoints, then centroids
    cells: np.ndarray        # (Nc, 4) node indices, CCW
    cell_xy: np.ndarray      # (Nc, 4, 2)
    bcell: np.ndarray        # (Mb,) quad index of each boundary half-edge
    bref: np.ndarray         # (Mb,) reference edge (0 or 3)
    bseg: np.ndarray         # (Mb,) segment index
    bnormal: np.ndarray      # (Mb, 2)
    parent_tri: np.ndarray   # (Nc,)


def split_to_quads(mesh, bnd: Boundary) -> QuadMesh:
    tris = np.asarray(mesh.tris, np.int64)
    n, E = len(mesh.nodes), len(tris)
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    uniq, inv = np.unique(np.sort(e, 1), axis=0, return_inverse=True)
    inv = inv.ravel()
    mid = n + inv
    cen = n + len(uniq) + np.arange(E)
    nodes = np.vstack([mesh.nodes, 0.5 * (mesh.nodes[uniq[:, 0]] + mesh.nodes[uniq[:, 1]]),
                       mesh.nodes[tris].mean(1)])
    a, b, c = tris.T
    m_ab, m_bc, m_ca = mid[:E], mid[E:2 * E], mid[2 * E:]
    cells = np.concatenate([np.stack([a, m_ab, cen, m_ca], 1),
                            np.stack([b, m_bc, cen, m_ab], 1),
                            np.stack([c, m_ca, cen, m_bc], 1)])
    parent = np.concatenate([np.arange(E)] * 3)
    # boundary half-edges: triangle edge (p, q) on the boundary -> quad of p (ref edge 0:
    # p -> m_pq) and quad of q (ref edge 3: m_pq -> q)
    bkey = {tuple(ed): k for k, ed in enumerate(bnd.edges.tolist())}
    corner = {0: (a, b, m_ab), 1: (b, c, m_bc), 2: (c, a, m_ca)}
    quad_of = {0: 0, 1: 1, 2: 2}          # which quad block starts at which corner vertex
    bcell, bref, bseg, bnorm = [], [], [], []
    for j in range(3):
        p, q, _ = corner[j]
        nxt = (j + 1) % 3
        for t in range(E):
            k = bkey.get((int(p[t]), int(q[t])))
            if k is None:
                continue
            bcell += [quad_of[j] * E + t, quad_of[nxt] * E + t]
            bref += [0, 3]
            bseg += [bnd.seg[k]] * 2
            bnorm += [bnd.normal[k]] * 2
    return QuadMesh(nodes=nodes, cells=cells, cell_xy=nodes[cells], bcell=np.array(bcell),
                    bref=np.array(bref), bseg=np.array(bseg), bnormal=np.array(bnorm),
                    parent_tri=parent)


def free_dofs_p1(fam_name: str, mesh) -> np.ndarray:
    """(n_node, 2) bool: component c of node i is an unconstrained test DOF."""
    con = dirichlet_dofs(fam_name, mesh)
    free = np.ones((len(mesh.nodes), 2), bool)
    free.ravel()[con] = False
    return free

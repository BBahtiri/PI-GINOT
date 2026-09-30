#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 10, S4 -- arm A: the strong (collocation) form, generalised to the five
families.

The preprint's loss (``physics/losses.py``, ``PhysicsLoss``) with its terms and
weights, minus the dog-bone-specific resultant anchor (that lives in the
trainer and runs only in A-pre):

    L = w_eq   mean |(L0/S0) Div P|^2                    interior collocation points
      + w_arc  mean |t/S0|^2        on curved free edges (hole, notch, inclusion, fillet)
      + w_top  mean |t/S0|^2        on straight free edges
      + w_part mean |t_nat/S0|^2    on mixed segments: the natural component only
                                    (symmetry-line shear, sliding-grip transverse)
      + w_bar  mean relu(j_min - min(J2D, J3D))^2   interior + fully free boundary points

Weights default to the preprint's (config.py: 100, 20, 2, 2, 1e3); L0 = 50 mm,
S0 = E.  Points are drawn fresh every step: interior points stratified in the
same triangulation the energy form uses (3 per triangle), boundary points
uniform on every polygon edge.

One disclosed departure (plan, rule 2): the distance-function Dirichlet layer
has an unbounded Laplacian where two Dirichlet pieces of the same component
meet (geometry/adf.py), and Div P needs second derivatives there.  Interior
points within two local element sizes of such a junction are excluded, on
every family.  (On the inclusion the arc meets x = 0 and y = 0; elsewhere no
two same-component Dirichlet pieces touch, so nothing is excluded.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch

from config import NONDIM_SCALES, TRAINING_CONFIG
from geometry.vpinn_mesh import base_triangulation, boundary

FEATURE_SEGMENTS = {"hole", "notch", "inclusion", "right_arc"}


@dataclass
class StrongConfig:
    n_elem: int = 1400
    n_per_elem: int = 3
    n_per_edge: int = 2
    w_eq: float = TRAINING_CONFIG["w_equilibrium"]
    w_arc: float = TRAINING_CONFIG["w_trac_arc"]
    w_top: float = TRAINING_CONFIG["w_trac_top"]
    w_part: float = TRAINING_CONFIG["w_traction_partial"]
    w_bar: float = TRAINING_CONFIG["w_barrier"]
    j_min: float = TRAINING_CONFIG["barrier_delta"]
    L0: float = NONDIM_SCALES["L0"]
    S0: float = NONDIM_SCALES["S0"]
    junction_factor: float = 2.0


class StrongGeometry:
    """Point sets and boundary bookkeeping for one geometry."""

    def __init__(self, fam_name: str, params: dict, cfg: StrongConfig, mesh=None, seed=0):
        self.cfg = cfg
        m = mesh if mesh is not None else base_triangulation(fam_name, params, cfg.n_elem)
        self.mesh = m
        b = boundary(fam_name, m)
        self.bnd = b
        self.rng = np.random.default_rng(seed)
        # edge categories: 0 feature (full), 1 straight free (full), 2 partial, -1 fully Dirichlet
        nat = ~b.dirichlet[b.seg]                                  # (M, 2)
        cat = np.full(len(b.edges), -1)
        full = nat.all(1)
        feat = np.array([b.names[s] in FEATURE_SEGMENTS for s in b.seg])
        cat[full & feat] = 0
        cat[full & ~feat] = 1
        cat[nat.any(1) & ~full] = 2
        self.cat, self.nat = cat, nat
        # junctions of same-component Dirichlet pieces
        seg_nodes = {i: set(np.asarray(m.boundary[s]).tolist()) for i, s in enumerate(b.names)}
        junc = []
        for i in range(len(b.names)):
            for j in range(i + 1, len(b.names)):
                if not (b.dirichlet[i] & b.dirichlet[j]).any():
                    continue
                junc += list(seg_nodes[i] & seg_nodes[j])
        self.junctions = np.array(sorted(set(junc)), np.int64)
        if len(self.junctions):
            h = np.sqrt(2 * np.asarray(m.areas))                   # element size
            nh = np.zeros(len(m.nodes))
            cnt = np.zeros(len(m.nodes))
            for a in range(3):
                np.add.at(nh, np.asarray(m.tris)[:, a], h)
                np.add.at(cnt, np.asarray(m.tris)[:, a], 1)
            self.junc_xy = np.asarray(m.nodes)[self.junctions]
            self.junc_r = cfg.junction_factor * (nh / np.maximum(cnt, 1))[self.junctions]
        else:
            self.junc_xy = np.zeros((0, 2))
            self.junc_r = np.zeros(0)

    def draw(self):
        """Fresh interior and boundary points for one step."""
        m, cfg = self.mesh, self.cfg
        n_e, k = m.n_elem, cfg.n_per_elem
        r1, r2 = self.rng.random((n_e, k)), self.rng.random((n_e, k))
        flip = (r1 + r2) > 1.0
        r1, r2 = np.where(flip, 1 - r1, r1), np.where(flip, 1 - r2, r2)
        v = np.asarray(m.nodes)[np.asarray(m.tris)]
        x = (v[:, None, 0] * (1 - r1 - r2)[..., None] + v[:, None, 1] * r1[..., None]
             + v[:, None, 2] * r2[..., None]).reshape(-1, 2)
        if len(self.junc_xy):
            d = np.linalg.norm(x[:, None, :] - self.junc_xy[None], axis=-1)
            x = x[(d > self.junc_r[None]).all(1)]
        b = self.bnd
        keep = self.cat >= 0
        e = b.edges[keep]
        t = self.rng.random((len(e), cfg.n_per_edge))
        a, c = m.nodes[e[:, 0]], m.nodes[e[:, 1]]
        xb = (a[:, None] + (c - a)[:, None] * t[..., None]).reshape(-1, 2)
        nb = np.repeat(b.normal[keep], cfg.n_per_edge, 0)
        mb = np.repeat(self.nat[keep], cfg.n_per_edge, 0)
        cb = np.repeat(self.cat[keep], cfg.n_per_edge, 0)
        return x, xb, nb, mb, cb


def div_P(P, x):
    """Div P at points x (requires_grad), P components with graph to x."""
    out = []
    for i in range(2):
        gi = []
        for j in range(2):
            p = P[2 * i + j]
            if not p.requires_grad:
                gi.append(torch.zeros(len(x), dtype=x.dtype))
                continue
            g, = torch.autograd.grad(p.sum(), x, create_graph=True, allow_unused=True)
            gi.append(torch.zeros(len(x), dtype=x.dtype) if g is None else g[:, j])
        out.append(gi[0] + gi[1])
    return out


def strong_loss(P_fn: Callable, x, xb, nb, mb, cb, cfg: StrongConfig, b=None, tb=None):
    """``P_fn(points_tensor_requiring_grad) -> ([P11, P12, P21, P22], J)``, each (N,).
    ``b`` (N, 2) body force and ``tb`` (Nb, 2) prescribed traction for manufactured
    problems (default 0: the physical problem).  Returns (loss, parts)."""
    dt = torch.get_default_dtype()
    xi = torch.as_tensor(x, dtype=dt).requires_grad_(True)
    P, J = P_fn(xi)
    fx, fy = div_P(P, xi)
    if b is not None:
        bt = torch.as_tensor(b, dtype=dt)
        fx, fy = fx + bt[:, 0], fy + bt[:, 1]
    s = cfg.L0 / cfg.S0
    parts = {"L_eq": torch.mean((s * fx) ** 2 + (s * fy) ** 2)}
    xbt = torch.as_tensor(xb, dtype=dt).requires_grad_(True)
    Pb, Jb = P_fn(xbt)
    n = torch.as_tensor(nb, dtype=dt)
    tx = Pb[0] * n[:, 0] + Pb[1] * n[:, 1]
    ty = Pb[2] * n[:, 0] + Pb[3] * n[:, 1]
    if tb is not None:
        tbt = torch.as_tensor(tb, dtype=dt)
        tx, ty = tx - tbt[:, 0], ty - tbt[:, 1]
    m = torch.as_tensor(mb, dtype=dt)
    res = (m[:, 0] * tx ** 2 + m[:, 1] * ty ** 2) / cfg.S0 ** 2
    cbt = torch.as_tensor(cb)
    zero = torch.zeros((), dtype=dt)
    for key, c in (("L_arc", 0), ("L_top", 1), ("L_part", 2)):
        sel = cbt == c
        parts[key] = res[sel].mean() if bool(sel.any()) else zero
    full = cbt <= 1
    Jall = torch.cat([J, Jb[full]]) if J is not None else None
    parts["L_barrier"] = (torch.mean(torch.relu(cfg.j_min - Jall) ** 2) if Jall is not None else zero)
    loss = (cfg.w_eq * parts["L_eq"] + cfg.w_arc * parts["L_arc"] + cfg.w_top * parts["L_top"]
            + cfg.w_part * parts["L_part"] + cfg.w_bar * parts["L_barrier"])
    return loss, parts

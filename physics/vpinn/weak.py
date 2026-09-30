#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 10, S3 -- the finite-strain Neo-Hookean weak form, tested four ways.

For a test function v and component c the weak residual is

    R = int_Omega P_cJ(F) d_J v dA        (reference configuration, F = I + grad u)

with P the first Piola-Kirchhoff stress of ``physics/neo_hookean.py`` -- the same
constitutive code the energy form and the FEM use.  There is no load term: the
problem is displacement-controlled through the hard Dirichlet layer, and every
natural boundary is traction-free.

Arms (docs/phase10_formulations_plan.md):
  C1   FastVPINNs / hp-VPINN: Legendre bubbles on the quads of the split
       triangulation, K per direction, Gauss (K+2)^2 per quad; bubbles vanish on
       every quad edge, so natural conditions get a traction penalty (beta).
  C1n  as C1, plus edge test functions on boundary quads (nonzero on their
       boundary edge) for the natural components; no penalty.
  C2   Lagrange test functions (degree 1 or 2) on the triangulation, one per
       unconstrained nodal DOF; natural conditions tested as in the FEM.
  C2I  C2 with the network first interpolated onto P1 on the h/2 nested
       refinement (Berrone et al.'s IVPINN), tested with P1 hats on the base mesh.

Normalisation -- every term approximates an integral, so it means the same at
every mesh level (plan S3).  With A = |Omega|, ell a fixed length (the family's
half-height), S0 the stress scale:
  C1 / C1n domain and edge terms   (ell^2 / (S0^2 A)) sum_e mean_k sum_c R^2 / |K_e|
                                     (~ (ell/S0)^2 * mean of the strong residual^2)
  C1 penalty                        beta / (S0^2 |Gamma_N|) int_{Gamma_N} |t_nat|^2 ds
  C2 / C2I, gamma = 'area'          (ell^2 / (S0^2 A)) sum_{free} R^2 / |supp phi_i|
  C2 / C2I, gamma = 'one'           (1 / (S0^2 A)) sum_{free} R^2      (Berrone et al.)
All arms add the energy form's J-barrier at their volume points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from geometry.vpinn_mesh import base_triangulation, boundary, free_dofs_p1, split_to_quads
from physics.neo_hookean import first_piola_kirchhoff_stress
from physics.vpinn.tensors import edge_tensors, quad_tensors, tri_tensors


@dataclass
class WeakConfig:
    variant: str = "C1"          # C1 | C1n | C2 | C2I
    n_elem: int = 1400           # base triangulation budget
    K: int = 2                   # C1/C1n: bubbles per direction
    Q: Optional[int] = None      # C1/C1n: Gauss points per direction (default K + 2)
    degree: int = 1              # C2: Lagrange degree
    quad_degree: int = 4         # C2: Dunavant degree
    beta: float = 20.0           # C1: traction penalty weight
    gamma: str = "area"          # C2/C2I: 'area' | 'one'
    edge_Q: int = 4              # C1: Gauss points per boundary edge for the penalty


def _t(a, dtype=None):
    return torch.as_tensor(np.asarray(a), dtype=dtype or torch.get_default_dtype())


def refine_nested(mesh):
    """Every triangle into four (same polygon); children blocks of length E in
    the order corner a, corner b, corner c, centre -- parent = tile(arange(E), 4)."""
    from verification.phase9_headroom import refine
    return refine(mesh)[0]


class WeakGeometry:
    """Everything one geometry needs for one weak arm, precomputed once."""

    def __init__(self, fam_name: str, params: dict, cfg: WeakConfig, ell: float,
                 mesh=None):
        self.cfg, self.ell = cfg, float(ell)
        m = mesh if mesh is not None else base_triangulation(fam_name, params, cfg.n_elem)
        self.mesh = m
        b = boundary(fam_name, m)
        self.bnd = b
        self.A = float(m.areas.sum())
        v = cfg.variant
        if v in ("C1", "C1n"):
            Q = cfg.Q or cfg.K + 2
            qm = split_to_quads(m, b)
            self.qm = qm
            T = quad_tensors(qm.cell_xy, cfg.K, Q)
            self.x = T.x
            self.nq = T.n_quad
            self.gx, self.gy = _t(T.gx), _t(T.gy)
            self.val_np = T.val
            self.cell_area = _t(T.area)
            self.n_cells = T.n_elem
            if v == "C1n":
                nat = ~b.dirichlet[qm.bseg]                      # (M, 2) natural components
                keep = nat.any(1)
                ed = np.stack([qm.bcell, qm.bref], 1)[keep]
                _, egx, egy, self.e_val_np = edge_tensors(qm.cell_xy, ed, cfg.K, Q)
                self.e_ed = ed
                self.e_cell = torch.as_tensor(ed[:, 0])
                self.e_mask = _t(nat[keep])
                self.egx, self.egy = _t(egx), _t(egy)
            else:
                t, w = np.polynomial.legendre.leggauss(cfg.edge_Q)
                a, c = m.nodes[b.edges[:, 0]], m.nodes[b.edges[:, 1]]
                pts = (a[:, None] + (c - a)[:, None] * ((t + 1) / 2)[None, :, None])
                self.xb = pts.reshape(-1, 2)
                wb = (w[None, :] * b.length[:, None] / 2).reshape(-1)
                nat = ~b.dirichlet[b.seg]                        # (M, 2)
                self.nb = _t(np.repeat(b.normal, cfg.edge_Q, 0))
                self.mb = _t(np.repeat(nat, cfg.edge_Q, 0))
                self.wb = _t(wb)
                self.LN = float(b.length[nat.any(1)].sum())
        elif v == "C2":
            T = tri_tensors(m.nodes, m.tris, cfg.degree, cfg.quad_degree)
            self.x = T.x
            self.nq = T.n_quad
            self.grad = _t(T.grad)                              # (E, nloc, Nq, 2)
            self.dof = torch.as_tensor(T.dof)
            self.n_dof = T.n_dof
            self.free = _t(self._free_c2(fam_name, m, b, T), torch.bool)
            self.supp = _t(np.bincount(T.dof.ravel(), weights=np.repeat(T.area, T.dof.shape[1]),
                                       minlength=T.n_dof))
            self.bdof = torch.as_tensor(self._boundary_dofs(m, b, T.n_dof))
        elif v == "C2I":
            f = refine_nested(m)
            self.fine = f
            self.x_nodes = np.asarray(f.nodes)
            self.f_tris = torch.as_tensor(np.asarray(f.tris, np.int64))
            self.f_grads = _t(f.grads)                          # (Ef, 3, 2)
            self.f_area = _t(f.areas)
            E = m.n_elem
            self.parent = torch.as_tensor(np.tile(np.arange(E), 4))
            self.c_grads = _t(m.grads)                          # (E, 3, 2) coarse hats
            self.dof = torch.as_tensor(np.asarray(m.tris, np.int64))
            self.n_dof = len(m.nodes)
            self.free = _t(free_dofs_p1(fam_name, m), torch.bool)
            self.supp = _t(np.bincount(np.asarray(m.tris).ravel(),
                                       weights=np.repeat(m.areas, 3), minlength=len(m.nodes)))
            self.bdof = torch.as_tensor(self._boundary_dofs(m, b, len(m.nodes)))
        else:
            raise ValueError(v)

    # ------------------------------------------------------------------
    @staticmethod
    def _free_c2(fam_name, m, b, T):
        free = np.ones((T.n_dof, 2), bool)
        free[:len(m.nodes)] = free_dofs_p1(fam_name, m)
        if T.n_dof > len(m.nodes):                               # P2 edge midpoints
            tris = np.asarray(m.tris, np.int64)
            e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
            uniq = np.unique(np.sort(e, 1), axis=0)
            key = {tuple(r): len(m.nodes) + k for k, r in enumerate(uniq.tolist())}
            for (a, c), s in zip(b.edges.tolist(), b.seg.tolist()):
                d = key[tuple(sorted((a, c)))]
                free[d] &= ~b.dirichlet[s]
        return free

    @staticmethod
    def _boundary_dofs(m, b, n_dof):
        """DOFs on the boundary polygon (vertices; P2 edge midpoints)."""
        out = np.zeros(n_dof, bool)
        out[np.unique(b.edges)] = True
        if n_dof > len(m.nodes):
            tris = np.asarray(m.tris, np.int64)
            e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
            uniq = np.unique(np.sort(e, 1), axis=0)
            key = {tuple(r): len(m.nodes) + k for k, r in enumerate(uniq.tolist())}
            for a, c in b.edges.tolist():
                out[key[tuple(sorted((a, c)))]] = True
        return out

    # ------------------------------------------------------------------
    def points(self):
        """Where the network's displacement gradients are needed (N, 2)."""
        if self.cfg.variant == "C2I":
            return self.x_nodes
        return self.x

    def residuals(self, P):
        """P = (P11, P12, P21, P22) at ``points()``; returns the residual objects."""
        v = self.cfg.variant
        if v in ("C1", "C1n"):
            P = [p.reshape(self.n_cells, self.nq) for p in P]
            Rx = torch.einsum("ekq,eq->ek", self.gx, P[0]) + torch.einsum("ekq,eq->ek", self.gy, P[1])
            Ry = torch.einsum("ekq,eq->ek", self.gx, P[2]) + torch.einsum("ekq,eq->ek", self.gy, P[3])
            out = {"R": torch.stack([Rx, Ry], -1)}
            if v == "C1n":
                Pe = [p[self.e_cell] for p in P]
                ex = torch.einsum("mkq,mq->mk", self.egx, Pe[0]) + torch.einsum("mkq,mq->mk", self.egy, Pe[1])
                ey = torch.einsum("mkq,mq->mk", self.egx, Pe[2]) + torch.einsum("mkq,mq->mk", self.egy, Pe[3])
                out["Re"] = torch.stack([ex, ey], -1)
            return out
        if v == "C2":
            E, nloc = self.dof.shape
            P = [p.reshape(E, self.nq) for p in P]
            gx, gy = self.grad[..., 0], self.grad[..., 1]         # (E, nloc, Nq)
            rx = torch.einsum("elq,eq->el", gx, P[0]) + torch.einsum("elq,eq->el", gy, P[1])
            ry = torch.einsum("elq,eq->el", gx, P[2]) + torch.einsum("elq,eq->el", gy, P[3])
            R = torch.zeros(self.n_dof, 2, dtype=rx.dtype)
            R[:, 0].index_add_(0, self.dof.reshape(-1), rx.reshape(-1))
            R[:, 1].index_add_(0, self.dof.reshape(-1), ry.reshape(-1))
            return {"R": R}
        raise ValueError("C2I residuals come from residuals_interp")

    def fine_grads(self, u_nodes):
        """C2I: u at the fine nodes (Nf, 2) -> the P1 interpolant's constant
        gradient per fine element, as (du_dx, du_dy, dv_dx, dv_dy), each (Ef,)."""
        ue = u_nodes[self.f_tris]                                 # (Ef, 3, 2)
        g = torch.einsum("eac,eaj->ecj", ue, self.f_grads)        # (Ef, comp, dir)
        grads = (g[:, 0, 0], g[:, 0, 1], g[:, 1, 0], g[:, 1, 1])
        return grads

    # ------------------------------------------------------------------
    def loss(self, P, S0, J=None, w_barrier=0.0, j_min=0.05, Pb=None,
             u_nodes=None, mu=None, lam=None, state=None, load=None):
        """Assemble the arm's loss from P at ``points()`` (C1, C1n, C2) or from
        the fine-node displacements (C2I).  Returns (loss, parts dict)."""
        v, cfg = self.cfg.variant, self.cfg
        parts = {}
        if v == "C2I":
            grads = self.fine_grads(u_nodes)
            P11, P12, P21, P22, J2D, J3D = first_piola_kirchhoff_stress(
                *[g.reshape(1, -1, 1) for g in grads], mu, lam, state, return_J3D=True)
            Pf = [p.reshape(-1) for p in (P11, P12, P21, P22)]
            wa = self.f_area[:, None]                             # (Ef, 1)
            cg = self.c_grads[self.parent]                        # (Ef, 3, 2)
            rx = wa * (cg[..., 0] * Pf[0][:, None] + cg[..., 1] * Pf[1][:, None])
            ry = wa * (cg[..., 0] * Pf[2][:, None] + cg[..., 1] * Pf[3][:, None])
            dof = self.dof[self.parent]
            R = torch.zeros(self.n_dof, 2, dtype=rx.dtype)
            R[:, 0].index_add_(0, dof.reshape(-1), rx.reshape(-1))
            R[:, 1].index_add_(0, dof.reshape(-1), ry.reshape(-1))
            J = torch.minimum(J2D, J3D).reshape(-1)
            if load is not None:
                R = R - load["R"]
        else:
            out = self.residuals(P)
            R = out["R"]
            if load is not None:
                R = R - load["R"]
                if "Re" in out:
                    out["Re"] = out["Re"] - load["Re"]
        if v in ("C1", "C1n"):
            scale = self.ell ** 2 / (S0 ** 2 * self.A)
            dom = scale * ((R ** 2).sum(-1).mean(1) / self.cell_area).sum()
            parts["L_domain"] = dom
            main = dom
            if v == "C1n":
                Re = out["Re"]
                edge = scale * (((Re ** 2) * self.e_mask[:, None, :]).sum(-1).mean(1)
                                / self.cell_area[self.e_cell]).sum()
                parts["L_edge"] = edge
                main = main + edge
            elif Pb is not None:
                pen = self.penalty(Pb, S0, None if load is None else load.get("tb"))
                parts["L_pen"] = pen
                main = main + pen
        else:
            if cfg.gamma == "area":
                terms = self.ell ** 2 / (S0 ** 2 * self.A) * R ** 2 / self.supp[:, None]
            else:
                terms = R ** 2 / (S0 ** 2 * self.A)
            terms = terms * self.free
            main = terms.sum()
            parts["L_domain"] = main
            # reported split (monitors): DOFs on the boundary polygon carry the
            # traction residual, a line integral, whose share grows as 1/h
            parts["L_bnd"] = terms[self.bdof].sum().detach()
            parts["L_int"] = (main.detach() - parts["L_bnd"])
        loss = main
        if J is not None and w_barrier > 0:
            lb = torch.mean(torch.relu(j_min - J) ** 2)
            parts["L_barrier"] = lb
            loss = loss + w_barrier * lb
        return loss, parts

    def penalty(self, Pb, S0, tb=None):
        """C1 traction penalty from P = (P11, P12, P21, P22) at ``xb``; ``tb``
        (Nb, 2) is a prescribed traction (manufactured problems), else 0."""
        n1, n2 = self.nb[:, 0], self.nb[:, 1]
        tx = Pb[0] * n1 + Pb[1] * n2
        ty = Pb[2] * n1 + Pb[3] * n2
        if tb is not None:
            tx, ty = tx - tb[:, 0], ty - tb[:, 1]
        t2 = self.mb[:, 0] * tx ** 2 + self.mb[:, 1] * ty ** 2
        return self.cfg.beta * (self.wb * t2).sum() / (S0 ** 2 * self.LN)

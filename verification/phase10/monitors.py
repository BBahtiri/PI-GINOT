#!/usr/bin/env python3
"""
Phase 10 -- the monitors every operator run carries (plan S5 G4b, S6, S7).

(i)  Energy gap.  Pi_net on the FEM reference mesh (Dunavant 4, the 9.2 rule)
     against the Richardson-extrapolated FEM energy from the nested levels
     h, h/2, h/4 of that mesh (the 9.2 method, generalised to any geometry and
     to the dog-bone).  gap = Pi_net / Pi_extrap - 1.
(ii) Residual ratio -- the spurious-solution monitor.  The arm's own loss
     without the barrier, evaluated on the trained model with
       q+2 : the volume rule raised (C1: Gauss K+4 per direction; C2: Dunavant +2)
       h/2 : the same test space on the nested refinement of the arm's mesh
     divided by its value on the training rule.  For A the analogue is the
     strong loss on a draw in the refined mesh (4x the points) over a draw in
     the training mesh.  B has none (its training objective is (i)).
Also: the K_t error on the geometry (family scorer, or compare_one on the
dog-bone), for reporting.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import pickle
import time
from types import SimpleNamespace

import numpy as np
import torch

from config import LOADING_CONFIG, MATERIAL_CONFIG
from physics.neo_hookean import strain_energy_density
from physics.strong_form import StrongGeometry, div_P
from verification.phase9_energy_error import BARY, WQ, energy_fem
from verification.phase9_headroom import refine

MU, LAM, STATE = MATERIAL_CONFIG["mu"], MATERIAL_CONFIG["lam"], MATERIAL_CONFIG["state"]
UD = LOADING_CONFIG["u_max"]
CACHE = "verification/results/phase10/fem_extrap"


# --------------------------------------------------------------------------
# FEM references
# --------------------------------------------------------------------------
def fem_reference(fam_name, p, tag):
    if fam_name == "dogbone":
        from verification.supervised_ceiling import _solve_cached
        return _solve_cached(p, f"p10_{tag}", 1.3)
    from geometry.families import FAMILIES
    from verification.family_fem import reference
    return reference(FAMILIES[fam_name], p, f"p10_{tag}")


def _dirichlet(fam_name, mesh):
    if fam_name == "dogbone":
        from verification.fem.solver import _dirichlet as d
        con, _ = d(mesh, UD)
        return con, (lambda f: d(mesh, UD * f)[1])
    from geometry.families import FAMILIES
    return FAMILIES[fam_name].fem_dirichlet(mesh, UD)


def fem_extrap(fam_name, p, tag, levels=3):
    """Energies on h, h/2, h/4 of the reference mesh and the Richardson limit."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{fam_name}_{tag}.json")
    if os.path.exists(path):
        r = json.load(open(path))
        assert r["params"] == json.loads(json.dumps(p, default=float)), "cache/params mismatch"
        return r
    from verification.fem.solver import newton
    sol = fem_reference(fam_name, p, tag)
    rows = [{"level": 0, "n_elem": int(sol.mesh.n_elem), "Pi": energy_fem(sol)}]
    mesh, u = sol.mesh, sol.u
    for lv in range(1, levels):
        t0 = time.time()
        mesh, uniq = refine(mesh)
        u0 = np.vstack([u, 0.5 * (u[uniq[:, 0]] + u[uniq[:, 1]])])
        con, vals = _dirichlet(fam_name, mesh)
        u, hist, it = newton(mesh, con, vals, u0, MU, LAM, STATE, n_steps=1, verbose=False)
        rows.append({"level": lv, "n_elem": int(mesh.n_elem),
                     "Pi": energy_fem(SimpleNamespace(mesh=mesh, u=u)),
                     "newton_iter": int(it), "residual": float(hist[-1]), "wall_s": time.time() - t0})
    P0, P1, P2 = (r["Pi"] for r in rows[:3])
    rate = math.log2((P0 - P1) / (P1 - P2)) if (P0 - P1) > 0 and (P1 - P2) > 0 else float("nan")
    ext = P2 - (P1 - P2) / (2 ** rate - 1) if np.isfinite(rate) and rate > 0 else float("nan")
    r = {"params": json.loads(json.dumps(p, default=float)), "levels": rows, "rate": rate,
         "Pi_extrap": ext}
    json.dump(r, open(path, "w"), indent=1)
    return r


# --------------------------------------------------------------------------
# (i) energy gap
# --------------------------------------------------------------------------
def energy_net(model, ctx, mesh, chunk=6000):
    from training.formulation_trainer import disp_grads
    verts = mesh.nodes[mesh.tris]
    pts = np.einsum("qa,ead->eqd", BARY, verts).reshape(-1, 2)
    w = (mesh.areas[:, None] * WQ[None, :]).reshape(-1)
    tot, jmin = 0.0, np.inf
    for lo in range(0, len(pts), chunk):
        sl = slice(lo, min(lo + chunk, len(pts)))
        _, _, G = disp_grads(model, ctx.z.detach(), ctx, pts[sl], create_graph=False)
        G = G.double()
        W = strain_energy_density(*[G[:, k].reshape(1, -1, 1) for k in range(4)], MU, LAM, STATE)
        tot += float((W.reshape(-1).numpy() * w[sl]).sum())
        J = (1 + G[:, 0]) * (1 + G[:, 3]) - G[:, 1] * G[:, 2]
        jmin = min(jmin, float(J.min()))
    return tot, jmin


# --------------------------------------------------------------------------
# (ii) residual ratio
# --------------------------------------------------------------------------
def _consistent(name, L, parts, f):
    """The loss with its boundary share rescaled by f = h_level / h_train.

    C2/C2I boundary-DOF residuals and C1n edge-function residuals carry the
    traction error as a line integral, normalised per area: their sum grows as
    1/h, so a boundary-dominated residual doubles on h/2 with nothing spurious
    going on.  Rescaling that share makes the levels comparable; the domain
    terms and C1's penalty are integrals already."""
    if name in ("C2", "C2I"):
        return parts["L_int"] + f * parts["L_bnd"]
    if name == "C1n":
        return parts["L_domain"] + f * parts["L_edge"]
    return L


def residual_ratio(arm, model, ctx, p):
    """{'train', 'q2', 'h2', 'ratio', 'raw'} for A and the C arms; None for B."""
    name = arm.name
    if name == "B":
        return None
    if name == "A":
        return _ratio_A(arm, model, ctx, p)
    cfg = arm.cfg
    g0 = arm.geo(p)
    L0, P0 = arm.evaluate(model, ctx, g0, w_bar=0.0)
    out = {"train": _consistent(name, L0, P0, 1.0)}
    raw = {"train": L0}
    cq = None
    if name in ("C1", "C1n"):
        cq = dataclasses.replace(cfg, Q=(cfg.Q or cfg.K + 2) + 2)
    elif name == "C2" and cfg.quad_degree + 2 in (2, 4, 6):
        cq = dataclasses.replace(cfg, quad_degree=cfg.quad_degree + 2)
    if cq is not None:
        Lq, Pq = arm.evaluate(model, ctx, arm.build(p, cq, mesh=g0.mesh), w_bar=0.0)
        out["q2"], raw["q2"] = _consistent(name, Lq, Pq, 1.0), Lq
    Lh, Ph = arm.evaluate(model, ctx, arm.build(p, cfg, mesh=refine(g0.mesh)[0]), w_bar=0.0)
    out["h2"], raw["h2"] = _consistent(name, Lh, Ph, 0.5), Lh
    out["ratio"] = max(v for k, v in out.items() if k != "train") / out["train"]
    out["raw"] = raw
    return out


def _strong_value(arm, model, ctx, geo, seed=11):
    """The strong loss without the barrier, no parameter graph, chunked."""
    from training.formulation_trainer import P_and_J, disp_grads, _chunks
    cfg = arm.cfg
    geo.rng = np.random.default_rng(seed)
    x, xb, nb, mb, cb = geo.draw()
    s2 = (cfg.L0 / cfg.S0) ** 2
    eq = 0.0
    for sl in _chunks(len(x), arm.chunk):
        q, _, G = disp_grads(model, ctx.z.detach(), ctx, x[sl], create_graph=True)
        P, _ = P_and_J(G)
        fx, fy = div_P(P, q)
        eq += float((s2 * (fx ** 2 + fy ** 2)).sum().detach()) / len(x)
    _, _, Gb = disp_grads(model, ctx.z.detach(), ctx, xb, create_graph=False)
    Pb, _ = P_and_J(Gb)
    n = torch.as_tensor(nb, dtype=Gb.dtype)
    tx = (Pb[0] * n[:, 0] + Pb[1] * n[:, 1]).numpy()
    ty = (Pb[2] * n[:, 0] + Pb[3] * n[:, 1]).numpy()
    res = (mb[:, 0] * tx ** 2 + mb[:, 1] * ty ** 2) / cfg.S0 ** 2
    L = cfg.w_eq * eq
    for c, w in ((0, cfg.w_arc), (1, cfg.w_top), (2, cfg.w_part)):
        if (cb == c).any():
            L += w * float(res[cb == c].mean())
    return L


def _ratio_A(arm, model, ctx, p):
    g0 = StrongGeometry(arm.fam_name, p, arm.cfg, mesh=arm.mesh_of(p))
    g2 = StrongGeometry(arm.fam_name, p, arm.cfg, mesh=refine(g0.mesh)[0])
    L0 = _strong_value(arm, model, ctx, g0)
    L2 = _strong_value(arm, model, ctx, g2)
    return {"train": L0, "h2": L2, "ratio": L2 / L0}


# --------------------------------------------------------------------------
# K_t
# --------------------------------------------------------------------------
def kt_err(model, fam_name, p, sol):
    if fam_name == "dogbone":
        from eval.compare_reference import compare_one
        from verification.heldout_sets import entries
        model.set_geometry(p)
        _, coll = entries([p], 7700)[0]
        r = compare_one(model, p, UD, None, "cpu", sample_id=30000, coll=coll, sol=sol)
        return {"err": float(r["scf_rel_err"]), "vm": float(r["vm_all"]["rel_L2"])}
    from geometry.families import FAMILIES
    from verification.family_score import score_geometry
    s = score_geometry(model, FAMILIES[fam_name], p, sol)
    return {"err": float(s["err"]), "vm": float(s["vm"])}


# --------------------------------------------------------------------------
class Monitor:
    """Callable for formulation_trainer.train(monitor=...)."""

    def __init__(self, arm_name, fam_name, geoms, tags, overrides=None, kt=True):
        from training.family_trainer import _sets
        from training.formulation_trainer import make_arm
        _, _, self.extent = _sets(fam_name)
        self.fam_name = fam_name
        self.arm = make_arm(arm_name, fam_name, self.extent, overrides)
        self.geoms = geoms
        self.sols = [fem_reference(fam_name, p, t) for p, t in zip(geoms, tags)]
        self.ext = [fem_extrap(fam_name, p, t)["Pi_extrap"] for p, t in zip(geoms, tags)]
        self.kt = kt

    def __call__(self, model, step):
        from training.formulation_trainer import Ctx
        was = model.training
        model.eval()
        out = []
        for p, sol, ext in zip(self.geoms, self.sols, self.ext):
            ctx = Ctx(model, self.fam_name, p, self.extent, torch.get_default_dtype())
            Pn, jmin = energy_net(model, ctx, sol.mesh)
            r = {"gap": Pn / ext - 1.0, "Pi_net": Pn, "Jmin": jmin}
            rr = residual_ratio(self.arm, model, ctx, p)
            if rr is not None:
                r["res"] = rr
            if self.kt:
                r.update(kt_err(model, self.fam_name, p, sol))
            out.append(r)
        model.train(was)
        return {"geoms": out}

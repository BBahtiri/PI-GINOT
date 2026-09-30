#!/usr/bin/env python3
"""
Phase 10, gate G4a -- every arm learns a manufactured solution.

On the first bank geometry of the open hole and the double notch (as in G2 and
G3), a manufactured field that satisfies the family's Dirichlet data exactly,

    u* = spec.apply(X, N*(X)),   N*_x = 0.3 u_d sin(pi X/L) cos(pi Y/(2H)),
                                 N*_y = -0.2 u_d (X/L)(Y/H),

is imposed through the body force b = -Div P(u*) and the tractions
t = P(u*) N_h on every natural component of the polygon (autograd, float64), so
u* is the exact solution of the polygonal problem.  A plain network (4 x 64
tanh, inputs X / max(L, H), the family's hard Dirichlet layer -- u* is in its
trial space) is trained from scratch by each arm, with the matching load terms:

  B    Pi(u) - int b.u - int_{Gamma_N} t.u      3 stratified points per triangle, fresh
                                                each step; traction work on Gauss 4
  A    the strong-form loss with Div P + b and (P N - t)_nat   (physics/strong_form.py)
  C1   bubbles on the coarse level, K = 2:  R - int b v;  penalty on P N - t
  C1n  bubbles + edge functions:  R - int b v,  Re - int b psi - int t psi ds
  C2   P1 hats on B's level, Dunavant 4:  R - int b phi - int t phi ds
  C2I  as C2 on the interpolant (P1 on the h/2 refinement)

Loads use the arm's own volume rule (8-point Gauss on edges).  4,000 Adam steps,
1e-3 with cosine decay to 1e-5, float32, seed 0.

Pass: relative H1-seminorm error to u* <= 2e-2 on an independent rule (the h/2
nested refinement of B's mesh, Dunavant 6).

    python -m verification.phase10.g4a_mms_train FAMILY ARM [ARM ...]
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

from config import LOADING_CONFIG, MATERIAL_CONFIG, NONDIM_SCALES
from geometry.families import FAMILIES
from physics.neo_hookean import first_piola_kirchhoff_stress, strain_energy_density
from physics.strong_form import StrongConfig, StrongGeometry, strong_loss
from physics.vpinn.basis import edge_functions, lagrange_tri
from physics.vpinn.quadrature import triangle_rule
from physics.vpinn.weak import WeakConfig, WeakGeometry, refine_nested

MU, LAM, STATE = MATERIAL_CONFIG["mu"], MATERIAL_CONFIG["lam"], MATERIAL_CONFIG["state"]
S0, UD = NONDIM_SCALES["S0"], LOADING_CONFIG["u_max"]
OUT_DIR = "verification/results/phase10/g4a"
ARMS = ["B", "A", "C1", "C1n", "C2", "C2I"]


def P_and_J(g):
    o = first_piola_kirchhoff_stress(*[t.reshape(1, -1, 1) for t in g], MU, LAM, STATE,
                                     return_J3D=True)
    return [t.reshape(-1) for t in o[:4]], torch.minimum(o[4], o[5]).reshape(-1)


def grads(fn, x, create_graph=True):
    u = fn(x)
    gu, = torch.autograd.grad(u[:, 0].sum(), x, create_graph=create_graph, retain_graph=True)
    gv, = torch.autograd.grad(u[:, 1].sum(), x, create_graph=create_graph)
    return [gu[:, 0], gu[:, 1], gv[:, 0], gv[:, 1]]


class Problem:
    def __init__(self, fam_name):
        fam = FAMILIES[fam_name]
        self.fam_name = fam_name
        self.p = fam.bank(64)[0]
        self.L, self.H = fam.extent(self.p)
        self.spec = fam.spec(self.p, UD)
        from training.family_trainer import build_loss
        self.mesh = build_loss(fam_name)._mesh(self.p)          # B's triangulation

    def u_star(self, X):
        L, H = self.L, self.H
        N = torch.stack([0.3 * UD * torch.sin(math.pi * X[:, 0] / L) * torch.cos(math.pi * X[:, 1] / (2 * H)),
                         -0.2 * UD * (X[:, 0] / L) * (X[:, 1] / H)], -1)
        return self.spec.apply(X, N)

    def b(self, x):
        """b = -Div P(u*) at numpy points (N, 2), float64."""
        with torch.enable_grad():
            X = torch.tensor(np.asarray(x), dtype=torch.float64, requires_grad=True)
            P, _ = P_and_J(grads(self.u_star, X))
            div = []
            for i in range(2):
                d = 0
                for j in range(2):
                    g, = torch.autograd.grad(P[2 * i + j].sum(), X, retain_graph=True)
                    d = d + g[:, j]
                div.append(d)
        return -torch.stack(div, -1).detach().numpy()

    def t(self, x, n):
        """t = P(u*) n at numpy points and normals, float64."""
        X = torch.tensor(np.asarray(x), dtype=torch.float64, requires_grad=True)
        P, _ = P_and_J(grads(self.u_star, X, create_graph=False))
        P = [q.detach().numpy() for q in P]
        n = np.asarray(n)
        return np.stack([P[0] * n[:, 0] + P[1] * n[:, 1], P[2] * n[:, 0] + P[3] * n[:, 1]], 1)


class Net(torch.nn.Module):
    def __init__(self, prob, width=64, depth=4):
        super().__init__()
        layers, d = [], 2
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
            d = width
        layers += [torch.nn.Linear(d, 2)]
        self.mlp = torch.nn.Sequential(*layers)
        self.spec = prob.spec
        self.scale = 1.0 / max(prob.L, prob.H)

    def forward(self, X):
        return self.spec.apply(X, UD * self.mlp(X * self.scale))


# --------------------------------------------------------------------------
# load vectors
# --------------------------------------------------------------------------
def c2_load(geo, prob, n_edge=8):
    """P1 (or C2I) load: int b phi (the arm's rule, or Dunavant 6 for C2I) plus
    int_{Gamma_N} t phi ds on every polygon edge (natural components only matter:
    Dirichlet rows are never used)."""
    m = geo.mesh
    tris = np.asarray(m.tris, np.int64)
    qd = geo.cfg.quad_degree if geo.cfg.variant == "C2" else 6
    r, s, w = triangle_rule(qd)
    N, _, _ = lagrange_tri(1, r, s)                             # (3, Nq)
    V = np.asarray(m.nodes)[tris]
    x = (V[:, None, 0] * (1 - r - s)[None, :, None] + V[:, None, 1] * r[None, :, None]
         + V[:, None, 2] * s[None, :, None])
    if geo.cfg.variant == "C2":
        assert np.allclose(x.reshape(-1, 2), geo.x)
    bq = prob.b(x.reshape(-1, 2)).reshape(len(tris), len(w), 2)
    c = np.einsum("lq,eq,eqc->elc", N, w[None] * np.asarray(m.areas)[:, None], bq)
    out = np.zeros((len(m.nodes), 2))
    for k in range(3):
        for cc in range(2):
            np.add.at(out[:, cc], tris[:, k], c[:, k, cc])
    bnd = geo.bnd
    t1, w1 = np.polynomial.legendre.leggauss(n_edge)
    sfr = (t1 + 1) / 2
    a, e = m.nodes[bnd.edges[:, 0]], m.nodes[bnd.edges[:, 1]]
    xe = a[:, None] + (e - a)[:, None] * sfr[None, :, None]
    tr = prob.t(xe.reshape(-1, 2), np.repeat(bnd.normal, n_edge, 0)).reshape(len(a), n_edge, 2)
    we = w1[None] * bnd.length[:, None] / 2
    for k, (i, j) in enumerate(bnd.edges.tolist()):
        out[i] += (we[k][:, None] * tr[k] * (1 - sfr)[:, None]).sum(0)
        out[j] += (we[k][:, None] * tr[k] * sfr[:, None]).sum(0)
    return out


def c1_loads(geo, prob, n_edge=8):
    load = {}
    bx = prob.b(geo.x).reshape(geo.n_cells, geo.nq, 2)
    load["R"] = np.einsum("ekq,eqc->ekc", geo.val_np, bx)
    if geo.cfg.variant == "C1":
        load["tb"] = prob.t(geo.xb, geo.nb.numpy())
        return load
    qm, ed = geo.qm, geo.e_ed
    keep = (~geo.bnd.dirichlet[qm.bseg]).any(1)
    nrm = qm.bnormal[keep]
    vol = np.einsum("mkq,mqc->mkc", geo.e_val_np, bx[ed[:, 0]])
    t1, w1 = np.polynomial.legendre.leggauss(n_edge)
    edge = np.zeros_like(vol)
    for mm, (cidx, ref) in enumerate(ed.tolist()):
        cxy = qm.cell_xy[cidx]
        if ref == 0:                                            # eta = -1: corner 0 -> 1
            xi, eta, pa, pb = t1, -np.ones_like(t1), cxy[0], cxy[1]
        else:                                                   # xi = -1: corner 0 -> 3
            xi, eta, pa, pb = -np.ones_like(t1), t1, cxy[0], cxy[3]
        psi, _, _ = edge_functions(geo.cfg.K, xi, eta, ref)     # (K, n_edge)
        pts = pa[None] + (pb - pa)[None] * ((t1 + 1) / 2)[:, None]
        tr = prob.t(pts, np.repeat(nrm[mm][None], n_edge, 0))
        edge[mm] = np.einsum("kq,qc->kc", psi * (w1 * np.linalg.norm(pb - pa) / 2)[None], tr)
    load["Re"] = vol + edge
    return load


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------
def build_arm(arm, prob, dt, seed=0, strong_cfg=None, level="B"):
    m = prob.mesh
    if arm == "B":
        from geometry.vpinn_mesh import boundary
        bnd = boundary(prob.fam_name, m)
        V = np.asarray(m.nodes)[np.asarray(m.tris)]
        wq = np.repeat(np.asarray(m.areas) / 3.0, 3)
        rng = np.random.default_rng(seed)
        t4, w4 = np.polynomial.legendre.leggauss(4)
        a, c = m.nodes[bnd.edges[:, 0]], m.nodes[bnd.edges[:, 1]]
        xb = (a[:, None] + (c - a)[:, None] * ((t4 + 1) / 2)[None, :, None]).reshape(-1, 2)
        wb = (w4[None] * bnd.length[:, None] / 2).reshape(-1)
        nat = np.repeat(~bnd.dirichlet[bnd.seg], 4, 0)
        tb = torch.tensor(prob.t(xb, np.repeat(bnd.normal, 4, 0)) * nat * wb[:, None], dtype=dt)
        xbt = torch.tensor(xb, dtype=dt)
        wqt = torch.tensor(wq, dtype=dt)
        A = float(np.asarray(m.areas).sum())

        def loss(net):
            r1, r2 = rng.random((len(V), 3)), rng.random((len(V), 3))
            f = (r1 + r2) > 1
            r1, r2 = np.where(f, 1 - r1, r1), np.where(f, 1 - r2, r2)
            x = (V[:, None, 0] * (1 - r1 - r2)[..., None] + V[:, None, 1] * r1[..., None]
                 + V[:, None, 2] * r2[..., None]).reshape(-1, 2)
            X = torch.tensor(x, dtype=dt, requires_grad=True)
            u = net(X)
            gu, = torch.autograd.grad(u[:, 0].sum(), X, create_graph=True)
            gv, = torch.autograd.grad(u[:, 1].sum(), X, create_graph=True)
            W = strain_energy_density(*[q.reshape(1, -1, 1) for q in
                                        (gu[:, 0], gu[:, 1], gv[:, 0], gv[:, 1])],
                                      MU, LAM, STATE).reshape(-1)
            bx = torch.tensor(prob.b(x), dtype=dt)
            Pi = (wqt * (W - (bx * u).sum(1))).sum() - (tb * net(xbt)).sum()
            return Pi / (S0 * A)
        return loss

    if arm == "A":
        geo = StrongGeometry(prob.fam_name, prob.p, strong_cfg or StrongConfig(), mesh=m, seed=seed)

        def loss(net):
            x, xb, nb, mb, cb = geo.draw()
            Lo, _ = strong_loss(lambda q: P_and_J(grads(net, q)), x, xb, nb, mb, cb, geo.cfg,
                                b=prob.b(x), tb=prob.t(xb, nb))
            return Lo
        return loss

    cfg = {"C1": WeakConfig("C1", K=2, n_elem=350), "C1n": WeakConfig("C1n", K=2, n_elem=350),
           "C2": WeakConfig("C2", degree=1, quad_degree=4), "C2I": WeakConfig("C2I")}[arm]
    if level == "h2" and not arm.startswith("C1"):
        m = refine_nested(m)                                    # diagnostic: the h/2 test mesh
    geo = WeakGeometry(prob.fam_name, prob.p, cfg, prob.H, mesh=None if arm.startswith("C1") else m)
    if arm.startswith("C1"):
        load = {k: torch.tensor(v, dtype=dt) for k, v in c1_loads(geo, prob).items()}
    else:
        load = {"R": torch.tensor(c2_load(geo, prob), dtype=dt)}
    pts = torch.tensor(geo.points(), dtype=dt)
    xb = torch.tensor(geo.xb, dtype=dt) if arm == "C1" else None

    def loss(net):
        if arm == "C2I":
            Lo, _ = geo.loss(None, S0, u_nodes=net(pts), mu=MU, lam=LAM, state=STATE, load=load)
            return Lo
        P, J = P_and_J(grads(net, pts.clone().requires_grad_(True)))
        Pb = P_and_J(grads(net, xb.clone().requires_grad_(True)))[0] if arm == "C1" else None
        Lo, _ = geo.loss(P, S0, J=J, Pb=Pb, load=load)
        return Lo
    loss.n_eq = (int(geo.free.sum()) if arm.startswith("C2") else
                 geo.n_cells * cfg.K ** 2 * 2 + (0 if arm == "C1" else int(geo.e_mask.sum()) * cfg.K))
    return loss


def h1_error(net, prob, dt):
    fine = refine_nested(prob.mesh)
    r, s, w = triangle_rule(6)
    V = np.asarray(fine.nodes)[np.asarray(fine.tris)]
    x = (V[:, None, 0] * (1 - r - s)[None, :, None] + V[:, None, 1] * r[None, :, None]
         + V[:, None, 2] * s[None, :, None]).reshape(-1, 2)
    ww = (w[None] * np.asarray(fine.areas)[:, None]).reshape(-1)
    X = torch.tensor(x, dtype=dt, requires_grad=True)
    g = torch.stack(grads(net, X, create_graph=False), -1).detach().double().numpy()
    Xs = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    gs = torch.stack(grads(prob.u_star, Xs, create_graph=False), -1).detach().numpy()
    e1 = np.sqrt((((g - gs) ** 2).sum(1) * ww).sum() / ((gs ** 2).sum(1) * ww).sum())
    with torch.no_grad():
        u = net(torch.tensor(x, dtype=dt)).double().numpy()
        us = prob.u_star(torch.tensor(x, dtype=torch.float64)).numpy()
    e0 = np.sqrt((((u - us) ** 2).sum(1) * ww).sum() / ((us ** 2).sum(1) * ww).sum())
    return float(e1), float(e0)


def run(fam_name, arm, steps=4000, seed=0, lr0=1e-3, lr1=1e-5, log_every=250, strong_cfg=None,
        save=None, level="B"):
    torch.manual_seed(seed)
    dt = torch.float32
    torch.set_default_dtype(dt)
    prob = Problem(fam_name)
    net = Net(prob)
    t0 = time.time()
    loss_fn = build_arm(arm, prob, dt, seed=seed, strong_cfg=strong_cfg, level=level)
    t_setup = time.time() - t0
    opt = torch.optim.Adam(net.parameters(), lr=lr0)
    hist = []
    t0 = time.time()
    for k in range(steps):
        lr = lr1 + 0.5 * (lr0 - lr1) * (1 + math.cos(math.pi * k / steps))
        for gp in opt.param_groups:
            gp["lr"] = lr
        opt.zero_grad()
        Lo = loss_fn(net)
        Lo.backward()
        opt.step()
        if (k + 1) % log_every == 0 or k == 0:
            e1, e0 = h1_error(net, prob, dt)
            hist.append({"step": k + 1, "loss": float(Lo.detach()), "h1": e1, "l2": e0,
                         "t": time.time() - t0})
            print(f"  {fam_name} {arm} {k + 1:5d} loss {float(Lo):+.4e} h1 {e1:.3e} l2 {e0:.3e} "
                  f"{time.time() - t0:.0f}s", flush=True)
    e1, e0 = h1_error(net, prob, dt)
    if save:
        torch.save(net.state_dict(), save)
    return {"h1_rel": e1, "l2_rel": e0, "wall_s": time.time() - t0, "setup_s": t_setup,
            "n_eq": getattr(loss_fn, "n_eq", None), "steps": steps, "hist": hist,
            "pass": e1 <= 2e-2}


def main():
    """Registered runs:   FAMILY [ARM ...]
    A diagnostics:     FAMILY A --diag NAME --steps N --weq W --wtrac W   (not the gate)"""
    import argparse
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("arms", nargs="*")
    ap.add_argument("--steps", type=int, default=int(os.environ.get("G4A_STEPS", 4000)))
    ap.add_argument("--diag", default=None)
    ap.add_argument("--weq", type=float, default=None)
    ap.add_argument("--wtrac", type=float, default=None)
    ap.add_argument("--level", default="B")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for arm in (a.arms or ARMS):
        sc = None
        if a.weq is not None or a.wtrac is not None:
            import dataclasses
            sc = StrongConfig()
            if a.weq is not None:
                sc = dataclasses.replace(sc, w_eq=a.weq)
            if a.wtrac is not None:
                sc = dataclasses.replace(sc, w_arc=a.wtrac, w_top=a.wtrac, w_part=a.wtrac)
        tag = f"diag_{a.diag}_{a.family}_{arm}" if a.diag else f"{a.family}_{arm}"
        r = run(a.family, arm, steps=a.steps, log_every=max(1, min(250, a.steps // 4)), strong_cfg=sc,
                save=f"{OUT_DIR}/{tag}.pt", level=a.level)
        if a.diag:
            r["diag"] = {"name": a.diag, "w_eq": a.weq, "w_trac": a.wtrac, "level": a.level}
            json.dump(r, open(f"{OUT_DIR}/diag_{a.diag}_{a.family}_{arm}.json", "w"), indent=1)
        elif a.steps == 4000:
            json.dump(r, open(f"{OUT_DIR}/{a.family}_{arm}.json", "w"), indent=1)
        print("RESULT", a.family, arm, {k: v for k, v in r.items() if k != "hist"}, flush=True)


if __name__ == "__main__":
    main()

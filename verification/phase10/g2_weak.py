#!/usr/bin/env python3
"""
Phase 10, gate G2 -- the weak-form losses are the weak form (no training).

All in float64, on each family's first bank geometry at the energy form's
budget (~1400 triangles), for the four weak arms C1, C1n, C2 (degree 1 and 2)
and C2I:

  (a) patch test -- an affine displacement (homogeneous F0):
        C1 bubbles and C2 interior DOFs: residual 0;
        C2 / C2I boundary DOFs and C1n edge functions: residual equals the exact
        traction work  int (P(F0) N_h)_c v ds  on the polygon edges.
  (b) FEM equivalence -- C2 degree 1 on a mesh with any piecewise-linear field
        (non-equilibrium, per-element gradients) equals the FEM's internal force
        (verification.fem.solver.assemble) at the free DOFs.
  (c) manufactured solution -- the sympy field of verification/mms.py with
        b = -Div P* and polygon-normal tractions t = P* N_h on every natural
        component (free edges, symmetry shear, sliding grip):
        R(u*) - int b.v - int t.v ds -> 0 as the quadrature order rises.
  (d) energy consistency -- R = d/deps Pi(u + eps v) under the same rule.
  (e) loss gradients match central finite differences (small network).
  (f) null-space probe -- reported: response of each loss vs the energy to
        cubic element bubbles added to the discrete FEM minimiser.
  (g) counts -- test equations per geometry against 573k parameters.

    python -m verification.phase10.g2_weak
"""
import json
import sys

import numpy as np
import sympy as sp
import torch

torch.set_default_dtype(torch.float64)

from config import MATERIAL_CONFIG, NONDIM_SCALES                      # noqa: E402
from geometry.vpinn_mesh import base_triangulation                     # noqa: E402
from physics.neo_hookean import (first_piola_kirchhoff_stress,         # noqa: E402
                                 strain_energy_density)
from physics.vpinn.basis import edge_functions, lagrange_tri, tensor_2d  # noqa: E402
from physics.vpinn.quadrature import square_rule, triangle_rule        # noqa: E402
from physics.vpinn.weak import WeakConfig, WeakGeometry                # noqa: E402

MU, LAM, STATE = MATERIAL_CONFIG["mu"], MATERIAL_CONFIG["lam"], MATERIAL_CONFIG["state"]
S0 = NONDIM_SCALES["S0"]
FAMS = ["dogbone", "open_hole", "inclusion", "double_notch", "single_notch"]
OUT = "verification/results/phase10/g2.json"
ARMS = {"C1": WeakConfig("C1", K=2), "C1n": WeakConfig("C1n", K=2),
        "C2p1": WeakConfig("C2", degree=1, quad_degree=4),
        "C2p2": WeakConfig("C2", degree=2, quad_degree=6),
        "C2I": WeakConfig("C2I")}


def params_of(fam):
    if fam == "dogbone":
        from geometry.banks import train_bank_params
        from config import get_fillet_geometry
        p = train_bank_params(64)[0]
        fi = get_fillet_geometry(p)
        return p, (fi["L_half"], fi["H_grip"])
    from geometry.families import FAMILIES
    f = FAMILIES[fam]
    p = f.bank(64)[0]
    return p, f.extent(p)


def P_of(g):
    """(du_dx, du_dy, dv_dx, dv_dy) tensors -> (P11, P12, P21, P22), J."""
    P11, P12, P21, P22, J2, J3 = first_piola_kirchhoff_stress(
        *[t.reshape(1, -1, 1) for t in g], MU, LAM, STATE, return_J3D=True)
    return [p.reshape(-1) for p in (P11, P12, P21, P22)], torch.minimum(J2, J3).reshape(-1)


def edge_quadrature(geo, n=8):
    """Gauss points on every polygon boundary edge: x (M, n, 2), w (M, n), seg, normal."""
    b = geo.bnd
    m = geo.mesh
    t, w = np.polynomial.legendre.leggauss(n)
    a, c = m.nodes[b.edges[:, 0]], m.nodes[b.edges[:, 1]]
    x = a[:, None] + (c - a)[:, None] * ((t + 1) / 2)[None, :, None]
    return x, w[None] * b.length[:, None] / 2, t


def p1_hat_on_edges(geo, x_edge, t):
    """Values of P1 hats of the two edge nodes along each edge: (M, n, 2)."""
    s = (t + 1) / 2
    return np.stack([np.broadcast_to(1 - s, x_edge.shape[:2]), np.broadcast_to(s, x_edge.shape[:2])], -1)


def expected_boundary_c2(geo, traction_fn, n=8):
    """sum over boundary edges of int t_c(x) phi_i ds, for the C2 DOFs (P1 or P2)."""
    x, w, t = edge_quadrature(geo, n)
    tr = traction_fn(x.reshape(-1, 2), np.repeat(geo.bnd.normal, n, 0)).reshape(len(x), n, 2)
    s = (t + 1) / 2
    ed = geo.bnd.edges
    out = np.zeros((geo.n_dof, 2))
    if geo.cfg.variant == "C2" and geo.cfg.degree == 2:
        m = geo.mesh
        tris = np.asarray(m.tris, np.int64)
        e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
        uniq = np.unique(np.sort(e, 1), axis=0)
        key = {tuple(r): len(m.nodes) + k for k, r in enumerate(uniq.tolist())}
        l0, l1 = 1 - s, s
        shp = [l0 * (2 * l0 - 1), l1 * (2 * l1 - 1), 4 * l0 * l1]
        for k, (a, c) in enumerate(ed.tolist()):
            dofs = [a, c, key[tuple(sorted((a, c)))]]
            for d, phi in zip(dofs, shp):
                out[d] += (w[k][:, None] * tr[k] * phi[:, None]).sum(0)
        return out
    for k, (a, c) in enumerate(ed.tolist()):
        out[a] += (w[k][:, None] * tr[k] * (1 - s)[:, None]).sum(0)
        out[c] += (w[k][:, None] * tr[k] * s[:, None]).sum(0)
    return out


# --------------------------------------------------------------------------
def gate_a(fam, geos):
    F0 = np.array([[1.012, 0.004], [-0.003, 0.991]])
    g0 = [F0[0, 0] - 1, F0[0, 1], F0[1, 0], F0[1, 1] - 1]
    Pc, _ = P_of([torch.tensor([v]) for v in g0])
    Pm = np.array([[float(Pc[0]), float(Pc[1])], [float(Pc[2]), float(Pc[3])]])
    traction = lambda x, nrm: nrm @ Pm.T                          # t_c = P_cJ N_J
    res = {}
    for name, geo in geos.items():
        pts = geo.points()
        g = [torch.full((len(pts),), v) for v in g0]
        if geo.cfg.variant == "C2I":
            u = torch.tensor(pts @ (F0 - np.eye(2)).T)
            gf = geo.fine_grads(u)
            gradsP, _ = P_of(gf)
            wa = geo.f_area[:, None]
            cg = geo.c_grads[geo.parent]
            rx = wa * (cg[..., 0] * gradsP[0][:, None] + cg[..., 1] * gradsP[1][:, None])
            ry = wa * (cg[..., 0] * gradsP[2][:, None] + cg[..., 1] * gradsP[3][:, None])
            R = torch.zeros(geo.n_dof, 2)
            dof = geo.dof[geo.parent]
            R[:, 0].index_add_(0, dof.reshape(-1), rx.reshape(-1))
            R[:, 1].index_add_(0, dof.reshape(-1), ry.reshape(-1))
            R = R.numpy()
            exp = expected_boundary_c2(geo, traction)
            scale = np.abs(exp).max()
            res[name] = {"max_dev_rel": float(np.abs(R - exp)[geo.free.numpy()].max() / scale)}
            continue
        P, _ = P_of(g)
        out = geo.residuals(P)
        scale_el = float(np.abs(Pm).max())
        if geo.cfg.variant in ("C1", "C1n"):
            R = out["R"].numpy()
            dev = np.abs(R).max(axis=(1, 2)) / (scale_el * np.sqrt(geo.cell_area.numpy()))
            r = {"bubble_max_rel": float(dev.max())}
            if geo.cfg.variant == "C1n":
                # expected: int over the boundary edge of (P N)_c psi ds, psi the edge function
                qm = geo.qm
                keep = (~geo.bnd.dirichlet[qm.bseg]).any(1)
                cells = qm.cell_xy[qm.bcell[keep]]
                refs = qm.bref[keep]
                nrm = qm.bnormal[keep]
                t, w = np.polynomial.legendre.leggauss(8)
                exp = np.zeros((keep.sum(), geo.cfg.K, 2))
                for m_, (cxy, e, nn) in enumerate(zip(cells, refs, nrm)):
                    if e == 0:
                        xi, eta = t, -np.ones_like(t)
                        L = np.linalg.norm(cxy[1] - cxy[0])
                    else:
                        xi, eta = -np.ones_like(t), t
                        L = np.linalg.norm(cxy[0] - cxy[3])
                    psi, _, _ = edge_functions(geo.cfg.K, xi, eta, e)
                    tr = Pm @ nn
                    exp[m_] = (psi * w[None] * L / 2).sum(1)[:, None] * tr[None, :]
                mask = geo.e_mask.numpy()[:, None, :]
                Re = out["Re"].numpy() * mask
                r["edge_max_rel"] = float(np.abs(Re - exp * mask).max() / np.abs(exp).max())
            res[name] = r
        else:
            R = out["R"].numpy()
            exp = expected_boundary_c2(geo, traction)
            scale = np.abs(exp).max()
            res[name] = {"max_dev_rel": float(np.abs(R - exp)[geo.free.numpy()].max() / scale)}
    ok = all(max(v.values()) < 1e-12 for v in res.values())
    return res, ok


def gate_b(fam, p, ell):
    from verification.fem.solver import assemble
    geo = WeakGeometry(fam, p, WeakConfig("C2", degree=1, quad_degree=4), ell)
    m = geo.mesh
    X = m.nodes
    L, H = X[:, 0].max(), X[:, 1].max()
    u = np.stack([0.02 * np.sin(3 * X[:, 0] / L) + 0.01 * X[:, 1] / H,
                  0.015 * np.cos(2 * X[:, 1] / H) * X[:, 0] / L], 1)
    f, _, _, _ = assemble(m, u, MU, LAM, STATE, want_tangent=False)
    ge = np.einsum("eai,eaj->eij", u[m.tris], m.grads)          # (E, comp, dir)
    nq = geo.nq
    g = [torch.tensor(np.repeat(ge[:, i, j], nq)) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
    P, _ = P_of(g)
    R = geo.residuals(P)["R"].numpy()
    free = geo.free.numpy()
    fr = f.reshape(-1, 2)
    rel = float(np.abs(R[free] - fr[free]).max() / np.abs(fr[free]).max())
    return {"max_rel": rel, "n_free": int(free.sum())}, rel < 1e-12


def mms_fns(ell_x, ell_y, amp=(0.35, 0.25)):
    from verification.mms import _lambdify, _symbolic_fields
    s = _symbolic_fields(STATE, ell_x, ell_y, amp)
    X, Y = s["X"], s["Y"]
    grads = [_lambdify(sp.diff(s["u"], X), X, Y), _lambdify(sp.diff(s["u"], Y), X, Y),
             _lambdify(sp.diff(s["v"], X), X, Y), _lambdify(sp.diff(s["v"], Y), X, Y)]
    Pf = [_lambdify(e, X, Y) for e in s["P"]]
    div = [_lambdify(e, X, Y) for e in s["div"]]
    ev = lambda fs, x: [np.broadcast_to(np.asarray(f(x[:, 0], x[:, 1]), float), (len(x),)).copy()
                        for f in fs]
    return (lambda x: ev(grads, x)), (lambda x: ev(Pf, x)), (lambda x: ev(div, x))


def gate_c(fam, p, ell, extent):
    gradf, Pf, divf = mms_fns(*extent)

    def traction(x, nrm):
        P = Pf(x)
        return np.stack([P[0] * nrm[:, 0] + P[1] * nrm[:, 1], P[2] * nrm[:, 0] + P[3] * nrm[:, 1]], 1)

    res = {}
    for name, cfg_list in {"C1": [WeakConfig("C1", K=2, Q=q) for q in (2, 4, 6)],
                           "C2p1": [WeakConfig("C2", degree=1, quad_degree=d) for d in (2, 4, 6)],
                           "C2p2": [WeakConfig("C2", degree=2, quad_degree=d) for d in (2, 4, 6)]}.items():
        errs = []
        for cfg in cfg_list:
            geo = WeakGeometry(fam, p, cfg, ell)
            x = geo.points()
            P, _ = P_of([torch.tensor(a) for a in gradf(x)])
            R = geo.residuals(P)["R"].numpy()
            b = -np.stack(divf(x), 1)                               # body force at the points
            if cfg.variant == "C1":
                # int b_c v : the value tensor (w detJ folded) against b
                from physics.vpinn.tensors import quad_tensors
                T = quad_tensors(geo.qm.cell_xy, cfg.K, cfg.Q)
                bb = b.reshape(T.n_elem, T.n_quad, 2)
                load = np.einsum("ekq,eqc->ekc", T.val, bb)
                e = np.abs(R - load).max() / np.abs(load).max()
            else:
                from physics.vpinn.tensors import tri_tensors
                T = tri_tensors(geo.mesh.nodes, geo.mesh.tris, cfg.degree, cfg.quad_degree)
                r_, s_, w_ = triangle_rule(cfg.quad_degree)
                N, _, _ = lagrange_tri(cfg.degree, r_, s_)
                wv = N[None] * (w_[None, None, :] * T.area[:, None, None])  # (E, nloc, Nq)
                bb = b.reshape(len(T.area), -1, 2)
                vol = np.zeros((T.n_dof, 2))
                contrib = np.einsum("elq,eqc->elc", wv, bb)
                for c in range(2):
                    np.add.at(vol[:, c], T.dof.ravel(), contrib[..., c].ravel())
                load = vol + expected_boundary_c2(geo, traction, n=10)
                free = geo.free.numpy()
                e = np.abs(R - load)[free].max() / np.abs(load[free]).max()
            errs.append(float(e))
        res[name] = errs
    # pass: the finest rule is at least 100x better than the coarsest, and decreasing
    ok = all(v[-1] < v[0] / 100 and v[1] <= v[0] for v in res.values())
    return res, ok


def gate_d(fam, p, ell, extent):
    """R vs d/deps of the rule's energy for a few C2 (P1, P2) and C1 test functions."""
    gradf, _, _ = mms_fns(*extent)
    rng = np.random.default_rng(0)
    res = {}
    for name, cfg in {"C2p1": WeakConfig("C2", degree=1, quad_degree=4),
                      "C2p2": WeakConfig("C2", degree=2, quad_degree=6),
                      "C1": WeakConfig("C1", K=2)}.items():
        geo = WeakGeometry(fam, p, cfg, ell)
        x = geo.points()
        g = [torch.tensor(a) for a in gradf(x)]
        P, _ = P_of(g)
        R = geo.residuals(P)["R"].numpy()
        worst = 0.0
        if cfg.variant == "C2":
            wq = geo.grad.numpy()                                   # (E, nloc, Nq, 2), w|T| folded
            r_, s_, w_ = triangle_rule(cfg.quad_degree)
            wts = (w_[None, :] * np.asarray(geo.mesh.areas)[:, None])  # (E, Nq)
            raw = wq / wts[:, None, :, None]
            dof = geo.dof.numpy()
            for i in rng.choice(geo.n_dof, 6, replace=False):
                for c in range(2):
                    eps = torch.zeros((), requires_grad=True)
                    gg = [t.clone().reshape(len(dof), -1) for t in g]
                    E_, L_ = np.nonzero(dof == i)
                    for e_, l_ in zip(E_, L_):
                        gx = torch.tensor(raw[e_, l_, :, 0])
                        gy = torch.tensor(raw[e_, l_, :, 1])
                        gg[2 * c][e_] = gg[2 * c][e_] + eps * gx
                        gg[2 * c + 1][e_] = gg[2 * c + 1][e_] + eps * gy
                    W = strain_energy_density(*[t.reshape(1, -1, 1) for t in gg], MU, LAM, STATE)
                    Pi = (W.reshape(len(dof), -1) * torch.tensor(wts)).sum()
                    dPi, = torch.autograd.grad(Pi, eps)
                    # relative to the residual scale of the tested set, not to one entry
                    # (an entry can be ~0 by symmetry, leaving only roundoff)
                    worst = max(worst, abs(float(dPi) - R[i, c]) / np.abs(R[geo.free.numpy()]).max())
        else:
            wdet = None
            from physics.vpinn.tensors import quad_tensors
            T = quad_tensors(geo.qm.cell_xy, cfg.K, cfg.K + 2)
            rawx, rawy = T.gx / T.wdet[:, None, :], T.gy / T.wdet[:, None, :]
            for e_ in rng.choice(T.n_elem, 4, replace=False):
                for k in range(T.n_test):
                    for c in range(2):
                        eps = torch.zeros((), requires_grad=True)
                        gg = [t.clone().reshape(T.n_elem, -1) for t in g]
                        gg[2 * c][e_] = gg[2 * c][e_] + eps * torch.tensor(rawx[e_, k])
                        gg[2 * c + 1][e_] = gg[2 * c + 1][e_] + eps * torch.tensor(rawy[e_, k])
                        W = strain_energy_density(*[t.reshape(1, -1, 1) for t in gg], MU, LAM, STATE)
                        Pi = (W.reshape(T.n_elem, -1) * torch.tensor(T.wdet)).sum()
                        dPi, = torch.autograd.grad(Pi, eps)
                        worst = max(worst, abs(float(dPi) - R[e_, k, c]) / np.abs(R[e_]).max())
        res[name] = worst
    return res, all(v < 1e-10 for v in res.values())


def gate_e(fam, p, ell):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 8),
                              torch.nn.Tanh(), torch.nn.Linear(8, 2))
    for q in net.parameters():
        q.data *= 0.3
    res = {}
    for name, cfg in {"C1": WeakConfig("C1", K=2, n_elem=350), "C1n": WeakConfig("C1n", K=2, n_elem=350),
                      "C2": WeakConfig("C2", degree=1, n_elem=350), "C2I": WeakConfig("C2I", n_elem=350)}.items():
        geo = WeakGeometry(fam, p, cfg, ell)
        scale = 1.0 / ell

        def loss_fn():
            if cfg.variant == "C2I":
                u = 0.01 * net(torch.tensor(geo.points()) * scale)
                L, _ = geo.loss(None, S0, u_nodes=u, mu=MU, lam=LAM, state=STATE)
                return L
            x = torch.tensor(geo.points(), requires_grad=True)
            u = 0.01 * net(x * scale)
            gu, = torch.autograd.grad(u[:, 0].sum(), x, create_graph=True)
            gv, = torch.autograd.grad(u[:, 1].sum(), x, create_graph=True)
            P, J = P_of([gu[:, 0], gu[:, 1], gv[:, 0], gv[:, 1]])
            Pb = None
            if cfg.variant == "C1":
                xb = torch.tensor(geo.xb, requires_grad=True)
                ub = 0.01 * net(xb * scale)
                gbu, = torch.autograd.grad(ub[:, 0].sum(), xb, create_graph=True)
                gbv, = torch.autograd.grad(ub[:, 1].sum(), xb, create_graph=True)
                Pb, _ = P_of([gbu[:, 0], gbu[:, 1], gbv[:, 0], gbv[:, 1]])
            L, _ = geo.loss(P, S0, J=J, w_barrier=0.0, Pb=Pb)
            return L

        params = list(net.parameters())
        net.zero_grad()
        L = loss_fn()
        L.backward()
        worst = 0.0
        rng = np.random.default_rng(1)
        gscale = max(float(q.grad.abs().max()) for q in params if q.grad is not None)
        for _ in range(6):
            pi = rng.integers(len(params))
            flat = params[pi].data.view(-1)
            j = rng.integers(flat.numel())
            h = 1e-4
            old = float(flat[j])
            flat[j] = old + h
            Lp = float(loss_fn().detach())
            flat[j] = old - h
            Lm = float(loss_fn().detach())
            flat[j] = old
            fd = (Lp - Lm) / (2 * h)
            gr = params[pi].grad
            ad = 0.0 if gr is None else float(gr.view(-1)[j])   # e.g. the output bias: no effect on grad u
            worst = max(worst, abs(fd - ad) / gscale)
        res[name] = worst
    return res, all(v < 1e-6 for v in res.values())


def gate_f(fam, p, ell):
    """Reported: loss response to cubic element bubbles on the P1 FEM minimiser."""
    if fam == "dogbone":
        return {"skipped": "dog-bone FEM on the training mesh not wired"}, True
    from geometry.families import FAMILIES
    from verification.family_fem import solve_family
    m = base_triangulation(fam, p, 1400)
    from config import LOADING_CONFIG
    sol = solve_family(FAMILIES[fam], p, LOADING_CONFIG["u_max"], mesh=m)
    uh = sol.u
    rng = np.random.default_rng(3)
    amp = rng.normal(size=(m.n_elem, 2))
    out = {}
    for name, cfg in {"C1": WeakConfig("C1", K=2), "C2p1": WeakConfig("C2", degree=1, quad_degree=6),
                      "C2I": WeakConfig("C2I")}.items():
        geo = WeakGeometry(fam, p, cfg, ell, mesh=m)
        rows = []
        for s in (0.0, 1e-4, 1e-3, 1e-2):
            if cfg.variant == "C2I":
                # fine nodes: coarse vertices and edge midpoints -- the cubic bubble is zero there
                un = torch.tensor(np.vstack([uh, 0.5 * (uh[np.unique(np.sort(np.concatenate(
                    [m.tris[:, [0, 1]], m.tris[:, [1, 2]], m.tris[:, [2, 0]]]), 1), axis=0)[:, 0]]
                    + uh[np.unique(np.sort(np.concatenate([m.tris[:, [0, 1]], m.tris[:, [1, 2]],
                                                           m.tris[:, [2, 0]]]), 1), axis=0)[:, 1]])]))
                L, _ = geo.loss(None, S0, u_nodes=un, mu=MU, lam=LAM, state=STATE)
                rows.append((s, float(L), None))
                continue
            x = geo.points()
            # element of each point and its barycentric coordinates
            if cfg.variant == "C1":
                tri_of = np.repeat(geo.qm.parent_tri, geo.nq)
            else:
                tri_of = np.repeat(np.arange(m.n_elem), geo.nq)
            T = np.asarray(m.nodes)[np.asarray(m.tris)[tri_of]]
            A = np.stack([T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]], -1)
            rs = np.linalg.solve(A, (x - T[:, 0])[..., None])[..., 0]
            l1, l2 = rs[:, 0], rs[:, 1]
            l0 = 1 - l1 - l2
            invT = np.linalg.inv(A).transpose(0, 2, 1)
            dl = np.stack([-invT.sum(-1), invT[..., 0], invT[..., 1]], 1)   # grad lambda_a (N,3,2)
            gb = 27 * (l1 * l2)[:, None] * dl[:, 0] + 27 * (l0 * l2)[:, None] * dl[:, 1] \
                + 27 * (l0 * l1)[:, None] * dl[:, 2]
            ge = np.einsum("eai,eaj->eij", uh[m.tris], m.grads)[tri_of]
            ge = ge + s * amp[tri_of][:, :, None] * gb[:, None, :]
            g = [torch.tensor(ge[:, i, j]) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
            P, J = P_of(g)
            L, _ = geo.loss(P, S0, J=J)
            rows.append((s, float(L), None))
        out[name] = rows
    # the energy on the fine rule for the same perturbations
    r_, s_, w_ = triangle_rule(6)
    Tn = np.asarray(m.nodes)[np.asarray(m.tris)]
    ge0 = np.einsum("eai,eaj->eij", uh[m.tris], m.grads)
    erow = []
    for s in (0.0, 1e-4, 1e-3, 1e-2):
        l0 = 1 - r_ - s_
        lam3 = np.stack([l0, r_, s_], 0)
        A = np.stack([Tn[:, 1] - Tn[:, 0], Tn[:, 2] - Tn[:, 0]], -1)
        invT = np.linalg.inv(A).transpose(0, 2, 1)
        dl = np.stack([-invT.sum(-1), invT[..., 0], invT[..., 1]], 1)    # (E,3,2)
        gb = 27 * (lam3[1] * lam3[2])[None, :, None] * dl[:, None, 0] \
            + 27 * (lam3[0] * lam3[2])[None, :, None] * dl[:, None, 1] \
            + 27 * (lam3[0] * lam3[1])[None, :, None] * dl[:, None, 2]      # (E,Nq,2)
        ge = ge0[:, None] + s * amp[:, None, :, None] * gb[:, :, None, :]
        g = [torch.tensor(ge[..., i, j].ravel()) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
        W = strain_energy_density(*[t.reshape(1, -1, 1) for t in g], MU, LAM, STATE).reshape(len(Tn), -1)
        erow.append((s, float((W.numpy() * w_[None] * np.asarray(m.areas)[:, None]).sum())))
    out["energy"] = erow
    return out, True


def gate_g():
    rows = {}
    for fam in FAMS:
        p, (L, H) = params_of(fam)
        r = {}
        for lvl, n in (("B", 1400), ("coarse", 350)):
            for name, cfg in {"C1_K2": WeakConfig("C1", K=2, n_elem=n), "C1_K4": WeakConfig("C1", K=4, n_elem=n),
                              "C2p1": WeakConfig("C2", degree=1, n_elem=n),
                              "C2p2": WeakConfig("C2", degree=2, n_elem=n),
                              "C2I": WeakConfig("C2I", n_elem=n)}.items():
                geo = WeakGeometry(fam, p, cfg, H)
                if cfg.variant == "C1":
                    neq = geo.n_cells * cfg.K ** 2 * 2
                    npts = geo.n_cells * geo.nq
                else:
                    neq = int(geo.free.sum())
                    npts = len(geo.points())
                r[f"{name}|{lvl}"] = {"equations": int(neq), "points": int(npts)}
        rows[fam] = r
    return rows


def main():
    out, allok = {}, True
    geos_cache = {}
    for fam in FAMS:
        p, (L, H) = params_of(fam)
        geos = {name: WeakGeometry(fam, p, cfg, H) for name, cfg in ARMS.items()}
        ra, oa = gate_a(fam, geos)
        rb, ob = gate_b(fam, p, H)
        rc, oc = gate_c(fam, p, H, (L, H))
        rd, od = gate_d(fam, p, H, (L, H))
        re_, oe = gate_e(fam, p, H)
        out[fam] = {"a_patch": ra, "a_pass": oa, "b_fem": rb, "b_pass": ob, "c_mms": rc, "c_pass": oc,
                    "d_energy": rd, "d_pass": od, "e_fd": re_, "e_pass": oe}
        allok &= oa and ob and oc and od and oe
        print(fam, "a", oa, "b", ob, "c", oc, "d", od, "e", oe, flush=True)
        print("   a:", {k: {kk: f"{vv:.1e}" for kk, vv in v.items()} for k, v in ra.items()})
        print("   b:", rb, " c:", {k: [f"{x:.1e}" for x in v] for k, v in rc.items()})
        print("   d:", {k: f"{v:.1e}" for k, v in rd.items()}, " e:", {k: f"{v:.1e}" for k, v in re_.items()}, flush=True)
    for fam in ("open_hole", "double_notch"):
        p, (L, H) = params_of(fam)
        rf, _ = gate_f(fam, p, H)
        out[fam]["f_nullspace"] = rf
        print(fam, "f:", rf, flush=True)
    out["g_counts"] = gate_g()
    out["G2"] = "PASS" if allok else "FAIL"
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("G2:", out["G2"])
    return allok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

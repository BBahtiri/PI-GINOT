#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 9.2 -- is the objective's minimum accurate?  (plan: 0f65e5c, sec. 9.2)

One network fine-tuned on one geometry's energy, to see whether driving the
energy down fixes K_t (H-opt) or not (H-obj).  Removing the constraint that
one network serve 64 geometries makes this an upper bound on what
optimisation can deliver for the operator.

Families: open hole and double notch.  Geometries, by the rule fixed in the
plan: among the 12 in-range geometries, those whose seed-0 |K_t err| after the
9.1 anneal arm (ck800) is >= 2%; the largest and the smallest of them (ties
to the lower index); if fewer than two qualify, the two largest overall.

Arms, both from the seed-0 anneal checkpoint, every trainable parameter
(decoder + parameter MLP; the unused encoder excluded):
  (a) L-BFGS, float64, history 50, strong Wolfe, 1000 iterations as ten
      100-iteration windows; quadrature points fixed within a window (6 per
      triangle of the training triangulation, twice the training count) and
      redrawn at each window with the history reset.  Tracked every 50.
  (b) Adam, float32, training quadrature resampled every step, 50-step warm-up
      3e-5 -> 3e-4 then cosine to 3e-6, 2000 steps.  Tracked every 100.

Tracked: K_t error, peak location, vm L2 (the Tier-1 scorer against the n_r 60
reference) and Pi_net on the 9.0b rule (n_r 60 mesh, 6-point Dunavant).

Guard: nested refinement of the reference mesh (each triangle split in four,
same polygon), FEM solved at h, h/2, h/4.  delta = Pi(h) - Pi(h/2) bounds the
h/2 error if the energy converges at rate >= 1 (checked with the third
level); a run whose Pi_net drops below Pi(h/2) - delta is flagged.

Verdicts: H-opt on a geometry if arm (a) ends with |err| <= |err_0| / 3 and
<= 1%; H-obj if |err| >= 2/3 |err_0| while Pi_net changed < 1e-6 relative over
the last 200 iterations.  Each hypothesis needs 3 of 4 geometries.

    python -m verification.phase9_headroom select
    python -m verification.phase9_headroom run --family open_hole --pick max --arm lbfgs
    python -m verification.phase9_headroom fem
    python -m verification.phase9_headroom report
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pickle
import time

import numpy as np
import torch

from eval.compare_reference import LAM, MU, STATE
from geometry.families import FAMILIES
from training.family_trainer import UD, FamilyEnergyLoss, build_loss, family_model
from training.manifest import set_deterministic, set_precision

ROOT = "verification/results/phase9/p92"
SEL = os.path.join(ROOT, "selection.json")
FEMF = os.path.join(ROOT, "fem_levels.json")
FAMS = ("open_hole", "double_notch")
P91 = "verification/results/phase9/p91"


# ------------------------------------------------------------------ selection
def select(wait=True):
    from verification.family_fem import reference
    from verification.family_score import score_geometry
    from training.family_trainer import load
    out = {}
    for fam_name in FAMS:
        d = os.path.join(P91, f"{fam_name}_s0_anneal")
        while wait and not os.path.exists(os.path.join(d, "result.json")):
            time.sleep(60)
        fam = FAMILIES[fam_name]
        model = load(fam_name, os.path.join(d, "ck800.pt"))
        errs = [score_geometry(model, fam, p, reference(fam, p, f"ir{i}"))["err"]
                for i, p in enumerate(fam.in_range())]
        a = np.abs(errs)
        q = [i for i in range(len(a)) if a[i] >= 0.02]
        if len(q) >= 2:
            big = min(q, key=lambda i: (-a[i], i))
            small = min(q, key=lambda i: (a[i], i))
        else:
            order = sorted(range(len(a)), key=lambda i: (-a[i], i))
            big, small = order[0], order[1]
        out[fam_name] = {"errs": [float(e) for e in errs], "qualifying": q,
                         "max": int(big), "min": int(small)}
        print(f"{fam_name}: |err| {np.round(100 * a, 2).tolist()}  -> max g{big}, min g{small}",
              flush=True)
    os.makedirs(ROOT, exist_ok=True)
    json.dump(out, open(SEL, "w"), indent=1)
    return out


# ------------------------------------------------------------------ FEM levels
def refine(mesh):
    """Nested refinement: every triangle into four; same polygon."""
    tris = mesh.tris.astype(np.int64)
    n, E = mesh.n_node, len(tris)
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    uniq, inv = np.unique(np.sort(edges, 1), axis=0, return_inverse=True)
    inv = inv.ravel()
    mid = n + np.arange(len(uniq))
    nodes = np.vstack([mesh.nodes, 0.5 * (mesh.nodes[uniq[:, 0]] + mesh.nodes[uniq[:, 1]])])
    m01, m12, m20 = mid[inv[:E]], mid[inv[E:2 * E]], mid[inv[2 * E:]]
    a, b, c = tris.T
    new = np.concatenate([np.stack([a, m01, m20], 1), np.stack([m01, b, m12], 1),
                          np.stack([m20, m12, c], 1), np.stack([m01, m12, m20], 1)])
    p = nodes[new]
    tw = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
          - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    assert (tw > 0).all(), "orientation lost in refinement"
    grads = np.empty((len(new), 3, 2))
    for k in range(3):
        k1, k2 = (k + 1) % 3, (k + 2) % 3
        grads[:, k, 0] = (p[:, k1, 1] - p[:, k2, 1]) / tw
        grads[:, k, 1] = (p[:, k2, 0] - p[:, k1, 0]) / tw
    count = np.bincount(inv, minlength=len(uniq))
    bnd = {}
    for s, idx in mesh.boundary.items():
        S = np.zeros(n, bool)
        S[idx] = True
        on = (count == 1) & S[uniq[:, 0]] & S[uniq[:, 1]]
        bnd[s] = np.concatenate([np.asarray(idx, np.int64), mid[on]])
    return dataclasses.replace(mesh, nodes=nodes, tris=new.astype(np.int32),
                               areas=0.5 * tw, grads=grads, boundary=bnd), uniq


def fem_levels(fam_name, i, levels=3):
    from verification.family_fem import reference
    from verification.fem.solver import newton
    from verification.phase9_energy_error import energy_fem
    from verification.fem.solver import FemSolution, assemble
    fam = FAMILIES[fam_name]
    p = fam.in_range()[i]
    sol = reference(fam, p, f"ir{i}")
    rows = [{"level": 0, "n_elem": int(sol.mesh.n_elem), "Pi": energy_fem(sol)}]
    mesh, u = sol.mesh, sol.u
    for lv in range(1, levels):
        t0 = time.time()
        mesh, uniq = refine(mesh)
        assert abs(mesh.areas.sum() - sol.mesh.areas.sum()) < 1e-9 * sol.mesh.areas.sum()
        u0 = np.vstack([u, 0.5 * (u[uniq[:, 0]] + u[uniq[:, 1]])])
        con, vals = fam.fem_dirichlet(mesh, UD)
        u, hist, it = newton(mesh, con, vals, u0, MU, LAM, STATE, n_steps=1, verbose=False)
        _, _, P, J2D = assemble(mesh, u, MU, LAM, STATE, want_tangent=False)
        s2 = FemSolution(mesh=mesh, u=u, u_delta=UD, P=P, J2D=J2D)
        rows.append({"level": lv, "n_elem": int(mesh.n_elem), "Pi": energy_fem(s2),
                     "newton_iter": int(it), "residual": float(hist[-1]),
                     "wall_s": time.time() - t0})
        print(f"  {fam_name} g{i} level {lv}: {mesh.n_elem} elements, Pi {rows[-1]['Pi']:.10f}, "
              f"{it} Newton iterations, {time.time() - t0:.0f} s", flush=True)
    P0, P1, P2 = (r["Pi"] for r in rows[:3])
    rate = math.log2((P0 - P1) / (P1 - P2)) if (P0 - P1) > 0 and (P1 - P2) > 0 else float("nan")
    extrap = P2 - (P1 - P2) / (2 ** rate - 1) if np.isfinite(rate) and rate > 0 else float("nan")
    return {"levels": rows, "delta": P0 - P1, "rate": rate, "Pi_extrap": extrap}


def _wait_sel():
    while not os.path.exists(SEL):
        time.sleep(30)
    return json.load(open(SEL))


def fem_all():
    sel = _wait_sel()
    res = json.load(open(FEMF)) if os.path.exists(FEMF) else {}
    for fam_name in FAMS:
        for pick in ("max", "min"):
            i = sel[fam_name][pick]
            key = f"{fam_name}|{i}"
            if key not in res:
                res[key] = fem_levels(fam_name, i)
                json.dump(res, open(FEMF, "w"), indent=1)
    return res


# ------------------------------------------------------------------ training
class WindowedLoss(FamilyEnergyLoss):
    """Stratified points drawn once per window, then held fixed.

    ``chunk`` (an index array into the window's points) restricts one call to
    a subset, so the closure can accumulate the gradient chunk by chunk: the
    float64 graph of all 8,400 points at once needs ~3.3 GB, and two such
    jobs overflowed the container's 8 GB (first attempt, OOM-killed).  The
    energy is a sum over points and the barrier a mean, so the chunked total
    is the same loss.
    """

    def __init__(self, family, **kw):
        super().__init__(family, **kw)
        self._fixed = None
        self.chunk = None

    def new_window(self):
        self._fixed = None

    def n_points(self, params, device):
        if self._fixed is None:
            self._fixed = super().quad_points(params, device)
        return self._fixed[0].shape[1]

    def quad_points(self, params, device, dtype=None):
        if self._fixed is None:
            self._fixed = super().quad_points(params, device, dtype)
        pts, w, area, mesh = self._fixed
        if self.chunk is None:
            return pts.detach().clone(), w, area, mesh
        c = torch.as_tensor(self.chunk)
        return pts[:, c].detach().clone(), w[c], area, mesh


def _track(model, fam, p, sol, rec):
    from verification.family_score import score_geometry
    from verification.phase9_energy_error import energy_net
    model.eval()
    s = score_geometry(model, fam, p, sol)
    Pn, jmin = energy_net(model, fam, p, sol.mesh)
    rec.update(err=s["err"], vm=s["vm"], u=s["u"], v=s["v"], N_err=s["N_err"],
               x_peak=s["x_peak_pinn"], x_peak_ref=s["x_peak_ref"], Pi_net=Pn, Jmin=jmin)
    model.train()
    return rec


def run(fam_name, pick, arm, threads=1, windows=10, half=50, adam_steps=2000,
        win=10, total=1000, track_every=50):
    from verification.family_fem import reference
    torch.set_num_threads(threads)
    sel = _wait_sel()
    i = sel[fam_name][pick]
    fam = FAMILIES[fam_name]
    p = fam.in_range()[i]
    sol = reference(fam, p, f"ir{i}")
    out_dir = os.path.join(ROOT, f"{fam_name}_g{i}_{arm}")
    os.makedirs(out_dir, exist_ok=True)
    set_precision("float64" if arm.startswith("lbfgs") else "float32")
    set_deterministic(0)
    dt = torch.get_default_dtype()
    model = family_model(fam_name)
    st = torch.load(os.path.join(P91, f"{fam_name}_s0_anneal", "ck800.pt"),
                    map_location="cpu", weights_only=False)["model_state_dict"]
    model.load_state_dict(st, strict=True)
    model = model.to(dt).train()
    params = list(model.decoder.parameters()) + list(model.param_mlp.parameters())
    for q in model.encoder.parameters():
        q.requires_grad_(False)
    L, Hs = fam.extent(p)
    u_d, x_m, y_m = (torch.tensor([v], dtype=dt) for v in (UD, L, Hs))
    bpc = torch.zeros(1, 1, 2, dtype=dt)
    hist = []
    t0 = time.time()
    rec = _track(model, fam, p, sol, {"it": 0, "wall_min": 0.0})
    hist.append(rec)
    print(f"[{fam_name} g{i} {arm}] it 0  err {100 * rec['err']:+.3f}%  Pi_net {rec['Pi_net']:.8f}",
          flush=True)

    if arm.startswith("lbfgs"):
        from config import NONDIM_SCALES, TRAINING_CONFIG
        from training.family_trainer import LAM as LAM_T, MU as MU_T, N_ELEM_Q, STATE as STATE_T
        loss_fn = WindowedLoss(fam, mu=MU_T, lam=LAM_T, stress_state=STATE_T,
                               n_elem_target=N_ELEM_Q, resample=True, n_per_elem=6, seed=2,
                               w_barrier=TRAINING_CONFIG["w_barrier"],
                               j_min=TRAINING_CONFIG["barrier_delta"],
                               E_scale=NONDIM_SCALES["S0"], S0=NONDIM_SCALES["S0"])
        n_eval = [0]

        def closure(n_chunks=4):
            opt.zero_grad()
            N = loss_fn.n_points(p, u_d.device)
            tot = torch.zeros((), dtype=dt)
            for c in np.array_split(np.arange(N), n_chunks):
                loss_fn.chunk = c
                o = loss_fn(model, p, bpc, u_d, x_m, y_m)
                part = o["L_energy"] + loss_fn.w_bar * (len(c) / N) * o["L_barrier"]
                part.backward()
                tot = tot + part.detach()
            loss_fn.chunk = None
            n_eval[0] += 1
            return tot

        it = 0
        if arm == "lbfgs10":
            # Deviation (see docs/phase9_results.md): the registered 100-iteration
            # windows let L-BFGS fit the fixed points -- the energy on the
            # independent rule rose within 50 iterations.  Fresh points every
            # `win` iterations (Jnini et al. resample SSBroyden every 10), history
            # reset with them.
            while it < total:
                loss_fn.new_window()
                opt = torch.optim.LBFGS(params, lr=1.0, max_iter=win, max_eval=3 * win,
                                        history_size=50, line_search_fn="strong_wolfe",
                                        tolerance_grad=0.0, tolerance_change=0.0)
                ls = float(opt.step(closure).detach())
                it += win
                if it % track_every == 0:
                    rec = _track(model, fam, p, sol, {"it": it, "loss": ls,
                                                      "n_eval": int(n_eval[0]),
                                                      "wall_min": (time.time() - t0) / 60})
                    hist.append(rec)
                    print(f"[{fam_name} g{i} {arm}] it {it:4d}  loss {ls:.10e}  "
                          f"err {100 * rec['err']:+.3f}%  Pi_net {rec['Pi_net']:.8f}  "
                          f"evals {n_eval[0]}  {rec['wall_min']:.1f} min", flush=True)
                    json.dump(hist, open(os.path.join(out_dir, "history.json"), "w"), indent=1)
        for w in range(windows if arm == "lbfgs" else 0):
            loss_fn.new_window()
            opt = torch.optim.LBFGS(params, lr=1.0, max_iter=half, max_eval=3 * half,
                                    history_size=50, line_search_fn="strong_wolfe",
                                    tolerance_grad=0.0, tolerance_change=0.0)
            for hh in range(2):
                ls = float(opt.step(closure).detach())
                n_it = opt.state[opt._params[0]]["n_iter"]
                it = 2 * half * w + half * (hh + 1)
                rec = _track(model, fam, p, sol, {"it": it, "loss": ls, "window": w,
                                                  "n_iter_in_window": int(n_it),
                                                  "n_eval": int(n_eval[0]),
                                                  "wall_min": (time.time() - t0) / 60})
                hist.append(rec)
                print(f"[{fam_name} g{i} lbfgs] it {it:4d} (window iters {n_it})  loss {ls:.10e}  "
                      f"err {100 * rec['err']:+.3f}%  Pi_net {rec['Pi_net']:.8f}  "
                      f"evals {n_eval[0]}  {rec['wall_min']:.1f} min", flush=True)
                json.dump(hist, open(os.path.join(out_dir, "history.json"), "w"), indent=1)
    else:
        loss_fn = build_loss(fam_name)
        loss_fn._rng = np.random.default_rng(2)
        opt = torch.optim.Adam(params, lr=3e-5)
        steps, warm = adam_steps, 50
        for k in range(steps):
            lr = (3e-5 + (3e-4 - 3e-5) * (k + 1) / warm if k < warm else
                  3e-6 + 0.5 * (3e-4 - 3e-6) * (1 + math.cos(math.pi * (k - warm) / (steps - warm - 1))))
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad()
            o = loss_fn(model, p, bpc, u_d, x_m, y_m)
            o["loss"].backward()
            opt.step()
            if (k + 1) % 100 == 0:
                rec = _track(model, fam, p, sol, {"it": k + 1, "loss": float(o["loss"].detach()),
                                                  "lr": lr, "wall_min": (time.time() - t0) / 60})
                hist.append(rec)
                print(f"[{fam_name} g{i} adam] step {k + 1:4d}  err {100 * rec['err']:+.3f}%  "
                      f"Pi_net {rec['Pi_net']:.8f}  {rec['wall_min']:.1f} min", flush=True)
                json.dump(hist, open(os.path.join(out_dir, "history.json"), "w"), indent=1)
    torch.save({"model_state_dict": model.state_dict()}, os.path.join(out_dir, "final.pt"))
    json.dump({"family": fam_name, "geometry": i, "pick": pick, "arm": arm,
               "wall_min": (time.time() - t0) / 60}, open(os.path.join(out_dir, "result.json"), "w"))


# ------------------------------------------------------------------ report
def report():
    sel = json.load(open(SEL))
    fem = json.load(open(FEMF)) if os.path.exists(FEMF) else {}
    print("Phase 9.2 -- one geometry, driven down its own energy")
    print("=" * 110)
    out, hopt, hobj = {}, 0, 0
    for fam_name in FAMS:
        for pick in ("max", "min"):
            i = sel[fam_name][pick]
            key = f"{fam_name}|{i}"
            f = fem.get(key)
            row = {"family": fam_name, "geometry": i, "pick": pick}
            if f:
                P1 = f["levels"][1]["Pi"]
                row.update(Pi_fem=[lv["Pi"] for lv in f["levels"]], delta=f["delta"],
                           rate=f["rate"], Pi_extrap=f["Pi_extrap"], floor=P1 - f["delta"])
            for arm in ("lbfgs", "lbfgs10", "adam"):
                hp = os.path.join(ROOT, f"{fam_name}_g{i}_{arm}", "history.json")
                if not os.path.exists(hp):
                    continue
                h = json.load(open(hp))
                e0, e1 = h[0]["err"], h[-1]["err"]
                pis = [r["Pi_net"] for r in h]
                d = {"err0": e0, "err_final": e1, "Pi0": pis[0], "Pi_final": pis[-1],
                     "iters": h[-1]["it"], "wall_min": h[-1]["wall_min"],
                     "err_traj": [r["err"] for r in h], "Pi_traj": pis,
                     "it_traj": [r["it"] for r in h]}
                if f:
                    ref = f["Pi_extrap"] if np.isfinite(f["Pi_extrap"]) else f["levels"][2]["Pi"]
                    d["gap0_rel"] = pis[0] / ref - 1
                    d["gap_final_rel"] = pis[-1] / ref - 1
                    d["guard_flag"] = bool(min(pis) < row["floor"])
                if arm == "lbfgs10" and len(h) >= 5:
                    settle = abs(h[-1]["Pi_net"] - h[-5]["Pi_net"]) / abs(h[-1]["Pi_net"])
                    d["Pi_change_last200"] = settle
                    d["H_opt"] = bool(abs(e1) <= abs(e0) / 3 and abs(e1) <= 0.01)
                    d["H_obj"] = bool(abs(e1) >= 2 / 3 * abs(e0) and settle < 1e-6)
                    hopt += d["H_opt"]; hobj += d["H_obj"]
                row[arm] = d
                print(f"{fam_name:<13} g{i:<3}{pick:<4}{arm:<6} err {100 * e0:+7.3f}% -> {100 * e1:+7.3f}%   "
                      f"Pi {pis[0]:.8f} -> {pis[-1]:.8f}"
                      + (f"   gap to FEM(extrap) {d['gap0_rel']:+.2e} -> {d['gap_final_rel']:+.2e}"
                         f"   guard {'FLAG' if d['guard_flag'] else 'ok'}" if f else "")
                      + (f"   H-opt {d['H_opt']}  H-obj {d['H_obj']}" if "H_opt" in d else ""))
            if f:
                print(f"{'':<21}FEM Pi h, h/2, h/4: {', '.join(f'{x:.8f}' for x in row['Pi_fem'])}  "
                      f"rate {row['rate']:.2f}  extrapolated {row['Pi_extrap']:.8f}")
            out[key] = row
    print("-" * 110)
    n = sum(1 for r in out.values() if "lbfgs10" in r and "H_opt" in r["lbfgs10"])
    verdict = ("H-opt" if hopt >= 3 else "H-obj" if hobj >= 3 else "in between") if n == 4 else "pending"
    print(f"H-opt on {hopt}/{n}, H-obj on {hobj}/{n}  ->  9.2 verdict: {verdict}")
    json.dump({"rows": out, "H_opt": hopt, "H_obj": hobj, "verdict": verdict},
              open(os.path.join(ROOT, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "run", "fem", "report"])
    ap.add_argument("--family")
    ap.add_argument("--pick", choices=["max", "min"])
    ap.add_argument("--arm", choices=["lbfgs", "lbfgs10", "adam"])
    a = ap.parse_args()
    if a.cmd == "select":
        select()
    elif a.cmd == "run":
        run(a.family, a.pick, a.arm)
    elif a.cmd == "fem":
        fem_all()
    else:
        report()

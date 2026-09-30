#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 10 -- one trainer for the three formulations (plan S5-S7).

The operator, data and optimiser are those of ``training/family_trainer.py``
(oracle-conditioned model, bank of 64, batch 4, batch order from
``default_rng(42)``, Adam, grad-norm clip 1.0), with two changes the plan fixes
(rule 4): no plateau scheduler, and an explicit schedule

    linear warm-up lr/10 -> lr over ``warmup`` steps, constant to ``t_const``,
    then cosine to ``lr_end`` at ``steps``

(S7: warmup 0, t_const 1600, steps 2400; S6/G4b: warmup 50, t_const 50).

Arms (``ARM_DEFAULTS``):
  B    the energy form -- ``build_loss(fam)`` unchanged (Phase 8/9).
  A    the strong form (physics/strong_form.py), backpropagated in chunks of
       interior points; the loss is a sum of per-point terms, so chunking is
       exact.
  C1, C1n, C2, C2I   the weak forms (physics/vpinn/weak.py).  The loss is a
       function of residuals that sum over points, so it does not separate:
       pass 1 computes the displacement gradients G at every point without a
       parameter graph, the loss and dL/dG; pass 2 recomputes G chunk by chunk
       with the graph and backpropagates dL/dG.  The geometry latent is a leaf
       in pass 2 and its gradient is sent through the encoder once.
Every arm carries the J-barrier (1e3 at J = 0.05) at its volume points.

Logs: batch indices, loss parts, min J, the clipped fraction, peak RSS, and the
monitors of ``verification/phase10/monitors.py`` on the monitor geometries.

    python -m training.formulation_trainer --arm C2 --family open_hole --seed 0
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import resource
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from config import LOADING_CONFIG, MATERIAL_CONFIG, NONDIM_SCALES, TRAINING_CONFIG
from physics.neo_hookean import first_piola_kirchhoff_stress
from physics.strong_form import StrongConfig, StrongGeometry, div_P
from physics.vpinn.weak import WeakConfig, WeakGeometry
from training.family_trainer import _sets, build_loss, family_model
from training.manifest import set_deterministic, set_precision

MU, LAM, STATE = MATERIAL_CONFIG["mu"], MATERIAL_CONFIG["lam"], MATERIAL_CONFIG["state"]
UD, S0 = LOADING_CONFIG["u_max"], NONDIM_SCALES["S0"]
W_BAR, J_MIN = TRAINING_CONFIG["w_barrier"], TRAINING_CONFIG["barrier_delta"]

ARM_DEFAULTS = {
    "B": {},
    "A": {"strong": StrongConfig()},
    "C1": {"weak": WeakConfig("C1", K=2, n_elem=350, beta=20.0)},
    "C1n": {"weak": WeakConfig("C1n", K=2, n_elem=350)},
    "C2": {"weak": WeakConfig("C2", degree=1, quad_degree=4)},
    "C2I": {"weak": WeakConfig("C2I")},
}


# --------------------------------------------------------------------------
# model access
# --------------------------------------------------------------------------
class Ctx:
    """One geometry's conditioning: latent and scalars."""

    def __init__(self, model, fam_name, p, extent, dt):
        model.set_geometry(p)
        L, Hs = extent(p)
        self.dt = dt
        self.x_m = torch.tensor([L], dtype=dt)
        self.y_m = torch.tensor([Hs], dtype=dt)
        self.u_d = torch.tensor([UD], dtype=dt)
        self.z = model.encode(torch.zeros(1, 1, 2, dtype=dt), self.x_m, self.y_m)


def disp(model, z, ctx, pts):
    q = torch.as_tensor(pts, dtype=ctx.dt)
    return model.decode(q.unsqueeze(0), z, ctx.u_d, ctx.x_m, ctx.y_m)[0]


def disp_grads(model, z, ctx, pts, create_graph):
    """(q leaf (N, 2), u (N, 2), G (N, 4) = du/dx, du/dy, dv/dx, dv/dy)."""
    q = torch.as_tensor(pts, dtype=ctx.dt).clone().requires_grad_(True)
    uv = model.decode(q.unsqueeze(0), z, ctx.u_d, ctx.x_m, ctx.y_m)[0]
    gu, = torch.autograd.grad(uv[:, 0].sum(), q, create_graph=create_graph, retain_graph=True)
    gv, = torch.autograd.grad(uv[:, 1].sum(), q, create_graph=create_graph,
                              retain_graph=create_graph)
    G = torch.stack([gu[:, 0], gu[:, 1], gv[:, 0], gv[:, 1]], -1)
    if not create_graph:
        return q, uv.detach(), G.detach()
    return q, uv, G


def P_and_J(G):
    o = first_piola_kirchhoff_stress(*[G[:, k].reshape(1, -1, 1) for k in range(4)],
                                     MU, LAM, STATE, return_J3D=True)
    return [t.reshape(-1) for t in o[:4]], torch.minimum(o[4], o[5]).reshape(-1)


def _key(p):
    return json.dumps(p, sort_keys=True, default=float)


def _chunks(n, size):
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------
class ArmB:
    name = "B"

    def __init__(self, fam_name, extent, **kw):
        self.loss_fn = build_loss(fam_name)
        self.extent = extent

    def step(self, model, p, scale, dt):
        L, Hs = self.extent(p)
        o = self.loss_fn(model, p, torch.zeros(1, 1, 2, dtype=dt), torch.tensor([UD], dtype=dt),
                         torch.tensor([L], dtype=dt), torch.tensor([Hs], dtype=dt))
        (o["loss"] * scale).backward()
        return {"loss": float(o["loss"].detach()), "L_energy": o["L_energy_log"],
                "L_barrier": o["L_barrier_log"], "Jmin": min(o["min_J2D_log"], o["min_J3D_log"])}


class ArmA:
    name = "A"

    def __init__(self, fam_name, extent, strong: StrongConfig = None, chunk=600, seed=0, **kw):
        self.fam_name, self.extent = fam_name, extent
        self.cfg = strong or StrongConfig()
        self.chunk = chunk
        self.rng = np.random.default_rng(seed)
        self.geos: Dict[tuple, StrongGeometry] = {}
        self.mesh_of = build_loss(fam_name)._mesh

    def geo(self, p):
        key = _key(p)
        g = self.geos.get(key)
        if g is None:
            g = self.geos[key] = StrongGeometry(self.fam_name, p, self.cfg, mesh=self.mesh_of(p))
            g.rng = self.rng                                     # one stream for all geometries
        return g

    def terms(self, model, ctx, z, x, xb, nb, mb, cb, scale, backward=True):
        """The strong loss, chunked; identical to strong_form.strong_loss."""
        cfg = self.cfg
        full = cb <= 1
        n_int, n_J = len(x), len(x) + int(full.sum())
        s2 = (cfg.L0 / cfg.S0) ** 2
        logs = {"L_eq": 0.0, "L_barrier": 0.0, "L_arc": 0.0, "L_top": 0.0, "L_part": 0.0}
        jmin = np.inf
        for sl in _chunks(n_int, self.chunk):
            q, _, G = disp_grads(model, z, ctx, x[sl], create_graph=True)
            P, J = P_and_J(G)
            fx, fy = div_P(P, q)
            eq = s2 * ((fx ** 2 + fy ** 2).sum()) / n_int
            bar = (torch.relu(cfg.j_min - J) ** 2).sum() / n_J
            Lc = cfg.w_eq * eq + cfg.w_bar * bar
            if backward:
                (Lc * scale).backward()
            logs["L_eq"] += float(eq.detach())
            logs["L_barrier"] += float(bar.detach())
            jmin = min(jmin, float(J.detach().min()))
        _, _, Gb = disp_grads(model, z, ctx, xb, create_graph=True)
        Pb, Jb = P_and_J(Gb)
        n = torch.as_tensor(nb, dtype=ctx.dt)
        tx = Pb[0] * n[:, 0] + Pb[1] * n[:, 1]
        ty = Pb[2] * n[:, 0] + Pb[3] * n[:, 1]
        m = torch.as_tensor(mb, dtype=ctx.dt)
        res = (m[:, 0] * tx ** 2 + m[:, 1] * ty ** 2) / cfg.S0 ** 2
        Lb = 0.0
        for key, c, w in (("L_arc", 0, cfg.w_arc), ("L_top", 1, cfg.w_top), ("L_part", 2, cfg.w_part)):
            sel = torch.as_tensor(cb == c)
            if bool(sel.any()):
                t = res[sel].mean()
                Lb = Lb + w * t
                logs[key] = float(t.detach())
        fb = torch.as_tensor(full)
        if bool(fb.any()):
            bar_b = (torch.relu(cfg.j_min - Jb[fb]) ** 2).sum() / n_J
            Lb = Lb + cfg.w_bar * bar_b
            logs["L_barrier"] += float(bar_b.detach())
            jmin = min(jmin, float(Jb[fb].detach().min()))
        if backward:
            (Lb * scale).backward()
        logs["loss"] = (cfg.w_eq * logs["L_eq"] + cfg.w_arc * logs["L_arc"] + cfg.w_top * logs["L_top"]
                        + cfg.w_part * logs["L_part"] + cfg.w_bar * logs["L_barrier"])
        logs["Jmin"] = jmin
        return logs

    def step(self, model, p, scale, dt):
        ctx = Ctx(model, self.fam_name, p, self.extent, dt)
        x, xb, nb, mb, cb = self.geo(p).draw()
        zl = ctx.z.detach().requires_grad_(True)
        logs = self.terms(model, ctx, zl, x, xb, nb, mb, cb, scale)
        ctx.z.backward(zl.grad)
        return logs


class ArmC:
    def __init__(self, fam_name, extent, weak: WeakConfig, chunk=3000, level="B", **kw):
        """``level`` 'h2' puts C2/C2I on the nested refinement of B's triangulation."""
        self.name = weak.variant
        self.fam_name, self.extent = fam_name, extent
        self.cfg = weak
        self.level = level
        self.chunk = chunk
        self.geos: Dict[tuple, WeakGeometry] = {}
        self.mesh_of = build_loss(fam_name)._mesh

    def geo(self, p):
        """The training geometry for p (cached)."""
        key = _key(p)
        g = self.geos.get(key)
        if g is None:
            g = self.geos[key] = self.build(p, self.cfg)
        return g

    def build(self, p, cfg, mesh=None):
        if mesh is None and cfg.n_elem == 1400:
            mesh = self.mesh_of(p)                               # B's triangulation, exactly
            if self.level == "h2":
                from physics.vpinn.weak import refine_nested
                mesh = refine_nested(mesh)
        _, Hs = self.extent(p)
        return WeakGeometry(self.fam_name, p, cfg, Hs, mesh=mesh)

    def _points(self, g):
        pts = g.points()
        if g.cfg.variant == "C1":
            pts = np.vstack([pts, g.xb])
        return pts

    def loss_of_G(self, g, G, u_nodes=None, w_bar=W_BAR):
        if g.cfg.variant == "C2I":
            return g.loss(None, S0, u_nodes=u_nodes, mu=MU, lam=LAM, state=STATE,
                          w_barrier=w_bar, j_min=J_MIN)
        n = len(g.points())
        P, J = P_and_J(G[:n])
        Pb = P_and_J(G[n:])[0] if g.cfg.variant == "C1" else None
        return g.loss(P, S0, J=J, w_barrier=w_bar, j_min=J_MIN, Pb=Pb)

    def evaluate(self, model, ctx, g, w_bar=W_BAR):
        """Loss value without a parameter graph (monitors)."""
        if g.cfg.variant == "C2I":
            with torch.no_grad():
                u = torch.cat([disp(model, ctx.z, ctx, g.points()[sl])
                               for sl in _chunks(len(g.points()), self.chunk)])
            L, parts = self.loss_of_G(g, None, u_nodes=u, w_bar=w_bar)
        else:
            pts = self._points(g)
            G = torch.cat([disp_grads(model, ctx.z, ctx, pts[sl], create_graph=False)[2]
                           for sl in _chunks(len(pts), self.chunk)])
            with torch.no_grad():
                L, parts = self.loss_of_G(g, G, w_bar=w_bar)
        return float(L), {k: float(v) for k, v in parts.items()}

    def step(self, model, p, scale, dt):
        g = self.geo(p)
        ctx = Ctx(model, self.fam_name, p, self.extent, dt)
        zl = ctx.z.detach().requires_grad_(True)
        if g.cfg.variant == "C2I":
            # forward only (no derivatives): a single pass fits in memory
            u = disp(model, zl, ctx, g.points())
            L, parts = self.loss_of_G(g, None, u_nodes=u)
            (L * scale).backward()
            jmin = None
        else:
            pts = self._points(g)
            G = torch.cat([disp_grads(model, zl.detach(), ctx, pts[sl], create_graph=False)[2]
                           for sl in _chunks(len(pts), self.chunk)])
            Gl = G.clone().requires_grad_(True)
            L, parts = self.loss_of_G(g, Gl)
            dG, = torch.autograd.grad(L, Gl)
            for sl in _chunks(len(pts), self.chunk):
                _, _, Gc = disp_grads(model, zl, ctx, pts[sl], create_graph=True)
                torch.autograd.backward(Gc, dG[sl] * scale)
            n = len(g.points())
            jmin = float(P_and_J(G[:n])[1].min())
        ctx.z.backward(zl.grad)
        out = {"loss": float(L.detach())}
        out.update({k: float(v.detach()) for k, v in parts.items()})
        if jmin is not None:
            out["Jmin"] = jmin
        return out


def make_arm(arm, fam_name, extent, overrides=None, seed=0):
    kw = dict(ARM_DEFAULTS[arm])
    kw.update(overrides or {})
    if arm == "B":
        return ArmB(fam_name, extent, **kw)
    if arm == "A":
        return ArmA(fam_name, extent, seed=seed, **kw)
    return ArmC(fam_name, extent, **kw)


# --------------------------------------------------------------------------
# schedule and loop
# --------------------------------------------------------------------------
def lr_at(k, lr, steps, warmup=0, t_const=None, lr_end=None):
    lr_end = lr / 100 if lr_end is None else lr_end
    t_const = steps if t_const is None else t_const
    if k < warmup:
        return lr / 10 + (lr - lr / 10) * k / warmup
    if k < t_const:
        return lr
    frac = (k - t_const) / max(1, steps - t_const)
    return lr_end + 0.5 * (lr - lr_end) * (1 + math.cos(math.pi * frac))


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def train(arm, fam_name, seed, out_dir, steps=2400, batch=4, lr=3e-4, lr_end=None,
          warmup=0, t_const=1600, clip=1.0, geoms=None, overrides=None, threads=1,
          monitor=None, monitor_every=400, ckpt_every=400, log_every=25):
    """``geoms``: the training geometries (default: the family bank of 64).
    ``monitor``: callable(model, step) -> dict, run at step 0, every
    ``monitor_every`` steps and at the end."""
    torch.set_num_threads(threads)
    set_precision("float32")
    set_deterministic(seed)
    dt = torch.get_default_dtype()
    model = family_model(fam_name)
    bank, _, extent = _sets(fam_name)
    geoms = bank if geoms is None else geoms
    a = make_arm(arm, fam_name, extent, overrides, seed=seed)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.default_rng(42)
    os.makedirs(out_dir, exist_ok=True)
    order, hist, mon, batches = [], [], [], []
    n_clip = 0
    t0 = time.time()
    if monitor is not None:
        mon.append(dict(step=0, **monitor(model, 0)))
    for step in range(steps):
        cur = lr_at(step, lr, steps, warmup, t_const, lr_end)
        for gp in opt.param_groups:
            gp["lr"] = cur
        model.train()
        opt.zero_grad()
        if len(order) < batch:
            order = list(range(len(geoms)))
            rng.shuffle(order)
        idx = [order.pop(0) for _ in range(min(batch, len(geoms)))]
        batches.append(idx)
        logs = [a.step(model, geoms[i], 1.0 / len(idx), dt) for i in idx]
        gn = float(torch.nn.utils.clip_grad_norm_(params, clip if clip else float("inf")))
        if not math.isfinite(gn):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        n_clip += int(clip is not None and clip > 0 and gn > clip)
        opt.step()
        if (step + 1) % log_every == 0 or step == 0:
            rec = {"step": step + 1, "lr": cur, "grad_norm": gn,
                   "wall_min": (time.time() - t0) / 60.0, "rss_gb": rss_gb()}
            for k in logs[0]:
                vals = [l[k] for l in logs if l.get(k) is not None]
                if vals:
                    rec[k] = float(np.min(vals)) if k == "Jmin" else float(np.mean(vals))
            hist.append(rec)
            print(f"  [{arm} {fam_name} s{seed}] {step + 1:5d} loss {rec['loss']:.5e} "
                  f"J {rec.get('Jmin', float('nan')):.3f} lr {cur:.1e} |g| {gn:.2e} "
                  f"{rec['wall_min']:.1f} min {rec['rss_gb']:.2f} GB", flush=True)
        if monitor is not None and ((step + 1) % monitor_every == 0 or step + 1 == steps):
            mon.append(dict(step=step + 1, **monitor(model, step + 1)))
            print(f"  [{arm} {fam_name} s{seed}] monitor {mon[-1]}", flush=True)
        if (step + 1) % ckpt_every == 0 or step + 1 == steps:
            torch.save({"model_state_dict": model.state_dict(), "step": step + 1, "arm": arm,
                        "family": fam_name, "seed": seed}, os.path.join(out_dir, f"ck{step + 1}.pt"))
            json.dump({"hist": hist, "monitor": mon, "batches": batches},
                      open(os.path.join(out_dir, "history.json"), "w"))
    wall = time.time() - t0
    res = {"arm": arm, "family": fam_name, "seed": seed, "steps": steps, "wall_min": wall / 60,
           "s_per_step": wall / steps, "clipped_frac": n_clip / steps, "peak_rss_gb": rss_gb(),
           "final": hist[-1], "monitor": mon, "overrides": {k: str(v) for k, v in (overrides or {}).items()}}
    json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
    torch.save({"model_state_dict": model.state_dict(), "step": steps, "arm": arm,
                "family": fam_name, "seed": seed}, os.path.join(out_dir, "last.pt"))
    return model, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARM_DEFAULTS))
    ap.add_argument("--family", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"verification/results/phase10/s7/{a.arm}_{a.family}_s{a.seed}"
    train(a.arm, a.family, a.seed, out, steps=a.steps)

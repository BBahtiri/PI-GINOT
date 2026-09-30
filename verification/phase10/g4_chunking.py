#!/usr/bin/env python3
"""
Phase 10, S5 -- the trainer's chunked backward is the plain backward.

For the operator (float64), on the open hole and the dog-bone:
  A    ArmA.terms (interior chunks of 100 points, then the boundary) against
       physics.strong_form.strong_loss in one pass, same points;
  C*   ArmC.step (pass 1 without a graph, pass 2 in chunks of 500 points, the
       latent as a leaf) against one autograd pass of WeakGeometry.loss;
  B    ArmB.step against build_loss(fam) called directly (same RNG state).
Pass: loss equal to 1e-12 relative and the full parameter gradient equal to
1e-10 relative (max-norm).

    python -m verification.phase10.g4_chunking
"""
import copy
import json
import sys

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from physics.strong_form import StrongConfig, StrongGeometry, strong_loss   # noqa: E402
from physics.vpinn.weak import WeakConfig                                   # noqa: E402
from training import formulation_trainer as FT                              # noqa: E402
from training.family_trainer import _sets, build_loss, family_model        # noqa: E402

OUT = "verification/results/phase10/g4_chunking.json"


def flat_grad(model):
    return torch.cat([(q.grad if q.grad is not None else torch.zeros_like(q)).reshape(-1)
                      for q in model.parameters()])


def compare(g_ref, g_new, L_ref, L_new):
    return {"loss_rel": abs(L_new - L_ref) / abs(L_ref),
            "grad_rel": float((g_new - g_ref).abs().max() / g_ref.abs().max())}


def case_A(fam, model, p, extent):
    dt = torch.float64
    cfg = StrongConfig()
    arm = FT.ArmA(fam, extent, strong=cfg, chunk=100)
    geo = StrongGeometry(fam, p, cfg, mesh=arm.mesh_of(p), seed=3)
    x, xb, nb, mb, cb = geo.draw()
    rs = np.random.default_rng(0)
    x = x[rs.choice(len(x), 300, replace=False)]
    sel = np.concatenate([rs.choice(np.where(cb == c)[0], min(30, (cb == c).sum()), replace=False)
                          for c in np.unique(cb)])
    xb, nb, mb, cb = xb[sel], nb[sel], mb[sel], cb[sel]
    model.zero_grad()
    ctx = FT.Ctx(model, fam, p, extent, dt)
    Lr, _ = strong_loss(lambda q: _pf(model, ctx, q), x, xb, nb, mb, cb, cfg)
    Lr.backward()
    g_ref = flat_grad(model).clone()
    model.zero_grad()
    ctx = FT.Ctx(model, fam, p, extent, dt)
    zl = ctx.z.detach().requires_grad_(True)
    logs = arm.terms(model, ctx, zl, x, xb, nb, mb, cb, 1.0)
    ctx.z.backward(zl.grad)
    return compare(g_ref, flat_grad(model), float(Lr), logs["loss"])


def _pf(model, ctx, q):
    """P_fn for strong_loss: q is the (N, 2) leaf strong_loss differentiates."""
    uv = model.decode(q.unsqueeze(0), ctx.z, ctx.u_d, ctx.x_m, ctx.y_m)[0]
    gu, = torch.autograd.grad(uv[:, 0].sum(), q, create_graph=True, retain_graph=True)
    gv, = torch.autograd.grad(uv[:, 1].sum(), q, create_graph=True, retain_graph=True)
    return FT.P_and_J(torch.stack([gu[:, 0], gu[:, 1], gv[:, 0], gv[:, 1]], -1))


def case_C(fam, model, p, extent, cfg):
    dt = torch.float64
    arm = FT.ArmC(fam, extent, weak=cfg, chunk=500)
    g = arm.geo(p)
    model.zero_grad()
    ctx = FT.Ctx(model, fam, p, extent, dt)
    if cfg.variant == "C2I":
        u = FT.disp(model, ctx.z, ctx, g.points())
        Lr, _ = arm.loss_of_G(g, None, u_nodes=u)
    else:
        pts = arm._points(g)
        q = torch.tensor(pts, requires_grad=True)
        uv = model.decode(q.unsqueeze(0), ctx.z, ctx.u_d, ctx.x_m, ctx.y_m)[0]
        gu, = torch.autograd.grad(uv[:, 0].sum(), q, create_graph=True, retain_graph=True)
        gv, = torch.autograd.grad(uv[:, 1].sum(), q, create_graph=True, retain_graph=True)
        Lr, _ = arm.loss_of_G(g, torch.stack([gu[:, 0], gu[:, 1], gv[:, 0], gv[:, 1]], -1))
    Lr.backward()
    g_ref = flat_grad(model).clone()
    model.zero_grad()
    logs = arm.step(model, p, 1.0, dt)
    return compare(g_ref, flat_grad(model), float(Lr), logs["loss"])


def case_B(fam, model, p, extent):
    dt = torch.float64
    arm = FT.ArmB(fam, extent)
    ref = build_loss(fam)
    rng_state = copy.deepcopy(arm.loss_fn._rng) if hasattr(arm.loss_fn, "_rng") else None
    if rng_state is not None:
        ref._rng = copy.deepcopy(rng_state)
    L, Hs = extent(p)
    model.zero_grad()
    o = ref(model, p, torch.zeros(1, 1, 2), torch.tensor([FT.UD]), torch.tensor([L]), torch.tensor([Hs]))
    o["loss"].backward()
    g_ref = flat_grad(model).clone()
    model.zero_grad()
    logs = arm.step(model, p, 1.0, dt)
    return compare(g_ref, flat_grad(model), float(o["loss"]), logs["loss"])


def main():
    torch.set_num_threads(2)
    out, ok = {}, True
    small = {"C1": WeakConfig("C1", K=2, n_elem=120), "C1n": WeakConfig("C1n", K=2, n_elem=120),
             "C2": WeakConfig("C2", degree=1, quad_degree=4, n_elem=200),
             "C2p2": WeakConfig("C2", degree=2, quad_degree=6, n_elem=120),
             "C2I": WeakConfig("C2I", n_elem=200)}
    for fam in ("open_hole", "dogbone"):
        torch.manual_seed(0)
        model = family_model(fam).double()
        bank, _, extent = _sets(fam)
        p = bank[0]
        r = {"A": case_A(fam, model, p, extent), "B": case_B(fam, model, p, extent)}
        for name, cfg in small.items():
            r[name] = case_C(fam, model, p, extent, cfg)
        for k, v in r.items():
            print(fam, k, {kk: f"{vv:.2e}" for kk, vv in v.items()}, flush=True)
            ok &= v["loss_rel"] < 1e-12 and v["grad_rel"] < 1e-10
        out[fam] = r
    out["pass"] = bool(ok)
    json.dump(out, open(OUT, "w"), indent=1)
    print("chunking:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

#!/usr/bin/env python3
"""
Phase 10, gate G3 -- arm A, the strong form.

  (a) manufactured solution (the sympy field of verification/mms.py, float64):
      on every family's first bank geometry, Div P computed by the arm's own
      autograd chain equals the independent sympy Div P* -- i.e.
      Div P(u*) + b = 0 with b = -Div P* -- and the traction residual
      P(u*) N equals the sympy P* N on every boundary point (1e-10 relative).
  (b) continuity with the preprint: on the dog-bone, the generalised loss with
      the preprint's terms reproduces ``physics.losses.PhysicsLoss`` for the same
      model and points (1e-10 relative).  The resultant anchor is not in either
      (it lives in the trainer; A-pre exercises it).

    python -m verification.phase10.g3_strong
"""
import json
import sys

import numpy as np
import sympy as sp
import torch

torch.set_default_dtype(torch.float64)

from config import MATERIAL_CONFIG, NONDIM_SCALES, TRAINING_CONFIG      # noqa: E402
from physics.neo_hookean import first_piola_kirchhoff_stress           # noqa: E402
from physics.strong_form import StrongConfig, StrongGeometry, div_P, strong_loss  # noqa: E402
from verification.phase10.g2_weak import FAMS, params_of               # noqa: E402

MU, LAM, STATE = MATERIAL_CONFIG["mu"], MATERIAL_CONFIG["lam"], MATERIAL_CONFIG["state"]
OUT = "verification/results/phase10/g3.json"


def mms_torch_P(Lx, Ly, amp=(0.35, 0.25)):
    a, b = amp

    def P_fn(x):
        X, Y = x[:, 0], x[:, 1]
        u = a * torch.sin(np.pi * X / Lx) * (Y / Ly)
        v = b * (X / Lx) * (Y / Ly) ** 2
        gu, = torch.autograd.grad(u.sum(), x, create_graph=True)
        gv, = torch.autograd.grad(v.sum(), x, create_graph=True)
        out = first_piola_kirchhoff_stress(*[t.reshape(1, -1, 1) for t in
                                             (gu[:, 0], gu[:, 1], gv[:, 0], gv[:, 1])],
                                           MU, LAM, STATE, return_J3D=True)
        P = [p.reshape(-1) for p in out[:4]]
        return P, torch.minimum(out[4], out[5]).reshape(-1)
    return P_fn


def gate_a():
    from verification.mms import _lambdify, _symbolic_fields
    res = {}
    for fam in FAMS:
        p, (L, H) = params_of(fam)
        geo = StrongGeometry(fam, p, StrongConfig())
        x, xb, nb, mb, cb = geo.draw()
        s = _symbolic_fields(STATE, L, H)
        Xs, Ys = s["X"], s["Y"]
        div = [np.asarray(_lambdify(e, Xs, Ys)(x[:, 0], x[:, 1]), float) for e in s["div"]]
        Pb_s = [np.broadcast_to(np.asarray(_lambdify(e, Xs, Ys)(xb[:, 0], xb[:, 1]), float), (len(xb),))
                for e in s["P"]]
        P_fn = mms_torch_P(L, H)
        xt = torch.tensor(x, requires_grad=True)
        P, _ = P_fn(xt)
        fx, fy = div_P(P, xt)
        dsc = max(np.abs(div[0]).max(), np.abs(div[1]).max())
        e_div = max(np.abs(fx.detach().numpy() - div[0]).max(), np.abs(fy.detach().numpy() - div[1]).max()) / dsc
        xbt = torch.tensor(xb, requires_grad=True)
        Pb, _ = P_fn(xbt)
        tx = (Pb[0] * torch.tensor(nb[:, 0]) + Pb[1] * torch.tensor(nb[:, 1])).detach().numpy()
        ty = (Pb[2] * torch.tensor(nb[:, 0]) + Pb[3] * torch.tensor(nb[:, 1])).detach().numpy()
        txs = Pb_s[0] * nb[:, 0] + Pb_s[1] * nb[:, 1]
        tys = Pb_s[2] * nb[:, 0] + Pb_s[3] * nb[:, 1]
        tsc = max(np.abs(txs).max(), np.abs(tys).max())
        e_t = max(np.abs(tx - txs).max(), np.abs(ty - tys).max()) / tsc
        res[fam] = {"div_rel": float(e_div), "traction_rel": float(e_t), "n_interior": len(x),
                    "n_boundary": len(xb), "n_junctions": int(len(geo.junctions)),
                    "excluded_frac": float(1 - len(x) / (geo.mesh.n_elem * geo.cfg.n_per_elem)),
                    "categories": {int(k): int((cb == k).sum()) for k in np.unique(cb)}}
    ok = all(r["div_rel"] < 1e-10 and r["traction_rel"] < 1e-10 for r in res.values())
    return res, ok


def gate_b():
    from training.family_trainer import family_model, _sets
    from physics.losses import PhysicsLoss
    from config import get_fillet_geometry
    torch.manual_seed(0)
    model = family_model("dogbone").double()
    bank, _, extent = _sets("dogbone")
    p = bank[0]
    Lh, Hg = extent(p)
    cfg = StrongConfig()
    geo = StrongGeometry("dogbone", p, cfg, seed=1)
    x, xb, nb, mb, cb = geo.draw()
    # the equivalence does not need every point: a subsample keeps the float64
    # second-order graph of the 573k-parameter operator affordable
    rs = np.random.default_rng(2)
    x = x[rs.choice(len(x), 400, replace=False)]
    sel = np.concatenate([rs.choice(np.where(cb == c)[0], min(40, (cb == c).sum()), replace=False)
                          for c in np.unique(cb)])
    xb, nb, mb, cb = xb[sel], nb[sel], mb[sel], cb[sel]
    torch.set_num_threads(2)
    dt = torch.float64
    u_d, x_m, y_m = (torch.tensor([v], dtype=dt) for v in (0.5, Lh, Hg))
    bpc = torch.zeros(1, 1, 2, dtype=dt)
    model.set_geometry(p)

    def P_fn(q):
        latent = model.encode(bpc, x_m, y_m)
        uv, a, b, c, d = model.predict_with_grad_latent(q.unsqueeze(0), latent, u_d, x_m, y_m)
        out = first_piola_kirchhoff_stress(a, b, c, d, MU, LAM, STATE, return_J3D=True)
        return [t.reshape(-1) for t in out[:4]], torch.minimum(out[4], out[5]).reshape(-1)

    L_ours, parts = strong_loss(P_fn, x, xb, nb, mb, cb, cfg)
    full = cb <= 1
    tags = np.where(cb[full] == 0, 1, 0)                     # PhysicsLoss: 0 gauge top, 1 arc
    part = cb == 2
    dirs = np.where(mb[part][:, 0], 0, 1)
    pl = PhysicsLoss(MU, LAM, w_equilibrium=cfg.w_eq, w_trac_top=cfg.w_top, w_trac_arc=cfg.w_arc,
                     w_traction_partial=cfg.w_part, w_barrier=cfg.w_bar, j_min=cfg.j_min,
                     L0=cfg.L0, S0=cfg.S0, stress_state=STATE).double()
    t = lambda a: torch.tensor(a, dtype=dt).unsqueeze(0)
    ref = pl(model, t(x).requires_grad_(True), t(xb[full]).requires_grad_(True), t(nb[full]),
             torch.tensor(tags), t(xb[part]).requires_grad_(True), t(nb[part]), torch.tensor(dirs),
             bpc, u_d, x_m, y_m)
    rel = abs(float(L_ours) - float(ref["loss"])) / abs(float(ref["loss"]))
    comp = {"L_eq": (float(parts["L_eq"]), float(ref["L_eq"])),
            "L_arc": (float(parts["L_arc"]), float(ref["L_trac_arc"])),
            "L_top": (float(parts["L_top"]), float(ref["L_trac_top"])),
            "L_part": (float(parts["L_part"]), float(ref["L_part"])),
            "L_barrier": (float(parts["L_barrier"]), float(ref["L_barrier"]))}
    return {"loss_rel": rel, "components": comp}, rel < 1e-10


def main():
    ra, oa = gate_a()
    for k, v in ra.items():
        print("a", k, v, flush=True)
    rb, ob = gate_b()
    print("b", rb, flush=True)
    out = {"a": ra, "a_pass": oa, "b": rb, "b_pass": ob, "G3": "PASS" if oa and ob else "FAIL"}
    json.dump(out, open(OUT, "w"), indent=1)
    print("G3:", out["G3"])
    return oa and ob


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

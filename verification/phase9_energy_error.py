#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 9.0b -- does a lower energy mean a better K_t?  (plan: 0f65e5c)

For every Tier-1 model (four new families x 3 seeds) and each of the 12
in-range scored geometries, the network's total potential energy is computed
with a fixed rule never used in training: the FEM reference mesh (n_r 60) and
the 6-point degree-4 Dunavant rule in every triangle.  The FEM's own energy on
that mesh (exact for constant-strain triangles) is computed alongside.  The
Dirichlet data are identical and there is no external work term (displacement
control), so Pi = integral of W(F) dA for both.

Test (prediction H-opt): for each geometry and each of its three seed pairs,
the lower-energy seed has the smaller |K_t err| in > 60% of the 144 pooled
pairs.  H-obj predicts about 50%.  K_t errors are the Tier-1 scores of the same
models on the same geometries (verification/results/phase8/tier1/scores.json).

    python -m verification.phase9_energy_error
"""

from __future__ import annotations

import itertools
import json
import os

import numpy as np
import torch
from scipy import stats

from eval.compare_reference import LAM, MU, STATE
from geometry.families import FAMILIES
from physics.neo_hookean import strain_energy_density
from training.family_trainer import UD, load
from verification.family_fem import reference
from verification.fem.solver import deformation_gradients

TIER1 = "verification/results/phase8/tier1"
OUT = "verification/results/phase9/p90b_energy_error.json"
FAMS = ["open_hole", "inclusion", "double_notch", "single_notch"]

# Dunavant degree-4, 6 points: (barycentric, weight as a fraction of the area)
_A, _B = 0.445948490915965, 0.091576213509771
_WA, _WB = 0.223381589678011, 0.109951743655322
BARY = np.array([[1 - 2 * _A, _A, _A], [_A, 1 - 2 * _A, _A], [_A, _A, 1 - 2 * _A],
                 [1 - 2 * _B, _B, _B], [_B, 1 - 2 * _B, _B], [_B, _B, 1 - 2 * _B]])
WQ = np.array([_WA] * 3 + [_WB] * 3)


def psi(F):
    """W(F) through the training loss's own energy density (float64)."""
    t = lambda a: torch.tensor(a, dtype=torch.float64).reshape(1, -1, 1)
    return strain_energy_density(t(F[:, 0, 0] - 1.0), t(F[:, 0, 1]), t(F[:, 1, 0]),
                                 t(F[:, 1, 1] - 1.0), MU, LAM, STATE).numpy().ravel()


def energy_fem(sol):
    return float((sol.mesh.areas * psi(deformation_gradients(sol.mesh, sol.u))).sum())


def energy_net(model, fam, p, mesh, chunk=6000):
    verts = mesh.nodes[mesh.tris]                           # (E, 3, 2)
    pts = np.einsum("qa,ead->eqd", BARY, verts).reshape(-1, 2)
    w = (mesh.areas[:, None] * WQ[None, :]).reshape(-1)
    dt = torch.get_default_dtype()
    model.set_geometry(p)
    L, Hs = fam.extent(p)
    x_m, y_m = torch.tensor([L], dtype=dt), torch.tensor([Hs], dtype=dt)
    with torch.no_grad():
        z = model.encode(torch.zeros(1, 1, 2, dtype=dt), x_m, y_m)
    tot, jmin = 0.0, np.inf
    for sl in np.array_split(np.arange(len(pts)), max(1, len(pts) // chunk)):
        q = torch.tensor(pts[sl], dtype=dt).unsqueeze(0).requires_grad_(True)
        _, a, b, c, d = model.predict_with_grad_latent(q, z, torch.tensor([UD], dtype=dt),
                                                       x_m, y_m)
        g = lambda t: t.detach().double()
        P = strain_energy_density(g(a), g(b), g(c), g(d), MU, LAM, STATE)[0, :, 0].numpy()
        tot += float((P * w[sl]).sum())
        J = ((1 + g(a)) * (1 + g(d)) - g(b) * g(c)).min().item()
        jmin = min(jmin, J)
    return tot, jmin


def compute(refine=0, fams=None, out=None):
    """``refine`` > 0 evaluates Pi_net on the nested refinement of the reference
    mesh (same polygon), to check the rule's own integration error (added after
    the independent check of the results; the registered rule is refine=0)."""
    torch.set_num_threads(1)
    out = out or OUT
    scores = json.load(open(f"{TIER1}/scores.json"))
    res = json.load(open(out)) if os.path.exists(out) else {}
    fem_cache = {}
    for fam_name in (fams or FAMS):
        fam = FAMILIES[fam_name]
        plist = fam.in_range()
        for s in range(3):
            key = f"{fam_name}|{s}"
            if key in res:
                continue
            model = load(fam_name, f"{TIER1}/{fam_name}_s{s}/last.pt")
            for prm in model.parameters():
                prm.requires_grad_(False)
            rows = []
            for i, p in enumerate(plist):
                sol = reference(fam, p, f"ir{i}")
                fk = (fam_name, i)
                if fk not in fem_cache:
                    fem_cache[fk] = energy_fem(sol)
                mesh = sol.mesh
                for _ in range(refine):
                    from verification.phase9_headroom import refine as _ref
                    mesh = _ref(mesh)[0]
                Pn, jmin = energy_net(model, fam, p, mesh)
                Pf = fem_cache[fk]
                rows.append({"i": i, "Pi_net": Pn, "Pi_fem": Pf, "gap_rel": Pn / Pf - 1.0,
                             "Jmin_net": jmin, "err": scores[key]["IR"][i]["err"]})
            res[key] = rows
            os.makedirs(os.path.dirname(out), exist_ok=True)
            json.dump(res, open(out, "w"), indent=1)
            print(f"{key}: median gap {np.median([r['gap_rel'] for r in rows]):+.3e}", flush=True)
    return res


def report(res):
    print("\nPhase 9.0b -- lower energy vs smaller |K_t err|, seed pairs per geometry")
    print("=" * 92)
    wins = tot = 0
    out = {}
    all_gap, all_err = [], []
    for fam_name in FAMS:
        rows = [res[f"{fam_name}|{s}"] for s in range(3)]
        w = n = 0
        for i in range(12):
            for a, b in itertools.combinations(range(3), 2):
                ra, rb = rows[a][i], rows[b][i]
                lower = ra if ra["Pi_net"] < rb["Pi_net"] else rb
                higher = rb if lower is ra else ra
                w += abs(lower["err"]) < abs(higher["err"])
                n += 1
        gaps = np.array([[r["gap_rel"] for r in rr] for rr in rows])
        errs = np.array([[abs(r["err"]) for r in rr] for rr in rows])
        all_gap += gaps.ravel().tolist(); all_err += errs.ravel().tolist()
        sp = stats.spearmanr(gaps.ravel(), errs.ravel())
        spread = np.median((gaps.max(0) - gaps.min(0)))
        print(f"{fam_name:<14} lower-energy seed more accurate in {w:2d}/{n} pairs "
              f"({100 * w / n:4.1f}%)   gap to FEM median {np.median(gaps):+.2e} "
              f"(seed spread of gap, median {spread:.1e})   Spearman(gap, |err|) {sp.correlation:+.2f}")
        out[fam_name] = {"wins": int(w), "pairs": int(n), "median_gap": float(np.median(gaps)),
                         "median_seed_spread_of_gap": float(spread),
                         "spearman_gap_abs_err": float(sp.correlation),
                         "spearman_p": float(sp.pvalue)}
        wins += w; tot += n
    bt = stats.binomtest(wins, tot, 0.5, alternative="greater")
    sp = stats.spearmanr(all_gap, all_err)
    frac = wins / tot
    print("-" * 92)
    print(f"pooled: {wins}/{tot} = {100 * frac:.1f}%  (one-sided binomial p vs 50%: {bt.pvalue:.3g})")
    print(f"pooled Spearman(gap to FEM, |K_t err|) over 144 model-geometries: "
          f"{sp.correlation:+.2f} (p {sp.pvalue:.2g})")
    verdict = "HOLDS" if frac > 0.60 else "FAILS"
    print(f"9.0b prediction (>60% of pairs, H-opt): {verdict}")
    out["pooled"] = {"wins": int(wins), "pairs": int(tot), "fraction": frac,
                     "binom_p_greater": float(bt.pvalue),
                     "spearman_gap_abs_err": float(sp.correlation),
                     "spearman_p": float(sp.pvalue), "verdict": verdict}
    return out


if __name__ == "__main__":
    import sys
    if "--refine" in sys.argv:
        # python -m verification.phase9_energy_error --refine [family ...]
        fams = [a for a in sys.argv[2:] if a in FAMS] or None
        ro = "verification/results/phase9/p90b_energy_error_refined.json"
        res = compute(refine=1, fams=fams, out=ro if not fams else ro.replace(".json", f"_{fams[0]}.json"))
        if not fams:
            summ = report(res)
            json.dump({"rows": res, "summary": summ},
                      open("docs/phase9_energy_error_refined.json", "w"), indent=1)
    else:
        res = compute()
        summ = report(res)
        json.dump({"rows": res, "summary": summ},
                  open("docs/phase9_energy_error.json", "w"), indent=1)

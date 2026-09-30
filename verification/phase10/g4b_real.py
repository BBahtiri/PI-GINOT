#!/usr/bin/env python3
"""
Phase 10, gate G4b -- each arm on one geometry of the real problem.

Per family, the first validation geometry; the operator, float32, default
settings (training/formulation_trainer.ARM_DEFAULTS), trained on that one
geometry (batch 1) for 1,000 steps: warm-up 50 steps, then cosine 3e-4 -> 3e-6,
clip 1.0, seed 0.  Monitors (verification/phase10/monitors.py) at 0, 250, 500,
750, 1000.

Gate, per arm and family (both must hold):
  (i)  the energy gap to the extrapolated FEM energy falls over training
       (final < step-0 value) and ends <= 3x B's final gap on the geometry;
  (ii) the residual ratio (q+2 and h/2 over the training rule) ends <= 2.
Reported: K_t error on the geometry; the K_t of a P1 FEM solved on B's
triangulation (the test mesh of C2 and the base of C2I).

    python -m verification.phase10.g4b_real run FAMILY ARM
    python -m verification.phase10.g4b_real timing FAMILY ARM     (5 steps: s/step, RSS)
    python -m verification.phase10.g4b_real report
"""
import json
import os
import sys
import time

import numpy as np

from training.family_trainer import _sets

ROOT = "verification/results/phase10/g4b"
FAMS = ["open_hole", "double_notch", "inclusion", "single_notch", "dogbone"]
ARMS = ["B", "A", "C1", "C1n", "C2", "C2I"]


def diag_overrides(name):
    """Diagnostic (non-gate) settings."""
    import dataclasses
    from physics.strong_form import StrongConfig
    if name == "long3k":                   # same settings, 3,000 steps
        return {}
    if name == "w10t20":                   # A: the boundary-weighted corner of its S6 grid
        return {"strong": dataclasses.replace(StrongConfig(), w_eq=10.0, w_arc=20.0, w_top=20.0,
                                              w_part=20.0)}
    raise KeyError(name)


def run(fam, arm, steps=1000, timing=False, diag=None):
    from training.formulation_trainer import train
    from verification.phase10.monitors import Monitor
    _, val, _ = _sets(fam)
    p = val[0]
    ov = diag_overrides(diag) if diag else None
    if diag == "long3k":
        steps = 3000
    out = f"{ROOT}/{'timing/' if timing else ''}{'diag_' + diag + '_' if diag else ''}{arm}_{fam}"
    mon = None if timing else Monitor(arm, fam, [p], ["val0"], overrides=ov)
    _, res = train(arm, fam, 0, out, steps=steps, batch=1, lr=3e-4, lr_end=3e-6, warmup=50,
                   t_const=50, geoms=[p], monitor=mon, monitor_every=250, overrides=ov,
                   ckpt_every=steps, log_every=1 if timing else 25)
    if timing:
        h = json.load(open(f"{out}/history.json"))["hist"]
        t = [r["wall_min"] * 60 for r in h]
        res["s_per_step_steady"] = float(np.median(np.diff(t)[1:]))
        json.dump(res, open(f"{out}/result.json", "w"), indent=1)
        print("TIMING", fam, arm, f"{res['s_per_step_steady']:.2f} s/step", f"{res['peak_rss_gb']:.2f} GB")
    return res


def fem_on_test_mesh(fam):
    """K_t of the P1 FEM on B's triangulation (what C2's test space can resolve)."""
    from training.family_trainer import build_loss
    _, val, _ = _sets(fam)
    p = val[0]
    m = build_loss(fam)._mesh(p)
    if fam == "dogbone":
        return None
    from geometry.families import FAMILIES
    from verification.family_fem import kt_of, solve_family
    from verification.phase10.monitors import fem_reference
    from config import LOADING_CONFIG
    f = FAMILIES[fam]
    sol = solve_family(f, p, LOADING_CONFIG["u_max"], mesh=m)
    ref = fem_reference(fam, p, "val0")
    k, kr = kt_of(f, p, sol), kt_of(f, p, ref)
    return {"Kt_test_mesh": k, "Kt_ref": kr, "err": k / kr - 1.0, "n_elem": int(m.n_elem)}


def report():
    rows, verdict = {}, {}
    for fam in FAMS:
        B = f"{ROOT}/B_{fam}/result.json"
        if not os.path.exists(B):
            continue
        gB = json.load(open(B))["monitor"][-1]["geoms"][0]["gap"]
        for arm in ARMS:
            path = f"{ROOT}/{arm}_{fam}/result.json"
            if not os.path.exists(path):
                continue
            r = json.load(open(path))
            m = r["monitor"]
            g0, g1 = m[0]["geoms"][0], m[-1]["geoms"][0]
            ok_i = g1["gap"] < g0["gap"] and g1["gap"] <= 3 * gB
            ratio = g1.get("res", {}).get("ratio")
            ok_ii = ratio is None or ratio <= 2.0
            rows[f"{fam}|{arm}"] = {"gap0": g0["gap"], "gap": g1["gap"], "gap_over_B": g1["gap"] / gB,
                                    "ratio": ratio, "res": g1.get("res"), "kt_err": g1.get("err"),
                                    "vm": g1.get("vm"), "Jmin": g1["Jmin"],
                                    "s_per_step": r["s_per_step"], "rss_gb": r["peak_rss_gb"],
                                    "clipped": r["clipped_frac"], "pass_i": ok_i, "pass_ii": ok_ii,
                                    "gap_traj": [x["geoms"][0]["gap"] for x in m]}
            verdict[f"{fam}|{arm}"] = "PASS" if ok_i and ok_ii else "FAIL"
    return rows, verdict


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "run":
        run(sys.argv[2], sys.argv[3], diag=sys.argv[4] if len(sys.argv) > 4 else None)
    elif mode == "timing":
        run(sys.argv[2], sys.argv[3], steps=6, timing=True)
    elif mode == "fem_test_mesh":
        out = {f: fem_on_test_mesh(f) for f in FAMS}
        json.dump(out, open(f"{ROOT}/fem_test_mesh.json", "w"), indent=1)
        print(out)
    else:
        rows, verdict = report()
        json.dump({"rows": rows, "verdict": verdict}, open(f"{ROOT}/report.json", "w"), indent=1)
        for k, v in rows.items():
            print(f"{k:22s} gap {v['gap0']:+.3e} -> {v['gap']:+.3e} ({v['gap_over_B']:.2f}x B)  "
                  f"ratio {v['ratio'] if v['ratio'] is None else round(v['ratio'], 2)}  "
                  f"Kt err {v['kt_err']}  {v['s_per_step']:.2f} s/step {v['rss_gb']:.2f} GB  "
                  f"{verdict[k]}")

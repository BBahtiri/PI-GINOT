#!/usr/bin/env python3
"""
Phase 10, S6 -- the tuning pilot (same budget for every arm, B included).

Families: open hole and double notch.  Operator, seed 0, bank of 64, batch 4,
800 steps: warm-up 50, then cosine to lr/100.  Tuning set: 4 fresh geometries
per family (``tuning_set``: default_rng(seed_base + 800), inside a 5% margin
of the training box; in neither the bank nor the scored sets), each with its
own FEM reference and extrapolated FEM energy.

Grids (plan S6, <= 8 per arm; the cost pre-filter G5 acts first):
  B    lr {1e-4, 3e-4, 1e-3} x clip {1.0, off}
  A    w_eq {10, 100} x w_top {2, 20} x lr {3e-4, 1e-3}   (w_arc 20; w_part = w_top)
  C1   level {B, coarse} x K {2, 4} x beta {5, 50}         (Q = K + 2)
  C1n  level {B, coarse} x K {2, 4}
  C2   degree {1, 2} x Dunavant {4, 6} x test mesh {B, h/2}; then gamma {area, one}
  C2I  base mesh {B, h/2}; then gamma {area, one}
G5 pre-filter: a configuration is dropped before it runs if its peak RSS
exceeds 3 GB or its estimated S7 cost (15 runs x 2,400 steps x batch 4 at the
measured s/step) exceeds 150 CPU-h.  Costs come from ``timing``.

Selection: lowest mean |K_t err| over the 8 tuning geometries; a
configuration is discarded if its tuning-set energy gap rises between steps
400 and 800, or its residual ratio (monitors (ii)) exceeds 2.  The top two are
re-run with seeds 1 and 2 and the lower three-seed mean wins.

    python -m verification.phase10.s6_pilot timing ARM CONFIG
    python -m verification.phase10.s6_pilot run ARM CONFIG FAMILY [SEED]
    python -m verification.phase10.s6_pilot report
"""
import dataclasses
import json
import os
import sys

import numpy as np

from physics.strong_form import StrongConfig
from physics.vpinn.weak import WeakConfig

ROOT = "verification/results/phase10/s6"
FAMS = ["open_hole", "double_notch"]
S7_RUNS, S7_STEPS, S7_BATCH = 15, 2400, 4
MAX_RSS_GB, MAX_S7_CPUH = 3.0, 150.0


def tuning_set(fam_name, n=4):
    from geometry.families import FAMILIES
    f = FAMILIES[fam_name]
    rng = np.random.default_rng(f.seed_base + 800)
    box = {k: (lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo)) for k, (lo, hi) in f.box.items()}
    return [f._draw(rng, box) for _ in range(n)]


def grid():
    g = {}
    for lr in (1e-4, 3e-4, 1e-3):
        for clip in (1.0, None):
            g[("B", f"lr{lr:g}_clip{'on' if clip else 'off'}")] = ({}, {"lr": lr, "clip": clip})
    for weq in (10.0, 100.0):
        for wt in (2.0, 20.0):
            for lr in (3e-4, 1e-3):
                sc = dataclasses.replace(StrongConfig(), w_eq=weq, w_top=wt, w_part=wt)
                g[("A", f"weq{weq:g}_wtop{wt:g}_lr{lr:g}")] = ({"strong": sc}, {"lr": lr})
    for lvl, n in (("B", 1400), ("coarse", 350)):
        for K in (2, 4):
            for beta in (5.0, 50.0):
                g[("C1", f"{lvl}_K{K}_beta{beta:g}")] = (
                    {"weak": WeakConfig("C1", K=K, n_elem=n, beta=beta)}, {})
            g[("C1n", f"{lvl}_K{K}")] = ({"weak": WeakConfig("C1n", K=K, n_elem=n)}, {})
    for deg in (1, 2):
        for qd in (4, 6):
            for lvl in ("B", "h2"):
                g[("C2", f"P{deg}_D{qd}_{lvl}")] = (
                    {"weak": WeakConfig("C2", degree=deg, quad_degree=qd), "level": lvl}, {})
    for lvl in ("B", "h2"):
        g[("C2I", lvl)] = ({"weak": WeakConfig("C2I"), "level": lvl}, {})
    return g


def gamma_variant(arm, cfg_name, gamma="one"):
    ov, kw = grid()[(arm, cfg_name)]
    ov = dict(ov)
    ov["weak"] = dataclasses.replace(ov["weak"], gamma=gamma)
    return ov, kw


def _resolve(arm, cfg_name):
    if cfg_name.endswith("+one"):
        return gamma_variant(arm, cfg_name[:-4])
    return grid()[(arm, cfg_name)]


def timing(arm, cfg_name, fam="open_hole", steps=4):
    """Steps on 4 bank geometries (batch 4; the same 4 every step, so per-geometry
    set-up is paid once, as it is amortised in S7) -> steady s/step, peak RSS."""
    from training.family_trainer import _sets
    from training.formulation_trainer import train
    ov, kw = _resolve(arm, cfg_name)
    bank = _sets(fam)[0][:4]
    out = f"{ROOT}/timing/{arm}_{cfg_name}_{fam}"
    _, res = train(arm, fam, 0, out, steps=steps, batch=4, warmup=50, t_const=50, geoms=bank,
                   overrides=ov, log_every=1, ckpt_every=10 ** 9, **kw)
    h = json.load(open(f"{out}/history.json"))["hist"]
    t = np.array([r["wall_min"] * 60 for r in h])
    sps = float(np.median(np.diff(t)[1:]))
    s7 = S7_RUNS * S7_STEPS * sps / 3600
    rec = {"arm": arm, "config": cfg_name, "s_per_step": sps, "peak_rss_gb": res["peak_rss_gb"],
           "s7_cpu_h": s7, "keep": bool(res["peak_rss_gb"] <= MAX_RSS_GB and s7 <= MAX_S7_CPUH)}
    json.dump(rec, open(f"{out}/timing.json", "w"), indent=1)
    print("TIMING", json.dumps(rec), flush=True)
    return rec


class TuneMonitor:
    """Energy gap and residual ratio on the 4 tuning geometries (not K_t)."""

    def __init__(self, arm, fam, overrides):
        from verification.phase10.monitors import Monitor
        ps = tuning_set(fam)
        self.m = Monitor(arm, fam, ps, [f"tune{i}" for i in range(len(ps))], overrides, kt=False)

    def __call__(self, model, step):
        if step not in (400, 800):
            return {"skipped": True}
        return self.m(model, step)


def run(arm, cfg_name, fam, seed=0, steps=800):
    from training.formulation_trainer import train
    from verification.phase10.monitors import fem_reference, kt_err
    ov, kw = _resolve(arm, cfg_name)
    out = f"{ROOT}/runs/{arm}_{cfg_name}_{fam}_s{seed}"
    if os.path.exists(f"{out}/score.json"):
        return json.load(open(f"{out}/score.json"))
    mon = TuneMonitor(arm, fam, ov)
    model, res = train(arm, fam, seed, out, steps=steps, batch=4, warmup=50, t_const=50,
                       overrides=ov, monitor=mon, monitor_every=400, ckpt_every=400, **kw)
    model.eval()
    rows = []
    for i, p in enumerate(tuning_set(fam)):
        rows.append(kt_err(model, fam, p, fem_reference(fam, p, f"tune{i}")))
    mons = [m for m in res["monitor"] if not m.get("skipped")]
    gap = {m["step"]: [g["gap"] for g in m["geoms"]] for m in mons}
    ratio = [g.get("res", {}).get("ratio") for g in mons[-1]["geoms"]] if mons else []
    sc = {"arm": arm, "config": cfg_name, "family": fam, "seed": seed, "kt": rows,
          "mean_abs_err": float(np.mean([abs(r["err"]) for r in rows])),
          "gap": gap, "ratio": ratio, "s_per_step": res["s_per_step"],
          "peak_rss_gb": res["peak_rss_gb"], "clipped_frac": res["clipped_frac"]}
    json.dump(sc, open(f"{out}/score.json", "w"), indent=1)
    print("SCORE", arm, cfg_name, fam, seed, f"{sc['mean_abs_err']:.4f}", flush=True)
    return sc


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "timing":
        timing(sys.argv[2], sys.argv[3], *(sys.argv[4:5]))
    elif mode == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 0)
    elif mode == "list":
        for k in grid():
            print(*k)

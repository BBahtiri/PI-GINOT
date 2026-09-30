#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 9.4 -- score the from-scratch energy runs and judge P9.4-1..3
(docs/phase9_4_plan.md).

    python -m verification.phase9_4_score            # score finished runs
    python -m verification.phase9_4_score --check    # scorer integrity (a stored 9.1 score)
    python -m verification.phase9_4_score --report
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from verification.phase9_score import FAMS, SEEDS, _load, _score, start_rows

ROOT = "verification/results/phase9/p94"
OUT = "verification/results/phase9/p94_scores.json"
P91 = "docs/phase9_p91_scores.json"


def score_all():
    torch.set_num_threads(1)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for fam in FAMS:
        for s in SEEDS:
            key = f"{fam}|{s}"
            d = f"{ROOT}/{fam}_s{s}"
            if key in res or not os.path.exists(f"{d}/result.json"):
                continue
            model = _load(fam, f"{d}/last.pt")
            res[key] = _score(fam, model, ("IR", "OOR"))
            json.dump(res, open(OUT, "w"), indent=1)
            e = np.abs([r["err"] for r in res[key]["IR"]]).mean()
            print(f"{key}: IR mean |err| {100 * e:.2f}%", flush=True)
    return res


def check():
    """G6-style integrity: re-score a stored 9.1 checkpoint."""
    torch.set_num_threads(1)
    ref = json.load(open(P91))["open_hole|0|anneal|800"]["IR"]
    m = _load("open_hole", "verification/results/phase9/p91/open_hole_s0_anneal/ck800.pt")
    new = _score("open_hole", m, ("IR",))["IR"]
    dev = max(abs(a["err"] - b["err"]) for a, b in zip(ref, new))
    print(f"scorer integrity: max |err difference| {dev:.2e} over {len(new)} geometries")
    return dev


def _rows(fam, s, src, res):
    if src == "9.4":
        return res[f"{fam}|{s}"]
    if src == "9.1":
        return json.load(open(P91))[f"{fam}|{s}|anneal|800"]
    return start_rows()[(fam, s)]


def boot(d, n=10000, seed=0):
    """Two-way cluster bootstrap of the mean of d (seeds x geometries)."""
    rng = np.random.default_rng(seed)
    S, G = d.shape
    m = np.empty(n)
    for k in range(n):
        si = rng.integers(0, S, S)
        gi = rng.integers(0, G, G)
        m[k] = d[np.ix_(si, gi)].mean()
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def summary(res):
    out = {}
    for fam in FAMS:
        if not all(f"{fam}|{s}" in res for s in SEEDS):
            continue
        f = {}
        E = {}
        for src in ("tier1", "9.1", "9.4"):
            rows = {s: _rows(fam, s, src, res) for s in SEEDS}
            Ei = np.array([[r["err"] for r in rows[s]["IR"]] for s in SEEDS])
            Eo = np.array([[r["err"] for r in rows[s]["OOR"]] for s in SEEDS])
            K = np.array([r["K_ref"] for r in rows[0]["IR"]])
            P = np.array([[r["K_pinn"] for r in rows[s]["IR"]] for s in SEEDS])
            r2 = [float(1 - ((P[s] - K) ** 2).sum() / ((K - K.mean()) ** 2).sum()) for s in SEEDS]
            E[src] = Ei
            f[src] = {"ir_mean_abs": float(np.abs(Ei).mean()),
                      "ir_seed_sd": float(np.sqrt(Ei.var(0, ddof=1).mean())),
                      "ir_bias": float(Ei.mean()), "ir_r2": float(np.mean(r2)),
                      "oor_mean_abs": float(np.abs(Eo).mean()),
                      "per_seed_ir": [float(np.abs(Ei[s]).mean()) for s in SEEDS]}
            if src == "9.4":
                for k in ("u", "v", "vm", "N_err"):
                    vals = [r[k] for s in SEEDS for r in rows[s]["IR"] if k in r]
                    if vals:
                        f[src][f"ir_{k}"] = float(np.mean(np.abs(vals)))
        for ref in ("tier1", "9.1"):
            d = np.abs(E["9.4"]) - np.abs(E[ref])
            mean, lo, hi = boot(d)
            f[f"d_vs_{ref}"] = {"mean": mean, "lo": lo, "hi": hi,
                                "resolved_better": hi < 0, "resolved_worse": lo > 0}
        runs = [json.load(open(f"{ROOT}/{fam}_s{s}/result.json")) for s in SEEDS]
        f["wall_min"] = [r.get("wall_min", float("nan")) for r in runs]
        f["clipped_frac"] = [r.get("clipped_frac", float("nan")) for r in runs]
        out[fam] = f
    return out


def verdicts(sm):
    if len(sm) < len(FAMS):
        return {"complete": False}
    lower = sum(v["9.4"]["ir_mean_abs"] < v["tier1"]["ir_mean_abs"] for v in sm.values())
    res_better = sum(v["d_vs_tier1"]["resolved_better"] for v in sm.values())
    worse91 = [k for k, v in sm.items() if v["d_vs_9.1"]["resolved_worse"]]
    ratio = {k: sm[k]["9.4"]["oor_mean_abs"] / sm[k]["9.4"]["ir_mean_abs"]
             for k in ("open_hole", "double_notch", "single_notch")}
    return {"complete": True,
            "P9.4-1": {"lower_than_tier1": lower, "resolved_better": res_better,
                       "holds": lower >= 4 and res_better >= 3},
            "P9.4-2": {"resolved_worse_than_9.1": worse91, "holds": not worse91},
            "P9.4-3": {"oor_over_ir": ratio, "holds": all(r > 2 for r in ratio.values())}}


def report(res):
    sm = summary(res)
    print("Phase 9.4 -- energy method from scratch (2,400 steps, anneal built in): in-range K_t")
    print("=" * 110)
    print(f"{'family':<14}{'source':<8}{'IR |err|':>10}{'seed sd':>9}{'bias':>8}{'R2':>7}{'OOR |err|':>11}"
          f"   paired d (9.4 - src), 95% CI")
    for fam, f in sm.items():
        for src in ("tier1", "9.1", "9.4"):
            v = f[src]
            ci = ""
            if src != "9.4":
                d = f[f"d_vs_{src}"]
                ci = f"   {100 * d['mean']:+.2f} [{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}]"
            print(f"{fam:<14}{src:<8}{100 * v['ir_mean_abs']:>9.2f}%{100 * v['ir_seed_sd']:>8.2f}%"
                  f"{100 * v['ir_bias']:>+7.2f}%{v['ir_r2']:>+7.2f}{100 * v['oor_mean_abs']:>10.2f}%{ci}")
        print(f"{'':<14}wall {', '.join(f'{w:.0f}' for w in f['wall_min'])} min; clipped "
              f"{', '.join(f'{c:.2f}' for c in f['clipped_frac'])}")
    vd = verdicts(sm)
    print(json.dumps(vd, indent=1))
    return sm, vd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.check:
        check()
    elif a.report:
        res = json.load(open(OUT))
        sm, vd = report(res)
        json.dump({"summary": sm, "verdicts": vd}, open("docs/phase9_4_summary.json", "w"), indent=1)
    else:
        score_all()

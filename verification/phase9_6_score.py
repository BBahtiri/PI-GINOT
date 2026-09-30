#!/usr/bin/env python3
"""
Phase 9.6 -- score the 4,800-step runs on the dog-bone, open hole and inclusion
and judge P9.6-1..4 (docs/phase9_6_plan.md).  Paired with 9.4.  Shares the
score file and helpers of verification/phase9_5_score.py.

    python -m verification.phase9_6_score            # score finished runs
    python -m verification.phase9_6_score --report
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from verification.phase9_4_score import boot
from verification.phase9_score import _load, _score

ROOT = "verification/results/phase9/p94x"
OUT = "verification/results/phase9/p94x_scores.json"
P94 = "verification/results/phase9/p94_scores.json"
FAMS = ("dogbone", "open_hole", "inclusion")
CONFIRM = {"dogbone": (0, 1, 2), "open_hole": (1, 2), "inclusion": (0, 1, 2)}   # OH s0: pilot


def score_all():
    torch.set_num_threads(1)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for fam in FAMS:
        for s in (0, 1, 2):
            k, d = f"{fam}|{s}", f"{ROOT}/{fam}_s{s}"
            if k in res or not os.path.exists(f"{d}/result.json"):
                continue
            res[k] = _score(fam, _load(fam, f"{d}/last.pt"), ("IR", "OOR"))
            json.dump(res, open(OUT, "w"), indent=1)
            print(k, "scored", flush=True)
    return res


def _val(fam, s, root, step=None):
    r = json.load(open(f"{root}/{fam}_s{s}/result.json"))
    m = r["monitor"][-1] if step is None else [x for x in r["monitor"] if x["step"] == step][0]
    return float(np.mean([g["gap"] for g in m["geoms"]])), r["wall_min"]


def summary(res):
    p94 = json.load(open(P94))
    out = {}
    for fam in FAMS:
        seeds_all = [s for s in (0, 1, 2) if f"{fam}|{s}" in res]
        f = {"seeds": seeds_all}
        for name, seeds in (("all", seeds_all), ("confirm", [s for s in CONFIRM[fam] if s in seeds_all])):
            if not seeds:
                continue
            row = {}
            for st in ("IR", "OOR"):
                L = np.array([[r["err"] for r in res[f"{fam}|{s}"][st]] for s in seeds])
                S = np.array([[r["err"] for r in p94[f"{fam}|{s}"][st]] for s in seeds])
                d = np.abs(L) - np.abs(S)
                m, lo, hi = boot(d)
                row[st] = {"long": float(np.abs(L).mean()), "p94": float(np.abs(S).mean()),
                           "d": m, "lo": lo, "hi": hi, "better_geoms": int((d < 0).sum()),
                           "n": int(d.size), "per_seed_long": [float(x) for x in np.abs(L).mean(1)],
                           "per_seed_p94": [float(x) for x in np.abs(S).mean(1)]}
            gaps = [(_val(fam, s, ROOT)[0], _val(fam, s, "verification/results/phase9/p94")[0]) for s in seeds]
            row["val_gap"] = {"long": [g[0] for g in gaps], "p94": [g[1] for g in gaps]}
            row["wall_min"] = [_val(fam, s, ROOT)[1] for s in seeds]
            f[name] = row
        out[fam] = f
    return out


def verdicts(sm):
    if not all(set(CONFIRM[f]) <= set(sm.get(f, {}).get("seeds", [])) for f in FAMS):
        return {"complete": False}
    c = {f: sm[f]["confirm"] for f in FAMS}
    return {"complete": True,
            "P9.6-1": c["open_hole"]["IR"]["long"] < c["open_hole"]["IR"]["p94"],
            "P9.6-2": c["open_hole"]["OOR"]["long"] < c["open_hole"]["OOR"]["p94"],
            "P9.6-3": {f: not (c[f]["IR"]["lo"] > 0) for f in ("dogbone", "inclusion")},
            "P9.6-4": {f: all(a < b for a, b in zip(c[f]["val_gap"]["long"], c[f]["val_gap"]["p94"]))
                       for f in FAMS}}


def report(res):
    sm = summary(res)
    for fam, f in sm.items():
        for name in ("confirm", "all"):
            if name not in f:
                continue
            r = f[name]
            print(f"{fam:13s} {name:8s}", end="")
            for st in ("IR", "OOR"):
                v = r[st]
                print(f"  {st} {100 * v['p94']:.2f}% -> {100 * v['long']:.2f}%  d {100 * v['d']:+.2f} "
                      f"[{100 * v['lo']:+.2f}, {100 * v['hi']:+.2f}] better {v['better_geoms']}/{v['n']}", end="")
            print(f"  val gap {[round(100 * g, 3) for g in r['val_gap']['p94']]} -> "
                  f"{[round(100 * g, 3) for g in r['val_gap']['long']]}")
    vd = verdicts(sm)
    print(json.dumps(vd, indent=1))
    return sm, vd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        sm, vd = report(json.load(open(OUT)))
        json.dump({"summary": sm, "verdicts": vd}, open("docs/phase9_6_summary.json", "w"), indent=1)
    else:
        score_all()

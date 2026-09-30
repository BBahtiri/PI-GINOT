#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 9.1, exploratory (not pre-registered): weight averaging.

9.2 showed K_t moving by 0.3-1.4 points between checks at nearly constant
energy -- a noise floor.  Stochastic weight averaging (the mean of the
weights along a constant-learning-rate trajectory) is the standard remedy for
exactly that, and the 9.1 control arm *is* such a trajectory: it saved ck200,
ck400, ck600 and ck800 at a constant 3e-4.  So SWA costs no training here --
average those four checkpoints and score the result like any other model.

    python -m verification.phase9_swa
"""

from __future__ import annotations

import json
import os

import torch

from verification.phase9_score import FAMS, ROOT, SEEDS, _score

OUT = "verification/results/phase9/p91_swa_scores.json"


def swa_model(fam, seed, cks=(200, 400, 600, 800)):
    from training.family_trainer import family_model
    d = os.path.join(ROOT, f"{fam}_s{seed}_control")
    sts = [torch.load(os.path.join(d, f"ck{c}.pt"), map_location="cpu",
                      weights_only=False)["model_state_dict"] for c in cks]
    avg = {k: (sum(s[k] for s in sts) / len(sts)) if sts[0][k].is_floating_point() else sts[0][k]
           for k in sts[0]}
    m = family_model(fam)
    m.load_state_dict(avg, strict=True)
    return m.eval()


def main():
    torch.set_num_threads(1)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for fam in FAMS:
        for s in SEEDS:
            key = f"{fam}|{s}|swa"
            if key in res or not os.path.exists(os.path.join(ROOT, f"{fam}_s{s}_control", "result.json")):
                continue
            print(f"scoring {key} ...", flush=True)
            res[key] = _score(fam, swa_model(fam, s), ("IR", "OOR"))
            json.dump(res, open(OUT, "w"), indent=1)
    return res




def report():
    import numpy as np
    from verification.phase9_score import OUT as SCORES
    sw = json.load(open(OUT))
    sc = json.load(open(SCORES))
    print("Phase 9.1 exploratory -- SWA of the control arm vs its last checkpoint vs anneal")
    print(f"{'family':<14}{'arm':<9}{'IR |err|':>10}{'seed sd':>9}{'OOR |err|':>11}")
    out = {}
    for fam in FAMS:
        rows = {"control": [sc.get(f"{fam}|{s}|control|800") for s in SEEDS],
                "swa": [sw.get(f"{fam}|{s}|swa") for s in SEEDS],
                "anneal": [sc.get(f"{fam}|{s}|anneal|800") for s in SEEDS]}
        out[fam] = {}
        for arm, rr in rows.items():
            if any(r is None for r in rr):
                continue
            E = np.array([[g["err"] for g in r["IR"]] for r in rr])
            O = np.array([[g["err"] for g in r["OOR"]] for r in rr])
            d = {"ir_mean_abs": float(np.abs(E).mean()),
                 "ir_seed_sd": float(np.sqrt(E.var(0, ddof=1).mean())),
                 "oor_mean_abs": float(np.abs(O).mean())}
            out[fam][arm] = d
            print(f"{fam:<14}{arm:<9}{100 * d['ir_mean_abs']:>9.2f}%{100 * d['ir_seed_sd']:>8.2f}%"
                  f"{100 * d['oor_mean_abs']:>10.2f}%")
    json.dump(out, open("verification/results/phase9/p91_swa_summary.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    import sys
    report() if "--report" in sys.argv else main()

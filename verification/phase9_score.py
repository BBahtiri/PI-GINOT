#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 9.1 -- score the anneal and control continuations, and judge
P9.1a-d as registered in docs/phase9_optimizer_plan.md (0f65e5c).

Scored exactly as the models they started from:
  * new families: verification.family_score.score_geometry against the cached
    n_r 60 references, 12 in-range + 8 out-of-range;
  * dog-bone: verification.retest.score_set (the 6.7 scorer), same sets.

Checkpoints: ck800 of both arms on IR and OOR; ck200 / ck400 / ck600 of the
control arm on IR, for the wander W (the sd of each geometry's K_t error over
the control's four checkpoints, ddof 1, averaged over geometries).

"start" rows are the scores of the checkpoints the runs began from (Tier-1
scores.json; the 6.7 retest scores for the dog-bone).

    python -m verification.phase9_score            # score what has finished
    python -m verification.phase9_score --report
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

ROOT = "verification/results/phase9/p91"
OUT = "verification/results/phase9/p91_scores.json"
FAMS = ["dogbone", "open_hole", "inclusion", "double_notch", "single_notch"]
SEEDS = (0, 1, 2)
ARMS = ("anneal", "control")


def _load(fam, path):
    from training.family_trainer import family_model
    m = family_model(fam)
    st = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    m.load_state_dict(st, strict=True)
    return m.eval()


def _score(fam, model, splits):
    out = {}
    if fam == "dogbone":
        from verification.heldout_sets import IR_SEED, OOR_SEED, entries, param_sets
        from verification.retest import score_set
        ir_p, oor_p, _ = param_sets()
        sets = {"IR": (entries(ir_p, IR_SEED), f"ir_{IR_SEED}"),
                "OOR": (entries(oor_p, OOR_SEED), f"oor_{OOR_SEED}")}
        for st in splits:
            rows = score_set(model, "energy", *sets[st])
            out[st] = [{"K_ref": r["scf_ref"], "K_pinn": r["scf_u"], "err": r["err_u"],
                        "u": r["u"], "v": r["v"], "vm": r["vm"]} for r in rows]
        return out
    from geometry.families import FAMILIES
    from verification.family_fem import reference
    from verification.family_score import score_geometry
    fam_o = FAMILIES[fam]
    for st in splits:
        plist = fam_o.in_range() if st == "IR" else fam_o.out_of_range()
        out[st] = [score_geometry(model, fam_o, p, reference(fam_o, p, f"{st.lower()}{i}"))
                   for i, p in enumerate(plist)]
    return out


def score_all():
    torch.set_num_threads(1)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for fam in FAMS:
        for s in SEEDS:
            for arm in ARMS:
                d = os.path.join(ROOT, f"{fam}_s{s}_{arm}")
                if not os.path.exists(os.path.join(d, "result.json")):
                    continue
                cks = [800] + ([200, 400, 600] if arm == "control" else [])
                for ck in cks:
                    key = f"{fam}|{s}|{arm}|{ck}"
                    if key in res:
                        continue
                    print(f"scoring {key} ...", flush=True)
                    model = _load(fam, os.path.join(d, f"ck{ck}.pt"))
                    res[key] = _score(fam, model, ("IR", "OOR") if ck == 800 else ("IR",))
                    os.makedirs(os.path.dirname(OUT), exist_ok=True)
                    json.dump(res, open(OUT, "w"), indent=1)
    return res


def start_rows():
    t1 = json.load(open("verification/results/phase8/tier1/scores.json"))
    rt = json.load(open("docs/phase6_7_retest_scores.json"))
    rows = {}
    for fam in FAMS:
        for s in SEEDS:
            if fam == "dogbone":
                rows[(fam, s)] = {st: [{"K_ref": r["scf_ref"], "K_pinn": r["scf_u"],
                                        "err": r["err_u"]} for r in rt[f"energy|{s}"][st]]
                                  for st in ("IR", "OOR")}
            else:
                rows[(fam, s)] = t1[f"{fam}|{s}"]
    return rows


def _E(rows_by_seed, st):
    return np.array([[r["err"] for r in rows_by_seed[s][st]] for s in SEEDS])


def summary(res):
    start = start_rows()
    out = {}
    for fam in FAMS:
        arms = {"start": {s: start[(fam, s)] for s in SEEDS}}
        for arm in ARMS:
            if all(f"{fam}|{s}|{arm}|800" in res for s in SEEDS):
                arms[arm] = {s: res[f"{fam}|{s}|{arm}|800"] for s in SEEDS}
        fam_out = {}
        for name, rows in arms.items():
            E, O = _E(rows, "IR"), _E(rows, "OOR")
            K = np.array([r["K_ref"] for r in rows[0]["IR"]])
            P = np.array([[r["K_pinn"] for r in rows[s]["IR"]] for s in SEEDS])
            r2 = [float(1 - ((P[s] - K) ** 2).sum() / ((K - K.mean()) ** 2).sum()) for s in SEEDS]
            fam_out[name] = {
                "ir_mean_abs": float(np.abs(E).mean()),
                "ir_seed_sd": float(np.sqrt(E.var(0, ddof=1).mean())),
                "ir_ensemble_mean_abs": float(np.abs(E.mean(0)).mean()),
                "ir_bias": float(E.mean()),
                "ir_r2_mean": float(np.mean(r2)), "ir_r2_sd": float(np.std(r2, ddof=1)),
                "oor_mean_abs": float(np.abs(O).mean()),
                "per_seed_ir_mean_abs": [float(np.abs(E[s]).mean()) for s in SEEDS]}
        if all(f"{fam}|{s}|control|{ck}" in res for s in SEEDS for ck in (200, 400, 600, 800)):
            W = []
            for s in SEEDS:
                T = np.array([[r["err"] for r in res[f"{fam}|{s}|control|{ck}"]["IR"]]
                              for ck in (200, 400, 600, 800)])
                W.append(float(T.std(0, ddof=1).mean()))
            # The seeds share one data stream, so part of the wander moves in
            # step across seeds and cannot create seed spread.  Remove the
            # across-seed mean at each checkpoint (rescaled by sqrt(3/2) for
            # the removed mean of three) -- added after the independent check.
            TT = np.array([[[r["err"] for r in res[f"{fam}|{s}|control|{ck}"]["IR"]]
                            for ck in (200, 400, 600, 800)] for s in SEEDS])
            D = TT - TT.mean(0, keepdims=True)
            Ws = float(D.std(1, ddof=1).mean(-1).mean() * np.sqrt(3 / 2))
            fam_out["wander"] = {"W_per_seed": W, "W": float(np.mean(W)),
                                 "W_seed_specific": Ws,
                                 "control_mean_over_checkpoints": float(np.abs(TT).mean())}
        out[fam] = fam_out
    return out


def report(res):
    sm = summary(res)
    print("Phase 9.1 -- anneal vs control (+800 steps), in-range K_t")
    print("=" * 104)
    print(f"{'family':<14}{'arm':<9}{'IR |err|':>10}{'seed sd':>9}{'3-seed avg':>11}{'bias':>8}"
          f"{'R2':>14}{'OOR |err|':>11}   per-seed IR |err|")
    for fam, d in sm.items():
        for arm in ("start",) + ARMS:
            if arm not in d:
                continue
            x = d[arm]
            print(f"{fam:<14}{arm:<9}{100 * x['ir_mean_abs']:>9.2f}%{100 * x['ir_seed_sd']:>8.2f}%"
                  f"{100 * x['ir_ensemble_mean_abs']:>10.2f}%{100 * x['ir_bias']:>+7.2f}%"
                  f"{x['ir_r2_mean']:>+8.2f}±{x['ir_r2_sd']:.2f}{100 * x['oor_mean_abs']:>10.2f}%   "
                  + " ".join(f"{100 * v:.2f}" for v in x["per_seed_ir_mean_abs"]))
        if "wander" in d:
            w = d["wander"]
            print(f"{'':<14}control wander W {100 * w['W']:.2f}% "
                  f"(per seed {' '.join(f'{100 * v:.2f}' for v in w['W_per_seed'])}); "
                  f"W / start seed sd {w['W'] / d['start']['ir_seed_sd']:.2f} "
                  f"(seed-specific {w['W_seed_specific'] / d['start']['ir_seed_sd']:.2f}); "
                  f"control mean over its checkpoints {100 * w['control_mean_over_checkpoints']:.2f}%"
                  + (f", W / control seed sd {w['W'] / d['control']['ir_seed_sd']:.2f}"
                     if "control" in d else ""))
    print("-" * 104)
    done = [f for f in FAMS if "anneal" in sm[f] and "control" in sm[f]]
    verdicts = {}
    if len(done) == len(FAMS):
        a = [sm[f]["anneal"]["ir_seed_sd"] <= 0.7 * sm[f]["control"]["ir_seed_sd"] for f in FAMS]
        verdicts["P9.1a"] = (sum(a), "HOLDS" if sum(a) >= 3 else "FAILS")
        b = [sm[f]["anneal"]["ir_mean_abs"] <= 0.85 * sm[f]["control"]["ir_mean_abs"] for f in FAMS]
        sn = sm["single_notch"]["anneal"]["ir_mean_abs"] <= 0.75 * sm["single_notch"]["control"]["ir_mean_abs"]
        verdicts["P9.1b"] = (sum(b), bool(sn), "HOLDS" if sum(b) >= 3 and sn else "FAILS")
        if all("wander" in sm[f] for f in FAMS):
            c = [sm[f]["wander"]["W"] >= 0.5 * sm[f]["start"]["ir_seed_sd"] for f in FAMS]
            verdicts["P9.1c"] = (sum(c), "HOLDS" if sum(c) >= 3 else "FAILS")
        d_ = [abs(sm[f]["anneal"]["oor_mean_abs"] / sm[f]["control"]["oor_mean_abs"] - 1) < 0.2
              for f in ("open_hole", "double_notch")]
        verdicts["P9.1d"] = (sum(d_), "HOLDS" if all(d_) else "FAILS")
        for f in FAMS:
            x, y = sm[f]["anneal"], sm[f]["control"]
            print(f"{f:<14} seed sd anneal/control {x['ir_seed_sd'] / y['ir_seed_sd']:.2f}   "
                  f"IR |err| anneal/control {x['ir_mean_abs'] / y['ir_mean_abs']:.2f}   "
                  f"OOR |err| anneal/control {x['oor_mean_abs'] / y['oor_mean_abs']:.2f}")
        print(f"P9.1a seed sd >=30% lower (anneal vs control) on {verdicts['P9.1a'][0]}/5 -> {verdicts['P9.1a'][1]}")
        print(f"P9.1b IR |err| >=15% lower on {verdicts['P9.1b'][0]}/5, single notch >=25% lower: "
              f"{verdicts['P9.1b'][1]} -> {verdicts['P9.1b'][2]}")
        if "P9.1c" in verdicts:
            print(f"P9.1c control wander >= 0.5 x start seed sd on {verdicts['P9.1c'][0]}/5 -> {verdicts['P9.1c'][1]}")
        print(f"P9.1d OOR within +-20% (open hole, double notch): {verdicts['P9.1d'][0]}/2 -> {verdicts['P9.1d'][1]}")
    else:
        print(f"verdicts pending: complete families {done}")
    return {"summary": sm, "verdicts": verdicts}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    r = json.load(open(OUT)) if a.report else score_all()
    out = report(r)
    json.dump(out, open("verification/results/phase9/p91_summary.json", "w"), indent=1,
              default=lambda o: o if not isinstance(o, (np.bool_,)) else bool(o))

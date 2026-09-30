#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 9.1 -- anneal against control (docs/phase9_optimizer_plan.md, 0f65e5c).

Continues an existing per-family operator for 800 steps with the Tier-1 family
loop (training/family_trainer.py) unchanged in everything but the learning
rate:

  * both arms restart Adam (the Tier-1 checkpoints hold no optimiser state;
    the dog-bone 6.7 checkpoints do, and it is deliberately discarded so every
    run restarts the same way) with a 50-step linear warm-up 3e-5 -> 3e-4;
  * ``anneal``: cosine 3e-4 -> 3e-6 over the remaining 750 steps;
  * ``control``: constant 3e-4;
  * ReduceLROnPlateau is not built, in either arm;
  * the batch stream continues the original one: ``default_rng(42)`` advanced
    past the 100 shuffles that 1600 steps x 4 geometries consumed;
  * the quadrature stream is a fresh stratified draw from ``EnergyLoss(seed=1)``
    (replaying the original generator's consumption is not practical); the
    same for every run, so the two arms see identical data;
  * checkpoints ck200 / ck400 / ck600 / ck800, history.json, and result.json
    last as the completion marker.

    python -m training.phase9_continue --family open_hole --seed 0 --arm anneal
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from config import TRAINING_CONFIG
from training.family_trainer import UD, _sets, build_loss, family_model
from training.manifest import set_deterministic, set_precision

ROOT = "verification/results/phase9/p91"
STEPS, WARM = 800, 50
LR_PEAK, LR_WARM0, LR_END = 3e-4, 3e-5, 3e-6
QUAD_SEED = 1
ORIG_STEPS, BATCH, BANK = 1600, 4, 64


def checkpoint(fam, seed):
    """The model each continuation starts from."""
    if fam == "dogbone":
        if seed == 0:
            return "verification/results/fixed_long/oracle-energy/last.pt"
        return f"verification/results/retest/energy_s{seed}/oracle-energy/last.pt"
    return f"verification/results/phase8/tier1/{fam}_s{seed}/last.pt"


def load_start(fam, seed):
    path = checkpoint(fam, seed)
    if fam == "dogbone":
        from verification.retest import load as retest_load
        return retest_load("energy", path)
    from training.family_trainer import load
    return load(fam, path)


def lr_at(k, arm):
    if k < WARM:
        return LR_WARM0 + (LR_PEAK - LR_WARM0) * (k + 1) / WARM
    if arm == "control":
        return LR_PEAK
    t = (k - WARM) / max(1, STEPS - WARM - 1)
    return LR_END + 0.5 * (LR_PEAK - LR_END) * (1.0 + math.cos(math.pi * t))


def run(fam, seed, arm, out_dir=None, threads=1, chunk=25, steps=STEPS):
    assert arm in ("anneal", "control")
    out_dir = out_dir or os.path.join(ROOT, f"{fam}_s{seed}_{arm}")
    torch.set_num_threads(threads)
    set_precision("float32")
    set_deterministic(seed)
    model = load_start(fam, seed).train()
    loss_fn = build_loss(fam)
    loss_fn._rng = np.random.default_rng(QUAD_SEED)
    bank, val, extent = _sets(fam)
    assert len(bank) == BANK
    opt = torch.optim.Adam(model.parameters(), lr=lr_at(0, arm))
    clip = TRAINING_CONFIG["grad_clip_norm"]
    rng = np.random.default_rng(42)
    for _ in range(ORIG_STEPS * BATCH // BANK):     # advance past the original run
        rng.shuffle(list(range(BANK)))
    dt = torch.get_default_dtype()
    u_d = torch.tensor([UD], dtype=dt)
    bpc = torch.zeros(1, 1, 2, dtype=dt)
    os.makedirs(out_dir, exist_ok=True)
    order, ema, hist = [], None, []
    t0 = time.time()

    def geo_loss(p):
        L, Hs = extent(p)
        return loss_fn(model, p, bpc, u_d, torch.tensor([L], dtype=dt),
                       torch.tensor([Hs], dtype=dt))

    for k in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(k, arm)
        if k % chunk == 0:
            model.eval()
            v = float(np.mean([(lambda o: o["L_energy_log"] + o["L_barrier_log"])(geo_loss(p))
                               for p in val]))
            ema = v if ema is None else 0.9 * ema + 0.1 * v
        model.train()
        opt.zero_grad()
        if len(order) < BATCH:
            order = list(range(BANK))
            rng.shuffle(order)
        idx = [order.pop(0) for _ in range(BATCH)]
        tot, e_sum, jmin = 0.0, 0.0, np.inf
        for i in idx:
            o = geo_loss(bank[i])
            (o["loss"] / BATCH).backward()
            tot += float(o["loss"].detach()) / BATCH
            e_sum += o["L_energy_log"] / BATCH
            jmin = min(jmin, o["min_J3D_log"])
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
        opt.step()
        if (k + 1) % chunk == 0 or k == 0:
            rec = {"step": k + 1, "loss": tot, "L_energy": e_sum, "min_J3D": jmin,
                   "val_ema": ema, "val": v, "lr": opt.param_groups[0]["lr"],
                   "grad_norm": gn, "wall_min": (time.time() - t0) / 60.0}
            hist.append(rec)
            print(f"  [{fam} s{seed} {arm}] step {k + 1:4d}  loss {tot:.6e}  "
                  f"val {v:.6e}  lr {rec['lr']:.2e}  |g| {gn:.2e}  "
                  f"{rec['wall_min']:.1f} min", flush=True)
        if (k + 1) % 200 == 0:
            torch.save({"model_state_dict": model.state_dict(), "step": k + 1,
                        "family": fam, "seed": seed, "arm": arm},
                       os.path.join(out_dir, f"ck{k + 1}.pt"))
            json.dump(hist, open(os.path.join(out_dir, "history.json"), "w"))
    wall = time.time() - t0
    json.dump({"family": fam, "seed": seed, "arm": arm, "steps": steps,
               "start": checkpoint(fam, seed), "wall_min": wall / 60.0,
               "s_per_step": wall / steps, "final_loss": hist[-1]["loss"]},
              open(os.path.join(out_dir, "result.json"), "w"), indent=2)
    print(f"  [{fam} s{seed} {arm}] done in {wall / 60:.1f} min", flush=True)
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", required=True, choices=["anneal", "control"])
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.family, a.seed, a.arm, out_dir=a.out, steps=a.steps)

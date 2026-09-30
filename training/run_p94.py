#!/usr/bin/env python3
"""
The energy method from scratch with the anneal built in (Phase 9.4-9.6).

    # the paper recipe (Phase 9.5/9.6): 4,800 steps, Adam 3e-4 for 3,200 then cosine
    python -m training.run_p94 double_notch 0 --steps 4800 --t-const 3200 \
        --root verification/results/phase9/p94x

    # Phase 9.4: 2,400 steps (1,600 at 3e-4, then cosine)
    python -m training.run_p94 double_notch 0

Families: dogbone, open_hole, inclusion, double_notch, single_notch.  Seeds 0-2
were used for the reported results.  Monitors on the family's two validation
geometries (energy gap to the Richardson-extrapolated FEM energy, K_t) run
every ``--monitor-every`` steps; checkpoints every 400.
"""
import argparse

from training.family_trainer import _sets
from training.formulation_trainer import train
from verification.phase10.monitors import Monitor

ROOT = "verification/results/phase9/p94"


def run(fam, seed, steps=2400, t_const=1600, monitor_every=400):
    _, val, _ = _sets(fam)
    mon = Monitor("B", fam, list(val), [f"val{i}" for i in range(len(val))])
    return train("B", fam, seed, f"{ROOT}/{fam}_s{seed}", steps=steps, batch=4, lr=3e-4,
                 lr_end=3e-6, warmup=0, t_const=t_const, clip=1.0, monitor=mon,
                 monitor_every=monitor_every, ckpt_every=400, log_every=25)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("seed", type=int)
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--t-const", type=int, default=1600,
                    help="steps at the constant rate before the cosine anneal")
    ap.add_argument("--monitor-every", type=int, default=None)
    ap.add_argument("--root", default=ROOT, help="output directory root")
    a = ap.parse_args()
    ROOT = a.root
    run(a.family, a.seed, steps=a.steps, t_const=a.t_const,
        monitor_every=a.monitor_every or (800 if a.steps > 2400 else 400))

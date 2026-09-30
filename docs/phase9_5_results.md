# Phase 9.5 — results: a longer anneal on the notch families

Pre-registration: `docs/phase9_5_plan.md` (`00a392c`).
- Scores: `verification/results/phase9/p94x_scores.json`.
- Summary and verdicts: `docs/phase9_5_summary.json`.
- Runs: `verification/results/phase9/p94x/`.

**Recipe.** As 9.4, but 4,800 steps: Adam 3e-4 for 3,200, then cosine to 3e-6
over 1,600.
- Everything else is 9.4's: bank, batch order, seeds, scorer, and the same
  monitors (every 800 steps rather than 400).
- Each run is paired with the 9.4 run of the same seed, and the two share the
  first 1,600 steps exactly. In all six pairs, `ck1600.pt` is bit-identical,
  every logged line and monitor to step 1,600 matches, and the batch indices
  match for all 2,400 steps. The runs diverge from step 1,625.

**Runs.**
- Double notch seeds 0 (the pilot), 1 and 2.
- Single notch seeds 0, 1 and 2.
- The predictions are judged on the confirmatory runs: double notch seeds 1–2
  and all single-notch seeds.

## Results

In-range and out-of-range K_t, 9.4 (2,400 steps) → 9.5 (4,800 steps). The
paired difference has a 95% two-way cluster-bootstrap interval, as in 9.4.

| family | seeds | in-range \|err\| | paired d, 95% CI | out-of-range \|err\| | paired d, 95% CI |
|---|---|---|---|---|---|
| double notch | 1–2 (confirmatory) | 3.26 → **2.54%** | −0.72 [−1.74, +0.06] (borderline) | 19.35 → **11.52%** | **−7.83 [−10.96, −5.37]** |
| double notch | 0–2 (incl. pilot) | 2.92 → **2.31%** | −0.61 [−1.48, +0.18] | 19.30 → **12.56%** | **−6.74 [−10.26, −3.56]** |
| single notch | 0–2 (confirmatory) | 3.61 → **2.20%** | **−1.41 [−2.52, −0.40]** | 14.44 → **11.33%** | **−3.11 [−5.46, −0.42]** |

- A bold interval is a resolved difference.
- **Borderline.** Across random streams, the double notch's confirmatory
  in-range upper end moves between +0.04 and +0.06, with 3.5–4% of resamples
  ≥ 0.
- **Two seeds.** With only two seeds, the seed resampling under-represents
  seed variance, so the confirmatory double-notch intervals are
  anti-conservative. The out-of-range call does not depend on this: 16 of 16
  geometry-seed pairs improve.
- **Geometry-seed pairs that improve, in range:** 17/24 (double notch,
  confirmatory), 24/36 (double notch, all seeds), 25/36 (single notch).

| family (3 seeds) | R² | bias | seed sd | per-seed in-range \|err\| | per-seed out-of-range \|err\| | von Mises / v rel. L2 | wall-clock |
|---|---|---|---|---|---|---|---|
| double notch, 9.4 | +0.943 | −0.21% | 2.21% | 2.23, 4.03, 2.50% | 19.2, 18.1, 20.6% | 1.67 / 4.99% | ~2.2 h |
| double notch, 9.5 | **+0.955** | +0.60% | 1.67% | 1.84, 2.90, 2.18% | 14.7, 10.1, 12.9% | 1.34 / 3.01% | 4.2–4.4 h |
| single notch, 9.4 | +0.911 | −0.72% | 4.72% | 2.69, 3.83, 4.30% | 17.7, 7.9, 17.8% | 1.55 / 5.47% | ~2.1 h |
| single notch, 9.5 | **+0.963** | +0.17% | 2.96% | 0.99, 3.10, 2.51% | 13.2, 7.5, 13.2% | 1.19 / 3.13% | 4.1–4.3 h |

**Validation energy gap** (to the Richardson FEM energy, mean of the two
validation geometries), at the end of each run:

| family | seed | 9.4 (2,400 steps) | 9.5 (4,800 steps) |
|---|---|---|---|
| double notch | 0 | 0.144% | 0.081% |
| double notch | 1 | 0.105% | 0.043% |
| double notch | 2 | 0.157% | 0.081% |
| single notch | 0 | 0.055% | 0.023% |
| single notch | 1 | 0.057% | 0.023% |
| single notch | 2 | 0.061% | 0.025% |

- The gap falls by 44–60% in every run: 9.5 ends at 0.41–0.56 of the 9.4
  value.
- It is still falling at step 4,800 in every run. From step 4,000 to 4,800,
  for example, the double notch goes 0.061 → 0.043% and the single notch
  0.035 → 0.023%.

## Verdicts

- **P9.5-1 (in-range error lower at 4,800): holds on both families.**
  - Single notch: 3.61 → 2.20%, resolved.
  - Double notch, confirmatory seeds: 3.26 → 2.54%, not resolved (upper end
    +0.06). With the pilot seed: 2.92 → 2.31%, also not resolved.
  - Every seed's mean improves in both families (6 of 6, in and out of range).
    Per geometry-seed pair, 17/24, 24/36 and 25/36 improve.
- **P9.5-2 (validation energy gap lower): holds in every confirmatory run**, and
  in the pilot.
- **P9.5-3 (double-notch out-of-range error lower): holds, and is resolved.**
  - 19.35 → 11.52% on the confirmatory seeds; 16 of 16 pairs improve.
  - With the pilot included, 23 of 24 pairs improve.
- **Unpredicted: the single notch's out-of-range error also falls**, 14.44 →
  11.33%, resolved.

**Decision rule** (registered): P9.5-1 holds on both families, so **the
4,800-step schedule is recommended for the notch families.**

## What this changes

1. **The notch rows of the paper table should come from 9.5.**
   - Double notch 2.31% (R² +0.955); single notch 2.20% (R² +0.963). 9.4 had
     2.92% and 3.61%.
   - The table then states the schedule per family, or all families move to
     4,800 steps (see 3).
2. **Part of the out-of-range error was under-training.**
   - Phase 9.1 concluded that extrapolation "is not an optimizer problem".
     - Its evidence (P9.1d) compared anneal and control at equal length.
       Their out-of-range ratio was 1.02; length was not tested.
     - That conclusion was too strong: doubling the length reduces the error.
   - The cut is −40% on the double notch (confirmatory seeds; −35% with the
     pilot) and −22% on the single notch.
   - It remains 5.4× the in-range error on the double notch (from 6.6×) and
     5.2× on the single notch (from 4.0×, because its in-range error fell
     faster).
   - The remaining error is consistent across seeds, not seed noise.
     - The single notch's out-of-range errors are all negative in both 9.4 and
       9.5 (bias −14.4% → −11.3%).
     - The double notch's errors are concentrated on two geometries (3-seed
       means +40% and +22%).
     - Whether more training reduces them further is untested. The energy gap
       is still falling at step 4,800.
3. **Open question: do the other families gain too?** Their validation energy
   gaps at 2,400 steps were already ~0.011–0.013%, 5–12× smaller than the
   notches', so less is expected. An exploratory open-hole seed-0 run at 4,800
   steps is reported below.
4. **Cost:** 4.1–4.4 h per run on the container, against ~2.1–2.2 h.

Every number above was recomputed by an independent check. Re-scoring the six
checkpoints reproduces the score file exactly, and its corrections are folded
in.
- **Pre-registration timeline.** The plan was committed at 10:57:43; the first
  confirmatory claim (double notch s2) followed 52 ms later.
- **Double notch seed 1** was running when the plan was committed (about step
  2,050). Its `result.json` and checkpoint appeared only at 13:29. Before the
  commit it had shown only monitors identical to 9.4's.

## Addendum: open-hole pilot at 4,800 steps (seed 0, exploratory)

Run after 9.5 finished, and not registered.

| | 9.4 seed 0 (2,400 steps) | 4,800 steps |
|---|---|---|
| in-range \|err\| | 1.59% | **1.26%** (9 of 12 geometries better) |
| out-of-range \|err\| | 8.89% | **6.98%** (8 of 8 better) |
| validation energy gap | 0.0116% | 0.0046% |

The open hole gains too, although its energy gap at 2,400 steps was already
small. This motivates moving every family to the longer schedule: 9.6,
registered separately.

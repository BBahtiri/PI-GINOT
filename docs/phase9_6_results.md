# Phase 9.6 — results: the 4,800-step schedule for every family, and the paper table

Pre-registration: `docs/phase9_6_plan.md` (`745c463`).
- Scores: `verification/results/phase9/p94x_scores.json` (shared with 9.5).
- Summary and verdicts: `docs/phase9_6_summary.json`.

**Recipe** (as 9.5): the energy form, from scratch, Adam 3e-4 for 3,200 steps,
then cosine to 3e-6 over 1,600 (4,800 steps). Bank of 64, batch 4.

**Pairing.** Each run is paired with the 9.4 run of the same seed (2,400
steps; the first 1,600 steps are identical).

**Runs.**
- dog-bone seeds 0–2;
- open hole seeds 1–2 (seed 0 was the exploratory pilot of the 9.5 addendum);
- inclusion seeds 0–2.

## Results against 9.4 (2,400 steps)

In-range and out-of-range mean |K_t err|, paired 95% two-way
cluster-bootstrap intervals (as in 9.4 and 9.5):

| family | seeds | in-range: 2,400 → 4,800 | paired d, 95% CI | out-of-range: 2,400 → 4,800 | paired d, 95% CI |
|---|---|---|---|---|---|
| dog-bone | 0–2 | 1.28 → 1.01% | **−0.28 [−0.51, −0.05]** | 2.47 → 2.00% | −0.47 [−2.42, +1.37] |
| open hole | 1–2 (confirmatory) | 1.72 → 1.30% | −0.42 [−0.99, +0.16] | 9.66 → 7.68% | **−1.99 [−2.71, −1.10]** |
| open hole | 0–2 (incl. pilot) | 1.68 → 1.29% | −0.39 [−0.82, +0.02] (borderline) | 9.40 → 7.45% | **−1.96 [−2.61, −1.28]** |
| inclusion | 0–2 | 1.61 → 1.75% | +0.14 [−0.34, +0.83] | 1.18 → 0.85% | −0.33 [−0.87, +0.26] |

- A bold interval is a resolved difference.
- **Borderline.** For the open hole, all seeds, in range: across random
  streams the upper end runs from +0.002 to +0.022, with 2.5–3.1% of resamples
  ≥ 0.
- **Two seeds.** The confirmatory open-hole intervals use only two seeds, so
  they are anti-conservative (as in 9.5). P9.6-2 rests on 16 of 16 pairs
  improving, 24 of 24 with the pilot.
- **Dog-bone out of range.** The drop is a point estimate and even at the pair
  level: 12 of 24 pairs improve. Seed 1 gets worse (0.90 → 2.29%); the drop
  comes from seed 2's 9.4 outlier (5.01 → 2.53%).

**Geometry-seed pairs that improve:**
- dog-bone: 26/36 in range, 12/24 out of range;
- open hole: 16/24 in range and 16/16 out of range (confirmatory); 25/36 and
  24/24 with the pilot;
- inclusion: 19/36 in range, 13/24 out of range.

**Validation energy gap** falls from 0.011–0.014% to 0.004–0.006% in every
run.

## Verdicts

- **P9.6-1 (open-hole in-range lower): holds** on the point estimate: 1.72 →
  1.30%, not resolved.
- **P9.6-2 (open-hole out-of-range lower): holds, and is resolved.** 9.66 →
  7.68%, and 16 of 16 pairs improve.
- **P9.6-3 (dog-bone and inclusion not resolved worse in range): holds.**
  - The dog-bone is in fact **resolved better**, 1.28 → 1.01%, which the plan
    did not predict.
  - The inclusion is slightly worse on the point estimate (+0.14 points), and
    unresolved.
- **P9.6-4 (validation energy gap lower): holds in every confirmatory run.**

Per the plan's rule ("if P9.6-3 holds, the paper table becomes the 4,800-step
results for all five families"), **the paper table is the 4,800-step table
below.**

## The paper table: energy method, 4,800 steps, 3 seeds, held-out geometries

| family | in-range \|K_t err\| | R² | bias | seed sd | predict-the-mean | out-of-range \|err\| | in-range u / v / von Mises rel. L2 | wall-clock per run |
|---|---|---|---|---|---|---|---|---|
| dog-bone | **1.01%** | +0.85 | −1.01% | 0.44% | 2.43% | 2.00% | 0.05 / 0.76 / 0.46% | 4.4–4.8 h |
| open-hole plate | **1.29%** | +0.95 | −1.18% | 0.67% | 5.94% | 7.45% | 0.18 / 0.99 / 1.32% | 3.9 h |
| bonded rigid inclusion | **1.75%** | −0.12 | −1.75% | 0.58% | 1.60% | 0.85% | 0.05 / 0.50 / 0.60% | 4.0–4.5 h |
| double-edge U-notch | **2.31%** | +0.96 | +0.60% | 1.67% | 12.28% | 12.56% | 0.25 / 3.01 / 1.34% | 4.2–4.4 h |
| single-edge U-notch, clamped | **2.20%** | +0.96 | +0.17% | 2.96% | 15.86% | 11.33% | 0.21 / 3.13 / 1.19% | 4.1–4.3 h |

- 12 in-range and 8 out-of-range held-out geometries per family, scored against
  the Level-3-verified FEM references.
- "Predict-the-mean" is the in-range error of always predicting the mean
  reference K_t.
- Wall-clock is on the container's CPU, one thread, two runs sharing two cores.
- Open hole, dog-bone and inclusion come from 9.6 (with the open-hole pilot as
  its seed 0); the notches come from 9.5. Every family uses the same recipe.

**Reading.**
- **In range, four of five families are skilful:** 2.4–7.2× below predicting
  the mean, with R² 0.85–0.96.
- **The inclusion is the exception, because its K_t barely varies.**
  - K_ref spans only 1.41–1.50 (CV 1.9%). Its bias (−1.75%; all 36
    geometry-seed errors negative, from −0.5% to −4.3%; scatter 0.84%) is
    larger than that spread, so predicting the mean (1.60%) beats it.
  - A one-signed bias alone is not what sets it apart: the dog-bone is also
    36/36 negative at 4,800 steps (bias −1.01%).
  - The bias persists with longer training (−1.54 → −1.75%, unresolved). So do
    the smaller biases of the dog-bone (−1.16 → −1.01%) and open hole
    (−1.24 → −1.18%); their gains come mainly from less scatter. The
    inclusion's scatter also shrinks (seed sd 0.96 → 0.58%).
  - The network's peak lies on the loading axis (θ < 5°) in 36 of 36 inclusion
    rows (32 of 36 at 2,400 steps). That is consistent with Phase 8's
    finding about the interface peak.
- **Against 2,400 steps,** the longer schedule:
  - is lower in range on 4 of 5 families (resolved on the dog-bone and single
    notch);
  - is lower out of range on all 5 point estimates (resolved on the open
    hole, double notch and single notch; the dog-bone's is 12 of 24 pairs).
- **Out of range is still the largest error for the hole and notches:** 7.5%
  (open hole), 11–13% (notches), consistent across seeds (see 9.5). The
  inclusion is the exception (0.85% out of range).

## Integrity

- **Reproducibility.** A container restart interrupted inclusion seeds 1 and 2
  (at steps ~2,175 and ~1,825); they were re-run from scratch.
  - Every logged step, loss, J, lr and |g| matches the re-runs (88 and 74
    lines), as do the monitors at steps 800 and 1,600.
  - The checkpoints are bit-identical: ck400–ck2000 (seed 1) and ck400–ck1600
    (seed 2).
  - The interrupted directories are kept as `_interrupted_*`.
- **Pairing.** ck1600 is bit-identical to 9.4's in all 9 pairs.
- **Timeline.** The plan was committed before any confirmatory run started.
  The open-hole pilot ran before the commit, as disclosed.
- **Scoring.** Scored with the same scorer as 9.4 and 9.5, whose integrity check
  reproduces a stored 9.1 score exactly. The independent check re-scored three
  4,800-step checkpoints and reproduced the stored scores exactly.
- **Independent check.** It recomputed every number in both tables (all
  reproduce), and its prose corrections are folded in above.
- **Predict-the-mean.** It is mean |K_mean/K_ref − 1| over the 12 in-range
  geometries, as in Phase 8 (`verification/family_score.py`).

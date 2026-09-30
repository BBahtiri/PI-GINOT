# Phase 9.4 — results: the energy method from scratch, with the anneal built in

Pre-registration: `docs/phase9_4_plan.md` (`7748070`), committed before any
run; git confirms the plan predates the first run and is unchanged. Every number
was recomputed by an independent check; its corrections are folded in.
- Scores: `verification/results/phase9/p94_scores.json`.
- Summary and verdicts: `docs/phase9_4_summary.json`.
- Runs: `verification/results/phase9/p94/`.

## The recipe

- **Model:** the oracle-conditioned operator, 573k parameters, float32; the
  family's hard Dirichlet layer.
- **Loss:** the energy form (stratified resampled quadrature, barrier 1e3).
- **Data:** bank of 64, batch 4, batch order from `default_rng(42)`.
- **Schedule:** Adam 3e-4 for 1,600 steps, then cosine to 3e-6 over 800
  (2,400 steps); clip 1.0.
- **Seeds:** 0, 1, 2 × five families, from scratch.

## Results: in-range K_t (12 held-out geometries × 3 seeds per family)

| family | Tier 1 | 9.1 anneal (continuation) | 9.4 (from scratch) | 9.4 − Tier 1, 95% CI | 9.4 − 9.1, 95% CI |
|---|---|---|---|---|---|
| dog-bone | 1.51% | 1.25% | 1.28% | −0.23 [−1.00, +0.42] | +0.03 [−0.13, +0.26] |
| open hole | 3.36% | 1.47% | 1.68% | −1.68 [−3.35, −0.24] ✓ | +0.20 [−0.04, +0.50] (borderline) |
| inclusion | 1.75% | 1.66% | 1.61% | −0.14 [−0.35, +0.13] | −0.04 [−0.44, +0.29] |
| double notch | 4.09% | 2.74% | 2.92% | −1.17 [−3.49, +0.41] | +0.18 [−0.36, +0.62] |
| single notch | 6.79% | 3.55% | 3.61% | −3.18 [−5.41, −0.99] ✓ | +0.05 [−0.50, +0.57] |

- Differences are in percentage points of mean |K_t err|.
- The intervals are paired 95% two-way cluster-bootstrap intervals (12
  geometries × 3 seeds, 10,000 resamples). ✓ marks a resolved difference.
- **Borderline.** Across independent random streams, the open hole's
  9.4 − 9.1 lower end moves between −0.05 and −0.03. About 5% of resamples fall
  below 0, so it is close to "resolved worse". No other call is close.

| family | R² | bias | seed sd | out-of-range \|err\| | u / v / von Mises rel. L2 | wall-clock per run |
|---|---|---|---|---|---|---|
| dog-bone | +0.75 | −1.16% | 0.65% | 2.47% | 0.08% / 1.2% / 0.63% | 118–122 min |
| open hole | +0.90 | −1.24% | 1.28% | 9.40% | 0.24% / 1.4% / 1.4% | 105–109 min |
| inclusion | −0.03 | −1.54% | 0.96% | 1.18% | 0.07% / 0.9% / 0.68% | 125–128 min |
| double notch | +0.94 | −0.21% | 2.21% | 19.30% | 0.35% / 5.0% / 1.7% | 129–130 min |
| single notch | +0.91 | −0.72% | 4.72% | 14.44% | 0.37% / 5.5% / 1.55% | 121–126 min |

- **Inclusion R².** The inclusion's K_ref varies little across geometries
  (CV 1.9%), so its R² is dominated by a uniform −1.5% bias. With that bias
  removed, R² would be +0.69.
- **Section force** (N_err, mean |·|): 0.25% (open hole), 0.17% (inclusion),
  0.37% (double notch) and 0.21% (single notch). The dog-bone scorer does not
  compute it.
- **The v error is heavy-tailed on the notches:** mean 5.0% and 5.5% against
  medians 3.5% and 3.1%.

- The clipped fraction is 0 in every run. The grad norm has a median of 9e-5,
  with a range from 4e-6 to 3.8e-3 (at step 1), far below 1.0.
- Wall-clock is measured with two runs sharing the container's two cores, 1
  thread each.

## Verdicts on the predictions

- **P9.4-1 (better than Tier 1): does not hold as registered.**
  - 9.4 is lower than Tier 1 on 5 of 5 families.
  - It is resolved lower on only 2: the open hole and the single notch. The
    prediction asked for ≥3.
  - **Power.** The same test applied to 9.1 against Tier 1 also resolves only
    the same two families: open hole −1.89 [−3.59, −0.43] and single notch
    −3.24 [−5.38, −1.25]. That check was run after registration, as a check of
    the scoring code.
  - The two-way design (3 seeds × 12 geometries) gives wide intervals, and on
    the double notch the width comes mostly from how much geometries differ:
    - resampling geometries alone gives [−2.75, +0.21];
    - resampling seeds alone gives [−2.22, −0.32];
    - the per-geometry differences run from −7.5 to +3.0 points.
  - The ≥3 threshold was too demanding for this design.
- **P9.4-2 (no worse than the 9.1 continuation): holds.**
  - No family is resolved worse.
  - The point estimates are within +0.20 points. 9.4 is slightly higher on 4
    of 5 families (+0.03, +0.20, +0.18, +0.05; the inclusion is −0.04).
  - The open hole (+0.20, 14% relative) is borderline.
  - Any penalty for restarting the optimiser is below this design's
    resolution.
- **P9.4-3 (extrapolation not fixed by the schedule): holds.**
  - Out-of-range error is 4.0–6.6× the in-range error on the open hole (9.4%),
    double notch (19.3%) and single notch (14.4%). It is unchanged from 9.1.
  - Against Tier 1, out-of-range error is resolved lower on the open hole
    (12.30 → 9.40%, −2.89 [−3.91, −1.51]), the dog-bone (−1.20 [−2.23, −0.27])
    and the inclusion (−0.61 [−1.14, −0.15]).
  - It is not resolved on the double notch (+2.05 [−3.11, +6.28]) or the
    single notch (−0.33 [−4.06, +2.63]).

**Exploratory: validation-geometry energy gap** (to the Richardson FEM energy,
the mean of the two validation geometries and three seeds):

| family | gap at step 400 | gap at step 2,400 | \|K_t err\| on the validation geometries |
|---|---|---|---|
| dog-bone | 0.17% | 0.013% | 1.6% |
| open hole | 0.22% | 0.011% | 3.7% |
| inclusion | 0.12% | 0.011% | 1.6% |
| double notch | 1.06% | 0.136% | 4.3% |
| single notch | 0.95% | 0.058% | 3.5% |

- At the end, the notch families are further from the energy minimum than the
  other three: the single notch 4.6–5.2×, the double notch 10.7–12.1×.
- The double notch's gap comes mostly from one of its two validation
  geometries: 0.19–0.29%, against 0.023–0.026% on the other.
- Over the last four monitors (steps 1,200 to 2,400) the three-seed mean is
  still falling: 0.34 → 0.20 → 0.19 → 0.14%.
  - Seeds 0 and 1 fall steadily.
  - Seed 2 ends where it was at step 1,600.
- This fits the notch families being the ones with room left, as Phase 9 found
  on single geometries.

## Integrity

- **Scorer.** It reproduces the stored 9.1 score `open_hole|0|anneal|800`
  exactly (max difference 0.0).
- **Reproducibility.** A container restart interrupted the single-notch seeds
  0 and 1 after their last logged steps, 1,975 and 1,825. They were re-run from
  scratch.
  - Every logged line of the interrupted runs matches the re-runs: 80 and 74
    lines of loss, J, lr and |g|.
  - The monitor gaps, the K_t errors (steps 0–1,600) and the batch indices
    are bit-identical.
  - The interrupted directories are kept as `_interrupted_*`.
- **Scores.** The independent check re-scored all 15 `last.pt` files and
  reproduced `p94_scores.json` exactly.
- **Pairing.**
  - For the four new families, seed s of 9.4 shares its initialisation and
    batch stream with Tier-1 seed s. Step-1 losses match Tier 1 to ≤5e-6
    relative (inclusion 3e-4).
  - The dog-bone's Tier 1 (and so its 9.1 start) is the 6.7 retest models,
    trained in a different loop (`training/trainer.py`), so its pairing is by
    index only.
  - Per-geometry K_t errors of 9.4 and of the 9.1 run of the same seed
    correlate at 0.85–0.98 (Pearson on signed error, pooled over 36 pairs per
    family; single seeds go as low as 0.73). Across different seeds the
    correlation averages only 0.19–0.60, which is why the seed pairing
    matters.
  - Where seeds differ materially, their ranking carries over: on the double
    notch, seed 1 is the worst in both runs (4.03% in 9.4, 4.25% in 9.1). On
    the open hole and the inclusion the per-seed means are nearly tied.
- **Not re-tuned.** B's settings are Phase 9.1's (Phase 10's S6 did not run).

## What this means

1. **9.4 is the paper's multi-geometry table.** It is a clean from-scratch
   protocol that matches the 9.1 continuation within this design's
   resolution:

   | | dog-bone | open hole | inclusion | double notch | single notch |
   |---|---|---|---|---|---|
   | in-range \|err\| | 1.28% | 1.68% | 1.61% | 2.92% | 3.61% |
   | in-range R² | +0.75 | +0.90 | −0.03 | +0.94 | +0.91 |

2. **Against Tier 1, the gain is resolved on two families** (open hole,
   single notch) and directionally consistent on all five.
3. **The remaining in-range error sits on the notches**, where the energy is
   also least converged.
4. **Extrapolation is not fixed.**
   - Out-of-range error stays 4–6.6× the in-range error on the hole and notches
     (9–19%), unchanged from 9.1. It is the largest error in this table.
   - Against Tier 1 it falls on the open hole (and the dog-bone and inclusion),
     but not on the notches.

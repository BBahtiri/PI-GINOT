# Phase 9 — results

**Provenance.**
- **Plan and predictions:** `docs/phase9_optimizer_plan.md`, committed as
  `0f65e5c` before any run.
- **Measurement code**, committed before its results: `9eff1ed` and `e514856`.
  The 9.1 runs started 24 s before `9eff1ed`.
- **Deviations**, each committed before its own results: `f258bec` and
  `4a76e1b`.
- **Exploratory, not pre-registered:** the SWA script (`37c09fd`), and the
  refined-rule re-evaluation of 9.0b, added after the independent check.

**Status:** **9.0b, 9.1 and 9.2 complete**, run 2026-09-24 in the container
(2 CPUs). An independent check re-derived every number. It found one
measurement problem (9.0b) and several overreaching sentences, all corrected
below (§ Corrections).

**Archived numbers:**
- 9.0b: `docs/phase9_energy_error.json`, `docs/phase9_energy_error_refined.json`
- 9.1: `docs/phase9_p91_scores.json`, `docs/phase9_p91_summary.json`
- SWA: `docs/phase9_p91_swa.json`
- 9.2: `docs/phase9_p92_report.json`, `docs/phase9_p92_selection.json`,
  `docs/phase9_p92_fem_levels.json`

---

## Summary

**Annealing the learning rate is the lever that worked.** Quasi-Newton polishing
of the kind the two papers recommend did not work here.

**9.1: anneal against a constant-rate control.** All four registered
predictions hold.
- 800 more steps with the rate decayed to 3e-6 improved in-range `K_t`
  accuracy on every family.
- The anneal beat both the Tier-1 models and the constant-rate control on
  every family.

| family | in-range \|err\|: Tier 1 → anneal (control) | anneal / Tier 1 | anneal / control | seed sd: Tier 1 → anneal | R²: Tier 1 → anneal | predict-the-mean |
|---|---|---|---|---|---|---|
| dog-bone | 1.51 → **1.25%** (1.73%) | 0.83 | 0.72 | 0.93 → 0.77% | +0.61 → **+0.76 ± 0.03** | 2.43% |
| open hole | 3.36 → **1.47%** (1.78%) | 0.44 | 0.83 | 3.13 → 1.08% | +0.60 → **+0.91 ± 0.01** | 5.94% |
| inclusion | 1.75 → **1.66%** (2.00%) | 0.95 | 0.83 | 1.09 → 0.95% | −0.18 → −0.14 ± 0.51 | 1.60% |
| double notch | 4.09 → **2.74%** (6.62%) | 0.67 | 0.41 | 3.99 → 2.74% | +0.85 → **+0.94 ± 0.04** | 12.28% |
| single notch | 6.79 → **3.55%** (7.90%) | 0.52 | 0.45 | 9.74 → 4.67% | +0.67 → **+0.91 ± 0.05** | 15.86% |

**What the numbers show:**
- **More steps alone** explain most of the open-hole gain: the control reaches
  1.78% there. On the two notch families the gain is the cooling, since the
  control ends worse than Tier 1.
- **The recipe costs 50% more steps than Tier 1** (2,400 against 1,600).
- **At the constant rate, the operator's `K_t` error wanders**: by 1.4–7.2
  points (sd) between checkpoints 200 steps apart.
  - The Tier-1 models were snapshots taken during that wander.
  - Annealing removed most of the Tier-1 seed variance on the open hole (88%)
    and single notch (77%), about half on the double notch (53%), and about a
    quarter to a third on the dog-bone (31%) and inclusion (24%).

**9.2: L-BFGS, as run here, did not work.**
- With fixed quadrature points it fit the points, and the true energy rose.
- With points redrawn every 10 iterations it stalled or went backwards.
- The network has about 573k parameters against 5–8k points per step, a ratio
  of about 70–110. The paper setting where the same resampling works has a
  ratio of about 0.5–8 (Jnini et al., Table 10).
- SSBFGS/SSBroyden were not run. They are excluded by memory (2.6 TB), not by
  this test.

**9.2: Adam with fresh points and a decaying rate drove single geometries most
of the way to the true energy minimum.**
- The gap fell 3.6–17×, and the `K_t` error fell to 1.2–1.5% (final values).
- Whether that is a floor is **not established**. The energy was still falling
  at the end: the gap shrank by 27–47% over the last five checks (400
  steps). `K_t` moved by 0.3–1.4 points between checks late in the anneal.

**9.0b: unresolved at its threshold.**
- On the registered energy rule, the lower-energy seed was the more accurate
  in 59.0% of pairs, against a prediction of >60%.
- On a rule refined once (same polygon), four of the 144 orderings flip and
  the figure is 61.8%. The rule's own integration error, up to 1.8e-4, is as
  large as the energy differences between seeds.

**Out of range**, the anneal matched the control within 20% on the open hole
and double notch, as predicted. It was worse than the control on the single
notch (13.8% against 9.4%, Tier 1 14.8%).

---

## 9.0b — Does a lower energy mean a better `K_t`?

**Method.**
- The 12 Tier-1 models (four families × 3 seeds) were evaluated on the 12
  in-range scored geometries.
- The energy used the registered rule: the n_r 60 FEM reference mesh with the
  6-point Dunavant rule, never used in training.
- For each geometry and seed pair, the test asked whether the lower-energy seed
  is also the more accurate.

| family | registered rule: lower energy is more accurate | refined rule (h/2, same polygon) | network energy above FEM(h), median | Spearman(gap, \|err\|) within family |
|---|---|---|---|---|
| open hole | 24 / 36 (67%) | 24 / 36 | 1.1e-4 | +0.48 (p 0.003) |
| inclusion | 20 / 36 (56%) | 22 / 36 | 8.4e-5 | +0.12 (p 0.49) |
| double notch | 16 / 36 (44%) | 17 / 36 | 1.9e-3 | −0.11 (p 0.54) |
| single notch | 25 / 36 (69%) | 26 / 36 | 8.2e-4 | +0.17 (p 0.31) |
| **pooled** | **85 / 144 (59.0%)** | **89 / 144 (61.8%)** | | |

**Registered verdict (>60% on the registered rule): FAILS.**
- Refining the evaluation mesh once changes Π_net by up to 1.0e-4 (open hole)
  and 1.8e-4 (double notch).
- That is as large as the seed-to-seed energy differences being ranked.
- Four orderings flip, all toward the prediction, and the refined rule gives
  61.8%.
- The independent check found the same four flips hold at a second refinement.

The registered rule was not fine enough for this test, so **the question is
unresolved at the 60% threshold**, not answered "no".

**What does hold:**
- The link is present on the open hole (Spearman +0.48) and single notch.
- It is absent on the double notch.
- The pairs are not independent (one seed is lowest-energy on 9 of 12
  geometries in two families), so binomial p-values (0.018; 0.003 refined)
  overstate the significance and are not used.

**The FEM(h) baseline.** "FEM(h)" is the n_r 60 reference. Its own energy sits
2.0–7.9e-4 above the extrapolated exact energy (9.2), so the gaps in this table
are to that mesh, not to the exact minimum.

---

## 9.2 — One geometry, driven down its own energy

**Geometries.** The selection rule was applied to the seed-0 anneal checkpoint
(`docs/phase9_p92_selection.json`):
- open hole g10 (\|err\| 5.53%) and g1 (2.74%);
- double notch g0 (5.95%) and g5 (2.23%).

Selecting the largest errors builds in some regression toward the mean.

**Reference energy.** A nested refinement (same polygon) at h, h/2 and h/4.
The FEM energy converges at rate 1.90–1.92 on all four geometries. The
Richardson-extrapolated energy is taken as the exact minimum on the polygon.

**Guard.** It never fired: no network energy fell below Π(h/2) − δ.

**Tracked energies.** They use the 9.0b rule. On these four geometries its own
error is about 4e-6 to 1e-5: small next to the starting gaps, but 15–25% of the
smallest final ones.

| geometry | arm | \|`K_t` err\|, start → end | last 5 checks: mean \|err\| (sd of signed err) | energy gap to exact, start → end |
|---|---|---|---|---|
| open hole g10 | Adam, fresh points, anneal | 5.53 → 1.20% | 1.46% (0.66) | 5.3e-4 → 1.3e-4 |
| | L-BFGS, 10-iteration windows | 5.53 → 4.06% | 2.52% (3.05) | 5.3e-4 → **6.3e-4** (rose) |
| open hole g1 | Adam | 2.74 → 1.53% | 1.75% (0.32) | 1.6e-4 → 4.3e-5 |
| double notch g0 | Adam | 5.95 → 1.19% | 1.62% (1.41) | 4.2e-4 → 2.5e-5 |
| | L-BFGS, 10-iteration windows | 5.95 → 8.02% | 7.24% (2.26) | 4.2e-4 → 1.9e-4 |
| | L-BFGS, 100-iteration windows (registered) | 5.95 → −2.10% at it 250, stopped | — | 4.2e-4 → **7.8e-4** (rose) |
| double notch g5 | Adam | 2.23 → 1.30% | 1.22% (0.41) | 3.8e-4 → 1.1e-4 |

### The L-BFGS arms

**The registered arm (100-iteration windows) was void.**
- It used fixed points within each window. Within 50 iterations the window loss
  fell while the energy on the independent rule *rose*, reaching +7e-4 by
  iteration 100.
- The `K_t` error swung 5.9 → 0.8 → 7.8 → 11.4 → −2.1%.
- The network, with 573k parameters against about 8k points × 4 gradient
  components, fits the points rather than the energy.
- **The run.** Only one run of this arm produced data: double notch g0, which
  ran unchunked. It started before the chunking fix `f258bec`, and survived
  because the other three were OOM-killed.
- **Stopping it.** It was stopped at iteration 250, and the chunked reruns of
  the other three were cancelled (`4a76e1b`).
- **The chunking check.** It was run interactively and is recorded in the
  `f258bec` message: loss identical, gradient to 3e-15.

**The replacement arm (fresh points every 10 iterations, history reset) did not
work either.** It is post hoc, and labelled as such. The cadence is the one
Jnini et al. use for SSBroyden.
- **Double notch g0.** The gap reached about 2.5e-4 by iteration 150, then
  drifted between 1.9 and 2.6e-4. `K_t` wandered between 3.4 and 9.3%.
- **Open hole g10.** It *raised* the energy.
- **Cancellations.** Two of the arm's four runs (open hole g1, double notch g5)
  were cancelled at about 07:14 UTC. The pilot's trajectory, at iteration 900,
  was the only basis; the second run had not started.

**The registered verdict (H-opt or H-obj on ≥3 of 4 deterministic runs) cannot
be issued.**
- The arm it rests on neither minimised the energy nor ran on four geometries.
- On the one geometry it did run, the registered prediction (L-BFGS reaching
  ≤1%) was not supported.

### The Adam arm

The Adam arm starts from the same checkpoint and draws fresh stratified points
at every step, with cosine decay to 3e-6.
- **Energy gap.** Cut **3.6–17×** on every geometry.
- **`K_t` error.** By final value it fell **1.7–5×**, to **1.2–1.5%**. Averaged
  over the last five checks, it is 1.2–1.75%.

**Post hoc**, applying the registered thresholds to this arm:
- **H-opt: 0 of 4.** The error falls ≥3× on two geometries, but none ends ≤1%.
- **H-obj: 0 of 4.** Every error falls below 2/3 of its start.

This is "in between", but not by the registered route.

**Interpretation (post hoc):**
- Driving the true energy down took the error from 2.2–5.9% to about 1.2–1.75%.
  How much of that is the energy and how much the learning rate going to zero
  is not separated: early in the double notch g0 run, \|err\| reached 10.8%
  while the energy was below its start.
- Whether 1.2–1.75% is a floor is **open**.
  - The energy was still falling: the gap shrank by 27–47% over the last five
    checks (400 steps). The plan's H-obj standard of <1e-6 change over 200
    iterations was not met: the change was 4e-6 to 2.2e-5 over the last 200
    steps.
  - At learning rates of 3e-5 to 3e-6, `K_t` still moves by 0.3–1.4 points (sd)
    between checks.
  - The reference resolves `K_t` to 0.5% (Phase 8, check M).
- By final value the double notch g0, with the smallest gap, is the most
  accurate. By the last-five average it is not.

---

## 9.1 — Anneal against control

**Setup.** Every one of the 15 Tier-1 and 6.7 checkpoints was continued for 800
steps in both arms.
- **Unchanged:** the family loop, bank, batch, quadrature rule and float32.
- **Quadrature stream:** a fresh random stream (seed 1), identical in every run.
- Adam was restarted with a 50-step warm-up.
- **Anneal:** cosine decay to 3e-6.
- **Control:** constant 3e-4, checkpoints every 200 steps.
- The scorers and held-out sets were unchanged.
- **Checks passed:**
  - scoring a start checkpoint reproduces its stored score exactly;
  - both arms saw identical data (matching step-1 loss and validation energy
    in all 15 pairs).
- Each run took 36–44 min; the 30 runs took about 10 h on two lanes.

| family | arm | IR \|err\| | seed sd | 3-seed average \|err\| | bias | R² | OOR \|err\| |
|---|---|---|---|---|---|---|---|
| dog-bone | Tier 1 | 1.51% | 0.93% | 1.40% | −1.34% | +0.61 ± 0.22 | 3.67% |
| | anneal | **1.25%** | **0.77%** | 1.18% | −1.12% | **+0.76 ± 0.03** | 2.36% |
| | control | 1.73% | 1.08% | 1.52% | −1.50% | +0.54 ± 0.13 | 2.49% |
| open hole | Tier 1 | 3.36% | 3.13% | 3.35% | −3.35% | +0.60 ± 0.24 | 12.30% |
| | anneal | **1.47%** | **1.08%** | 1.26% | −0.91% | **+0.91 ± 0.01** | 9.35% |
| | control | 1.78% | 1.58% | 1.53% | +0.15% | +0.91 ± 0.02 | 9.20% |
| inclusion | Tier 1 | 1.75% | 1.09% | 1.66% | −1.66% | −0.18 ± 0.29 | 1.80% |
| | anneal | **1.66%** | 0.95% | 1.66% | −1.66% | −0.14 ± 0.51 | 1.16% |
| | control | 2.00% | 0.86% | 1.97% | −1.97% | −0.67 ± 0.62 | 1.43% |
| double notch | Tier 1 | 4.09% | 3.99% | 3.52% | +1.55% | +0.85 ± 0.10 | 17.26% |
| | anneal | **2.74%** | **2.74%** | 2.43% | +0.11% | **+0.94 ± 0.04** | 19.09% |
| | control | 6.62% | 4.48% | 6.47% | −6.12% | +0.71 ± 0.25 | 18.67% |
| single notch | Tier 1 | 6.79% | 9.74% | 3.22% | +0.11% | +0.67 ± 0.30 | 14.77% |
| | anneal | **3.55%** | **4.67%** | 2.07% | −0.23% | **+0.91 ± 0.05** | 13.80% |
| | control | 7.90% | 11.94% | 3.68% | −2.71% | +0.51 ± 0.64 | 9.39% |

**Verdicts** (all anneal against control, on the plan's exact criteria):

| prediction | result | verdict |
|---|---|---|
| **P9.1a**: seed sd ≥30% lower on ≥3 of 5 | ratios 0.72 / 0.69 / 1.10 / 0.61 / 0.39; three pass, the minimum. The open hole passes at 0.688 and the dog-bone misses at 0.72, so this is fragile at three seeds | **HOLDS** |
| **P9.1b**: IR \|err\| ≥15% lower on ≥3 of 5, single notch ≥25% | ratios 0.72 / 0.83 / 0.83 / 0.41 / **0.45**; five of five | **HOLDS** |
| **P9.1c**: control wander W ≥ 0.5 × the Tier-1 seed sd on ≥3 of 5 | W = 1.66 / 1.75 / 1.36 / 5.83 / 7.21%, i.e. 1.79 / 0.56 / 1.25 / 1.46 / 0.74 × the seed sd; five of five | **HOLDS** |
| **P9.1d**: OOR within ±20% on the open hole and double notch | ratios 1.02 / 1.02 | **HOLDS** |

**P9.1c, refined.** The seeds share one data stream, so part of W moves in step
across seeds and cannot create seed spread. That shared part is large on the
inclusion. Counting only the seed-specific part, the ratios are
1.61 / 0.44 / 0.56 / 1.51 / 0.75. P9.1c would still hold (4 of 5), but the open
hole's and inclusion's ratios shrink.

**What the control shows.**
- It is **worse than Tier 1 on 4 of 5 families**, even averaged over its four
  checkpoints: dog-bone 1.70, inclusion 1.89, double notch 6.74 and single
  notch 8.66%, against 1.51, 1.75, 4.09 and 6.79%. The open hole is the
  exception (2.43 against 3.36%).
- Two things could explain that:
  - chance (three seeds);
  - a penalty from restarting Adam, which the anneal arm would then partly
    recover from.
- **Anneal against control can therefore overstate the effect of cooling.**
  Anneal against Tier 1 is free of that penalty, and the anneal still wins on
  all five families: 0.83 / 0.44 / 0.95 / 0.67 / 0.52.
- The checkpoints wander. The double notch seed 0 reads 8.9, 3.9, 10.0 and 3.3%
  mean error at steps 200, 400, 600 and 800.
- **Validation energy.** The anneal reaches the lower value than the control in
  all 15 seed pairs.
  - Relative to Tier 1, the smoothed value falls by 3.9e-4 (open hole), 3.3e-4
    (inclusion), 1.8e-3 (double notch) and 1.2e-3 (single notch).
  - For the control the falls are 1.4e-4, 2.6e-4, 9.9e-4 and 6.9e-4.
  - The Tier-1 comparison crosses quadrature streams, so it carries a small,
    unquantified offset.

**What annealing did not fix.**
- **The inclusion** still does not beat predicting the mean: R² −0.14, and
  1.66% against 1.60%. It beats the constant on 2 of 3 seeds. Its problem is the
  misplaced interface peak (Phase 8).
- **Out of range**, the anneal is within 20% of the control on the hole and
  double notch.
  - It is worse than the control on the single notch (13.8% against 9.4%).
  - The dog-bone's and inclusion's out-of-range falls relative to Tier 1 come
    mostly from the extra steps: the control fell too (2.49%, 1.43%).

**Exploratory, not pre-registered: SWA.** Averaging the control's four
checkpoints (stochastic weight averaging, at no training cost) does less than
annealing.

| family | control | SWA | anneal |
|---|---|---|---|
| dog-bone | 1.73% | 1.40% | 1.25% |
| open hole | 1.78% | 2.02% (worse) | 1.47% |
| inclusion | 2.00% | 1.69% | 1.66% |
| double notch | 6.62% | 5.83% | 2.74% |
| single notch | 7.90% | 6.85% | 3.55% |

Not a substitute.

---

## What this means for the optimizer question

1. **Adopt the anneal as the default schedule.** It improves every family in
   range, against both Tier 1 and the matched control.
   - As run it is 50% more steps than Tier 1. Whether a from-scratch schedule
     of the same length does as well is 9.4.
   - The paper's multi-geometry table should come from 9.4, not from these
     continuations.
2. **Do not pursue L-BFGS polishing in the form tested.**
   - With fixed points it fits the quadrature.
   - Redrawn every 10 iterations, it loses to Adam.
   - This is shown for L-BFGS only, on one or two geometries, at one point
     count. SSBFGS/SSBroyden are excluded by memory, not by test.
   - Untested remedies: more points per window, full-bank windows, overlapping
     multi-batch L-BFGS.
   - The regime difference is itself worth a sentence in the paper: about 70–110
     parameters per quadrature point here, against about 0.5–8 where the papers'
     resampled SSBroyden works (Jnini et al., Table 10).
3. **The energy natural gradient (9.3 iii)** remains untested.
   - It draws fresh points each step.
   - Its line search on one sample carries the same risk of fitting the sample.
   - Comparing the operator after annealing with single-geometry Adam suggests
     room of about 2–3× on the notches:
     - double notch: 2.74% against about 1.2–1.6% on its two worst geometries;
     - single notch: no single-geometry data.
   - The comparison suggests little room on the hole: 1.47% against 1.2–1.75%.
   - This is not a bound: it sets family means against worst-case geometries
     whose energy was still falling.
   - Per the (post hoc) "in between" rule, it would start as a one-family,
     one-seed pilot, on a GPU.
4. **The cheaper next step is 9.4.** Retrain the five families from scratch with
   the anneal built in, possibly longer. This gives a clean protocol for the
   paper and tests whether a longer anneal closes the remaining notch gap.
5. **Below about 1.5%, whether the optimizer or the objective limits is open.**
   On single geometries the energy was still falling when the runs ended. The
   objective-side levers stay in reserve: adaptive quadrature at the peak and
   the Phase 7 C-series on the free edges.
6. **Extrapolation is not an optimizer problem.** On the hole and double notch
   the out-of-range error barely moved, as the plan predicted.

---

## Deviations from the plan

1. **9.2 arm (a), `f258bec` and `4a76e1b`.**
   - Two of the first attempts were OOM-killed. The loss is now accumulated in
     four chunks.
   - The arm was then stopped as void: the quadrature was being exploited. Its
     only data are from the double notch g0 run, which was unchunked.
   - It was replaced post hoc by 10-iteration windows. Two of those four runs
     were cancelled on the pilot's trajectory.
   - `4a76e1b` also moved the H-opt/H-obj computation to the replacement arm.
2. **9.2 ran before most of 9.1**, to decide earlier whether 9.3 is worth a GPU.
3. **The 9.2 Adam arm** is tracked every 100 steps, not every 50, and has a
   50-step warm-up that is not in the plan.
4. **9.0b** evaluated the 12 models of the four new families. The plan said
   "all 15 final models", but its test pools only those four families. A
   refined-rule re-evaluation was added after the check.
5. **9.1** used a fresh quadrature random stream (seed 1). The rule itself was
   unchanged.

## Corrections from the independent check (before this version)

**Changed:**
- **9.0b**, from "FAILS, narrowly; only weakly linked" to "FAILS on the
  registered rule; unresolved at the threshold". The rule's own error is as
  large as the effect, and the refined rule gives 61.8%.
- **The cancellation of two 9.2 runs.** It rested on the pilot alone, not on
  "the pilot and the second run".

**Toned down:**
- "roughly in half" (the effect sizes are now given against both baselines);
- "the seed spread was largely this wander";
- "quasi-Newton does not transfer" (now "L-BFGS as tested"), and the regime
  claim;
- "below 1.5% the limit is not the optimizer" (now open);
- the "at most 2×" bound for 9.3 (now "about 2–3×, not a bound");
- "out of range nothing changed".

**Fixed numbers:**
- FEM(h) above the exact energy: 2.0–7.9e-4, not 3–8e-4.
- Run times: 36–44 min, not 36–39.

**Added:**
- the possible Adam-restart penalty;
- the seed-specific wander;
- the unlisted deviations.

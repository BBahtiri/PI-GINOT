# Phase 9.4 — the energy method from scratch, with the anneal built in (pre-registered)

Committed before any 9.4 run. The recipe is Phase 9.1's winner, and this is also
arm B of Phase 10 (S7), which is now run alone (Phase 10 closed after S5).

## Recipe

- **Model:** the oracle-conditioned operator, 573k parameters, float32; the
  family's hard Dirichlet layer (the shipped layer for the dog-bone).
- **Loss:** the energy form. It is `training.family_trainer.build_loss`,
  unchanged, called through `training.formulation_trainer` arm B:
  - stratified resampling on the ~1,400-triangle quadrature mesh;
  - barrier 1e3 at J = 0.05; E_scale = E.
- **Data:** bank of 64 per family, batch 4, batch order from
  `default_rng(42)` (as Tier 1).
- **Schedule:** Adam at 3e-4 for 1,600 steps, then cosine to 3e-6 over 800
  (2,400 steps). No plateau scheduler; grad-norm clip 1.0.
- **Seeds:** 0, 1, 2. The same seed gives the Tier-1 initialisation and the same
  batch stream, so for the four new families 9.4 differs from Tier 1 only in
  the schedule.
- **Families:** dog-bone, open hole, inclusion, double notch, single notch
  (15 runs).
- **Monitors:** every 400 steps, on the two validation geometries: the energy
  gap to the Richardson-extrapolated FEM energy, and K_t.
- **Checkpoints:** every 400 steps.

B was not re-tuned (Phase 10's S6 did not run). Its settings are those Phase 9.1
validated.

## Scoring

Scored exactly as Tier 1 and 9.1 (`verification.phase9_score._score`):
- the four new families by `verification.family_score` against the cached
  FEM references, on 12 in-range and 8 out-of-range geometries;
- the dog-bone by the 6.7 scorer on its held-out sets.

Integrity check: the scorer must reproduce one stored 9.1 score exactly
(`open_hole|0|anneal|800`).

**Statistics.**
- Per family, 36 paired differences in |K_t err| (12 geometries × 3 seeds)
  between 9.4 and a reference.
- The interval is a 95% two-way cluster bootstrap (geometries and seeds
  resampled independently, 10,000 resamples, seed 0).
- A difference is **resolved** when the interval excludes 0.

References:
- **Tier 1:** the Phase 8 scores; for the dog-bone, the 6.7 retest energy
  models.
- **9.1 anneal:** the Tier-1 checkpoints plus 800 annealed steps, ck800.

## Predictions

- **P9.4-1 (better than Tier 1).** In-range mean |K_t err| is lower than
  Tier 1's on ≥4 of 5 families, and resolved lower on ≥3.
- **P9.4-2 (no worse than the 9.1 continuation).** 9.4 is resolved worse than
  the 9.1 anneal on no family.
  - Same length. No optimiser restart, and no plateau-scheduler phase.
- **P9.4-3 (extrapolation not fixed by the schedule).** Out-of-range mean
  |K_t err| exceeds 2× the in-range mean on the open hole, double notch and
  single notch.

**Exploratory:**
- R², bias, u / v / von Mises L2 error, N_err;
- wall-clock per run;
- the validation energy gap over training;
- the clipped fraction.

## Budget

B costs 2.8 s per step at batch 4 (Phase 10 G5), so about 1.9 h per run and
about 28 CPU-h in total. That is about 14–15 h on the container's two lanes.

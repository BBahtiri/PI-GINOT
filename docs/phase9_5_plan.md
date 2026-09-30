# Phase 9.5 — a longer anneal on the notch families (pre-registered)

**Why.** After 9.4 the remaining in-range error sits on the notches (double
notch 2.92%, single notch 3.61%). They are also the families furthest from the
energy minimum at step 2,400: validation energy gap 0.14% and 0.06%, against
~0.01% elsewhere.

**Pilot, not part of the test.** Double notch seed 0 at 4,800 steps, run while
9.4 finished:
- in-range 2.23% → 1.84% (better on 7 of 12 geometries);
- out-of-range 19.2% → 14.7%;
- validation energy gap still falling at step 4,800.

**Recipe.** As 9.4, but 4,800 steps: Adam 3e-4 for 3,200, then cosine to 3e-6
over 1,600. Monitors every 800 steps. Same bank, batch order, seeds and scorer.
Seed s pairs with 9.4 seed s: same initialisation, and the same batch stream
for the first 2,400 steps.

**Runs.**
- Double notch seeds 1 and 2. Seed 1 was started before this file, but none of
  its scores had been looked at.
- Single notch seeds 0, 1 and 2.
- The pilot (double notch s0) is reported with the family's results, but it is
  not used to judge the predictions.

**Predictions** (confirmatory runs only; paired with 9.4, same seed and
geometries):
- **P9.5-1.** In-range mean |K_t err| is lower at 4,800 than at 2,400 on the
  single notch (3 seeds), and on the double notch (seeds 1–2). Point estimates;
  paired intervals are reported.
- **P9.5-2.** The validation energy gap at 4,800 is below its 9.4 value at
  2,400 in every confirmatory run.
- **P9.5-3.** Double-notch out-of-range mean |err| is lower at 4,800 (seeds 1–2).
  No prediction for the single notch's out-of-range error.

**Decision rule.** If P9.5-1 holds on both families, the longer schedule is
recommended for the notch families. The paper table then states the schedule
per family. If it holds on neither, 2,400 steps stays the recipe.

**Cost.** About 4.3 h per run; 5 runs is about 21 CPU-h, 10–11 h on two lanes.

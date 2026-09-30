# Phase 9.6 — the 4,800-step schedule for the remaining families (pre-registered)

**Why.**
- 9.5 showed that doubling the schedule improves both notch families in
  range: double notch 2.92 → 2.31%, single notch 3.61 → 2.20%. It also cuts
  their out-of-range error by 22–40%.
- An exploratory open-hole seed-0 run shows the same direction: 1.59 → 1.26%
  in range, 8.89 → 6.98% out of range.
- To give the paper one schedule for every family, the other families get the
  same schedule.

**Recipe.** As 9.5: 4,800 steps, Adam 3e-4 for 3,200, then cosine to 3e-6
over 1,600. Everything else is 9.4's. Runs are paired with the 9.4 run of the
same seed.

**Runs** (8):
- dog-bone seeds 0, 1, 2;
- open hole seeds 1, 2 (seed 0 is the pilot: reported, not used for the
  verdicts);
- inclusion seeds 0, 1, 2.

**Predictions** (confirmatory runs only; paired with 9.4):
- **P9.6-1.** Open-hole in-range mean |K_t err| is lower at 4,800 (seeds 1–2).
- **P9.6-2.** Open-hole out-of-range mean |err| is lower at 4,800 (seeds 1–2).
- **P9.6-3.** The dog-bone and the inclusion are not resolved worse in range.
  No improvement is predicted: their errors are bias-dominated (inclusion) or
  already near the Phase 9 single-geometry level (dog-bone).
- **P9.6-4.** The validation energy gap is lower at 4,800 in every
  confirmatory run.

**Use.** If P9.6-3 holds, the paper table becomes the 4,800-step results for
all five families. 9.5 supplies the notches; the table states the protocol
once.

**Cost.** About 4.3 h per run; 8 runs is about 34 CPU-h, ~17–18 h on two
lanes.

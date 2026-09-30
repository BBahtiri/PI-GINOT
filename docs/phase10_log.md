# Phase 10 — implementation log and gate record

Plan: `docs/phase10_formulations_plan.md`, commit `b04bb66`. This log records
each gate's outcome, and every deviation from the plan, as it happens.

## S0 / G0a — reference library

**Environment.** A Python 3.10 venv (`/home/claude/venvs/fv310`) with:
- `fastvpinns==1.0.2`;
- `tensorflow-cpu==2.13.0` (the plan said `tensorflow==2.13.0`; the CPU
  build is the same code without the CUDA wheels);
- `numpy==1.23.5`, `pandas==1.4.3`, `meshio==5.3.4`, `gmsh==4.11.0`.

The gmsh wheel needed the system libraries `libglu1-mesa`, `libxcursor1`,
`libxinerama1`, `libxft2` and `libxrender1` (installed with apt).

**Deviation.** The v1.0.2 examples are not in the wheel or the sdist, and the
GitHub archive is not reachable from this session. As the plan's fallback,
the reference is the library's own API
(`verification/phase10/g0_library.py`), run on the paper's benchmark:
- u = −sin(2πx) sin(2πy) on the unit square;
- 2×2 cells, 15 test functions per direction, 40×40 Gauss–Legendre points;
- 3×30 tanh, Adam at 1e-3, β = 10, float32, 20,000 steps.

**G0a: PASS.**
- Relative L2 3.08e-3, max 1.16e-2, in 184 s.
- The paper reports errors of order 3e-3 at 100,000 iterations (Fig. 8).

## S1 / G0b — the port against the library, float64

**PASS**, with the criterion amended. Two meshes: the library's internal 4×4
rectangles (K = 5, Q = 6), and a randomly distorted 6×6 quad mesh read from a
`.mesh` file (K = 4, Q = 5).

| quantity | internal | distorted |
|---|---|---|
| quadrature points | 5.6e-17 | 0 |
| test values, x- and y-gradients (w·det J folded in) | ≤2.1e-15 | ≤1.3e-15 |
| forcing matrix | 1.6e-15 | 4.7e-16 |
| residual matrix, from the library's own network derivatives | 1.6e-15 | 4.7e-16 |
| loss, from the same | 2.5e-15 | 8.0e-16 |

**The amendment.** The plan asked for the residual and loss to agree to 1e-10
with each framework evaluating the network from copied weights.
- End to end, they agree to 4e-9 (residual) and 2.2e-10 (loss).
- That gap is **TensorFlow's**: its float64 network output differs from a
  plain numpy float64 evaluation of the same weights by 3.5e-8. PyTorch's
  differs by 9e-16.
- So the port is exact, and the gate now checks the port in isolation: the
  contraction fed the library's own derivatives, to 1e-12. It also checks that
  PyTorch's network equals numpy's.

## S1 / G0c — the port trains the benchmark

**PASS.** On the same problem and settings as G0a, PyTorch float32 gives:
- relative L2 3.09e-3, max 1.22e-2, in 177 s;
- the library, 3.08e-3 in 184 s;
- a ratio of 1.002. The threshold was ≤2.

## S2 / G1 — weak-form meshes of every geometry

**PASS.** Five families × (64 bank + 2 validation + 12 in-range + 8
out-of-range) = 430 geometries, each at two levels: B's budget (843–2,318
triangles) and coarse (264–524). All of these hold on every geometry:

| check | worst case |
|---|---|
| det J > 0 at every 6×6 Gauss point | min/max det ratio ≥ 1.1e-5, all positive |
| Σ w·det J against the polygon area | ≤ 4.4e-16 |
| boundary topology | every boundary node of degree 2; one closed loop; every edge named |
| outward normals: \|∮ n ds\| and ∮ x·n ds − 2A | ≤ 6e-17 and ≤ 5e-16 |
| quad half-edges tile the boundary | ≤ 6.4e-16 |

Also on every geometry, the Dirichlet DOFs derived from the segment flags equal
`fem_dirichlet`'s set.

**Reported, not gated.** The largest turn between consecutive chord normals is
0.131 rad on the hole, notch and inclusion arcs, and 0.034–0.076 on the
dog-bone fillet. The chord normal is off the arc normal by at most half that.

## S3 / G2 — the weak-form losses are the weak form

**PASS on all five families**, float64, at B's mesh level.

| sub-gate | result (worst over families) |
|---|---|
| (a) patch test, homogeneous F₀ | C1 bubble residuals ≤1.3e-14 (relative to P·\|K\|^½). C1n edge residuals equal the exact edge traction work to 5.3e-15. C2 P1/P2 and C2I boundary residuals equal ∮(P N_h)·φ to ≤3.1e-14. |
| (b) FEM equivalence, C2 P1, smooth non-equilibrium P1 field | against `assemble`'s internal force at 1,268–2,283 free DOFs: ≤1.3e-15 |
| (c) MMS (sympy field; b = −Div P*; polygon-normal tractions on every natural component) | C1 (Q = 2/4/6): 0.84 → 4e-9 → 1e-13. C2 P1 (Dunavant 2/4/6): 2e-5 → 8e-9 → 4e-13. C2 P2: 8e-4 → 8e-7 → 5e-11. |
| (d) R = d/dε Π(u + εv) under the same rule | ≤3e-15 (C2); ≤1e-12 (C1) |
| (e) loss gradients against central differences (h = 1e-4) | ≤1.6e-8 of the largest gradient, all four arms |

**Criterion fixes (not loosened).** The first run failed (d) and (e) through
normalisation, not through the maths. Both are now relative to the scale of
the set being tested.
- **(d)** Some bubble residuals are ≈0 by symmetry, so a per-entry relative
  error was pure roundoff.
- **(e)** At h = 1e-6 the losses (~1e-11) gave roundoff-dominated differences.
  Also, parameters with no effect on ∇u (the output bias, for C2I) have zero
  gradient.
- With that, the direct check matches finite differences to 1e-8.

**(f) Null-space probe (reported).**
- **Setup.** Cubic element bubbles s·a_T·27λ₁λ₂λ₃ were added to the P1 FEM
  minimiser on the training triangulation (open hole, double notch).

| s | energy change | C1 loss | C2 P1 loss | C2I loss |
|---|---|---|---|---|
| 1e-4 | +4.9e-2 | 2.5e-3 | 6e-11 | unchanged |
| 1e-3 | +4.9 | 0.25 | 6e-7 | unchanged |

- **C1** sees within-element modes at first order in the residual, like the
  energy.
- **C2 P1** sees them only at second order: ∫_T ∇b = 0 against a constant
  hat gradient. Its loss is about 4e7× less sensitive.
- **C2I is exactly blind:** the bubble vanishes at every interpolation node.
- **Consequence:** C2 and C2I leave within-element oscillations of the network
  unpenalised. The network's own stress is what gets scored, so the
  spurious-solution monitors of S5/S6 matter most for these two arms.

**(g) Test equations per geometry** (B level; coarse level in the JSON), against
573k parameters:

| arm | equations | points |
|---|---|---|
| C1, K = 2 | 29k–54k | 58k–108k |
| C1, K = 4 | 117k–216k | 131k–244k |
| C2 P1 | 1.3k–2.3k | 7k–14k |
| C2 P2 | 5k–9k | 7k–14k |
| C2I | 1.3k–2.3k | 2.5k–4.6k nodes |

At B's level, C1 needs 14–26× B's quadrature points (≈4.2k), which S6's cost
pre-filter will act on.

## S4 / G3 — arm A, the strong form

**PASS.**

**(a) Manufactured solution, every family.** The arm's autograd Div P against
the independent sympy Div P*:
- Div P: ≤1.0e-14 relative.
- Traction P(u*)N against the sympy P*N at every boundary point: ≤1.3e-14.

**Dirichlet-junction exclusion (plan rule 2).**
- Only the inclusion has same-component Dirichlet pieces that meet: two
  junctions, where the arc meets x = 0 and y = 0.
- 0.65% of its interior points are excluded; nothing is excluded elsewhere.

**Boundary categories per family.**

| family | curved free | straight free | mixed |
|---|---|---|---|
| dog-bone | 70 | 46 | 110 |
| open hole | 24 | 80 | 110 |
| double notch | 24 | 88 | 126 |
| single notch | 24 | 172 | 32 |

- "Mixed" means only the natural component is tested (symmetry shear, grip
  transverse).
- The inclusion's arc is fully Dirichlet, so it has no curved free edge.

**(b) Continuity with the preprint.** On the dog-bone, the generalised loss with
the preprint's terms and weights (100 / 20 / 2 / 2 / 1e3) reproduces
`PhysicsLoss` exactly: relative difference 0.0, and every component equal. The
test used 400 interior and 120 boundary points, float64, on the 573k-parameter
operator.

**Note for S6.** The full 4,605-point float64 evaluation of the second-order
graph did not finish in 10 minutes on this container, while 400 points took 4
s. Arm A needs chunked evaluation, or a check that its float32 cost is as
Phase 2 measured, before its runs are costed.

## S5 — End-to-end sanity (in progress)

### Loads for manufactured problems

Every arm now accepts a body force b and a prescribed traction t (A: `b`, `tb`
in `strong_loss`; C: precomputed load vectors `R`, `Re`, `tb` in
`WeakGeometry.loss`; B: Π − ∫b·u − ∫t·u in the G4a driver). Check in float64,
with u* substituted for the network (`g4a_mms_train.build_arm`):

- A, C1, C1n, C2, C2I: loss at u* ≤ 1.3e-20 (A exactly 0), against 2e-7 to
  7e-3 for perturbations of size 1e-3 to 1e-2. So u* is a global minimiser of
  every residual loss.
- B: stationary at u*. The slope along a test direction is 3e-5 to 2.5e-4 of
  the curvature scale, with a sign that changes with the sampling seed: Monte
  Carlo noise, not bias.

### Chunked backward (trainer check)

`training/formulation_trainer.py` backpropagates A in chunks of interior points
(its loss is a sum of per-point terms) and the weak arms in two passes:
1. G = ∇u at every point without a parameter graph, then the loss and dL/dG;
2. G again chunk by chunk with the graph, backpropagating dL/dG.

`verification/phase10/g4_chunking.py` (float64, the operator, open hole and
dog-bone) compares this with one plain autograd pass. For A, B, C1, C1n, C2 P1,
C2 P2 and C2I the loss difference is 0 and the parameter gradient differs by
≤1.03e-15 (relative, max-norm). **PASS.**

### G4a — trained manufactured solution

**Setup.**
- Geometry: the first bank geometry of the open hole and of the double notch.
- Manufactured field: u* = spec.apply(X, N*), with smooth N*, so it
  satisfies the family's Dirichlet data exactly and lies in the network's trial
  space.
- Loads: b = −Div P(u*) and t = P(u*)N_h on every natural component (polygon
  normals).
- Network: a plain 4×64 tanh net with the family's hard Dirichlet layer.
- Training: 4,000 Adam steps, 1e-3 cosine to 1e-5, float32, seed 0.
- Levels: C1/C1n on the coarse level (K = 2, Q = 4); C2 P1 Dunavant 4 and C2I
  on B's triangulation.
- Error: measured on the h/2 refinement with Dunavant 6.

**Registered result: FAIL** (relative H¹ error; pass ≤ 2e-2).

| arm | open hole | double notch |
|---|---|---|
| B | 0.0052 | 0.0064 |
| A | **0.074** | **0.041** |
| C1 | 0.0093 | 0.0167 |
| C1n | **0.041** | **0.048** |
| C2 | 0.0078 | **0.0237** |
| C2I | 0.0080 | **0.0214** |

**Diagnostics** (not part of the gate). The plan says a failure means the
implementation or the conditioning is wrong. The implementation was already
excluded above: every loss is exactly 0 at u*. These runs separate conditioning
from budget:

| diagnostic | open hole | double notch | reading |
|---|---|---|---|
| A, preprint weights, 12,000 steps | 0.048 | 0.028 | not a budget problem |
| A, w_eq 10, w_trac 20 (a corner of A's S6 grid), 4,000 steps | **0.0106** | **0.0053** | the weights |
| C1n, 12,000 steps | **0.0056** | **0.0077** | budget |
| C2, 12,000 steps | — | **0.0069** | budget |
| C2I, 12,000 steps | — | **0.0066** | budget |
| C2 on the h/2 test mesh, 4,000 steps | — | 0.057 | a finer test space is slower |
| C2I on the h/2 base, 4,000 steps | — | 0.052 | same |

- **A (the preprint's weights).** With these weights the traction on the
  natural edges of the trained open-hole field is 5.0% off, about equal to its
  H¹ error. With w_eq 10 and w_trac 20 it is 1.1% off.
  - The error is spread over the domain, weighted slightly towards the free
    edges: 72% of the squared H¹ error lies within 4h of a free edge, which
    covers 65% of the area.
  - So these weights under-enforce the traction conditions on this problem.
    It is a property of the loss's balance, not a code error.
  - The balanced weights are already in A's S6 grid (w_eq {10, 100} × w_trac
    {2, 20}).
- **C1n, C2, C2I** converge with 3× the steps. At the registered budget they
  are slower than B and C1, not wrong.
- **A finer test space** (C2/C2I at h/2) made the error worse at a fixed budget,
  not better. On this problem, resolving the test space is not what limits
  C2/C2I.

**Decision.** No arm is dropped.
- Every failure is conditioning (A's weights) or optimisation budget (C1n,
  C2, C2I on the double notch), not implementation.
- The registered FAIL stands, with these reasons.
- This bears on S6: A's grid must include the boundary-weighted corner (it
  does), and the h/2 test-mesh levels are expected to lose at 800 steps.

### Monitor (ii): made level-consistent (change before any operator run)

At initialisation, the ratio L(h/2)/L(h) of C2, C2I and C1n was already 2.0–2.07
with nothing spurious going on.
- **Cause.** These arms' boundary test functions carry the traction residual as
  a line integral, R_i ≈ ∫(PN − t)φ_i ds ~ Δt·h. The loss normalises it per
  area (R_i²/|supp φ_i| ~ Δt²), and there are ~1/h such DOFs. So the boundary
  share of the loss grows like 1/h, and a boundary-dominated residual doubles
  on h/2 by construction.
- **Change.** The monitor compares L_int + (h_level/h_train)·L_bnd, which
  removes that scaling:
  - L_bnd is the share from DOFs on the boundary polygon (C2, C2I) or from
    the edge functions (C1n);
  - the domain terms and C1's penalty are integrals already and are left as
    they are.
  - At initialisation the corrected ratios are 1.01–1.08 (C1: 1.15).
- **Unchanged.** The training loss itself (the interior/boundary balance is
  what S6's γ factor tunes). The raw ratios are logged next to the corrected
  ones.
- **Code.** `WeakGeometry.loss` now also reports the `L_int` / `L_bnd` split
  (the C2 loss is the same as before to 1.4e-16). The chunking check was
  re-run after the change: PASS.

### G5 cost data: the operator timing pilot

**Method.** Operator, float32, batch 4 on the same 4 open-hole bank geometries
every step (so per-geometry set-up is paid once, as it is amortised in S7), 4
steps. The figure is the median of the last steps. Two single-threaded
processes shared the container's 2 cores. Full data:
`docs/phase10_gates/g5_timing.json`.

| arm, configuration | s/step | × B | peak RSS | S7 estimate (15 × 2,400 steps) |
|---|---|---|---|---|
| B | 2.80 | 1.0 | 1.16 GB | 28 CPU-h |
| A (3 points per triangle) | 32.4 | 11.6 | 1.58 GB | 324 CPU-h |
| C1 / C1n, coarse, K = 2 | 17.3–17.5 | 6.2 | 1.07 GB | 173–175 CPU-h |
| C1 / C1n, coarse, K = 4 | 40.5–40.6 | 14.4 | 1.11 GB | 405 CPU-h |
| C1 / C1n, B's level, K = 2 | 60–82 | 21–29 | 1.10 GB | 600–820 CPU-h |
| C2 P1 or P2, Dunavant 4, B's mesh | 10.6–10.8 | 3.8–3.9 | 1.06 GB | 106–108 CPU-h |
| C2 P1 or P2, Dunavant 6, B's mesh | 19.7–20.8 | 7.0–7.4 | 1.07 GB | 196–208 CPU-h |
| C2 P1, Dunavant 4 or 6, h/2 mesh | 28–57 | 10–20 | 1.08 GB | 285–565 CPU-h |
| C2I, B's mesh | 0.62 | 0.22 | 0.80 GB | 6 CPU-h |
| C2I, h/2 | 2.00 | 0.71 | 1.09 GB | 20 CPU-h |

- Not timed: C2 P2 on h/2, and C1/C1n at K = 4 on B's level. They have at least
  as many points as configurations above that already cost 10× or more; the
  runs were stopped to free the lanes for G4b.
- Peak memory is ≤ 1.6 GB everywhere (chunking works), so only the cost limit
  binds.

**Why the estimates were off.**
- **A is 11.6× B, not 5×.** Div P needs four reverse passes through the
  first-derivative graph of the 573k-parameter operator.
- **C1 coarse K = 2 is 6×, not 3×.** It uses 4.4× B's points (16.8k volume
  points + 1.6k penalty points), and the two-pass backward adds pass 1 (~27%
  of pass 2, profiled).
- **C2 P1 Dunavant 4 is 3.9×, not 1.75×.** 1.75× B's points, the two passes,
  and a slightly higher per-point cost of chunked backward.

**Consequence: the G5 pre-filter as written would remove A and every C1/C1n
configuration.** Its limit is 150 CPU-h per arm in S7. The comparison would then
be B against C2 alone. That contradicts the plan's purpose: A, B, C1 and C2
must all be compared.

The pre-filter is not applied to whole arms. The choice is left open
(hardware, or cheaper variants of A and C1) and is reported to the user. See
the Phase 10 status for the options and budget figures.

### G4b — one geometry, the real problem (open hole and double notch)

**Setup.**
- Geometry: each family's first validation geometry.
- Model: the operator, float32, default settings (`ARM_DEFAULTS`).
- Training: from scratch on that one geometry (batch 1), 1,000 steps: warm-up
  50 steps, then cosine 3e-4 → 3e-6, clip 1.0, seed 0.
- Monitors at steps 0, 250, 500, 750 and 1,000:
  - (i) the energy gap to the Richardson-extrapolated FEM energy (h, h/2, h/4
    of the reference mesh; rates 1.76–1.93 on the five validation
    geometries);
  - (ii) the level-consistent residual ratio;
  - the K_t error and von Mises error against the FEM reference.
- Full data: `docs/phase10_gates/g4b.json`.

| run | s/step | energy gap: start → end | K_t err | vm err | ratio (ii) | gate |
|---|---|---|---|---|---|---|
| B, open hole | 0.74 | 7.3% → **0.074%** | −7.6% | 1.8% | — | PASS |
| B, double notch | 0.82 | 8.0% → 2.33% | −78% | 11% | — | (reference) |
| A, open hole | 7.7 | 7.3% → 4.46% | −64% | 12% | 1.0 | FAIL (60× B) |
| A, double notch | 8.6 | 8.0% → 5.49% | −77% | 14% | 1.0 | "PASS" (2.4× B) — see below |
| C1, open hole | 4.6 | 7.3% → **40.5%** | −61% | 56% | 25 | FAIL |
| C1, double notch | 5.6 | 8.0% → **40.1%** | −75% | 55% | 48 | FAIL |
| C1n, open hole | 4.5 | 7.3% → **39.6%** | −58% | 58% | 13 | FAIL |
| C1n, double notch | 5.5 | 8.0% → **36.0%** | −73% | 55% | 35 | FAIL |
| C2, open hole | 1.7 | 7.3% → 5.96% | −65% | 22% | 86 | FAIL |
| C2, double notch | 2.0 | 8.0% → 17.9% | −76% | 37% | 48 | FAIL |
| C2I, open hole | 0.15 | 7.3% → 6.14% | −64% | 23% | 84 | FAIL |
| C2I, double notch | 0.15 | 8.0% → 18.8% | −76% | 39% | 51 | FAIL |

**Diagnostic runs** (not the gate):

| run | energy gap: start → end | K_t err | vm err |
|---|---|---|---|
| B, double notch, 3,000 steps | 8.0% → **0.024%** | +12% | 1.3% |
| A, w_eq 10 / w_trac 20, open hole | 7.3% → 2.4% (step 250) → 4.58% | −63% | 18% |
| A, w_eq 10 / w_trac 20, double notch | 8.0% → 6.63% | −76% | 22% |

**Reading.**

1. **Every arm starts at the same place: the uniform stretch.** The Dirichlet
   layer's lift is u_d·x/L, and the network's contribution starts small. This
   is a plateau of every loss.
   - B leaves it at about step 250 on the open hole.
   - On the double notch (sharp root, rt 0.30), B has not left it by step
     1,000. It leaves at about step 1,000–1,200 with the 3,000-step schedule,
     then converges (gap 0.024%).
   - So on the double notch, criterion (i) compared against a B that had not
     yet converged. A's "PASS" there is an artefact. Against B at 3,000 steps,
     A's gap is 230× larger, and its K_t is still the uniform-stress value.
2. **A converges to the trivial solution.**
   - The trained open-hole field has von Mises between 18.6 and 19.5 MPa
     everywhere (FEM: 0.9 to 53.9 MPa). It is the homogeneous deformation:
     Div P = 0 exactly, and the section force is right (N err −0.2%).
   - The only cost is the traction term on the hole, which the loss's balance
     tolerates. The loss is flat from step 250.
   - With w_eq 10 / w_trac 20 the loss falls further, but the energy gap rises
     (2.4% → 4.6%). The field lowers the boundary term by giving up
     equilibrium, and still has no concentration.
3. **C1 and C1n are spurious.**
   - Their training residual is small (1e-5), but the energy is 36–40% above
     the true minimum and the von Mises error is 55–58%.
   - The residual on the h/2 test space is 13–48× the training value, and it
     grows over training.
   - At q+2 (same mesh, raised quadrature) the ratio stays ~1.02, so the
     network is not exploiting quadrature points: it lives in the null space
     of the test space.
   - The field is not an oscillation at the scale of the points. Its stress is
     near zero over much of the domain (median |P| 1.6 MPa against B's 17 MPa)
     and concentrated elsewhere, yet its moments against the four bubbles of
     every quad nearly vanish.
4. **C2 and C2I are spurious** on both geometries (ratio 48–86, von Mises error
   22–39%). On the open hole their energy gap falls (to 6%) while the h/2
   residual rises. On the double notch the energy rises (8% → 18%).
5. **The monitors did their job.** Every spurious run is flagged by (ii) by
   step 250. No arm needed the von Mises error to be found out.
6. **Independent recomputation.** The end values were recomputed through the
   Phase 9 code path: `phase9_energy_error.energy_net`, which differs from the
   monitor's in the model call, plus the Tier-1 scorer. The energy gaps agree
   to all printed digits, and the K_t and von Mises errors agree.
   - It adds the section force: B +0.8% / +2.0%; A −0.2% / −0.6%; C1 and C1n
     +55% to +60%; C2 and C2I +18% to +36%.
   - It adds the u displacement error: B 0.6%; C1 and C1n 36–38%; C2 and C2I
     16–28%.
   - The weak arms' fields do not even carry the right load.

**Why the weak arms fail here but passed G4a.**
- G4a's network had 13k parameters against 1.3k–9.7k test equations.
- The operator has 573k parameters against the same equations (G2(g): C1
  coarse ~9k, C2 P1 ~1.3k).
- A weak-form loss with a fixed test space is zero on a large set of fields;
  the network finds one of them before the true solution. The energy is a
  minimum principle with a unique minimiser, and B's quadrature is resampled
  every step, so B has no such null space.
- G2(f) predicted the direction (C2 is ~4e7× less sensitive than the energy
  to within-element modes; C2I exactly blind). The operator shows it.

**G4b verdict: FAIL for A, C1, C1n, C2 and C2I on both geometries; PASS for B.**
- The inclusion, single notch and dog-bone runs are held, pending the budget
  decision below. Their claims are in
  `verification/results/phase10/g4b/claims/*/HELD_for_budget_decision`.

**What this means for S6.**
- The registered S6 would spend ~250 CPU-h, most of it tuning arms that fail
  at default settings. The possible rescues are expensive:
  - C1 at B's level: 21–29× B's cost, with 29k–54k equations, still ≪ 573k
    parameters.
  - C2 on h/2 or P2: 10–20× B.
  - A: more steps at 11.6× B per step.
- Neither the registered pre-filter (G5) nor the registered grids anticipated
  this. The next step needs a decision from the user: hardware, and whether to
  run S6 as registered or a smaller rescue pilot first.

## Phase 10 closed after S5 (user decision, 2026-09-25)

The user chose to stop the comparison at S5 and continue with the energy method.
S6 (tuning pilot) and S7 (main runs of A, C1*, C2*) are **not run**. Nothing
in §5's predictions P10-1 to P10-5 is judged: those were stated for tuned arms
at 2,400 steps on the bank, which this evidence does not reach.

**What Phase 10 established** (S0–S5, qualitative, one seed, default settings):
- The weak-form machinery is correct:
  - the FastVPINNs tensor form agrees with the library to 2.5e-15;
  - the losses pass the patch, FEM-equivalence, manufactured-solution,
    energy-consistency and gradient gates;
  - the strong form reproduces the preprint's loss exactly.
- With a small network (13k parameters), every formulation can learn a smooth
  manufactured solution. The preprint's strong-form weights under-enforce
  traction, and the weak forms converge more slowly than the energy form.
- **With the operator (573k parameters), on the real problem, only the energy
  form (B) converged in 1,000 single-geometry steps.**
  - The strong form (A) converged to the homogeneous stretch, with no stress
    concentration.
  - Every weak form (C1, C1n, C2, C2I) found a spurious field with near-zero
    training residual: von Mises error 22–58%, section force off by 18–60%.
  - This is consistent with 573k parameters against 1.3k–9.7k test equations.
- **Cost per step against B:** A 11.6×, C1/C1n 6.2× (coarse, K = 2), C2
  3.9×, C2I 0.2×.

**Not established.**
- Whether tuning (other weights for A; richer test spaces for C: C1 at B's
  level, C2 P2 or h/2) or much longer training rescues A or the weak forms.
- Any comparison on the inclusion, single notch or dog-bone (G4b held).

**Code kept for reuse.**
- `physics/vpinn/`, `physics/strong_form.py`, `training/formulation_trainer.py`
  (arm B is the energy form with an explicit schedule) and
  `verification/phase10/monitors.py` (the energy gap to the extrapolated FEM
  energy).
- The energy-method work that follows (9.4) uses the trainer's arm B and the
  monitors.

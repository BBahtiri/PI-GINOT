# Phase 10 — Three formulations, one operator: strong form, energy (Ritz), weak form (hp-VPINN / FastVPINNs)

Status: **plan, pre-registered.** Nothing below has been run. Predictions and
gates are fixed here before any implementation or run.

An independent review of the first draft found:
- two design defects that would have made the verdicts meaningless;
- cost figures that contradicted the gates;
- several source misattributions.

All are corrected in this version (§8).

---

## 1. What is being compared

Three ways of putting the same physics into the loss. Everything else is held
fixed: network, hard Dirichlet layer, data, schedule, scorer.

| arm | formulation | tested against | natural (traction-free) BCs | network derivatives |
|---|---|---|---|---|
| **A — strong form** (the initial approach, arXiv 2607.23299) | pointwise Div P = 0 at collocation points, plus P·N = 0 at boundary points, as weighted penalties | Dirac deltas at points | penalised | second |
| **B — energy / Ritz** (current, Phases 2–9) | minimise Π = ∫ W(F) dA | implicitly the network's tangent functions ∂u/∂θᵢ | natural, at the minimiser | first |
| **C — weak form / VPINN** | squared residuals of ∫ P : ∇v dA = 0 against a fixed set of test functions v | chosen polynomials | depends on the test space | first |

### Arm C variants

**C1 family — FastVPINNs / hp-VPINN.** Element-local Legendre bubbles
v = (P_{n+1} − P_{n−1})(ξ)·(P_{m+1} − P_{m−1})(η) on quadrilaterals, with a
bilinear map and a Jacobian per quadrature point, in the FastVPINNs tensor
form.
- The bubbles vanish on every element edge, so no natural boundary condition
  is tested. Two ways to supply them are compared in the pilot:
  - **C1:** a traction penalty on the natural boundaries (the same term as arm
    A).
  - **C1-n (no penalty):** boundary elements also carry edge test functions.
    These are nonzero on their boundary edge: a (1 ± ξ)/2 factor normal to the
    edge times a bubble along it. The natural conditions are then tested
    weakly.
- One caveat. For bubbles, ∫_K P:∇v = −∫_K Div P·v, so C1 is a
  bubble-projected strong residual plus A's traction term: partly a strong-form
  method. C1-n removes the penalty half of that overlap.

**C2 family — Lagrange test functions on triangles** (Berrone, Canuto &
Pintore).
- Piecewise Lagrange polynomials of degree 1 or 2, one per unconstrained nodal
  degree of freedom.
- Nodes on free edges, symmetry lines and sliding grips carry test functions
  that do not vanish there, so the natural conditions are tested
  automatically, as in the FEM.
- The Dirichlet degrees of freedom are dropped; the hard layer imposes them.
- With degree 1 on the FEM's own mesh, the residual is the FEM internal-force
  vector evaluated with the network's stress.
- Two variants are compared in the pilot:
  - **C2:** the network is tested directly (Berrone et al.'s plain VPINN).
  - **C2-I:** the network is first interpolated onto a finer piecewise-linear
    space (their IVPINN). They show it removes the spurious zero-loss solutions
    that the plain VPINN admits.

### Contrasts the design isolates

- **A vs C1:** the interior residual, pointwise against projected; boundary
  terms identical.
- **C2 vs B:** squared residual against energy, same test/trial logic.
- **C1 vs C1-n:** penalised against weakly tested natural conditions.
- **A vs A-pre:** the generalised strong form against the preprint's exact
  loss. A-pre is `loss_form: penalty` through `training/trainer.py`, with its
  resultant anchor (`physics/uniaxial.py`) and adaptive weights, on the
  dog-bone only.

---

## 2. What the sources give, and what we must add

| | hp-VPINN (Kharazmi, Zhang & Karniadakis, CMAME 374, 2021) | FastVPINNs (Anandh, Ghose, Jain & Ganesan, SIAM J. Sci. Comput., doi 10.1137/24M1658620; arXiv 2404.12063) | Berrone, Canuto & Pintore (J. Sci. Comput. 2022; arXiv 2109.02035) |
|---|---|---|---|
| **test functions** | P_{k+1} − P_{k−1}, tensor products, local to an element | the same. In the library, `fe_type` "jacobi"/"legendre" → P_{n+1} − P_{n−1}; "jacobi_plain" → P_{n−1}, which does not vanish on edges and has no edge-flux term (used only to cross-check the library) | Lagrange degree 1–2 on triangles |
| **elements** | axis-aligned rectangles, constant Jacobian | quadrilaterals, bilinear map, Jacobian per quadrature point (the library uses \|det J\|); triangles are future work | triangles |
| **loss** | Σ_e (1/K_e) Σ_k R² + τ_b · penalty | paper eq. 10: (1/N_e) Σ; library and Alg. 3: Σ_e mean_k R² + β · Dirichlet penalty (library default β = 10) | Σ_i r_i² / γ_i, with γ_i = 1 in their experiments |
| **Dirichlet** | penalty | penalty (the library also has a hard-constraint model) | exact, via a function vanishing on the Dirichlet boundary, like our distance-function layer |
| **Neumann / natural** | not treated | not treated | treated, through Γ_N and the test space |
| **scope** | scalar; Burgers with Chebyshev test functions (non-integrated form) | scalar Poisson, Helmholtz, convection–diffusion, inverse | linear elliptic, plus one semilinear example (Sec. 7) |
| **warnings** | more test functions can hurt optimisation; τ an open problem | choose element count, test functions and quadrature "carefully" | lowest test degree plus high-precision quadrature; interpolating the network prevents spurious solutions |

**Library.** `fastvpinns` 1.0.2, TensorFlow ≤ 2.13. Repository
github.com/airexlab/fastvpinns; the package metadata still names
cmgcds/fastvpinns.

**New here, to be built and verified:**
1. The finite-strain Neo-Hookean weak form: first Piola–Kirchhoff stress
   against test-function gradients, two components per test function.
2. Natural conditions on free edges, symmetry lines and sliding grips, in every
   arm.
3. Quadrilateral meshes of the five families (triangle → 3 quads at the
   centroid and edge midpoints).
4. A PyTorch port of the tensor form. The library is TensorFlow; it stays the
   reference for G0.
5. Operator training over 64 geometries, with per-geometry tensor caches.

**Boundary bookkeeping**, checked against `fem_dirichlet` and the ADF specs for
every family. "Dirichlet" means imposed exactly; everything else is natural.

| family | Dirichlet components | everything else |
|---|---|---|
| dog-bone, open hole, double notch | x on the left symmetry line; y on the bottom; x = u_δ on a frictionless grip | natural |
| inclusion | as above, plus both components on the inclusion arc | natural |
| single notch | x on the left symmetry line; both components on the clamped grip | bottom edge fully free |

---

## 3. Rules that make the comparison fair

1. **The same model everywhere.** Oracle-conditioned operator, about 573k
   trainable parameters, float32. Only trainable parameters go to the
   optimizer.
2. **The same hard Dirichlet layer everywhere** (`geometry/adf.py`).
   - One exception, disclosed in advance. R-equivalence has an unbounded
     Laplacian where two Dirichlet pieces meet (`adf.py` notes this), and arm A
     needs second derivatives.
   - So A's collocation points exclude a neighbourhood of two local element
     sizes around those junctions, on every family.
   - A on the inclusion (arc meeting x = 0 and y = 0) is reported as possibly
     confounded.
3. **The same data and a paired design.**
   - Bank of 64, batch 4, batch order from `default_rng(42)`, 2 validation
     geometries.
   - The same seed gives the same initialisation and the same batch stream in
     every arm. The indices are logged and checked.
   - Scored on the Phase 8 in-range (12) and out-of-range (8) sets of each
     family.
4. **The same schedule, from scratch.**
   - Adam at 3e-4 for 1,600 steps, then cosine to 3e-6 over 800 (2,400 in
     total). This is the Phase 9.1 result, so arm B here is Phase 9.4.
   - No plateau scheduler: it is removed from the new trainer.
   - Gradient clipping at 1.0 as before, with the fraction of clipped steps
     reported per arm.
5. **The same scorer:** `verification.family_score`, and the 6.7 scorer for the
   dog-bone.
6. **Statistics.**
   - Per family, the 36 paired per-geometry differences in |K_t err| (12
     geometries × 3 seeds) between two arms.
   - A 95% cluster bootstrap over geometries and seeds (10,000 resamples).
   - A difference is **resolved** when the interval excludes 0.
   - Seeds vary only the initialisation, since data and batches are fixed, so
     single-arm seed sd understates run-to-run variance. That is why
     comparisons are paired.
7. **Equal tuning, including B.**
   - Every arm, B included, gets the same pilot (S6): ≤8 configurations,
     including learning rate and clip on/off.
   - Selection is on 4 fresh geometries per tuning family, drawn with a
     separate seed. They are in neither the bank nor the scored sets, and each
     has its own FEM reference.
8. **Budget.**
   - Primary: matched steps (2,400).
   - Secondary: **matched compute.** Arm A and any arm costing >2× B are rerun
     on two families with the step count scaled to B's compute and the same
     schedule shape. This replaces reading off intermediate checkpoints, which
     would compare an unannealed model (9.1 measured 1.4–7.2 points of wander
     at a constant rate).

---

## 4. Implementation, step by step, with quality gates

Each gate states pass criteria fixed now and what happens on failure.

### S0 — Reference library environment

**Work.**
- A Python 3.10 virtual environment (via `uv`; the container has 3.10). Pins:
  `fastvpinns==1.0.2`, `tensorflow==2.13.0`, `numpy==1.23.5`,
  `pandas<1.4.4`, `meshio==5.3.4`, `gmsh==4.11.0`. On Python 3.11 the pandas
  pin has no wheel.
- Fetch the v1.0.2 examples from the GitHub source; the wheel ships none.
- If the gmsh wheel needs system GL libraries, install them or use `--no-deps`
  with the pins by hand.

**G0a.** The library's Poisson examples (unit square; the circle quad mesh)
run, and reproduce their own reported error to within the repository's test
tolerance.
*On fail:* an analytic manufactured Poisson solution replaces the examples as
the reference.

### S1 — PyTorch port of the tensor form (`physics/vpinn/`)

**Work.**
- `basis.py`:
  - Legendre bubbles and plain P_{n−1}, tensor products, values and gradients;
  - edge test functions for C1-n;
  - Lagrange P1 and P2 on triangles.
- `quadrature.py`: Gauss–Legendre and Gauss–Lobatto on [−1,1]²; Dunavant
  degrees 2, 4 and 6.
- `mapping.py`: bilinear quadrilaterals and affine triangles, with a *signed*
  det J asserted > 0, and J⁻ᵀ gradients.
- `tensors.py`: per geometry, quadrature points and G_x, G_y
  (N_e × N_t × N_q) with w·det J folded in.
- `interp.py`: C2-I's interpolation of the network onto a finer P1 space.

**G0b (agreement, float64).**
- On the library's meshes, our G_x and G_y equal its tensors to 1e-12.
- For a fixed network with weights copied across frameworks, our Poisson
  residual matrix and loss equal the library's to 1e-10.

**G0c (agreement in training).**
- Problem: our port trains the library's Poisson benchmark (ω = 2π on the unit
  square, 2×2 elements, 15 test functions per direction, 40×40 quadrature
  points, 3×30 network, 20,000 Adam steps).
- Reference: the library run on the same problem in the venv.
- Pass: our port's final L2 error is within 2× of the library's.

*On fail at any G0 gate:* stop and fix; nothing uses the port until it passes.

### S2 — Meshes for the weak forms (`geometry/vpinn_mesh.py`)

**Work.**
- Triangle → 3 quads of the family triangulations at named levels. Each level
  is stated as element and quadrature-point counts per geometry.
- Boundary edges keep the shape's segment names and their Dirichlet/natural
  component flags (the table in §2).
- Outward unit normals per polygon edge.

**G1 (geometry)**, for every bank, validation, tuning and scored geometry of
all five families:
- det J > 0 at every quadrature point;
- Σ w·det J = polygon area (1e-12 relative);
- the boundary edges form closed chains whose names match `Shape.segments`;
- the edge normals equal the chord normals (1e-12) and point outward;
- the chord-vs-arc normal deviation on the hole, notch and inclusion arcs is
  reported, not gated;
- the Dirichlet degrees of freedom dropped by C2 equal `fem_dirichlet`'s set.

*On fail:* fix the split, or use Gmsh quad meshes for that family.

### S3 — The weak-form losses (`physics/vpinn/loss.py`)

**Residuals.**
- R_{e,k,c} = Σ_q P_{cJ}(F(x_q)) · ∂_J v_k(x_q) · w_q det J_q, with
  `first_piola_kirchhoff_stress` from `physics/neo_hookean.py`.
- C2 uses the same expression per nodal test function.

**Normalisation: every term is written as an integral**, so weights mean the
same thing at every mesh level.
- The C1 domain term is Σ_e mean_k Σ_c (R_{e,k,c}/(S0·|K_e|^{1/2}))².
- The C1 traction penalty is (1/(S0²·|Γ_N|)) ∫_{Γ_N} |t|² ds, with the
  integral evaluated by edge quadrature.
- For C2, γ_i ∈ {1, per-node scaling} (S6 factor). Berrone et al. use 1.

**Quadrature: fixed Gauss rules only.**
- Naive fresh sampling is biased for a squared residual:
  E[R̂²] = R² + Var(R̂). The variance term penalises stress variation inside
  an element, which is exactly the peak being scored.
- If the S5/S6 monitors show that fixed points are being exploited, the
  fallback is the unbiased two-draw estimator R̂₁·R̂₂. It gets its own gate,
  G2(h), before use.

**Memory.**
- C1 backpropagates element-chunked.
- C2's loss does not separate by element. It takes two passes: R without a
  graph, then backpropagate Σ_i (2R_i/s_i²)·R_i in chunks.

**Barrier.** The same J-barrier as B, on every arm.

**G2 (consistency; no training).**
- **(a) Patch test.** Affine displacement, homogeneous F₀.
  - C1: residuals exactly 0 on every element with Q ≥ ⌈(K+2)/2⌉ Gauss points
    per direction (≤1e-12 relative).
  - C2: interior-node residuals exactly 0. Boundary-node residuals equal
    +∮(P(F₀)N_h)_c φ_i ds, with N_h the polygon normal.
  - C1-n: its edge test functions give the same boundary integral.
- **(b) FEM equivalence (C2, degree 1, on the FEM's own mesh).**
  - The field: a *non-equilibrium* piecewise-linear field (the first Newton
    iterate of the cached solve), with ∇u_h per element.
  - Pass: C2's residual equals `solver.assemble`'s internal force at the free
    degrees of freedom to 1e-12 relative. Any rule whose weights sum to the
    area is exact here.
  - This verifies the code, not the training mesh.
- **(c) Manufactured solution, finite strain.**
  - Fields: the independent sympy fields of `verification/mms.py`, extended to
    the five family geometries with the ADF Dirichlet data built in.
  - Loads: body force b; tractions t on natural boundaries from the *polygon*
    normals, including the mixed segments (symmetry shear, sliding grip).
  - Pass: the weak residuals of u* → 0 as quadrature order rises, at the
    rule's rate, for C1, C1-n, C2 and C2-I.
- **(d) Energy consistency.** For any smooth u and each C2 test function,
  R_{i,c} = d/dε Π(u + εφ_i e_c) under the same rule (autograd, 1e-10).
- **(e) Gradients.** Loss gradients match central finite differences on a
  small network (1e-6, float64).
- **(f) Null-space probe.**
  - Add element bubbles s·b_T to the FEM field.
  - Report, for each loss, the change in loss against the change in energy as
    s grows.
  - This measures how blind each loss is to within-element modes, the
    mechanism behind spurious solutions (reported, not gated).
- **(g) Counts.** Report the number of test equations per geometry for each
  arm and level, against 573k parameters.

*On fail at (a)–(e):* stop.

### S4 — The strong form for five families (`physics/strong_form.py`)

**Work.**
- **Interior.** Div P at points drawn fresh each step, stratified in the
  triangulation. Same count as B (≈1,400 × 3). A neighbourhood of two element
  sizes around Dirichlet-piece junctions is excluded (rule 2).
- **Traction.** The natural components only, per segment, with polygon
  normals, via `physics/equilibrium.py`.
- **Weights (the preprint's):** w_eq 100; w_trac 20 on feature edges, 2 on
  straight free edges.
- **A-pre** runs through `training/trainer.py` unchanged, with the §3
  schedule.

**G3.**
- **(a)** For the sympy u*, Div P(u*) + b = 0 at every collocation point
  (1e-10). The traction residual equals P(u*)N − t (1e-10).
- **(b)** On the dog-bone, the generalised A with A-pre's terms switched on
  reproduces `PhysicsLoss` for the same model and points (1e-10). This covers
  only the terms in `physics/losses.py`; the anchor is exercised by A-pre's
  run.

### S5 — End-to-end sanity before any operator run

**G4a (trained manufactured solution).**
- Each arm (A, B, C1, C1-n, C2, C2-I) trains a plain network on the
  manufactured problem, on the open hole and double notch geometries, for
  4,000 steps.
- B minimises Π − ∫ b·u − ∫ t·u.
- Pass: relative H¹ error to u* ≤ 2e-2.
- *On fail:* that arm's implementation or conditioning is wrong. Fix it, or
  drop the variant with the reason recorded.

**G4b (one geometry, the real problem)**, per family validation geometry,
default settings. Both of the following must hold:
- **(i)** The independent-rule energy gap to the extrapolated FEM energy (the
  9.2 method) falls over training and ends ≤3× B's gap on the same geometry.
- **(ii)** Re-evaluated at quadrature order +2 and on an h/2 (or P2) test
  space, the trained residual is ≤2× its training value. This is the
  spurious-solution monitor.

**Reported, not gated:**
- the K_t of a P1 FEM solved on C2's own test mesh (what the test space alone
  can resolve);
- each arm's small-load open-hole K_t against the Level-3 handbook reference.

### S6 — Tuning pilot (same budget for every arm, B included)

**Setup.**
- Families: open hole and double notch.
- Operator, seed 0, 800 steps (50 warm-up, then cosine).
- Tuning set: 4 fresh geometries per family, with FEM references.

**Pre-filter (G5, before running).** A configuration is dropped from the grid
before it runs if it breaks either limit:
- peak RSS > 3 GB, so two lanes fit in 8 GB;
- estimated S7 cost for that arm > 150 CPU-h.

**Grids (≤8 each):**
- **B:** learning rate {1e-4, 3e-4, 1e-3} × clip {on, off}.
- **A:** w_eq {10, 100} × w_trac {2, 20} × learning rate {3e-4, 1e-3}.
- **C1 / C1-n:** mesh level {matched to B's triangulation; coarse} × test
  functions per direction {2, 4}, with Q = K+2 per direction. C1 adds β
  {5, 50}; C1-n has no β.
- **C2 / C2-I:** degree {1, 2} × Dunavant degree {4, 6} × test-mesh level
  {B's triangulation; h/2}; γ_i {1, per-node} on the winner.

**Selection.** Lowest mean |K_t err| over the 8 tuning geometries.
- The top two configurations are re-run with seeds 1 and 2, and the lower
  three-seed mean wins.
- **Spurious rule:** a configuration is discarded if its independent-rule
  energy rises over the last 400 steps, or if its G4b(ii) ratio exceeds 2.
- **C winners:** the C1 family and the C2 family each yield one winner, C1*
  and C2*.

### S7 — Main runs

**Runs.**
- A, B, C1* and C2* × 5 families × 3 seeds, 2,400 steps, paired.
- A-pre × 3 seeds on the dog-bone.
- **Matched-compute runs:** A, and any C costing >2× B, on the open hole and
  double notch × 3 seeds, with steps scaled to B's compute and the same
  schedule shape.

**Every run logs:**
- batch indices;
- clipped fraction;
- the independent-rule energy gap and G4b(ii) ratio on the 2 validation
  geometries, every 400 steps.

**G6 (integrity).**
- Every run completes with no NaN.
- Batch indices are identical across arms per seed.
- The scorer reproduces a known checkpoint's stored score exactly (a 9.1
  checkpoint).
- The monitors stay within G4b limits. A run that breaches them is reported
  and flagged, not silently kept.
- *On fail:* rerun once; >2 failures in an arm → the arm is reported as not
  robust.

### S8 — Scoring, verdicts, independent check

**Scoring**, per arm, family and seed:
- in-range and out-of-range K_t |err|, R² and bias;
- u, v and vm L2;
- free-edge traction residual |P·N|/σ_nom (`verification/family_residual.py`);
- N_err;
- time per step, peak memory and clipped fraction.

**Cross-evaluation matrix.** Every arm's final models evaluated under every
loss: energy gap, C1 residual, C2 residual, strong residual. This shows what
each formulation actually optimised.

**Verdicts** on §5, with the paired statistics of rule 6.

**G7.** An independent agent recomputes every number from the raw scores; its
corrections are folded in before the results are used.

---

## 5. Predictions (fixed now)

In range, paired per rule 6. "Resolved" means the 95% cluster-bootstrap
interval of the paired difference excludes 0.

- **P10-1 (B beats A at matched steps).** B is resolved better than A on ≥3 of
  5 families, and A is resolved better than B on none.
  - Phase 2's 2× stress advantage for the energy form was at matched
    wall-clock (1,750 vs 350 epochs), where A had the smaller budget. At
    matched steps A gets about 5× B's compute, so this is a real test.
- **P10-2 (A is better at the free edge).** A has resolved lower free-edge
  traction residual than B on ≥3 of 5 families.
- **P10-3 (FastVPINN/hp-VPINN does not beat the energy form).** C1* is resolved
  better than B on at most 1 family. *Falsified* if it is on ≥2.
- **P10-4 (Lagrange test functions do not beat the energy form).** C2* is
  resolved better than B on at most 1 family. *Falsified* if it is on ≥2.
- **P10-5 (extrapolation).** For every arm, out-of-range mean |err| exceeds
  in-range mean |err| on the open hole, double notch and single notch.

**Exploratory, no prediction:**
- C1* against C2*;
- C1 against C1-n;
- A against A-pre;
- cost, which is fixed by the S6 choices;
- the matched-compute contrasts.

---

## 6. Budget

Rates from this project: B at 2.7–3.3 s per step in 9.1 (36–44 min per 800
steps), so 1.8–2.2 h per 2,400-step run; A about 5× B (Phase 2).

| arm | one run (estimate) | S7 total |
|---|---|---|
| B | 1.8–2.2 h | ~30 CPU-h |
| A (+ A-pre) | ~9–11 h | ~150 + 30 CPU-h |
| C1* | ≤150 CPU-h by the pre-filter; likely 4–9 h | ~60–135 CPU-h |
| C2* | 2–4× B's points (Dunavant 4–6), so ~4–8 h | ~60–120 CPU-h |
| matched-compute runs | at B's cost | ~25 CPU-h |

**Totals:**
- **S0–S6:** about 60–150 CPU-h, i.e. 1.5–3 days on the container's two
  lanes, plus implementation time.
- **S7:** about 350–490 CPU-h, i.e. **7–10 days on the container.**
  - On one GPU this should be **1–2 days**. That is an estimate, to be
    confirmed by a timing pilot in S6.
  - A cheaper staging: S7a on the dog-bone, open hole and double notch (about
    60%), then S7b on the inclusion and single notch.

---

## 7. What the comparison will and will not tell us

**It will tell us** which formulation the same operator learns best from:
- at equal steps, and at equal compute on two families;
- at equal tuning effort;
- on five geometries against verified FEM, and at what cost.

We know of no comparison of all three on a finite-strain operator with a
stress-concentration QoI. Related comparisons exist:
- energy against collocation on single problems: Samaniego et al., CMAME 2020;
- variational losses with compactly supported test functions: VarNet,
  Khodayi-Mehr & Zavlanos, L4DC 2020.

**It will not tell us** how each formulation performs at its own best budget,
or at GPU scale. C1's natural-boundary treatments (penalty or edge test
functions) are our extensions, not the papers'.

---

## 8. Changes after the independent review (before commit)

**Critical:**
1. Naive quadrature resampling removed from C: it biases squared residuals
   against stress peaks. Fixed Gauss rules are used, with an unbiased two-draw
   fallback that has its own gate.
2. P10-3 was unfalsifiable at the stated seed sd; P10-1 had no threshold. They
   are replaced by a paired design with cluster-bootstrap intervals.
3. Cost figures contradicted G5, and C2's point count was misstated. G5 is now
   a pre-filter with stated limits; the budget is recomputed from the measured
   9.1 rates.
4. The spurious-solution risk of the plain VPINN is now addressed:
   - the C2-I (IVPINN) variant;
   - the null-space probe;
   - test-equation counts;
   - quadrature/test-space refinement monitors in every run;
   - a tightened G4;
   - a P1-FEM-on-test-mesh baseline.
5. P10-1's basis was misstated: Phase 2 matched wall-clock, not steps.
   Matched-compute runs replace checkpoint read-offs.

**Also changed:**
- **Tuning:** B now tuned too; learning rate and clip are factors; tuning
  geometries are fresh; top-two run-off with three seeds.
- **Arms:** C1-n added, since C1's penalty makes it partly a strong-form
  method; the contrasts are stated.
- **Arm A:** ADF junctions excluded from its collocation (and the inclusion
  flagged).
- **G1:** now checks chord normals.
- **MMS:** gates use independent sympy fields, polygon normals and the mixed
  segments, with the sign fixed. A trained MMS for all arms was added.
- **G2(b):** now uses a non-equilibrium field.
- **Environment:** Python 3.10 pins.
- **Memory:** chunking specified for C1 and C2.
- **Sources corrected:** Berrone et al. do treat Neumann conditions and use
  γ_i = 1; FastVPINNs is published in SIAM J. Sci. Comput.; library details.
- **Loose ends:** the plateau scheduler is removed; A-pre is defined through
  the trainer; G6 wording fixed; ambiguous predictions made exploratory.

---

## References

- E. Kharazmi, Z. Zhang, G.E. Karniadakis, *hp-VPINNs: Variational
  physics-informed neural networks with domain decomposition*, CMAME 374
  (2021) 113547.
- T. Anandh, D. Ghose, H. Jain, S. Ganesan, *FastVPINNs: Tensor-driven
  acceleration of VPINNs for complex geometries*, SIAM J. Sci. Comput., doi
  10.1137/24M1658620; arXiv 2404.12063. Library `fastvpinns` 1.0.2.
- S. Berrone, C. Canuto, M. Pintore, *Variational physics informed neural
  networks: the role of quadratures and test functions*, J. Sci. Comput.
  (2022), doi 10.1007/s10915-022-01950-4; arXiv 2109.02035.
- S. Rojas et al., *Robust variational physics-informed neural networks*, CMAME
  (2024); arXiv 2308.16910. The dual-norm loss is a follow-up if C's residual
  norms prove ill-conditioned.
- E. Samaniego et al., *An energy approach to the solution of partial
  differential equations in computational mechanics via machine learning*,
  CMAME (2020).
- R. Khodayi-Mehr, M. Zavlanos, *VarNet: Variational neural networks for the
  solution of partial differential equations*, L4DC (2020); arXiv 1912.07443.

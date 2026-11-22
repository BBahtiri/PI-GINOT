# Energy-PI-GINOT — solution-label-free variational operator learning for stress-concentration families

Energy-PI-GINOT learns **family-specific parametric solution operators for finite-strain stress-concentration problems** on notched, holed, filleted, and inclusion-containing specimens. Exact geometry descriptors and spatial coordinates are mapped to displacement fields, from which stresses and stress-concentration quantities are derived.

Training is performed by minimizing the potential energy of a compressible Neo-Hookean solid using per-step resampled quadrature. **No finite-element displacement or stress solution labels enter the training objective.** Finite-element solutions are used for monitoring, verification, and post-training evaluation.

The repository contains the five geometry families, finite-strain plane-stress formulation, admissible trial-field construction, geometry-conditioned neural operator, training workflow, verified finite-element reference solver, held-out evaluation sets, diagnostic studies, and the records underlying the reported results.

<p align="center">
 <img src="docs/pi_ginot_architecture_animated.gif" alt="Energy-PI-GINOT architecture: geometry conditioning, spatial decoder, admissible trial field and variational energy training" width="100%">
</p>

The reported accuracy measures are evaluated against **separately solved and verified finite-element references on held-out geometries**, using three network initializations per family.

---

## Main results

The reported models use one separately trained operator for each geometry family.

The family-appropriate stress-concentration factor is denoted by `K`:

- `K_g` for the filleted dog-bone;
- `K_N` for the plate families.

| Specimen family | In-range MARE in K | Mean R2 | Reference-mean predictor | Out-of-range MARE in K |
|---|---:|---:|---:|---:|
| Filleted dog-bone | **1.01%** | 0.848 | 2.43% | 2.00% |
| Open-hole plate | **1.29%** | 0.948 | 5.94% | 7.45% |
| Bonded rigid inclusion | **1.75%** | -0.121 | 1.60% | 0.85% |
| Double-edge U-notch | **2.31%** | 0.955 | 12.28% | 12.56% |
| Single-edge U-notch | **2.20%** | 0.963 | 15.86% | 11.33% |

### Evaluation protocol

- **Training geometries:** 64 per family.
- **Monitoring geometries:** 2 per family.
- **Held-out in-range geometries:** 12 per family.
- **Directed out-of-range geometries:** 8 per family.
- **Initializations:** 3 per family.
- **In-range von Mises field errors:** approximately 0.46–1.34%.

The reference-mean predictor is included only as descriptive context for the amount of geometry-driven response variation; it is not a trained competing model.

Four of the five families track in-range geometry-driven variation well, with mean seedwise `R2 = 0.848–0.963`.

The bonded inclusion is the important exception. Its reference concentration factor varies only weakly across the tested geometries, so a small, nearly uniform bias produces a negative raw `R2` despite a concentration MARE of only 1.75%. The corresponding trend remains visible in the Pearson correlation and demeaned coefficient of determination reported in the manuscript.

Beyond the training range, the hole and notch families show substantial degradation. The directed extrapolation experiments are therefore treated as **coverage diagnostics**, not as evidence for arbitrary out-of-distribution robustness.

---

## Verification philosophy

The central objective of the project is not only to obtain low average errors, but to determine **which diagnostics are required before a learned mechanics field can be trusted for a localized stress concentration**.

The study separates the following verification levels:

1. variational energy;
2. displacement field;
3. distributed stress field;
4. section force;
5. stress-concentration magnitude;
6. peak location;
7. in-range versus out-of-range geometric coverage.

**A good result at one level does not certify the next.**

Observed examples include:

- a fixed quadrature rule can yield an artificially low sampled energy while the stress field is severely wrong;
- sub-percent displacement errors can coexist with significantly larger stress-concentration errors;
- accurate section force does not guarantee accurate peak stress;
- accurate peak magnitude does not guarantee the correct critical location;
- low interpolation error does not guarantee reliable geometric extrapolation.

These failure modes form a central part of the scientific contribution.

---

## The method

### Geometry conditioning

The reported operators use the **exact low-dimensional geometry parameters available for each family**, rather than asking a point-cloud encoder to rediscover those dimensions.

Each parameter is normalized using the bounds of the corresponding training interval:

```text
p_norm = (p - p_train_min) / (p_train_max - p_train_min)
```

Training and in-range geometries therefore satisfy:

```text
0 <= p_norm <= 1
```

for every parameter.

The same transformation is retained for directed out-of-range evaluation. Therefore, a parameter intentionally taken beyond the training range can produce:

```text
p_norm < 0
```

or

```text
p_norm > 1
```

No clipping to the training interval is applied.

The conditioning architecture contains:

- a parameter encoder with two hidden GELU layers of width 96;
- 32 geometry tokens of width 64;
- a six-octave Fourier encoding of the spatial query coordinates;
- six residual cross-attention blocks;
- four attention heads;
- FiLM conditioning through the pooled geometry latent;
- approximately 573k trainable parameters.

The five geometry families use the same architecture and training specification but **do not share network weights**.

Point-cloud geometry encoders are retained in the repository as controlled methodological diagnostics, but they are not the representation used for the final reported multi-family operators.

---

### Admissible trial fields

The raw neural output is converted into a displacement field that satisfies the prescribed essential boundary conditions exactly.

For displacement component `c`:

```text
u_theta,c(X; p)
    = g_c(X; p)
    + [phi_c(X; p) / ell_c(p)] N_theta,c(X; p)
```

Here:

- `g_c` extends the prescribed displacement values;
- `phi_c = 0` on the corresponding essential boundary pieces;
- `ell_c` is a characteristic length;
- `N_theta,c` is the unconstrained neural output.

Straight boundaries use distance primitives, while the bonded inclusion additionally uses a circular distance representation.

Multi-boundary constraints are combined using approximate-distance/R-function and transfinite-interpolation constructions.

Traction-free boundaries arise naturally from the variational formulation rather than through an additional boundary penalty.

---

## Finite-strain mechanics

The material is a homogeneous compressible Neo-Hookean solid with:

```text
E  = 760 MPa
nu = 0.23
```

The in-plane deformation gradient is:

```text
F_parallel = I + grad_X(u)
```

with:

```text
J2 = det(F_parallel)
```

The three-dimensional deformation gradient retains an out-of-plane stretch `s = F_33`:

```text
F = diag(F_parallel, s)
```

and:

```text
J = s J2
```

The out-of-plane stretch is determined pointwise from the plane-stress condition:

```text
P_33 = 0
```

which gives the scalar closure:

```text
mu s^2 + lambda ln(J2 s) - mu = 0
```

The closure is solved using twelve detached Newton iterations followed by two differentiable Newton refinement steps.

The compressible Neo-Hookean stored-energy density is:

```text
Psi(F)
    = (mu/2) [tr(F^T F) - 3]
      - mu ln(J)
      + (lambda/2) [ln(J)]^2
```

with:

```text
mu     = E / [2(1 + nu)]
lambda = E nu / [(1 + nu)(1 - 2 nu)]
```

Because the loading is displacement controlled and the remaining boundaries are traction-free, the total potential equals the internal stored energy.

---

## Resampled variational objective

Each geometry is equipped with a graded triangular integration mesh containing approximately 1,400 triangles, refined near the stress-concentrating feature.

At every energy evaluation, three new points are sampled uniformly inside every triangle.

The stochastic area-weighted energy estimator can be written schematically as:

```text
Pi_hat_theta(p)
    = sum over elements e:
        (A_e / 3)
        sum over q = 1,2,3:
            Psi_hat[F_parallel(u_theta(X_e,q; p))]
```

where:

- `A_e` is the area of triangle `e`;
- `X_e,q` are newly sampled integration points.

Per-step resampling is important.

A controlled fixed-quadrature diagnostic showed that the neural field can exploit gaps between persistent integration points, producing an artificially low sampled energy while retaining very large field errors.

In the most severe diagnostic case:

```text
sampled energy / reference energy = 0.007
von Mises field error             = approximately 270%
```

The training objective additionally contains a determinant guard against strongly nonphysical local Jacobians.

The guard uses:

```text
J_min = 0.05
beta  = 1000
```

and acts as a numerical regularizer rather than a proof of positive Jacobian everywhere.

**No FEM displacement or stress field enters the training objective.**

---

## Training specification

The final family-specific operators use:

- 64 training geometries per family;
- minibatches of 4 geometries;
- Adam optimization;
- initial learning rate `3 x 10^-4`;
- 3,200 constant-learning-rate steps;
- cosine decay to `3 x 10^-6` over a further 1,600 steps;
- 4,800 total optimization steps;
- gradient-norm clipping at 1.0;
- float32 arithmetic;
- three network initializations per family.

All reported scores use the **last checkpoint** rather than selecting a checkpoint retrospectively from test-set performance.

Training requires several hours per model on the reference CPU setup. Runtime is hardware dependent, and no hardware-matched FEM-versus-operator inference benchmark or speedup claim is made.

---

## Geometry families

| Family | Training parameter box | Directed out-of-range perturbation |
|---|---|---|
| Filleted dog-bone | `L_total = 40–70 mm`; `W_grip = 16–26 mm`; `W_gauge = 6–14 mm`; `R = 8–20 mm` | `W_gauge/W_grip >= 0.78`, while retaining the raw dimensional bounds |
| Open hole | `W = 16–26 mm`; `d/W = 0.20–0.50`; `L_h/W = 1.5–2.5` | `d/W = 0.55–0.60` |
| Bonded rigid inclusion | Same geometric intervals as the open-hole family | `d/W = 0.55–0.60` |
| Double U-notch | `W = 16–26 mm`; `t/W = 0.10–0.30`; `rho/t = 0.25–1.0`; `L_h/W = 1.5–2.5` | `rho/t = 0.15–0.25` |
| Single U-notch | `W = 16–26 mm`; `t/W = 0.10–0.30`; `rho/t = 0.25–1.0`; `L_h/W = 2.0–3.0` | `t/W = 0.30–0.35` |

The out-of-range sets deliberately perturb a physically relevant concentration-controlling ratio while retaining the remaining descriptors inside the training box.

They are mechanism-oriented extrapolation tests rather than unrestricted samples from arbitrary geometries.

---

## Finite-element reference

The verification suite contains a nonlinear finite-strain finite-element solver using P1 triangular elements, the same constitutive formulation, and the same component-wise displacement boundary data as the neural formulation.

The reference implementation is checked through:

- patch tests;
- manufactured solutions;
- uniaxial tests;
- mesh-refinement studies;
- classical stress-concentration benchmarks for open holes, shallow notches, and bonded rigid inclusions.

Representative reference checks include:

- finite-width open-hole factors reproduced within approximately `+0.01% to +0.95%`;
- bonded rigid-inclusion benchmark reproduced within approximately `+1.24%`;
- shallow double-notch benchmark reproduced within approximately `-1.11%`;
- shallow single-notch benchmark reproduced within approximately `-1.05%`.

Refining the sharpest in-range plate geometries from 60 to 85 elements per feature radius changes the reference concentration factors by approximately:

```text
Open hole:        0.036%
Rigid inclusion:  0.090%
Double U-notch:   0.298%
Single U-notch:   0.066%
```

The neural and finite-element solutions use different approximation spaces and solution procedures, although they share geometry-generation and meshing infrastructure.

The FEM solutions are therefore treated as **verified numerical references**, not as training labels.

Relevant implementation:

```text
verification/fem/
verification/family_fem.py
verification/mms.py
```

---

## Quantities of interest

### Stress-concentration factor

For the plate families:

```text
K_N = max(sigma_vm) / (N / A_gross)
```

where:

- `sigma_vm` is the von Mises equivalent Cauchy stress;
- `N` is the axial section force per unit reference thickness;
- `A_gross` is the gross reference section area per unit thickness.

For the dog-bone:

```text
K_g = max(sigma_vm) / mean_gauge(sigma_vm)
```

These are precisely defined finite-deformation quantities used consistently for neural and FEM fields.

They should **not** be interpreted as conventional small-strain textbook `K_t` values.

---

## Field errors

The evaluation records include:

- axial displacement error `e_u`;
- transverse displacement error `e_v`;
- area-weighted von Mises field error `e_vm`;
- section-force error `e_N`;
- concentration-factor error `e_K`;
- sampled peak-stress error `e_peak`.

In range, the final operators give approximately:

```text
e_u:   0.05–0.25%
e_v:   0.50–3.13%
e_vm:  0.46–1.34%
```

The relatively larger transverse-displacement percentages reflect the weaker energetic sensitivity of that displacement component rather than necessarily larger physical displacement errors.

---

## Peak-location diagnostics

The evaluation records also contain the reference and predicted peak-element centroids.

Peak localization is assessed using:

1. discrete peak-element coincidence;
2. Euclidean peak offset:

```text
||Delta X_peak||_2
```

3. peak offset normalized by the relevant feature radius:

```text
||Delta X_peak||_2 / r_f
```

4. FEM reference stress at the operator-predicted peak location:

```text
chi_peak
    = sigma_vm,FEM(X_peak,operator)
      / sigma_vm,FEM,max
```

`chi_peak = 1` means that the operator-predicted critical location carries the true FEM peak stress.

A small value indicates that the operator has identified a mechanically noncritical location as the maximum.

This distinction is important because a peak magnitude can be numerically accurate even when its location is wrong.

---

## Held-out evaluation

Each family contains:

```text
12 held-out in-range geometries
 8 directed out-of-range geometries
 3 network initializations
```

This gives:

```text
36 in-range geometry-seed evaluations per family
24 out-of-range geometry-seed evaluations per family
```

The in-range sets are stratified with respect to the concentration-controlling geometric ratio.

The out-of-range sets change one physically important descriptor beyond its training coverage while retaining the others within their prescribed bounds.

The purpose is to distinguish interpolation from directed geometric extrapolation, not to claim robustness to arbitrary topology.

---

## Peak-localization findings

The peak-location diagnostics reveal an important distinction between peak magnitude and peak location.

### Open-hole plate

In range:

```text
Peak-element coincidence: 75.0%
Mean peak offset:          0.019 mm
Maximum peak offset:       0.130 mm
```

Beyond coverage:

```text
Peak-element coincidence: 66.7%
Mean peak offset:          0.041 mm
Maximum peak offset:       0.191 mm
```

Non-coincident peaks remain close to the physical stress concentration.

### Bonded rigid inclusion

In range:

```text
Peak-element coincidence:  0.0%
Mean peak offset:           1.999 mm
Maximum peak offset:        2.847 mm
```

The FEM reference maximum lies near 30 degrees around the bonded interface, consistent with the classical solution.

The operator instead places the maximum close to the loading axis in all in-range evaluations.

This occurs despite:

```text
Concentration MARE: approximately 1.75%
Von Mises field error: approximately 0.60%
```

The inclusion therefore demonstrates that an accurate peak magnitude does not guarantee an accurate peak location.

### Double U-notch beyond coverage

For the two sharpest out-of-range double-notch geometries, the predicted maximum moves from the notch root to the notch flank near the notch mouth.

The displaced points carry only approximately:

```text
0.4–2.6%
```

of the true FEM peak stress.

At the same time, section-force errors remain below approximately 0.6%.

This is a clear example of accurate global load transfer coexisting with severe local extrapolation failure.

---

## Section-force versus peak-stress accuracy

For the plate families, the concentration factor combines the sampled peak stress and section force.

In range, mean absolute section-force errors are approximately:

```text
0.10–0.17%
```

while peak-stress errors are approximately:

```text
1.30–2.29%
```

Beyond coverage, the separation becomes much stronger.

For the double notch:

```text
Mean section-force error: 0.29%
Mean peak-stress error:  12.53%
```

For the single notch:

```text
Mean section-force error: 0.86%
Mean peak-stress error:  10.57%
```

Accurate integrated load transfer therefore does not certify the local maximum.

---

## Controlled methodological diagnostics

Several additional studies are retained because they help explain the observed success and failure modes.

### Geometry representation

A point-cloud/attention-based geometry representation was examined on the dog-bone family.

Under the energy objective it captured the global gauge-to-grip ratio well:

```text
Linear-probe R2: 0.974
```

but captured the local fillet radius poorly:

```text
Linear-probe R2: approximately 0.03–0.30
```

Replacing the point-cloud representation by the exact four geometry parameters at matched model capacity reduced the held-out concentration error after 1,600 steps from approximately:

```text
6.87% -> 2.26%
```

and substantially improved the transverse displacement field.

This motivates the use of exact descriptors when such parameters are available.

It is not a claim that point-cloud encoders are unsuitable for non-parameterized geometries.

### Supervised representability

The same conditional field architecture was also fitted directly to FEM data in a controlled diagnostic.

Direct displacement supervision reached approximately:

```text
Axial displacement error:       0.14%
Transverse displacement error:  0.22%
```

but still produced:

```text
Von Mises field error: 4.4%
Concentration MARE:    3.15%
```

Direct stress supervision reduced the concentration error substantially.

This demonstrates that the architecture can represent concentrated stresses and that the difficulty lies partly in transmitting derivative-sensitive information through a displacement field trained by a global energy objective.

### Optimization schedule

Cosine annealing stabilizes terminal predictions and reduces initialization sensitivity for several families.

Paired 2,400-step and 4,800-step evaluations quantify the influence of optimization budget.

The final concentration MAREs changed as follows:

```text
                         In range          Out of range
Dog-bone:               1.28 -> 1.01%     2.47 -> 2.00%
Open hole:              1.68 -> 1.29%     9.40 -> 7.45%
Rigid inclusion:        1.61 -> 1.75%     1.18 -> 0.85%
Double U-notch:         2.92 -> 2.31%    19.30 -> 12.56%
Single U-notch:         3.61 -> 2.20%    14.44 -> 11.33%
```

Longer optimization therefore improves many cases but does not eliminate the geometric extrapolation gap.

### Alternative-formulation probes

Strong-form and fixed-test-space variational residual formulations are retained as diagnostic implementations.

The tested strong-form formulation requires nested spatial differentiation and was substantially more expensive per optimization step than the first-derivative energy evaluation.

Fixed low-dimensional weak test spaces can possess blind directions for oscillatory modes.

These studies are used to understand formulation behaviour and are **not** presented as matched competitive benchmarks against the final energy formulation.

---

## Repository layout

```text
config.py
    Material, loading, network, and training configuration.

main.py
    Main training entry point retained from the development workflow.

models/
    Parameter encoder, spatial decoder, cross-attention/FiLM modules,
    and positional/Fourier encoding.

geometry/
    Parametric geometry families, meshing, and approximate-distance-function
    boundary constructions.

physics/
    Compressible Neo-Hookean model, plane-stress mechanics, energy form,
    strong-form and weak-form diagnostic implementations.

training/
    Family-specific trainers and final training workflows.

verification/
    FEM reference solver, manufactured-solution checks, held-out geometry
    generators, scoring routines, and verification studies.

verification/phase10/
    Diagnostic strong-form and variational-formulation studies.

eval/
    Field-comparison and dense-evaluation utilities.

tests/
    Physics and numerical unit tests.

docs/
    Development records, evaluation summaries, and machine-readable score data.

agent/
llm_agents/
    Optional inference and agentic interfaces. These are not part of the
    scientific results reported in the manuscript.
```

---

## Quick start

Install the core dependencies and run the tests:

```bash
pip install -r requirements-core.txt
python -m pytest -q tests
```

Train one family using the final 4,800-step recipe:

```bash
python -m training.run_p94 double_notch 0 \
    --steps 4800 \
    --t-const 3200 \
    --root verification/results/phase9/p94x
```

Available family identifiers are:

```text
dogbone
open_hole
inclusion
double_notch
single_notch
```

The training workflow stores checkpoints, optimization histories, and monitoring records under the configured results directory.

The verification routines solve and cache the required FEM references on first use and subsequently evaluate the trained operators against the corresponding held-out geometries.

Example scoring commands:

```bash
python -m verification.phase9_5_score
python -m verification.phase9_6_score
```

Some optional diagnostics require additional dependencies beyond the core environment. See the corresponding module documentation for those cases.

---

## Development record

The repository developed through a sequence of documented numerical and methodological stages.

The `docs/` directory retains the associated plans, verification notes, evaluation records, and result summaries for transparency and reproducibility.

Major stages include:

| Stage | Main topic |
|---|---|
| 0–1 | Correctness audit, FEM reference, and verification hierarchy |
| 2 | Variational/energy formulation |
| 3–5 | Transverse field, geometry representation, and numerical stabilization |
| 6 | Metrics, numerical noise, supervised representability, and re-evaluation |
| 7 | Boundary conditions and grip idealization |
| 8 | Multi-family geometry formulation and admissible boundary construction |
| 9 | Optimization schedule and final operator recipe |
| 10 | Diagnostic strong-form and weak-form investigations |

These records document the development history of the implementation.

The scientific results of the manuscript are defined by the final evaluation protocol described in the paper.

---

## Optional inference interfaces

The repository also contains optional inference and agentic interfaces.

`agent/` provides reliability-gated inference utilities based on mechanics diagnostics.

`llm_agents/` contains an experimental LangGraph/Streamlit interface for querying trained models through predictor, optimization, diagnosis, and reporting components.

These interfaces are **not part of the quantitative evidence or claims of the accompanying manuscript**.

---

## Scope and limitations

The evidence in this repository supports the reported conclusions only within the tested scope.

- One separately trained operator is used for each geometry family.
- No cross-family weight transfer is demonstrated.
- One compressible Neo-Hookean material parameter set is considered.
- One imposed displacement level is evaluated.
- The reported mechanics use finite-strain plane stress.
- Generalization to new materials, load amplitudes, stress states, boundary-condition topologies, or arbitrary CAD geometry is not demonstrated.
- Distributed validation focuses primarily on the von Mises equivalent stress rather than all stress components and principal directions.
- Peak stresses are evaluated on the common FEM centroid sampling set rather than as converged continuum suprema.
- Directed out-of-range errors remain substantially larger than in-range errors for the hole and notch families.
- No hardware-matched FEM/operator timing benchmark or speedup claim is made.

These limitations define the domain of the reported evidence rather than being hidden assumptions.

---

## Reproducibility

The repository retains:

- geometry-generation routines;
- training configurations;
- trained-model evaluation utilities;
- verified FEM reference routines;
- held-out geometry definitions;
- machine-readable score records;
- scripts for reproducing aggregate statistics and data-driven figures;
- controlled diagnostic implementations.

The training objective contains no FEM displacement or stress labels.

FEM references are used for:

- monitoring;
- verification;
- post-training assessment.

All reported operators are family specific.

No result demonstrates:

- cross-family weight transfer;
- arbitrary-topology generalization;
- transfer across material parameters;
- transfer across load magnitudes;
- transfer across stress states;
- transfer across boundary-condition classes.

---

## Citation

This repository accompanies the manuscript:

**A. Dean, B. Bahtiri, M. K. Khraisheh**

**Energy-trained geometry-conditioned neural operators for stress concentrations in finite-strain solids: verification hierarchy, failure modes and extrapolation limits**

Citation details will be updated upon publication.

---

## License

MIT — see `LICENSE`.

---

## Authors

Betim Bahtiri and Aamir Dean

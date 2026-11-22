# Energy-PI-GINOT — solution-label-free variational operator learning for stress-concentration families

Energy-PI-GINOT learns **family-specific parametric solution operators for finite-strain stress-concentration problems** on notched, holed, filleted and inclusion-containing specimens. Exact geometry descriptors and spatial coordinates are mapped to displacement fields, from which stresses and stress-concentration quantities are derived.

Training is performed by minimizing the potential energy of a compressible Neo-Hookean solid using per-step resampled quadrature. **No finite-element displacement or stress solution labels enter the training objective.** Finite-element solutions are used for monitoring, verification and post-training evaluation.

The repository contains the five geometry families, finite-strain plane-stress formulation, admissible trial-field construction, geometry-conditioned neural operator, training workflow, verified finite-element reference solver, held-out evaluation sets, diagnostic studies and the records underlying the reported results.

<p align="center">
 <img src="docs/pi_ginot_architecture_animated.gif" alt="Energy-PI-GINOT architecture: geometry conditioning, spatial decoder, admissible trial field and variational energy training" width="100%">
</p>

The reported accuracy measures are evaluated against **separately solved and verified finite-element references on held-out geometries**, using three network initializations per family.

---

## Main results

The reported models use one separately trained operator for each geometry family. The family-appropriate stress-concentration factor is denoted by \(K\): \(K_g\) for the filleted dog-bone and \(K_N\) for the plate families.

| Specimen family | In-range MARE in \(K\) | Mean \(R^2\) | Reference-mean predictor | Out-of-range MARE in \(K\) |
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
- The reference-mean predictor is included only as descriptive context for the amount of geometry-driven response variation; it is not a trained competing model.

Four of the five families track in-range geometry-driven variation well, with mean seedwise \(R^2=0.848\)–0.963. The bonded inclusion is the important exception: its reference concentration factor varies only weakly across the tested geometries, so a small nearly uniform bias produces a negative raw \(R^2\) despite a concentration MARE of only 1.75%. The corresponding trend remains visible in the Pearson correlation and demeaned coefficient of determination reported in the manuscript.

Beyond the training range, the hole and notch families show substantial degradation. The directed extrapolation experiments are therefore treated as **coverage diagnostics**, not as evidence for arbitrary out-of-distribution robustness.

---

## Verification philosophy

The central objective of the project is not only to obtain low average errors, but to determine **which diagnostics are required before a learned mechanics field can be trusted for a localized stress concentration**.

The study separates the following levels:

1. variational energy;
2. displacement field;
3. distributed stress field;
4. section force;
5. stress-concentration magnitude;
6. peak location;
7. in-range versus out-of-range geometric coverage.

A good result at one level does not certify the next.

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

The parameters are affinely normalized using the bounds of the training parameter box. Training and in-range geometries therefore lie within the normalized training cube, while directed out-of-range geometries may extend beyond it. No clipping to the training interval is applied.

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

The raw neural output is converted into a displacement field that satisfies the prescribed essential boundary conditions exactly:

\[
u_{\theta,c}(X;p)
=
g_c(X;p)
+
\frac{\phi_c(X;p)}{\ell_c(p)}
N_{\theta,c}(X;p).
\]

Here \(g_c\) extends the prescribed displacement values and \(\phi_c\) vanishes on the corresponding essential boundary pieces.

Straight boundaries use distance primitives, while the bonded inclusion additionally uses a circular distance representation. Multi-boundary constraints are combined using approximate-distance/R-function and transfinite-interpolation constructions.

Traction-free boundaries arise naturally from the variational formulation rather than through an additional boundary penalty.

---

### Finite-strain mechanics

The material is a homogeneous compressible Neo-Hookean solid with

\[
E = 760\ {\rm MPa},
\qquad
\nu = 0.23.
\]

The three-dimensional deformation gradient is represented as

\[
F =
\begin{bmatrix}
F_{\parallel} & 0\\
0 & s
\end{bmatrix},
\qquad
J=s\,\det F_{\parallel},
\]

and the out-of-plane stretch \(s=F_{33}\) is obtained pointwise from the plane-stress condition

\[
P_{33}=0.
\]

The scalar closure is solved with detached Newton iterations followed by differentiable refinement steps so that the reduced plane-stress mechanics remain compatible with automatic differentiation.

The stored-energy density is

\[
\Psi(F)
=
\frac{\mu}{2}
\left[
\operatorname{tr}(F^T F)-3
\right]
-\mu\ln J
+
\frac{\lambda}{2}(\ln J)^2.
\]

Because the loading is displacement controlled and the remaining boundaries are traction-free, the total potential is the internal stored energy.

---

### Resampled variational objective

Each geometry is equipped with a graded triangular integration mesh containing approximately 1,400 triangles, refined near the stress-concentrating feature.

At every energy evaluation, three new points are sampled uniformly within every triangle. The resulting stochastic area-weighted energy estimator is

\[
\widehat{\Pi}_{\theta}(p)
=
\sum_e
\frac{A_e}{3}
\sum_{q=1}^{3}
\widehat{\Psi}
\left(
F_{\parallel}
\left(
u_\theta(X_{e,q};p)
\right)
\right).
\]

Per-step resampling is important. A controlled fixed-quadrature diagnostic showed that the neural field can exploit gaps between persistent integration points, producing an artificially low sampled energy while retaining very large field errors.

The training objective additionally contains a determinant guard against strongly nonphysical local Jacobians. The guard is a numerical regularizer rather than a proof of positive Jacobian everywhere.

No FEM displacement or stress field enters the training objective.

---

## Training specification

The final family-specific operators use:

- 64 training geometries per family;
- minibatches of 4 geometries;
- Adam optimization;
- learning rate \(3\times10^{-4}\) for 3,200 steps;
- cosine decay to \(3\times10^{-6}\) over a further 1,600 steps;
- 4,800 total optimization steps;
- gradient-norm clipping at 1.0;
- float32 arithmetic;
- three initializations per family.

All reported scores use the **last checkpoint** rather than selecting a checkpoint retrospectively from test-set performance.

Training requires several hours per model on the reference CPU setup. Runtime is hardware dependent, and no hardware-matched FEM-versus-operator inference benchmark or speedup claim is made.

---

## Geometry families

| Family | Training parameter box | Directed out-of-range perturbation |
|---|---|---|
| Filleted dog-bone | \(L_{\rm total}=40\)–70 mm, \(W_{\rm grip}=16\)–26 mm, \(W_{\rm gauge}=6\)–14 mm, \(R=8\)–20 mm | \(W_{\rm gauge}/W_{\rm grip}\ge 0.78\) while retaining the raw dimensional bounds |
| Open hole | \(W=16\)–26 mm, \(d/W=0.20\)–0.50, \(L_h/W=1.5\)–2.5 | \(d/W=0.55\)–0.60 |
| Bonded rigid inclusion | Same geometric intervals as the open-hole family | \(d/W=0.55\)–0.60 |
| Double U-notch | \(W=16\)–26 mm, \(t/W=0.10\)–0.30, \(\rho/t=0.25\)–1.0, \(L_h/W=1.5\)–2.5 | \(\rho/t=0.15\)–0.25 |
| Single U-notch | \(W=16\)–26 mm, \(t/W=0.10\)–0.30, \(\rho/t=0.25\)–1.0, \(L_h/W=2.0\)–3.0 | \(t/W=0.30\)–0.35 |

The out-of-range sets deliberately perturb a physically relevant concentration-controlling ratio while retaining the remaining descriptors inside the training box. They are mechanism-oriented extrapolation tests rather than unrestricted samples from arbitrary geometries.

---

## Finite-element reference

The verification suite contains a nonlinear finite-strain finite-element solver using P1 triangular elements, the same constitutive formulation and the same component-wise displacement boundary data as the neural formulation.

The reference implementation is checked through:

- patch tests;
- manufactured solutions;
- uniaxial tests;
- mesh-refinement studies;
- classical stress-concentration benchmarks for open holes, shallow notches and bonded rigid inclusions.

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

For the plate families,

\[
K_N
=
\frac{\max \sigma_{\rm vm}}
     {N/A_{\rm gross}},
\]

where \(N\) is the axial section force per unit reference thickness.

For the dog-bone,

\[
K_g
=
\frac{\max \sigma_{\rm vm}}
     {\langle \sigma_{\rm vm}\rangle_{\rm gauge}}.
\]

These are precisely defined finite-deformation quantities used consistently for neural and FEM fields; they should not be interpreted as conventional small-strain textbook \(K_t\) values.

---

### Field errors

The evaluation records include:

- axial displacement error;
- transverse displacement error;
- area-weighted von Mises field error;
- section-force error;
- concentration-factor error;
- sampled peak-stress error.

---

### Peak-location diagnostics

The evaluation also records the reference and predicted peak-element centroids.

Peak localization is assessed using:

- discrete peak-element coincidence;
- Euclidean peak offset;
- offset normalized by the relevant feature radius;
- the FEM reference stress evaluated at the operator-predicted peak location.

The final quantity distinguishes a harmless mesh-scale displacement of a near-equal maximum from a genuinely incorrect critical location.

---

## Held-out evaluation

Each family contains:

- 12 held-out in-range geometries;
- 8 directed out-of-range geometries;
- 3 initialization seeds.

The in-range sets are stratified with respect to the concentration-controlling geometric ratio.

Out-of-range sets change one physically important descriptor beyond its training coverage while retaining the others within their prescribed bounds.

The purpose is to distinguish interpolation from directed geometric extrapolation, not to claim robustness to arbitrary topology.

---

## Controlled methodological diagnostics

Several additional studies are retained because they help explain the observed success and failure modes.

### Geometry representation

A point-cloud/attention-based geometry representation was examined on the dog-bone family. Under the energy objective it captured the global taper ratio much more effectively than the local fillet radius.

Replacing the point-cloud representation by the exact four geometry parameters substantially reduced the held-out concentration and transverse-displacement errors at matched model capacity.

This motivates the use of exact descriptors when such parameters are available. It is not a claim that point-cloud encoders are unsuitable for non-parameterized geometries.

### Supervised representability

The same conditional field architecture can represent concentrated stresses accurately when directly supervised on stress data.

Direct displacement supervision can nevertheless produce very small displacement errors while retaining appreciably larger stress and concentration errors, demonstrating that displacement accuracy alone does not certify derivative-sensitive quantities.

### Optimization schedule

Cosine annealing stabilizes terminal predictions and reduces the seed spread for several families.

Paired 2,400- and 4,800-step evaluations are retained to quantify the influence of optimization budget.

### Alternative-formulation probes

Strong-form and fixed-test-space variational residual formulations are retained as diagnostic implementations.

They are used to study derivative cost and weak-form blind directions, not as matched competitive benchmarks against the final energy formulation.

---

## Repository layout

```text
config.py
    Material, loading, network and training configuration.

main.py
    Main training entry point retained from the development workflow.

models/
    Parameter encoder, spatial decoder, cross-attention/FiLM modules and
    positional/Fourier encoding.

geometry/
    Parametric geometry families, meshing and approximate-distance-function
    boundary constructions.

physics/
    Compressible Neo-Hookean model, plane-stress mechanics, energy form,
    strong-form and weak-form diagnostic implementations.

training/
    Family-specific trainers and final training workflows.

verification/
    FEM reference solver, manufactured-solution checks, held-out geometry
    generators, scoring routines and verification studies.

verification/phase10/
    Diagnostic strong-form and variational-formulation studies.

eval/
    Field-comparison and dense-evaluation utilities.

tests/
    Physics and numerical unit tests.

docs/
    Development records, evaluation summaries and machine-readable score data.

agent/
llm_agents/
    Optional inference and agentic interfaces; these are not part of the
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

The training workflow stores checkpoints, optimization histories and monitoring records under the configured results directory.

The verification routines solve and cache the required FEM references on first use and subsequently evaluate the trained operators against the corresponding held-out geometries.

Example scoring commands:

```bash
python -m verification.phase9_5_score
python -m verification.phase9_6_score
```

Some optional diagnostics require additional dependencies beyond the core environment. See the corresponding module documentation for those cases.

---

## Development record

The repository developed through a sequence of documented numerical and methodological stages. The `docs/` directory retains the associated plans, verification notes and result summaries for transparency and reproducibility.

Major stages include:

| Stage | Main topic |
|---|---|
| 0–1 | Correctness audit, FEM reference and verification hierarchy |
| 2 | Variational/energy formulation |
| 3–5 | Transverse field, geometry representation and numerical stabilization |
| 6 | Metrics, numerical noise, supervised representability and re-evaluation |
| 7 | Boundary conditions and grip idealization |
| 8 | Multi-family geometry formulation and admissible boundary construction |
| 9 | Optimization schedule and final operator recipe |
| 10 | Diagnostic strong-form and weak-form investigations |

These records document the development history of the implementation. The scientific results of the manuscript are defined by the final evaluation protocol described in the paper.

---

## Optional inference interfaces

The repository also contains optional inference and agentic interfaces.

`agent/` provides reliability-gated inference utilities based on mechanics diagnostics.

`llm_agents/` contains an experimental LangGraph/Streamlit interface for querying trained models through predictor, optimization, diagnosis and reporting components.

These interfaces are **not part of the quantitative evidence or claims of the accompanying manuscript**.

---

## Scope and limitations

The evidence in this repository supports the reported conclusions only within the tested scope.

- One separately trained operator is used for each geometry family.
- No cross-family weight transfer is demonstrated.
- One compressible Neo-Hookean material parameter set is considered.
- One imposed displacement level is evaluated.
- The reported mechanics use finite-strain plane stress.
- Generalization to new materials, load amplitudes, stress states, boundary-condition topologies or arbitrary CAD geometry is not demonstrated.
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

The training objective contains no FEM displacement or stress labels. FEM references are used for monitoring, verification and post-training assessment.

All reported operators are family specific.

---

## Citation

This repository accompanies the manuscript:

**A. Dean, B. Bahtiri, M. K. Khraisheh**

**“Energy-trained geometry-conditioned neural operators for stress concentrations in finite-strain solids: verification hierarchy, failure modes and extrapolation limits.”**

Citation details will be updated upon publication.

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Authors

**Betim Bahtiri and Aamir Dean**  

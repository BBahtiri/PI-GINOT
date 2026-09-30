# PI-GINOT — a physics-informed neural operator for stress concentrations

PI-GINOT learns the **solution operator of finite-strain hyperelasticity on
families of notched and holed specimens**: geometry in, displacement and stress
fields out. It is trained **without any simulation data**. The loss is the
**potential energy** of a compressible Neo-Hookean solid, integrated over each
specimen with resampled quadrature (the Deep Ritz / energy form). The
Dirichlet conditions are imposed exactly by a distance-function layer.

<p align="center">
 <img src="docs/pi_ginot_architecture_animated.gif" alt="PI-GINOT architecture: geometry branch, neural-operator trunk, and energy-based physics loss" width="100%">
</p>

Every number below is scored against an **independent, self-verified finite
element reference**, on **held-out geometries**, over **three seeds**. The
comparisons behind them were pre-registered.

## Results

**Stress concentration factor K_t**, the energy method trained from scratch
for 4,800 steps, one operator per family:

| specimen family | in-range \|K_t error\| | R² | predict-the-mean baseline | out-of-range \|K_t error\| |
|---|---|---|---|---|
| dog-bone (fillet) | **1.01%** | +0.85 | 2.43% | 2.00% |
| plate with a central hole | **1.29%** | +0.95 | 5.94% | 7.45% |
| plate with a bonded rigid inclusion | **1.75%** | −0.12 | 1.60% | 0.85% |
| double-edge U-notch | **2.31%** | +0.96 | 12.28% | 12.56% |
| single-edge U-notch, clamped grips | **2.20%** | +0.96 | 15.86% | 11.33% |

- **Test sets.** 12 in-range and 8 out-of-range held-out geometries per family.
  "Predict-the-mean" is the error of always predicting the family's mean K_t.
- **In range**, four of five families are 2.4–7.2× better than that baseline.
  Von Mises fields are within 0.5–1.3% relative L2 of the FEM reference.
- **The inclusion is the known exception.** Its K_t barely varies across
  geometries (coefficient of variation 1.9%), and the model carries a
  one-signed −1.75% bias.
- **Out of range** (the hole and notch parameters pushed beyond the training
  box), errors are 7–13%. That is the largest remaining error.
- **Cost:** about 4–5 h per run on two CPU cores, no GPU.

Full results, intervals and verdicts are in
[`docs/phase9_6_results.md`](docs/phase9_6_results.md) and
[`docs/phase9_5_results.md`](docs/phase9_5_results.md).

**What did not work.** Documented in [`docs/phase10_log.md`](docs/phase10_log.md)
and [`docs/phase9_results.md`](docs/phase9_results.md):
- **The strong form** (the original residual loss, arXiv 2607.23299) converges
  to the trivial uniform stretch on the operator.
- **Weak forms with fixed test spaces** (hp-VPINN / FastVPINNs, and Lagrange
  test functions) find spurious fields with near-zero residual at operator
  scale: 573k parameters against a few thousand test equations.
- **L-BFGS polishing** overfits the fixed quadrature points.

## The method

**Model.** A branch–trunk operator:
- a geometry encoder (point cloud, or the family's shape parameters);
- a cross-attention decoder with FiLM conditioning and NeRF positional
  encoding;
- a hard Dirichlet layer built from approximate distance functions
  (`geometry/adf.py`), so u = ū holds exactly on every Dirichlet segment;
- about 573k trainable parameters.

The reported results condition the decoder on the family's shape parameters
(`verification/supervised_ceiling.OracleConditioned`).

**Loss: the energy form** (`physics/weak_form.py`):
- Π(u) = ∫ W(F) dA, compressible Neo-Hookean, plane stress, E = 760 MPa,
  ν = 0.23. Displacement-controlled, so there is no external work term.
- Integrated on a ~1,400-triangle mesh of each specimen, with 3 stratified
  points per triangle, redrawn every step.
- A barrier penalises J < 0.05.

**Training** (`training/run_p94.py`):
- a bank of 64 geometries per family, batch 4;
- Adam 3e-4 for 3,200 steps, then cosine to 3e-6 over 1,600;
- grad-norm clip 1.0, float32.

**Specimen families** (`geometry/families.py`, and `config.py` for the dog-bone):

| family | parameters (training box) | out-of-range direction |
|---|---|---|
| dog-bone | L 40–70, W_grip 16–26, W_gauge 6–14, R_fillet 8–20 mm | gauge/grip taper beyond the bank |
| open hole | W 16–26 mm, d/W 0.2–0.5, L/W 1.5–2.5 | d/W 0.55–0.60 |
| rigid inclusion | as the open hole, bonded (u = 0) on the arc | d/W 0.55–0.60 |
| double U-notch | W 16–26 mm, t/W 0.1–0.3, ρ/t 0.25–1.0, L/W 1.5–2.5 | ρ/t 0.15–0.25 |
| single U-notch | W 16–26 mm, t/W 0.1–0.3, ρ/t 0.25–1.0, L/W 2.0–3.0 | t/W 0.30–0.35 |

## Verification apparatus

- **FEM reference** (`verification/fem/`, `verification/family_fem.py`):
  - a P1 finite-strain Newton solver;
  - checked against manufactured solutions, patch tests and mesh convergence;
  - Level-3 checks against handbook K_t for shallow notches and holes.
- **Held-out sets** (`verification/heldout_sets.py`, `geometry/families.py`):
  in-range sets stratified inside the training box, and out-of-range sets
  beyond it. Both use seeds never used for training.
- **Scoring:**
  - `verification/family_score.py` scores K_t, the section force, and u / v /
    von Mises relative L2;
  - the energy gap to a Richardson-extrapolated FEM energy (h, h/2, h/4) is a
    convergence monitor.
- **Statistics:**
  - paired comparisons (same seed, initialisation and batch stream), with a
    two-way cluster bootstrap over geometries and seeds;
  - since Phase 6, predictions are registered before the runs
    (`docs/*_plan.md`);
  - the Phase 9 and 10 results were independently recomputed, and the
    corrections are folded into each results document.

## Quick start

```bash
pip install -r requirements-core.txt        # torch, numpy, scipy, matplotlib, pytest
python -m pytest -q tests                   # physics unit tests

# train one family (the paper recipe; ~4-5 h on 2 CPU cores)
python -m training.run_p94 double_notch 0 --steps 4800 --t-const 3200 \
    --root verification/results/phase9/p94x

# score the finished 4,800-step runs against the FEM references
# (references are solved and cached on first use, under verification/results/)
python -m verification.phase9_5_score              # notch families
python -m verification.phase9_6_score              # dog-bone, open hole, inclusion
```

- Families: `dogbone`, `open_hole`, `inclusion`, `double_notch`,
  `single_notch`.
- Outputs (checkpoints, histories, monitors) go under
  `verification/results/`, which is not versioned.
- The score files behind every reported number are in `docs/`:
  `docs/phase9_5_scores.json` holds the 4,800-step scores of all five
  families, and `docs/phase9_4_scores.json` the 2,400-step scores.
- The manufactured-solution checks (`verification/mms.py`) and the Phase 10
  weak-form gates also need `sympy`.
- The FastVPINNs library comparison (G0) runs in a separate environment with
  `fastvpinns` and TensorFlow.

## Repository layout

```
config.py                  material, loading, network and training configuration
main.py                    original training entry point (dog-bone, residual/energy forms)
models/                    encoder, decoder, attention/FiLM modules, positional encoding
geometry/                  specimen shapes and families, meshing, distance-function BCs
physics/                   Neo-Hookean model, energy form, strong form, weak forms (vpinn/)
training/                  trainers: dog-bone, per-family, formulation comparison, 9.4-9.6 runner
verification/              FEM solver, manufactured solutions, held-out sets, scorers, studies
  phase10/                 strong vs energy vs weak-form gates (G0-G5)
eval/                      field comparison against the FEM reference, frozen dense eval set
tests/                     physics unit tests
docs/                      the research log: plans, results and score files, phase by phase
agent/, llm_agents/        reliability-gated inference and an LLM agent interface (Streamlit)
```

## Research log

The project was run as a sequence of pre-registered phases. Each phase has a
plan and a results document in `docs/`:

| phase | topic | document |
|---|---|---|
| 0–1 | correctness audit; FEM reference and verification hierarchy | `phase0_verification.md`, `phase1_verification.md` |
| 2 | the energy form | `phase2_energy_form.md`, `phase2_operator.md` |
| 3–5 | transverse field, geometry encoder, hardening | `phase3_*`, `phase4_*`, `phase5_*` |
| 6 | metrics, noise floor, supervised ceiling, re-test | `phase6_*.md` |
| 7 | boundary conditions and grip model | `phase7_boundary_conditions.md` |
| 8 | five specimen families, distance-function BCs | `phase8_multi_geometry.md`, `phase8_tier1.md` |
| 9 | optimizer: annealing, L-BFGS, the final recipe | `phase9_*.md` |
| 10 | strong form vs energy vs weak forms (hp-VPINN / FastVPINNs) | `phase10_formulations_plan.md`, `phase10_log.md` |

## Agentic interface (optional)

`agent/` wraps inference with reliability gates. The equilibrium residual, the
section-force consistency and a confidence score decide whether a prediction
is accepted.

`llm_agents/` is a LangGraph and Streamlit multi-agent app for querying the
model:
- **Predictor**: predictions on new geometries.
- **Optimizer**: searches the geometry space.
- **Diagnostician**: explains low-confidence predictions.
- **Reporter**: writes up results.

It is deployable with `docker compose up --build` or as a Databricks App
(`app.yaml`). It needs a trained checkpoint at `checkpoints/best.pt` and an LLM
API key (see `.env.example`). This layer was built for the original dog-bone
model.

## Limitations

- **Out-of-range error** (7–13% on the hole and notches) is the largest
  error. It falls with longer training, but part of it is systematic.
- **The inclusion's small bias** (−1.75%, one-signed) does not respond to
  training.
- **Plane stress, a single load level** (1 mm end displacement) and
  two-dimensional quarter or half models.
- **One operator per family.** A single operator across families is future
  work.

## Citation

The original residual-form model is described in arXiv:2607.23299. The
energy-form results in this repository supersede its numbers. Please cite the
preprint and this repository.

## License

MIT — see [LICENSE](LICENSE).

## Authors

Betim Bahtiri and Aamir Dean

# CLAUDE.md

This file is the project-level engineering constitution for MTO-Net. Claude Code must treat it as the first source of truth when implementing, auditing, testing, or extending the repository.

The goal is not to build another molecular property predictor. The goal is to implement a full DetaNet-based MTO-Net system that can support a Nature-level scientific claim:

> Local equivariant tensor fields can be explicitly assembled, under symmetry constraints, into stable, transferable, and chemically meaningful molecule-level response modes.

Every code change, experiment, test, figure, and report must serve this claim or a clearly named supporting claim.

---

## 1. Core Scientific Position

MTO-Net is a representation principle inserted between local equivariant molecular features and final molecular observables.

The intended architecture is:

```text
molecular graph + 3D coordinates
  -> DetaNet backbone
  -> atom-centred equivariant tensor fields H_i^(l)
  -> MTO molecular tensor mode assembly O_k^(l)
  -> Clebsch-Gordan tensor interactions + invariant nonlinear gates
  -> task readouts / MLP heads
  -> scalar, vector, tensor, or spectral targets
```

The project must never be framed in code, reports, or comments as merely "a better MLP head", "attention pooling", or "DetaNet with extra parameters". The MTO layer is an explicit local-to-global molecular tensor mode assembly layer.

### 1.1 What MTO Is

- A bank of molecule-level latent tensor modes assembled from atom-level equivariant tensor fields.
- A symmetry-constrained structured pooling mechanism that preserves mode index and tensor order.
- A testable response representation: it can be audited for equivariance, ablated, frozen, transferred, aligned across seeds, and chemically interrogated.
- A learned nonlinear response-oriented analogue of orbital-like local-to-global composition in equivariant representation space.

### 1.2 What MTO Is Not

- Not a quantum-chemical molecular orbital.
- Not a Kohn-Sham orbital, wavefunction, Hamiltonian eigenstate, density matrix, or electronic state.
- Not a replacement for solving the Schrodinger equation.
- Not a claim that learned latent modes are physically real orbitals.
- Not scalar-only pooling with an orbital name attached.

Reports may use the "molecular tensor orbital" language, but must always keep this boundary explicit.

---

## 2. Non-Negotiable Engineering Rules

These rules are hard constraints. If any rule is violated, stop and fix the implementation before running large experiments.

### 2.1 True Tensor MTO Only

MTO must consume true DetaNet tensor irreducible representation channels.

Required input conceptually:

```text
h0: [num_atoms, C0] or [num_atoms, C0, 1]      # l = 0
h1: [num_atoms, C1, 3]                         # l = 1
h2: [num_atoms, C2, 5]                         # l = 2
h3: [num_atoms, C3, 7] if DetaNet exposes it   # l = 3
```

Scalar-only MTO is allowed only as an ablation baseline named explicitly as `scalar_only_mto`. It must never be reported as the main model.

### 2.2 Preserve Representation Types

Never arbitrarily concatenate different tensor orders and pass them through an ordinary MLP as if they were scalar features.

Allowed operations:

- linear mixing within the same irreducible order;
- invariant routing weights generated from scalar/invariant information;
- signed invariant coefficients;
- Clebsch-Gordan tensor products;
- tensor norms and scalar contractions;
- invariant scalar gates multiplying equivariant tensors;
- readouts that respect output tensor type.

Disallowed operations:

- mixing l=0, l=1, l=2, l=3 as ordinary channels without representation-aware operations;
- applying elementwise nonlinearities directly to m-components of non-scalar irreps;
- using ordinary batch norm or layer norm over tensor components unless equivariance is proven and tested;
- silently flattening tensor orders into a scalar vector before the MTO layer.

### 2.3 Equivariance Is a Test Requirement, Not a Comment

Every core tensor module must have numerical tests.

Required audits:

- DetaNet tensor adapter reconstruction error;
- Wigner-D transformation test for each exposed tensor order;
- MTO internal mode equivariance;
- output equivariance for scalar, vector, and rank-2 tensor readouts;
- translation invariance/equivariance as appropriate;
- permutation invariance under same-molecule atom reordering.

Target tolerance:

```text
float64 / deterministic small test: ideally 1e-7 to 1e-6
float32 / normal model path: ideally <= 1e-5
```

If a tolerance must be looser, document why in the audit report.

### 2.4 Matched Baselines Are Mandatory

Do not claim the MTO principle works unless it is compared against matched alternatives using the same DetaNet backbone and comparable training budget.

Minimum baselines:

- DetaNet original/direct readout;
- DetaNet + sum pooling;
- DetaNet + mean pooling;
- DetaNet + scalar attention pooling;
- DetaNet + tensor pooling without mode bank;
- DetaNet + global token or Set Transformer readout;
- DetaNet + scalar-only MTO;
- DetaNet + tensor MTO without CG coupling;
- DetaNet + tensor MTO without signed routing;
- DetaNet + tensor MTO without invariant gates.

Parameter count, training steps, dataset split, optimizer, random seeds, and metrics must be recorded.

---

## 3. Model Architecture Specification

### 3.1 DetaNet Backbone

Use DetaNet as the local equivariant tensor field generator.

The backbone must expose atom-level scalar and tensor channels before final property-specific readout. If the public DetaNet code does not expose these features cleanly, create a minimal adapter/wrapper rather than rewriting DetaNet wholesale.

The adapter must:

- preserve DetaNet's original behavior when MTO is disabled;
- expose typed tensor features by order;
- document the exact tensor layout;
- provide split/reconstruct utilities;
- include tests showing that splitting and reconstruction are exact up to numerical precision.

### 3.2 MTO Tensor Mode Assembly

For atom i, tensor order l, and molecular mode k, the MTO layer constructs:

```text
O_k^(l) = sum_i c_ki^(l) W_l H_i^(l)
```

where:

- `H_i^(l)` is the DetaNet atom-centred tensor feature;
- `W_l` is a representation-compatible channel mixing map within order l;
- `c_ki^(l)` is an invariant signed routing coefficient;
- `O_k^(l)` is the assembled molecule-level tensor mode.

Routing coefficients must be generated only from scalar/invariant information such as:

- atom scalar features;
- atom type embedding;
- learned mode embedding;
- tensor norms or scalar contractions;
- optional molecular scalar context.

The routing mechanism should support:

- normalized positive attention weights;
- signed modulation through `tanh` or equivalent bounded sign network;
- optional sparsity or entropy regularization;
- optional valence-adaptive masking over mode count K.

### 3.3 Mode Capacity K

Implement at least two K strategies:

```text
fixed_k: fixed number of molecular modes for all molecules
valence_adaptive_k: K based on molecular valence-electron count, with padding/masking
```

Do not claim each mode is an electron or occupied orbital. K is a capacity prior, not a physical occupation statement.

Recommended first implementation:

- start with `fixed_k` for robust batching and debugging;
- add `valence_adaptive_k` after tensor audits pass;
- run K-scaling experiments after the main model is stable.

### 3.4 Clebsch-Gordan Coupling

After initial mode assembly, implement representation-legal cross-order interactions:

```text
O_k^(l1) x_CG O_j^(l2) -> O_new^(L)
where |l1 - l2| <= L <= l1 + l2
```

The implementation may use e3nn or existing repository utilities. Prefer a proven library implementation over hand-coded CG coefficients unless the repo already has a tested local implementation.

CG coupling must be independently tested for equivariance.

### 3.5 Invariant Gating and Nonlinearity

Nonlinearity must preserve equivariance.

Allowed pattern:

```text
invariant_features = scalar_features + tensor_norms + scalar_contractions
gamma_k_l = MLP(invariant_features)
O_tilde_k^(l) = gamma_k_l * O_k^(l)
```

Never apply arbitrary nonlinear activation to non-scalar tensor components.

Gate ablations must be implemented:

- no gate;
- scalar-only gate;
- tensor-information gate using tensor norms/contractions;
- optional regularized gate.

### 3.6 Readouts

Readout heads must match target type.

Scalar targets:

- read from l=0 modes and invariant summaries;
- output invariant scalars.

Vector targets:

- read from l=1 modes;
- output vector components with equivariance tests.

Rank-2 tensor targets:

- combine isotropic scalar component and l=2 traceless component where appropriate;
- audit Cartesian conversion carefully;
- report Frobenius, isotropic, anisotropic, eigenvalue, and eigenvector metrics when labels permit.

Spectra:

- read from invariant summaries of the full mode bank;
- use spectral MSE/MAE, cosine similarity, peak position error, top-k peak recall, and optional Wasserstein distance;
- save representative spectra and failure cases.

---

## 4. Dataset Strategy

### 4.1 Main Dataset: QM9S

QM9S is the main dataset for the first full MTO-Net manuscript.

Primary tasks:

```text
Stage A:
  - dipole moment
  - polarizability / polarizability tensor
  - quadrupole, octupole, first hyperpolarizability if clean

Stage B:
  - IR spectrum
  - Raman spectrum
  - UV-Vis spectrum

Stage C:
  - NMR only if labels and alignment are clean
```

Before training, run a dataset audit:

- number of molecules;
- element set;
- atom count distribution;
- target availability;
- tensor target shapes;
- spectral grid definitions;
- missing values;
- NaN/inf counts;
- duplicate molecules;
- molecule ID alignment;
- train/validation/test split files.

Never assume a label exists because a paper mentions it. Inspect local files and record the actual available targets.

### 4.2 QMe14S as Nature-Level Upgrade

QMe14S should be used after QM9S mainline is stable.

Purpose:

- cross-dataset transfer;
- element OOD split;
- functional-group OOD split;
- IR/Raman/NMR spectral generalization;
- evidence that MTO is not a QM9S-specific trick.

Do not block the first implementation on QMe14S.

### 4.3 QM7-X as Conformer / Non-Equilibrium Upgrade

Use QM7-X after the main representation is stable.

Purpose:

- conformer perturbation;
- equilibrium-to-non-equilibrium transfer;
- smoothness of MTO subspaces under geometry changes;
- response alignment under structural deformation.

This is strong evidence for Nature-level generality, but not required for the first smoke-tested implementation.

---

## 5. Experiment Roadmap

### Phase 0: Repository and Data Audit

Deliverables:

```text
outputs/audit/code_state.json
outputs/audit/environment.txt
outputs/audit/dataset_audit.json
outputs/audit/target_table.csv
outputs/audit/detanet_tensor_layout.json
```

Do not begin full training until the code state and data state are auditable.

### Phase 1: Tensor Adapter and Mathematical Correctness

Required tests:

- tensor layout split/reconstruct;
- Wigner-D transformation for each tensor order;
- MTO internal mode equivariance;
- output equivariance;
- permutation invariance;
- translation handling.

Deliverables:

```text
outputs/audit/tensor_reconstruction_error.json
outputs/audit/tensor_equivariance_audit.json
outputs/audit/mto_internal_equivariance.json
outputs/audit/output_equivariance.json
```

Success criterion:

```text
All mathematical tests pass before any large experiment.
```

### Phase 2: Smoke Training

Run tiny controlled experiments before full QM9S:

```text
num_molecules: 16, 32, 128
epochs: 2 to 5
targets: mu, alpha, one spectral target if available
models: DetaNet direct, scalar-only MTO, true tensor MTO
```

Success criterion:

- forward/backward pass works;
- no NaN or exploding gradients;
- training loss decreases;
- tensor audits still pass after integration;
- saved checkpoints reload correctly.

### Phase 3: QM9S Mainline

Run at least 5 seeds for the final mainline if compute allows.

Models:

- DetaNet direct;
- pooling baselines;
- attention/global-token baselines;
- scalar-only MTO;
- full tensor MTO.

Targets:

- dipole moment;
- polarizability and tensor response targets;
- IR;
- Raman;
- UV-Vis.

Metrics:

- scalar MAE/RMSE/R2;
- tensor Frobenius MAE;
- isotropic/anisotropic error;
- spectral MAE/MSE;
- spectral cosine similarity;
- peak-position MAE;
- top-k peak recall.

### Phase 4: Full Ablation

Ablations:

- no tensor channels;
- no signed routing;
- no CG coupling;
- no invariant gates;
- no mode bank;
- different K;
- fixed K vs valence-adaptive K;
- l=0 only, l<=1, l<=2, l<=3.

All ablations must be run under comparable parameter and training budgets.

### Phase 5: Representation Stability

Run across seeds and compare MTO subspaces.

Use metrics that are invariant to:

- mode permutation;
- sign flips;
- rotations within degenerate or near-degenerate subspaces.

Recommended metrics:

- principal angles;
- subspace overlap;
- Procrustes alignment;
- CKA/SVCCA for representation comparison.

Do not compare raw mode slots naively and call the result instability.

### Phase 6: Reuse and Transfer

Required:

- frozen MTO probe;
- frozen DetaNet hidden-state probe;
- frozen pooling baseline probe;
- stage transfer between tensor properties and spectra;
- low-data fine-tuning.

Main question:

> Does MTO learn reusable response information rather than task-specific hidden variables?

### Phase 7: Chemical Meaning and Falsifiability

Interpretability must be tested, not asserted.

Required analyses:

- functional group or motif enrichment;
- atom-type fallback only if RDKit/SMARTS labels are unavailable, and label it honestly;
- matched molecular pairs;
- fragment intervention;
- mode masking;
- attribution robustness across methods.

Control for confounders:

- molecule size;
- atom count;
- heteroatom count;
- composition;
- target magnitude;
- training-set frequency.

Never claim "functional-group enrichment" if the analysis only used atom-type proxies.

### Phase 8: Nature-Level Upgrade

Only after the QM9S story is stable:

- QMe14S external transfer;
- QMe14S element OOD;
- QMe14S functional-group OOD;
- QMe14S IR/Raman/NMR spectral generalization;
- QM7-X conformer/non-equilibrium validation;
- curated experimental spectra case study;
- optional DFT/TDDFT design loop.

---

## 6. Figure and Report Outputs

All figures must be generated from saved result tables, not manually typed numbers.

Recommended main figures:

```text
Fig. 1: Concept and architecture
Fig. 2: Tensor adapter and equivariance audit
Fig. 3: QM9S response prediction
Fig. 4: Matched baselines and ablations
Fig. 5: Seed stability and frozen probes
Fig. 6: Chemical interpretability and interventions
Fig. 7: QMe14S/QM7-X generalization
Fig. 8: Experimental spectra or design-level validation
```

Every generated figure should have:

- source data CSV/JSON;
- plotting script;
- figure in PNG and PDF/SVG if possible;
- short caption draft;
- claim supported by the figure;
- limitations or failure cases.

Reports should be saved under:

```text
outputs/reports/
```

Use clear filenames:

```text
tensor_audit_report.md
qm9s_mainline_report.md
ablation_report.md
stability_report.md
transfer_report.md
interpretability_report.md
artifact_audit_report.md
```

---

## 7. Coding Standards

### 7.1 Work Style

Before editing:

- inspect existing code patterns;
- identify the DetaNet feature extraction path;
- identify existing train/eval/config conventions;
- prefer local abstractions already used in the repo.

During editing:

- keep changes scoped;
- preserve backwards compatibility;
- avoid broad rewrites;
- write tests next to the modules they protect;
- use typed configs when the repo already uses config objects;
- keep tensor shapes explicit in docstrings and assertions.

After editing:

- run unit tests;
- run mathematical audits;
- run smoke training;
- save a concise report.

### 7.2 Shape Discipline

Every tensor-heavy module must document shapes.

Example:

```python
# h_l: [num_atoms, channels_l, 2*l + 1]
# coeff: [num_modes, num_atoms, channels_or_1]
# o_l: [num_modes, channels_out_l, 2*l + 1]
```

Use assertions for:

- tensor order dimension equals `2*l + 1`;
- molecule batch indices are valid;
- mode masks match K;
- no NaN/inf in inputs and outputs during debug mode;
- reconstructed tensor shape matches original flat DetaNet feature.

### 7.3 Reproducibility

Every experiment must save:

- command;
- config;
- seed;
- data split;
- git commit/hash if available;
- environment summary;
- model parameter count;
- training curves;
- checkpoint path;
- metrics JSON/CSV.

Use deterministic settings where practical for audits. For full training, record any nondeterminism.

### 7.4 Logging

Training logs must include:

- train/validation loss;
- target-specific metrics;
- gradient norm;
- learning rate;
- gate statistics;
- routing entropy;
- mode usage statistics;
- NaN/inf checks;
- early plateau detection.

Mode collapse should be detectable from logs.

---

## 8. Testing Requirements

Minimum test groups:

```text
tests/test_detanet_tensor_adapter.py
tests/test_mto_shapes.py
tests/test_mto_equivariance.py
tests/test_cg_coupling_equivariance.py
tests/test_gating_equivariance.py
tests/test_readout_equivariance.py
tests/test_training_smoke.py
tests/test_checkpoint_reload.py
```

The exact filenames may follow repo conventions, but the coverage must exist.

Tests must include:

- CPU small cases;
- single-molecule and batched-molecule cases;
- l=0 only;
- l=1;
- l=2;
- l=3 if supported;
- molecules with different atom counts;
- same-element atom permutation;
- random rotation/reflection;
- checkpoint save/load.

Do not run full experiments if these tests fail.

---

## 9. Claim Discipline

Every claim in reports must map to evidence.

Allowed claims after tensor audits pass:

- "The implemented MTO modes obey the tested equivariance constraints."
- "The model consumes true tensor channels rather than scalar-only features."

Allowed claims after matched baselines:

- "Explicit molecular tensor mode assembly improves selected response tasks relative to matched readouts."

Allowed claims after stability/transfer:

- "MTO modes form a reusable response representation under the tested conditions."

Allowed claims after chemical controls:

- "Certain modes are statistically associated with chemically meaningful motifs under controlled analyses."

Avoid:

- "MTOs are real molecular orbitals."
- "The model discovers quantum wavefunctions."
- "MTO is proven universal."
- "MTO is Nature-level" without cross-dataset and real-use evidence.
- "Functional group enrichment" when only atom-type proxies were used.

---

## 10. Immediate Implementation Plan for Claude Code

When starting from the repository, proceed in this order.

### Step 1: Repository Reconnaissance

Inspect:

- model files;
- DetaNet implementation/wrapper;
- training scripts;
- dataset loaders;
- config system;
- current tests;
- current reports/outputs.

Write a short implementation map before editing.

### Step 2: DetaNet Tensor Adapter

Implement or repair:

- typed tensor extraction;
- split/reconstruct utilities;
- shape reporting;
- adapter tests;
- tensor transformation audit.

Stop here until tests pass.

### Step 3: Minimal Tensor MTO Layer

Implement:

- fixed K;
- invariant routing;
- signed routing;
- l-wise channel mixing;
- tensor mode output;
- shape tests;
- internal equivariance tests.

Do not add CG complexity until minimal mode assembly is correct.

### Step 4: CG Coupling and Gating

Implement:

- representation-legal CG interactions;
- invariant tensor-information gates;
- ablation switches;
- equivariance tests.

### Step 5: Readouts

Implement:

- scalar readout;
- vector readout;
- rank-2 tensor readout;
- spectral readout;
- output equivariance tests.

### Step 6: Smoke Training

Run tiny experiments:

- DetaNet direct;
- scalar-only MTO;
- true tensor MTO;
- at least one scalar target and one response/spectral target.

Save smoke report.

### Step 7: Formal Experiment Harness

Implement config-driven training/evaluation for:

- 5-seed QM9S mainline;
- matched baselines;
- ablations;
- stability metrics;
- frozen probe;
- stage transfer;
- interpretability analyses.

### Step 8: Artifact Audit

Before calling work complete, audit:

- tests;
- configs;
- checkpoints;
- metrics;
- figures;
- source data;
- reports;
- claim-evidence table.

---

## 11. Definition of Done

A task is not done when code runs once.

A task is done only when:

- implementation follows symmetry constraints;
- unit tests pass;
- equivariance audits pass;
- smoke training passes;
- outputs are reproducible;
- claims are documented conservatively;
- failure cases are recorded;
- no scalar-only path is accidentally used as the main model;
- reports explain what was proven and what remains unproven.

For the full project, the minimum serious manuscript-ready package is:

```text
1. True tensor MTO implementation
2. DetaNet tensor adapter audit
3. Internal MTO equivariance audit
4. QM9S main response prediction
5. Matched readout baselines
6. Full MTO ablation
7. Seed subspace stability
8. Frozen probe / stage transfer
9. Chemical interpretability with controls
10. Reproducible figures and source data
```

The Nature-level upgrade additionally requires:

```text
1. QMe14S cross-dataset transfer
2. QMe14S element / functional-group OOD
3. QM7-X conformer or non-equilibrium validation
4. Experimental spectra or DFT/TDDFT design-level evidence
5. Strong failure-case analysis
```

---

## 13. Next-Stage Nature Evidence Worktrees

The following worktrees (wt-06 through wt-14) implement the Nature-level evidence roadmap beyond the QM9S mainline.

| Worktree | Branch | Local | Server | Purpose |
|----------|--------|-------|--------|---------|
| wt-06-data-foundry | feat/06-data-foundry | yes | yes | Dataset registry, download manifests, ingestion/audit plans for QM7-X, QMe14S, experimental spectra, non-chemical data |
| wt-07-stability-transfer | feat/07-stability-transfer | yes | yes | Seed subspace overlap, principal angles, frozen probes, stage transfer, early plateau diagnostics |
| wt-08-chemical-interrogation | feat/08-chemical-interrogation | yes | yes | RDKit SMARTS, functional-group enrichment, matched molecular pairs, mode masking, attribution robustness |
| wt-09-qm9s-spectra | feat/09-qm9s-spectra | optional | yes | QM9S IR/Raman/UV-Vis spectra training, spectral metrics, peak-level analysis |
| wt-10-experimental-spectra | feat/10-experimental-spectra | yes | yes | NIST/SDBS/nmrshiftdb2/NMRexp registry, molecule matching, spectrum preprocessing, curated experimental case study |
| wt-11-external-generalization | feat/11-external-generalization | optional | yes | QM7-X conformer/non-equilibrium validation, QMe14S transfer, element OOD, functional-group OOD, spectra generalization |
| wt-12-universal-tensor-assembly | feat/12-universal-tensor-assembly | yes | yes | Non-chemical Tensor Mode Assembly: synthetic SO(3) multipole tasks, optional ModelNet40/ShapeNet later |
| wt-13-figures-reporting | feat/13-figures-reporting | yes | optional | Source-data-driven Nature-style figures, reports, source tables, claim-to-evidence maps |
| wt-14-final-artifact-audit | feat/14-final-artifact-audit | yes | yes | Final reproducibility audit, artifact manifest, result consistency checks, manuscript evidence audit |

### 13.1 Allowed Work per Worktree

**wt-06-data-foundry**: Write registry manifests, download scripts, checksum files, split definitions, ingestion/audit plans. Store only scripts, manifests, checksums, splits, and reports in Git. Treat acquired datasets as read-only. Do NOT download large datasets in setup tasks.

**wt-07-stability-transfer**: Run subspace overlap computations, principal angle analysis, frozen probe training, stage transfer experiments, plateau diagnostics. Compare MTO subspaces across seeds using principal angles, Procrustes alignment, CKA/SVCCA.

**wt-08-chemical-interrogation**: Implement RDKit SMARTS matching, functional-group enrichment with statistical controls, matched molecular pair analysis, mode masking experiments, attribution robustness (Integrated Gradients, occlusion, attention rollout). Always control for molecule size, atom count, heteroatom count, composition, target magnitude, and training-set frequency.

**wt-09-qm9s-spectra**: Train on QM9S IR/Raman/UV-Vis spectra. Implement spectral MSE/MAE, cosine similarity, peak position MAE, top-k peak recall, optional Wasserstein distance. Save representative spectra and failure cases.

**wt-10-experimental-spectra**: Curate 50-500 clean molecules from NIST/SDBS/nmrshiftdb2/NMRexp. Match by SMILES/InChIKey. Align spectra grids. Document solvent/phase/instrument limitations. Compare predicted vs calibrated spectra, peak positions, intensities, and MTO attributions. Include failure cases.

**wt-11-external-generalization**: Run QM7-X conformer perturbation, equilibrium-to-non-equilibrium transfer, QMe14S element OOD, functional-group OOD, spectra generalization. This is strong evidence for Nature-level generality.

**wt-12-universal-tensor-assembly**: Build synthetic SO(3)-structured local-to-global multipole tasks. Compare sum/mean pooling, attention pooling, global token, tensor pooling, and full TMA. Optional: rotated ModelNet40 point-cloud classification. Call the generic method "Tensor Mode Assembly", not MTO.

**wt-13-figures-reporting**: Generate all figures from saved result tables, not manually typed numbers. Each figure: source data CSV/JSON, plotting script, PNG and PDF/SVG, short caption draft, claim supported, limitations. Save under `outputs/reports/`.

**wt-14-final-artifact-audit**: Audit tests, configs, checkpoints, metrics, figures, source data, reports, claim-evidence table. Verify all claims map to evidence. Confirm no scalar-only path was accidentally used as the main model.

### 13.2 Forbidden Work

- No large dataset downloads without explicit user request.
- No Slurm/training jobs from wt-06 (data registry only).
- No modification of existing wt-00 through wt-05 worktrees.
- No deletion or renaming of existing branches or worktrees.
- No server operations outside `/data/home/scwc008/run/xxy`.
- No "functional-group enrichment" claims using only atom-type proxies (wt-08).
- No "MTOs are real molecular orbitals" claims (all worktrees).

### 13.3 Done Criteria per Worktree

- wt-06: Dataset manifests, download scripts, checksums, split files, audit plan committed; QM9S confirmed present on server.
- wt-07: Subspace overlap, principal angles, frozen probes, stage transfer, plateau diagnostics all run and saved as JSON/CSV.
- wt-08: SMARTS enrichment with controls, MMP analysis, mode masking, attribution robustness all run; no confounded claims.
- wt-09: QM9S spectra training complete with all spectral metrics; representative spectra and failure cases saved.
- wt-10: Curated experimental spectra registry, molecule matching, preprocessing, comparison plots, and case-study report.
- wt-11: QM7-X conformer and QMe14S transfer results with OOD metrics; failure cases documented.
- wt-12: Synthetic SO(3) multipole results showing TMA improvement over baselines; optional ModelNet40 results.
- wt-13: All figures generated from source data; claim-evidence table complete; reports saved.
- wt-14: Full artifact audit; reproducibility confirmed; claim-evidence mapping verified; manuscript evidence package ready.

---

## 14. Evidence Map

Each evidence claim maps to specific worktrees that must produce verifying results.

| Claim | Worktree(s) | Evidence Type |
|-------|-------------|---------------|
| Symmetry correctness | wt-02, wt-03 | Tensor reconstruction error, Wigner-D tests, MTO internal equivariance, output equivariance |
| Assembly necessity | wt-04 | Matched baselines: direct, sum/mean pooling, attention, global token, scalar-only MTO, no-CG, no-signed-routing, no-gate |
| Main training evidence | wt-05 | QM9S 5-seed mainline: dipole, polarizability, tensor response, scalar/vector/rank-2 readouts |
| Data foundation | wt-06 | Dataset registry, manifests, checksums, splits for QM7-X, QMe14S, experimental spectra |
| Stability and reuse | wt-07 | Subspace overlap across seeds, principal angles, frozen probes, stage transfer, plateau diagnostics |
| Chemical meaning | wt-08 | SMARTS enrichment with controls, MMP, mode masking, attribution robustness |
| Spectra on QM9S | wt-09 | IR/Raman/UV-Vis training, spectral metrics, peak analysis, representative spectra |
| Experimental reality | wt-10 | Curated NIST/SDBS/nmrshiftdb2 matching, spectrum comparison, MTO attribution on real spectra |
| External generalization | wt-11 | QM7-X conformer, QMe14S element/fg-OOD, spectra generalization |
| Universal method | wt-12 | Synthetic SO(3) multipole tasks, TMA vs baselines outside chemistry |
| Reporting | wt-13 | Source-data-driven figures, claim-to-evidence maps, Nature-style reports |
| Final audit | wt-14 | Reproducibility, artifact manifest, consistency checks, manuscript evidence package |

---

## 15. Planning Documents

Detailed planning documents are maintained alongside this CLAUDE.md:

- `docs/NATURE_WORKTREE_ROADMAP.md` — Full worktree roadmap with phases, dependencies, and milestones.
- `docs/DATA_FOUNDRY_PLAN.md` — Dataset registry, download manifests, and ingestion strategy.
- `docs/EXPERIMENTAL_SPECTRA_PLAN.md` — Curated experimental spectra case-study plan.
- `docs/UNIVERSAL_TENSOR_ASSEMBLY_PLAN.md` — Non-chemical Tensor Mode Assembly plan.

---

## 12. Final Reminder

The central engineering question is always:

> Does this code help prove that local equivariant tensor fields can be assembled into stable, transferable, chemically meaningful molecular response modes?

If the answer is no, the change is probably not part of the main project path.


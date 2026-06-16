# MTO-Net Nature Mainline Experiment Manual

Version: 2026-06-16

Companion file: `MTO-Net_Nature_Experiment_Plan.md`

This manual is designed as the Nature-mainline execution blueprint for MTO-Net. The companion experiment plan defines the general validation logic. This file translates that logic into a concrete evidence system: which datasets to use, what experiments to run, what figures to draw, what conclusions can be claimed, and how each result supports the central narrative of a Nature-level manuscript.

---

## 0. Nature-Level Central Thesis

The paper should not be framed as:

> MTO-Net is another molecular property prediction model with better MAE.

It should be framed as:

> Local equivariant tensor fields can be explicitly assembled, under symmetry constraints, into stable, transferable and chemically meaningful molecule-level response modes.

The central scientific object is therefore not only the final prediction error. It is the existence and usefulness of the MTO representation itself.

### 0.1 Four Main Claims

| Claim | Short name | What must be proven |
|---|---|---|
| C1 | Symmetry correctness | Internal MTO modes obey the intended SO(3)/O(3), translation and permutation transformation rules. |
| C2 | Assembly necessity | Explicit tensor-mode assembly is better than direct pooling, attention pooling, global tokens or scalar-only routing under matched conditions. |
| C3 | Response representation | MTO modes preserve information needed for molecular response: dipole, polarizability, higher-order tensors and spectra. |
| C4 | Scientific meaning | MTO modes are stable, transferable and chemically interrogable rather than arbitrary hidden states. |

### 0.2 Nature-Level Evidence Standard

For a Nature-mainline manuscript, every major claim needs at least two independent evidence types:

| Claim | Primary evidence | Secondary evidence |
|---|---|---|
| Symmetry correctness | Internal equivariance audit | Output equivariance audit and tensor shape audit |
| Assembly necessity | Matched architectural ablation on QM9S | K-scaling, size/complexity-stratified analysis |
| Response representation | QM9S tensor and spectra tasks | QMe14S external response transfer |
| Scientific meaning | Seed stability and frozen probes | Chemical interventions and experimental spectra case studies |
| Generality | QM7-X conformer perturbation | QMe14S element/function-group OOD |

---

## 1. Dataset System

### 1.1 Dataset Tiers

| Tier | Dataset | Role in paper | Why it matters |
|---|---|---|---|
| Main | QM9S | Main response benchmark | Contains scalar, vector, second-order, third-order tensor properties and IR/Raman/UV-Vis spectra for about 130K QM9-derived molecules. |
| Stability | QM7-X | Conformer and non-equilibrium validation | Contains about 4.2M equilibrium and non-equilibrium structures with 42 physicochemical properties, including response quantities. |
| External generalization | QMe14S | Element, functional-group and spectral OOD validation | Contains 186,102 molecules, 14 elements, 47 functional groups and IR/Raman/NMR spectra. |
| Real-spectrum check | NIST / SDBS / nmrshiftdb2 | Small curated experimental case study | Tests whether DFT-trained response modes remain useful for real experimental spectra. |
| Optional design loop | Custom DFT/TDDFT set | Demonstration of scientific use | Tests whether MTO modes can guide molecule editing or response optimization. |

### 1.2 Primary Sources to Cite

Use these sources in the manuscript and data section.

| Dataset | Source note |
|---|---|
| QM9S | Figshare dataset: QM9Spectra based on about 130K QM9 molecules, B3LYP/def-TZVP, scalar/vector/tensor properties, IR/Raman/UV-Vis spectra. |
| DetaNet/QM9S paper | Nature Computational Science / PubMed record: DetaNet predicts molecular spectra and high-order tensorial properties. |
| QM7-X | Scientific Data: about 4.2M equilibrium and non-equilibrium structures, 42 properties, response quantities including polarizability tensors. |
| QMe14S | JPCL / Figshare: 186,102 molecules, 14 elements, 47 functional groups, B3LYP/TZVP, IR/Raman/NMR and tensor properties. |
| NIST WebBook | NIST Chemistry WebBook: IR spectra for over 16,000 compounds and UV/Vis spectra for over 1,600 compounds. |
| SDBS | AIST Spectral Database for Organic Compounds: FT-IR, Raman, 1H NMR, 13C NMR, MS and ESR records. |
| nmrshiftdb2 | Open NMR database with organic structures, assigned spectra, raw data and peak lists under open-content terms. |

---

## 2. Global Experimental Principles

### 2.1 Fixed Protocols

All formal experiments should use locked protocols.

Required:

- fixed train/validation/test split per dataset;
- at least 5 random seeds for main QM9S experiments;
- at least 3 random seeds for expensive QMe14S experiments;
- identical backbone, depth, hidden size and training budget across readout variants;
- parameter-matched baselines whenever possible;
- raw metric and normalized metric both reported;
- uncertainty shown as mean +/- standard deviation or bootstrap confidence interval;
- all figures generated from saved result tables, not manually edited numbers.

### 2.2 Metrics

| Target type | Primary metrics | Secondary metrics |
|---|---|---|
| Scalar properties | MAE, RMSE, R2 | normalized MAE, rank correlation |
| Vector properties | vector MAE, norm MAE, angular error | equivariance error |
| Tensor properties | Frobenius MAE, isotropic error, anisotropy error | eigenvalue/eigenvector error |
| Spectra | spectral MSE/MAE, cosine similarity, peak-position MAE | Wasserstein distance, top-k peak recall |
| Representation stability | subspace overlap, principal angles, Procrustes distance | SVCCA/CKA |
| Interpretability | enrichment odds ratio, intervention effect size | matched-control delta, permutation p-value |
| OOD generalization | ID vs OOD metric gap | calibration error |

### 2.3 Baseline Families

Every major comparison should include these families:

| Baseline | Purpose |
|---|---|
| DetaNet direct readout | Tests whether DetaNet backbone alone is sufficient. |
| Mean/sum pooling readout | Tests ordinary permutation-invariant compression. |
| Attention pooling | Tests whether scalar attention is enough. |
| Set Transformer / global token readout | Tests whether a generic global latent is enough. |
| Scalar-only MTO | Tests whether tensor information is actually used. |
| Tensor pooling without explicit mode bank | Tests whether tensor aggregation alone is enough. |
| MTO without signed routing | Tests cancellation and reinforcement. |
| MTO without CG coupling | Tests cross-order tensor interaction. |
| MTO without invariant gates | Tests nonlinear response modulation. |
| MTO with different K | Tests mode capacity and valence-adaptive prior. |

---

## 3. Manuscript Figure Plan

### Main Figures

| Figure | Core question | Required panels |
|---|---|---|
| Fig. 1 | What is the new principle? | local tensor fields, direct pooling, MTO assembly, response readout, chemical analogy |
| Fig. 2 | Is the representation mathematically valid? | tensor layout, internal equivariance, output equivariance, failure of scalar-only interpretation |
| Fig. 3 | Does MTO improve molecular response learning? | QM9S scalar/vector/tensor tasks, IR/Raman/UV-Vis spectra, complexity-stratified performance |
| Fig. 4 | Is explicit assembly necessary? | matched baselines, ablations, K scaling, parameter-control plot |
| Fig. 5 | Are MTO modes stable and reusable? | seed subspace overlap, frozen probe, stage transfer, task similarity map |
| Fig. 6 | Are MTO modes chemically interrogable? | functional group enrichment, matched molecular pairs, mode masking, fragment intervention |
| Fig. 7 | Does the principle generalize? | QM7-X conformer perturbation, QMe14S element OOD, functional-group OOD, spectra transfer |
| Fig. 8 | Does it touch real chemical use? | NIST/SDBS/nmrshiftdb2 case study or DFT design loop, mode-guided molecule editing |

### Extended Data Figures

| Extended figure | Content |
|---|---|
| Extended Data Fig. 1 | Dataset audit tables and target distributions |
| Extended Data Fig. 2 | Training curves across seeds and early plateau analysis |
| Extended Data Fig. 3 | Full equivariance audit by tensor order and transformation type |
| Extended Data Fig. 4 | Full ablation table for all tasks |
| Extended Data Fig. 5 | Spectral examples and failure cases |
| Extended Data Fig. 6 | More functional group and matched-control analyses |
| Extended Data Fig. 7 | QMe14S OOD split definitions and distributions |
| Extended Data Fig. 8 | Experimental spectra matching and preprocessing checks |

---

## 4. Phase A: Dataset and Tensor Audit

### Experiment A1. Dataset Audit

| Item | Description |
|---|---|
| Dataset | QM9S, QM7-X, QMe14S |
| Purpose | Ensure every reported claim is based on actual available labels and clean molecule alignment. |
| Procedure | Parse all files, count molecules, atoms, elements, targets, tensor shapes, spectra shapes, missing values and duplicate identifiers. |
| Output files | `dataset_audit.json`, `target_table.csv`, `target_statistics.csv`, `dataset_summary.pdf` |
| Figure | Extended Data Fig. 1 |
| Expected conclusion | The training/evaluation system is auditable, reproducible and aligned with response-rich molecular labels. |
| Supports | C1, C3 |

Checklist:

- confirm QM9S has usable `mu`, `alpha`, tensor targets, IR, Raman and UV-Vis;
- confirm whether NMR labels are directly usable or should be excluded from the main text;
- confirm QMe14S label format and spectra grid;
- build canonical molecule IDs using SMILES/InChIKey when available;
- store split files, never regenerate them silently.

### Experiment A2. Tensor Adapter Audit

| Item | Description |
|---|---|
| Dataset | small QM9S subset |
| Purpose | Prove MTO consumes true tensor channels, not scalar proxies. |
| Procedure | Split DetaNet hidden tensors into irreps; reconstruct the flat tensor; rotate input; check Wigner-D transformation for each order. |
| Figure | Fig. 2a-b and Extended Data Fig. 3 |
| Success criterion | reconstruction error approximately zero; equivariance error around numerical precision to 1e-5 depending on precision. |
| Expected conclusion | The MTO layer is built on real equivariant tensor information. |
| Supports | C1, C2 |

### Experiment A3. Internal MTO Equivariance Audit

| Item | Description |
|---|---|
| Dataset | 1,000 QM9S molecules |
| Transformations | random rotations, reflections/inversion, translations, same-element permutations, combined transformations |
| Check | `O_k^l(gX) = D^l(g) O_k^l(X)` up to numerical error |
| Figure | Fig. 2c-d |
| Expected conclusion | MTO modes are themselves symmetry-valid molecular tensors. |
| Supports | C1 |

Do not only test final scalar outputs. The main claim requires internal MTO tensors to be audited.

---

## 5. Phase B: QM9S Main Response Suite

### Experiment B1. Scalar, Vector and Tensor Response Prediction

| Item | Description |
|---|---|
| Dataset | QM9S |
| Targets | energy, dipole moment, quadrupole moment, polarizability tensor, octupole moment, first hyperpolarizability, Hessian if usable |
| Models | MTO-Net, DetaNet direct, pooling baselines, attention pooling, global token |
| Metrics | MAE/RMSE/R2 for scalar; vector/tensor Frobenius errors; isotropic and anisotropic decomposition for tensor targets |
| Figure | Fig. 3a-c |
| Expected result | MTO advantage should grow for anisotropic, high-order and collective response targets. |
| Main conclusion | Explicit tensor-mode assembly is most useful when the target depends on molecule-wide cooperative response. |
| Supports | C2, C3 |

Important stratifications:

- molecule size;
- number of heavy atoms;
- number of heteroatoms;
- estimated conjugation length;
- polar vs nonpolar molecules;
- functional group count;
- isotropic vs anisotropic polarizability.

### Experiment B2. Spectral Prediction

| Item | Description |
|---|---|
| Dataset | QM9S |
| Targets | IR, Raman, UV-Vis |
| Models | same as B1 |
| Metrics | spectral MAE/MSE, cosine similarity, peak-position MAE, top-k peak recall, Wasserstein distance |
| Figure | Fig. 3d-f |
| Expected result | MTO should improve spectral shape and peak localization, especially for molecules with multiple interacting motifs. |
| Main conclusion | MTO modes provide a structured interface between local tensor fields and whole-molecule spectra. |
| Supports | C2, C3 |

Required plots:

- averaged metrics with uncertainty;
- representative spectra examples;
- failure cases;
- peak-level error distribution;
- spectral error versus molecular complexity.

### Experiment B3. Response Complexity Analysis

| Item | Description |
|---|---|
| Dataset | QM9S |
| Purpose | Show that MTO is not merely better on average, but specifically better where pooling should fail. |
| Groups | simple saturated molecules, polar molecules, conjugated molecules, donor-acceptor molecules, carbonyl/nitrile-containing molecules, high-anisotropy molecules |
| Figure | Fig. 3g-h |
| Expected result | MTO should show larger gains on high-complexity and high-anisotropy subsets. |
| Main conclusion | The gain is mechanistically aligned with the local-to-global response hypothesis. |
| Supports | C2, C3, C4 |

---

## 6. Phase C: Architecture Necessity and Ablation

### Experiment C1. Matched Readout Baseline Study

| Item | Description |
|---|---|
| Dataset | QM9S |
| Models | DetaNet direct, sum pooling, mean pooling, attention pooling, Set Transformer/global token, tensor pooling, MTO-Net |
| Control | same backbone, comparable parameter count, same training schedule |
| Figure | Fig. 4a-b |
| Expected result | MTO-Net should outperform generic readouts on response-rich targets. |
| Main conclusion | The improvement comes from explicit tensor-mode assembly, not only from backbone strength. |
| Supports | C2 |

### Experiment C2. Internal MTO Ablation

| Ablation | Scientific question |
|---|---|
| scalar-only MTO | Is tensor information essential? |
| no signed routing | Is constructive/destructive contribution important? |
| no CG coupling | Is cross-order tensor interaction important? |
| no invariant gate | Is nonlinear response modulation important? |
| no tensor norm features in gate | Does tensor-information gating matter? |
| fixed K | Does mode capacity affect response learning? |
| valence-adaptive K | Does chemically motivated capacity help? |

| Item | Description |
|---|---|
| Dataset | QM9S |
| Figure | Fig. 4c-e |
| Expected result | Removing tensor channels, CG coupling or gates should hurt tensor and spectra tasks more than simple scalar tasks. |
| Main conclusion | Each architectural component corresponds to a necessary part of the response-mode hypothesis. |
| Supports | C2, C3 |

### Experiment C3. Parameter and Compute Control

| Item | Description |
|---|---|
| Dataset | QM9S subset and full QM9S |
| Purpose | Prevent reviewer objection that MTO wins only because it has more parameters. |
| Procedure | Match parameter count by widening baseline readouts; compare training time, memory and accuracy. |
| Figure | Fig. 4f |
| Expected result | MTO remains favorable at matched or comparable parameter budgets. |
| Main conclusion | The gain is architectural, not only due to capacity. |
| Supports | C2 |

---

## 7. Phase D: Representation Stability and Reuse

### Experiment D1. Seed Subspace Stability

| Item | Description |
|---|---|
| Dataset | QM9S |
| Seeds | at least 5 |
| Representation | MTO mode bank before readout |
| Metrics | principal angles, subspace overlap, Procrustes distance, CKA/SVCCA |
| Figure | Fig. 5a-b |
| Expected result | Independent runs recover related subspaces after allowing slot permutation, sign freedom and subspace rotation. |
| Main conclusion | MTO is more stable than arbitrary hidden states or attention weights. |
| Supports | C4 |

Important:

- do not compare mode slots naively;
- align subspaces, not individual mode labels;
- separately analyze scalar, vector and rank-2 components.

### Experiment D2. Frozen Probe

| Item | Description |
|---|---|
| Dataset | QM9S |
| Procedure | Train MTO on one response family; freeze backbone and MTO; train linear or shallow heads for other response targets. |
| Pairs | `mu -> alpha`, `alpha -> IR/Raman`, `IR/Raman -> polarizability`, `UV-Vis -> dipole/transition-like labels if available` |
| Figure | Fig. 5c |
| Expected result | Frozen MTO should retain transferable response information better than frozen pooled baselines. |
| Main conclusion | MTO is a reusable response representation, not only a task-specific readout. |
| Supports | C3, C4 |

### Experiment D3. Stage Transfer

| Item | Description |
|---|---|
| Dataset | QM9S |
| Procedure | Pretrain on one stage, fine-tune on another with limited labels. |
| Stages | Stage A: tensor properties; Stage B: spectra; Stage C: external spectra or QMe14S |
| Figure | Fig. 5d-e |
| Expected result | MTO pretraining should improve low-data fine-tuning and convergence stability. |
| Main conclusion | MTO modes encode response coordinates shared across molecular observables. |
| Supports | C3, C4 |

### Experiment D4. Early Plateau and Optimization Stability

| Item | Description |
|---|---|
| Dataset | QM9S |
| Purpose | Address observed seed instability and early plateau behavior. |
| Procedure | Compare training curves, gradient norms, gate entropy, routing entropy and MTO norm statistics across seeds. |
| Figure | Extended Data Fig. 2 |
| Expected result | Failed seeds should correspond to identifiable collapse modes; improved initialization/gate regularization should reduce failures. |
| Main conclusion | Stability risks are understood and controlled. |
| Supports | C4 |

---

## 8. Phase E: Chemical Interpretability and Falsifiability

### Experiment E1. Functional Group Enrichment

| Item | Description |
|---|---|
| Dataset | QM9S, then QMe14S |
| Groups | carbonyl, nitrile, amine, alcohol, ether, aromatic, conjugated path, donor-acceptor motif, heteroatom-rich fragment |
| Method | RDKit SMARTS if available; fallback atom-type analysis only as supplementary |
| Figure | Fig. 6a-b |
| Expected result | Specific modes should enrich for chemically meaningful groups in property-specific ways. |
| Main conclusion | MTO modes are not only visually appealing; they statistically align with chemical motifs. |
| Supports | C4 |

Do not call atom-type enrichment functional-group enrichment. If RDKit SMARTS is unavailable, label it as atom-type proxy only.

### Experiment E2. Matched Molecular Pair Analysis

| Item | Description |
|---|---|
| Dataset | QM9S/QMe14S |
| Pairs | same scaffold with carbonyl addition/removal, nitrile substitution, donor/acceptor substitution, conjugation extension, heteroatom replacement |
| Figure | Fig. 6c-d |
| Expected result | Chemically targeted edits should cause targeted movement in a small number of MTO modes and corresponding response shifts. |
| Main conclusion | MTO modes behave like response coordinates under controlled chemical changes. |
| Supports | C4 |

### Experiment E3. Mode Masking and Intervention

| Item | Description |
|---|---|
| Dataset | QM9S |
| Procedure | Mask, amplify or swap selected MTO modes; measure change in predicted property and spectra. |
| Controls | random mode masking, random atom masking, composition-matched molecules |
| Figure | Fig. 6e-f |
| Expected result | Masking response-associated modes should selectively perturb relevant targets. |
| Main conclusion | MTO modes have causal predictive roles, not merely post-hoc visualization value. |
| Supports | C4 |

### Experiment E4. Attribution Robustness

| Item | Description |
|---|---|
| Dataset | QM9S/QMe14S |
| Procedure | Compare route weights, gradient attribution, integrated gradients and perturbation attribution. |
| Figure | Extended Data Fig. 6 |
| Expected result | Chemically meaningful modes should be robust across attribution definitions. |
| Main conclusion | Interpretation is not an artifact of one visualization method. |
| Supports | C4 |

---

## 9. Phase F: QM7-X Conformer and Non-Equilibrium Validation

### Experiment F1. Equilibrium to Non-Equilibrium Transfer

| Item | Description |
|---|---|
| Dataset | QM7-X |
| Procedure | Train on equilibrium conformers; test on non-equilibrium conformers. |
| Targets | dipole, polarizability, energy/forces as auxiliary if needed |
| Baselines | DetaNet direct, pooling, attention, MTO |
| Figure | Fig. 7a-b |
| Expected result | MTO should show better robustness for response quantities under structural perturbation. |
| Main conclusion | MTO modes are not tied to one optimized geometry. |
| Supports | C3, C4 |

### Experiment F2. Conformer Consistency of MTO Subspaces

| Item | Description |
|---|---|
| Dataset | QM7-X |
| Procedure | For each molecule, compare MTO subspaces across conformers as a function of RMSD or energy displacement. |
| Metrics | subspace distance, property distance, geometry distance correlation |
| Figure | Fig. 7c |
| Expected result | MTO subspace changes should be smooth and property-aligned rather than random. |
| Main conclusion | MTO captures geometry-dependent molecular response coordinates. |
| Supports | C4 |

### Experiment F3. Perturbation Linearity

| Item | Description |
|---|---|
| Dataset | QM7-X |
| Procedure | Analyze small normal-mode displacements and MTO changes. |
| Figure | Fig. 7d |
| Expected result | Small perturbations should induce structured movement in tensor modes, especially for dipole and polarizability. |
| Main conclusion | MTO behaves like a response coordinate under local structural perturbations. |
| Supports | C3, C4 |

---

## 10. Phase G: QMe14S External Generalization

### Experiment G1. QM9S to QMe14S Transfer

| Item | Description |
|---|---|
| Source | QM9S |
| Target | QMe14S |
| Procedure | Pretrain on QM9S response tasks; fine-tune on QMe14S. |
| Tasks | dipole, polarizability, Hessian-derived spectra, IR, Raman, NMR if clean |
| Figure | Fig. 7e-f |
| Expected result | MTO pretraining should improve low-data and medium-data QMe14S performance. |
| Main conclusion | MTO response modes transfer across related but broader chemical spaces. |
| Supports | C3, C4 |

### Experiment G2. Element OOD Split

| Item | Description |
|---|---|
| Dataset | QMe14S |
| ID elements | H, C, N, O, F |
| OOD elements | B, Al, Si, P, S, Cl, As, Se, Br |
| Procedure | Train on ID elements, fine-tune/test on OOD elements under low-data regimes. |
| Figure | Fig. 7g |
| Expected result | MTO should transfer better than generic readouts when new elements participate in response patterns. |
| Main conclusion | The representation principle is not confined to QM9 chemistry. |
| Supports | C3, C4 |

### Experiment G3. Functional-Group OOD Split

| Item | Description |
|---|---|
| Dataset | QMe14S |
| Groups | use the reported functional-group categories and RDKit SMARTS |
| Procedure | Hold out selected functional groups or group combinations. |
| Figure | Fig. 7h |
| Expected result | MTO should be more robust for unseen motif combinations than direct pooling. |
| Main conclusion | MTO learns composable response modes rather than memorizing fixed groups. |
| Supports | C2, C4 |

### Experiment G4. QMe14S Spectral Generalization

| Item | Description |
|---|---|
| Dataset | QMe14S |
| Targets | IR, Raman, NMR |
| Procedure | Train/evaluate spectra on ID and OOD splits. |
| Figure | Extended Data Fig. 7 |
| Expected result | MTO should improve spectral shape, peak recall and low-data transfer. |
| Main conclusion | MTO mode assembly is useful across multiple response modalities. |
| Supports | C3, C4 |

---

## 11. Phase H: Experimental Spectra and Design-Level Validation

This phase is not the first thing to run. It is the Nature-level closing evidence once the computational story is stable.

### Experiment H1. Curated Experimental Spectra Transfer

| Item | Description |
|---|---|
| Sources | NIST WebBook, SDBS, nmrshiftdb2 |
| Scale | 50 to 500 clean molecules, depending on data quality |
| Procedure | Match molecules by InChIKey/SMILES; align spectra grids; compare model predictions and mode attributions with experimental peaks. |
| Figure | Fig. 8a-c |
| Expected result | MTO-trained response modes should partially transfer to experimental spectral changes after calibration. |
| Main conclusion | MTO captures response structure beyond one DFT label system. |
| Supports | C3, C4 |

Use this only as a case study unless the data curation is extremely clean.

### Experiment H2. Mode-Guided Molecular Editing

| Item | Description |
|---|---|
| Dataset | generated molecule series plus DFT/TDDFT validation |
| Families | carbonyl/nitrile substitution, donor-pi-acceptor chromophores, conjugation extension, high-polarizability candidates |
| Procedure | Use MTO attribution to propose edits; screen candidates; validate selected molecules with DFT/TDDFT. |
| Scale | 500 to 5,000 generated candidates; 50 to 300 DFT validations |
| Figure | Fig. 8d-f |
| Expected result | Mode-guided edits should shift target response more efficiently than random or fingerprint-guided edits. |
| Main conclusion | MTO is not only predictive; it can guide chemical reasoning and design. |
| Supports | C4 and broad impact |

### Experiment H3. Failure Case Analysis

| Item | Description |
|---|---|
| Sources | QM9S, QMe14S, experimental spectra |
| Procedure | Identify molecules where MTO fails: flexible molecules, unusual elements, strong solvent effects, spectra with experimental condition mismatch. |
| Figure | Extended Data Fig. 8 |
| Expected result | Failure cases should be chemically understandable. |
| Main conclusion | The representation has clear scope and falsifiable limits. |
| Supports | manuscript credibility |

---

## 12. Final Evidence-to-Narrative Map

| Narrative | Minimal evidence | Strong Nature evidence | Main figures |
|---|---|---|---|
| Symmetry as axiom | internal equivariance audit | tensor order, output and transformation-type audit across datasets | Fig. 2 |
| Local-to-global composition | MTO beats pooling/attention/global token | gains increase with response complexity and anisotropy | Fig. 3, Fig. 4 |
| Structured pooling beyond compression | tensor and spectra gains | K-scaling, CG/gate/sign ablations, parameter controls | Fig. 4 |
| Quantified chemical intuition | functional group enrichment | matched pairs, mode intervention, experimental spectra | Fig. 6, Fig. 8 |
| Reusable scientific representation | seed stability, frozen probes | QM7-X conformer consistency and QMe14S transfer | Fig. 5, Fig. 7 |
| Broader method | QM9S response suite | QM7-X + QMe14S + real-spectrum/design validation | Fig. 7, Fig. 8 |

---

## 13. Execution Priority

### 13.1 Must-Have for a Serious Submission

Run these first:

1. QM9S data and tensor audit.
2. Internal MTO equivariance audit.
3. QM9S scalar/vector/tensor response prediction.
4. QM9S IR/Raman/UV-Vis spectral prediction.
5. Matched readout baselines.
6. Full MTO ablation.
7. Seed stability.
8. Frozen probe and stage transfer.
9. Functional-group enrichment with real SMARTS.
10. Mode masking/intervention.

Without these, the manuscript is a strong method paper but not yet Nature-mainline.

### 13.2 Strong Nature Upgrade

Run these after the must-have package:

1. QM7-X equilibrium to non-equilibrium transfer.
2. QM7-X conformer subspace consistency.
3. QMe14S QM9S-to-QMe14S transfer.
4. QMe14S element OOD split.
5. QMe14S functional-group OOD split.
6. QMe14S IR/Raman/NMR spectral generalization.

These experiments transform the story from "works on QM9S" to "a general response representation principle".

### 13.3 High-Impact Closing Evidence

Run these if time/resources allow:

1. Curated NIST/SDBS/nmrshiftdb2 experimental spectra case study.
2. Mode-guided molecular editing.
3. DFT/TDDFT validation of selected candidates.

These experiments are not necessary to prove the architecture, but they make the manuscript feel like chemistry rather than only machine learning.

---

## 14. Go/No-Go Decision Rules

### 14.1 Claims We Can Make Only If Results Support Them

| Result | Allowed claim |
|---|---|
| MTO wins on response tasks and ablations are clean | Explicit tensor-mode assembly improves response learning. |
| MTO modes are stable across seeds | MTO is a stable response subspace, not arbitrary mode slots. |
| Frozen probes transfer across tasks | MTO is reusable across molecular observables. |
| Interventions selectively change predictions | MTO modes have causal predictive roles. |
| QM7-X conformer consistency holds | MTO tracks geometry-dependent response. |
| QMe14S OOD transfer holds | MTO generalizes beyond QM9-like chemistry. |
| Experimental spectra case works | MTO captures response information relevant to real spectra. |

### 14.2 Claims to Avoid

Avoid these claims unless separately proven:

- MTOs are real quantum-chemical molecular orbitals.
- MTO directly recovers HOMO/LUMO.
- MTO solves the Schrodinger equation internally.
- MTO is universally interpretable for every molecule.
- MTO is guaranteed stable at the individual mode-slot level.
- MTO is the first use of symmetry in molecular learning.

Preferred language:

> MTOs are learned, nonlinear, symmetry-constrained response modes in equivariant representation space.

---

## 15. Suggested Main-Text Conclusion Template

If the full evidence chain succeeds, the main conclusion can be written as:

> Across response-rich molecular datasets, MTO-Net shows that atom-centred equivariant tensor fields can be reorganized into molecule-level tensor modes that are symmetry-correct, more effective than direct pooling, stable across training seeds, reusable across response tasks, chemically interrogable through controlled perturbations, and transferable beyond the original QM9S chemical space. These results support tensor-mode assembly as a general local-to-global representation principle for molecular response learning.

Shorter Nature-style version:

> MTO-Net turns local equivariant tensor fields into testable molecular response coordinates.

---

## 16. Reference Notes

Dataset and source facts used in this manual should be verified again at manuscript writing time.

Key public sources:

- QM9S dataset, Figshare: `https://figshare.com/articles/dataset/QM9S_dataset/24235333`
- DetaNet molecular spectra paper, PubMed: `https://pubmed.ncbi.nlm.nih.gov/38177591/`
- QM7-X, Scientific Data: `https://www.nature.com/articles/s41597-021-00812-2`
- QMe14S, JPCL: `https://pubs.acs.org/doi/10.1021/acs.jpclett.5c00839`
- QMe14S dataset, Figshare: `https://figshare.com/s/889262a4e999b5c9a5b3`
- NIST Chemistry WebBook: `https://webbook.nist.gov/`
- SDBS, AIST: `https://sdbs.db.aist.go.jp/`
- nmrshiftdb2: `https://nmrshiftdb.nmr.uni-koeln.de/`


# MTO-Net Nature-Level Experiment Plan

## 0. Central Claim

This experiment plan is designed to support the core narrative of MTO-Net:

> MTO-Net uses symmetry as a deductive constraint to assemble atom-centred equivariant tensor fields into stable, reusable and chemically meaningful molecule-level response modes.

The goal is not merely to prove that MTO-Net has lower MAE than another model. The goal is to prove a stronger representation-level statement:

> Atom-centred equivariant tensor fields can be reorganized into molecular tensor modes that act as explicit local-to-global response coordinates.

This is the key difference between MTO-Net and ordinary molecular graph neural networks, pooling readouts, attention pooling or global tokens.

---

## 1. Four Narratives and Required Evidence

| Narrative | Scientific claim | Required evidence |
|---|---|---|
| Symmetry as axiom | MTO assembly is constrained by SO(3)/O(3) irreducible representations, Clebsch-Gordan tensor products and invariant gates | Equivariance audit under rotations, reflections, translations and atom permutations |
| Local-to-global composition | MTO replaces terminal pooling with explicit molecular tensor mode assembly | Matched comparison against sum/mean pooling, attention pooling, tensor pooling and global-token readouts |
| Structured pooling beyond compression | MTO preserves tensor order, mode index, cancellation, anisotropy and long-range cooperation | Complexity-stratified benchmarks, molecule-size/generalization tests and K-scaling |
| Quantified chemical intuition | MTO modes align with motifs such as carbonyls, conjugation, donor-acceptor paths, polar fragments and anisotropic response | Functional-group enrichment, matched controls, chemical interventions and mode masking |
| Reusable scientific representation | MTO modes are not arbitrary hidden states, but stable and transferable response subspaces | Seed subspace stability, frozen probes, stage transfer and cross-dataset transfer |

The final paper should establish:

> MTO-Net reveals that local equivariant tensor fields can be assembled into stable, transferable and chemically meaningful molecular response subspaces.

---

## 2. Data System

### 2.1 Main Dataset: QM9S

QM9S should be the primary dataset for the first paper because it directly supports the molecular spectra and tensor-response story. Public descriptions report that QM9S was constructed from about 130K organic molecules based on QM9, with re-optimized geometries and spectral/property labels.

Main tasks:

```text
Stage A:
  - Dipole moment, mu
  - Polarizability, alpha

Stage B:
  - IR spectrum
  - Raman spectrum
  - UV-Vis spectrum

Stage C, optional or supplementary:
  - 1H NMR
  - 13C NMR
```

Recommended first-paper focus:

```text
mu, alpha, IR, Raman, UV-Vis
```

NMR can be included if label alignment is clean. If not, it should be moved to Supplementary Information or a follow-up paper.

### 2.2 External Spectral Data

Use external experimental spectra only after the core QM9S pipeline is stable.

Candidate sources:

```text
- nmrshiftdb2
- NIST Chemistry WebBook
- SpectraBase subsets
- NMRNet-style benchmark datasets
```

Purpose:

> Test whether response modes learned from quantum-chemical labels transfer to experimental spectral domains.

This is a Nature-level strengthening experiment, not required for the minimum viable first paper.

### 2.3 Optional Design Validation Data

For a stronger Nature-level story, add a small DFT/TDDFT validation loop.

Candidate molecular families:

```text
- Donor-pi-acceptor chromophores
- Carbonyl/nitrile/amine substitution series
- Push-pull conjugated molecules
- TADF/OLED-like emitters
- High polarizability anisotropy candidates
```

Minimum validation scale:

```text
Generate 500-5000 edited candidates.
Screen using MTO-Net.
Select top 50-300 candidates.
Validate by DFT or TDDFT.
```

---

## 3. Phase 0: Data, Code and Tensor Audit

### 3.1 Code State Lock

Before formal experiments, record:

```text
- Repository URL
- Branch
- Commit hash
- Python version
- PyTorch version
- CUDA version
- e3nn version
- DetaNet source version
- MTO-Net config
- Training config hash
```

Deliverables:

```text
outputs/audit/code_state.json
outputs/audit/environment.txt
outputs/audit/git_diff.patch
```

Success criterion:

```text
All reported results must be traceable to exact code and config states.
```

### 3.2 QM9S Data Audit

Inspect the actual local dataset rather than assuming label availability from papers.

Report:

```text
- Number of molecules
- Element set
- Available scalar targets
- Available vector targets
- Available tensor targets
- Available spectral targets
- Missing labels
- NaN/inf counts
- mol_id alignment
- Target shape table
- Fixed train/validation/test split
```

Special checks:

```text
mu
alpha
Hessian
IR
Raman
UV-Vis
1H NMR
13C NMR
```

Deliverables:

```text
outputs/audit/qm9s_dataset_audit.json
outputs/audit/qm9s_target_table.csv
outputs/audit/qm9s_label_statistics.pdf
```

### 3.3 DetaNet Tensor Adapter Audit

MTO must not use scalar features only. It must consume true tensor irreducible representations.

Expected feature layout:

```text
h0: [num_atoms, C0]
h1: [num_atoms, C1, 3]
h2: [num_atoms, C2, 5]
h3: [num_atoms, C3, 7], if available
```

Required tests:

```text
- Verify tensor split and reconstruction.
- Verify l=1, l=2 and l=3 channels transform under Wigner-D matrices.
- Verify routed MTO tensors preserve the expected tensor order.
```

Deliverables:

```text
outputs/audit/detanet_tensor_shapes.json
outputs/audit/tensor_reconstruction_error.json
outputs/audit/tensor_equivariance_audit.json
```

Success criterion:

```text
Tensor reconstruction error should be approximately zero.
Equivariance error should be around 1e-5 to 1e-7 depending on precision.
```

---

## 4. Phase 1: Mathematical Correctness

### 4.1 MTO Equivariance Audit

For each molecule \(X\), construct transformed inputs:

\[
X' = gX
\]

where \(g\) includes:

```text
- Random SO(3) rotations
- O(3) inversion/reflection
- Translation
- Same-element atom permutation
- Rotation plus permutation
- Rotation plus inversion
```

Check:

\[
\mathcal{O}^{(l)}_k(gX) \approx D^{(l)}(g)\mathcal{O}^{(l)}_k(X)
\]

Relative error:

\[
\epsilon_l =
\frac{
\|\mathcal{O}^{(l)}(gX)-D^{(l)}(g)\mathcal{O}^{(l)}(X)\|
}{
\|\mathcal{O}^{(l)}(X)\|+\delta
}.
\]

Report:

```text
- l=0 scalar invariance error
- l=1 vector equivariance error
- l=2 tensor equivariance error
- l=3 tensor equivariance error, if used
- Translation invariance
- Same-element permutation consistency
- Output-level equivariance/invariance
```

Figures:

```text
Fig. 1b: Equivariance error by tensor order
Fig. 1c: Error before and after MTO assembly
Fig. 1d: Error under rotation, inversion, translation and permutation
```

Success criterion:

```text
float32: 1e-5 to 1e-6
float64: 1e-7 or better
```

If the error is above 1e-4, stop and fix the implementation before doing scientific experiments.

### 4.2 Component-Level Equivariance Tests

Test each component independently:

```text
- Signed routing
- Clebsch-Gordan tensor product
- Scalar gate
- Tensor-information gate
- Activity gate
- Readout
```

Recommended test files:

```text
tests/test_mto_equivariance.py
tests/test_cg_coupling.py
tests/test_signed_routing.py
tests/test_tensor_gate.py
tests/test_readout_equivariance.py
```

---

## 5. Phase 2: Prediction and Structured Pooling Comparison

### 5.1 Main Question

Does MTO improve molecular response learning because of explicit tensor mode assembly, rather than because of parameter count or training details?

### 5.2 Fair Comparison Setup

All models should share:

```text
- Same DetaNet backbone
- Same data split
- Same optimizer
- Same learning-rate schedule
- Same batch size
- Same epoch budget
- Same early stopping rule
- Same seeds
- Matched or reported parameter count
```

### 5.3 Baselines

Compare:

```text
B0: DetaNet original readout
B1: DetaNet + sum pooling
B2: DetaNet + mean pooling
B3: DetaNet + scalar attention pooling
B4: DetaNet + tensor pooling
B5: DetaNet + global token
B6: DetaNet + Set Transformer readout
B7: DetaNet + scalar-only MTO
B8: DetaNet + tensor MTO without CG coupling
B9: DetaNet + MTO without signed routing
B10: DetaNet + MTO without gates
B11: Full MTO-Net
```

### 5.4 Metrics

For \(\mu\) and \(\alpha\):

```text
- MAE
- RMSE
- R2
- Relative error
- Polarizability anisotropy error
```

For IR, Raman and UV-Vis:

```text
- Spectrum MAE
- Spectrum RMSE
- Cosine similarity
- Spearman correlation
- Peak position error
- Peak intensity error
- Integrated intensity error
- Top-k peak recall
```

### 5.5 Deliverables

```text
outputs/benchmarks/stage_a_mu_alpha.csv
outputs/benchmarks/stage_b_spectra.csv
outputs/figures/fig2_benchmark_main.pdf
outputs/reports/benchmark_summary.md
```

### 5.6 Success Criteria

Minimum:

```text
- Full MTO-Net is competitive with or better than DetaNet original readout.
- Full MTO-Net is clearly better than scalar-only MTO.
- MTO shows stronger gains on collective response tasks than on simple scalar tasks.
```

Ideal:

```text
- MTO improves mu and alpha robustly.
- MTO improves IR/Raman/UV-Vis more strongly.
- MTO advantage increases for large, polar, conjugated or heteroatom-rich molecules.
```

Important interpretation:

> MTO does not need to win every task. It must win the representation story: stability, reuse and chemical meaning.

---

## 6. Phase 3: Local-to-Global Stress Tests

### 6.1 Molecule-Complexity Stratification

Stratify QM9S by:

```text
- Heavy atom count
- Graph diameter
- Number of heteroatoms
- Number of functional groups
- Conjugation length
- Donor-acceptor distance
- Polarizability anisotropy
```

Compare:

```text
pooling baselines vs global token vs full MTO-Net
```

Figures:

```text
Fig. 3a: Error vs heavy atom count
Fig. 3b: Error vs graph diameter
Fig. 3c: Error vs conjugation length
Fig. 3d: Error vs polarizability anisotropy
```

Expected conclusion:

> MTO is most useful when molecular response requires long-range cooperation, cancellation or anisotropic tensor information.

### 6.2 Controlled Fragment-Pair Task

Construct a small synthetic or semi-synthetic dataset:

```text
fragment A - bridge length n - fragment B
```

Targets:

```text
- Scalar target depending on A/B interaction
- Vector target along donor-acceptor axis
- Tensor target aligned with molecular anisotropy
```

Success criterion:

```text
As bridge length or graph diameter increases, pooling/global-token performance should degrade faster than MTO.
```

This is a mechanism experiment. It does not need to be the main benchmark.

---

## 7. Phase 4: Seed Stability and Response Subspaces

### 7.1 Main Question

Are MTO modes arbitrary hidden vectors, or do they form stable molecular response subspaces?

### 7.2 Training Protocol

Train:

```text
10 random seeds
same split
same architecture
same hyperparameters
```

For each seed \(s\), molecule \(X_n\), mode \(k\) and tensor order \(l\), save:

\[
\mathcal{O}^{(l)}_{k,s}(X_n)
\]

### 7.3 Subspace Metric

Do not compare slot \(k\) directly. MTO slots can permute, flip signs or rotate within degenerate subspaces.

For each molecule and order \(l\), construct:

\[
O_s^{(l)}(X)\in \mathbb{R}^{K\times d_l}
\]

Obtain orthonormal bases \(Q_s^{(l)}\) by QR or SVD.

Projection overlap:

\[
S_{\text{sub}}(s,t,l)=
\frac{1}{r}
\mathrm{Tr}
\left(
Q_s^{(l)}Q_s^{(l)\top}
Q_t^{(l)}Q_t^{(l)\top}
\right)
\]

Principal angles:

\[
\cos \theta_i =
\sigma_i
\left(
Q_s^{(l)\top}Q_t^{(l)}
\right)
\]

### 7.4 Controls

```text
- Ordinary hidden layer subspace
- Attention pooling weights
- Global-token representation
- Random projection
- Scalar-only MTO
- MTO without gates
- MTO without signed routing
- MTO without CG coupling
```

### 7.5 Figures

```text
Fig. 4a: Seed-to-seed subspace overlap heatmap
Fig. 4b: Principal angle distribution
Fig. 4c: Stability vs prediction error
Fig. 4d: Good seed vs bad seed training curves
Fig. 4e: MTO stability vs hidden-state stability
```

### 7.6 Success Criteria

Minimum:

```text
Full MTO subspace overlap is higher than hidden state/global token controls.
Bad seeds show lower subspace stability and early plateau.
```

Ideal:

```text
Individual MTO slots are not perfectly stable, but the response subspace is stable.
```

This is scientifically acceptable and even desirable, because representation subspaces can have gauge freedom.

---

## 8. Phase 5: Frozen Probe and Stage Transfer

### 8.1 Main Question

Are MTO modes reusable response representations rather than task-specific fitting artifacts?

### 8.2 Frozen Probe

Train source models on:

```text
- mu
- alpha
- mu + alpha
- IR + Raman
- UV-Vis
```

Then freeze:

```text
DetaNet backbone + MTO block
```

Train only a small readout head for target tasks:

```text
mu -> alpha
alpha -> mu
mu + alpha -> IR
mu + alpha -> Raman
IR/Raman -> alpha
UV-Vis -> HOMO-LUMO gap or excitation proxy, if available
```

### 8.3 Controls

```text
- Frozen DetaNet hidden state + probe
- Frozen attention pooling + probe
- Frozen global token + probe
- Frozen random MTO + probe
- Training from scratch
- Full fine-tuning upper bound
```

### 8.4 Low-Data Transfer

Use target-task training fractions:

```text
1%, 5%, 10%, 25%, 50%, 100%
```

Metrics:

```text
- MAE/RMSE/R2
- Sample efficiency
- Learning curve slope
- Frozen-probe gap
- Full-finetune gap
```

Figures:

```text
Fig. 5a: Transfer matrix
Fig. 5b: Low-data learning curves
Fig. 5c: Frozen MTO vs frozen hidden vs global token
Fig. 5d: Source-target similarity map
```

Success criterion:

```text
MTO frozen probe should outperform pooling/global-token frozen probes, especially under low-data target settings.
```

---

## 9. Phase 6: Chemical Meaning and Functional-Group Enrichment

### 9.1 Main Question

Can MTO modes quantify chemical intuition?

Specifically:

```text
- Functional groups
- Conjugation
- Polar fragments
- Donor-acceptor motifs
- Delocalized response
- Anisotropic molecular axes
```

### 9.2 RDKit Motif Labels

Use RDKit SMARTS to label:

```text
- Carbonyl
- Hydroxyl
- Amine
- Nitrile
- Ether
- Fluoro
- Aromatic ring
- Conjugated segment
- Donor fragment
- Acceptor fragment
- Heteroatom-rich fragment
- Saturated alkyl control
```

If RDKit is unavailable, do not call it functional-group enrichment. Downgrade the claim to atom-type enrichment.

### 9.3 Atom Contribution

For atom \(i\), mode \(k\):

\[
a_{ik} =
\sum_l
\left\|
c_{ki}^{(l)}W_lh_i^{(l)}
\right\|
\]

Functional-group enrichment:

\[
E_{g,k} =
\frac{
\mathrm{mean}(a_{ik}: i\in g)
}{
\mathrm{mean}(a_{ik}: i\in \text{matched background})
}
\]

Matched background should control for:

```text
- Atom type
- Degree
- Ring membership
- Molecule size
- Heavy atom count
- Heteroatom count
- Local environment
```

### 9.4 Statistical Tests

```text
- Permutation test
- Bootstrap confidence interval
- FDR correction
- Matched null model
- Random MTO baseline
- Random atom-attribution baseline
```

Figures:

```text
Fig. 6a: Functional-group enrichment heatmap
Fig. 6b: Volcano plot
Fig. 6c: Matched-control enrichment
Fig. 6d: Representative molecule visualizations
```

Success criterion:

```text
Selected MTO modes show significant enrichment for chemically meaningful motifs after matched controls and FDR correction.
```

---

## 10. Phase 7: Chemical Intervention

### 10.1 Main Question

Do changes in MTO modes explain property changes under chemically meaningful edits?

### 10.2 Matched Molecular Pairs

Construct or mine pairs:

```text
- carbonyl -> alcohol
- nitrile -> amine
- ether -> alkane
- donor group substitution
- acceptor group substitution
- conjugated bond -> saturated bond
- longer pi bridge
- heteroatom position migration
- fluorination / defluorination
```

For pair \((X,X')\):

\[
\Delta \mathcal{O}_k = \mathcal{O}_k(X')-\mathcal{O}_k(X)
\]

\[
\Delta y = y(X')-y(X)
\]

Analyze:

```text
- Correlation between Delta MTO and Delta mu
- Correlation between Delta MTO and Delta alpha
- Correlation between Delta MTO and IR peak shift
- Correlation between Delta MTO and UV-Vis red-shift
```

### 10.3 Mode Intervention

Apply:

```text
- Mask mode k
- Scale mode k by lambda
- Swap mode k between matched molecules
- Zero fragment contribution to mode k
```

Measure:

```text
- Delta mu
- Delta alpha
- Delta IR
- Delta Raman
- Delta UV-Vis
```

Figures:

```text
Fig. 7a: Matched molecular pairs
Fig. 7b: Delta MTO vs Delta property
Fig. 7c: Mode masking effects
Fig. 7d: Fragment-level intervention case study
```

Success criterion:

```text
A small number of MTO modes should capture chemically meaningful response changes, and masking those modes should selectively affect related properties.
```

---

## 11. Phase 8: Relationship to Molecular Orbitals and Electronic Structure

### 11.1 Main Question

MTOs are not quantum-chemical molecular orbitals. But do they align with orbital-like chemical effects in controlled molecular families?

Safe claim:

> MTOs are not quantum orbitals, but their learned response axes align with chemically recognized orbital effects in controlled cases.

### 11.2 Additional Labels

For 5,000-10,000 small molecules, obtain:

```text
- HOMO energy
- LUMO energy
- HOMO-LUMO gap
- Orbital localization index
- Dipole moment
- Polarizability anisotropy
- TDDFT excitation energy, optional
- Oscillator strength, optional
- NTO or transition-density proxy, optional
```

### 11.3 Analyses

Do not compare individual MTO slots to HOMO/LUMO directly.

Instead use:

```text
- CCA between MTO subspace and orbital descriptor space
- Linear probe: MTO -> HOMO/LUMO/gap
- Linear probe: orbital descriptors -> MTO-associated response
- Case studies: carbonyl n-to-pi*, donor-acceptor, aromatic delocalization
```

Figures:

```text
Fig. 8a: MTO-orbital descriptor CCA
Fig. 8b: MTO linear probe for orbital-related labels
Fig. 8c: Controlled chemical case studies
```

---

## 12. Phase 9: Cross-Dataset and Experimental Spectral Transfer

### 12.1 Main Question

Can response modes learned from quantum-chemical data transfer to experimental spectral domains?

### 12.2 Workflow

```text
1. Train MTO-Net on QM9S mu/alpha/IR/Raman/UV-Vis.
2. Freeze DetaNet + MTO.
3. Fine-tune a small readout on experimental NMR/IR/UV-Vis data.
4. Compare with scratch training and frozen pooling baselines.
```

Controls:

```text
- 2D fingerprint baseline
- 2D GNN baseline
- Frozen DetaNet hidden state
- Frozen attention pooling
- Frozen global token
- Frozen MTO
- Full fine-tuning
```

Metrics:

```text
- Experimental spectrum MAE
- Peak position error
- Peak intensity error
- Cosine similarity
- Top-k peak recall
- Low-data performance
- MTO subspace overlap before and after fine-tuning
```

Figures:

```text
Fig. 9a: QM9S -> experimental transfer workflow
Fig. 9b: Low-data transfer curves
Fig. 9c: Experimental spectrum examples
Fig. 9d: MTO mode consistency before/after fine-tuning
```

Success criterion:

```text
MTO frozen or fine-tuned models outperform pooling/global-token baselines under low-data experimental transfer.
```

---

## 13. Phase 10: MTO-Guided Molecular Design Loop

### 13.1 Main Question

Can MTO modes guide molecular editing and design?

### 13.2 Design Options

Option A: Increase polarizability anisotropy

```text
Start from QM9S/GDB molecules.
Identify MTO mode associated with alpha anisotropy.
Apply conjugation extension or donor-acceptor substitution.
Predict improved alpha anisotropy.
Validate top candidates by DFT.
```

Option B: UV-Vis red-shift

```text
Identify delocalized UV-associated MTO mode.
Extend pi bridge or strengthen donor-acceptor structure.
Predict absorption red-shift.
Validate top candidates by TDDFT.
```

Option C: IR peak engineering

```text
Identify functional-group MTO mode associated with target frequency.
Modify carbonyl/nitrile/amine environment.
Predict peak shift or intensity change.
Validate top candidates by DFT.
```

### 13.3 Validation Scale

```text
Generate 500-5000 candidates by rule-based edits.
Screen with MTO-Net.
Select top 50-300 candidates.
Validate by DFT/TDDFT.
```

Figures:

```text
Fig. 10a: MTO-guided design workflow
Fig. 10b: Predicted vs DFT-validated improvement
Fig. 10c: Representative designed molecules
Fig. 10d: Mode-level explanation of design rule
```

Success criterion:

```text
MTO-guided candidates show stronger validated improvement than random or heuristic edits, and the selected modes explain why the edits work.
```

---

## 14. Full Ablation Suite

Ablation must evaluate representation quality, not only prediction accuracy.

Report five dimensions:

```text
1. Prediction accuracy
2. Equivariance error
3. Seed stability
4. Transferability
5. Chemical enrichment/intervention
```

### 14.1 Architecture Ablations

```text
- Full MTO-Net
- Scalar-only MTO
- Tensor-only MTO
- Without l=2
- Without l=3
- Without CG coupling
- Without signed routing
- Positive-only attention
- Without scalar gate
- Without tensor gate
- Without nonlinear gate
- Without activity gate
- Fixed K
- K = valence electron count
- Constant small K
- Constant large K
- Without bottleneck
- Without diversity regularization
- Center-based assembly
- Center-free assembly
- Direct pooling
- Global token
```

### 14.2 Output Tables

```text
Table 1: Prediction metrics
Table 2: Equivariance errors
Table 3: Stability metrics
Table 4: Transfer metrics
Table 5: Interpretability metrics
```

Success criterion:

```text
Full MTO-Net should be best or near-best on the combined evidence profile.
Tensor MTO should clearly outperform scalar-only MTO.
CG coupling, signed routing and gates should each contribute to at least one key dimension: prediction, stability, transfer or interpretability.
```

---

## 15. Recommended Main Figures

### Fig. 1: Concept and Architecture

Message:

```text
Local equivariant fields -> MTO assembly -> molecular response modes -> properties/spectra
```

Panels:

```text
- MTO-Net architecture
- SO(3)/O(3) irreps
- CG coupling
- Signed routing
- Scalar/tensor gates
```

### Fig. 2: Symmetry Audit and Benchmark

Panels:

```text
- Equivariance error by tensor order
- mu/alpha benchmark
- IR/Raman/UV-Vis benchmark
- Parameter count and training cost
```

### Fig. 3: MTO as Structured Pooling

Panels:

```text
- MTO vs pooling/global token/attention
- Error vs molecule complexity
- Error vs conjugation length
- K-scaling curve
```

### Fig. 4: Stable Response Subspaces

Panels:

```text
- Seed subspace overlap heatmap
- Principal angle distributions
- Hidden-state vs MTO stability
- Good seed vs bad seed trajectories
```

### Fig. 5: Transfer and Reuse

Panels:

```text
- Frozen probe
- Stage transfer matrix
- Low-data learning curves
- Source-target similarity
```

### Fig. 6: Chemical Meaning

Panels:

```text
- Functional-group enrichment
- Matched controls
- Representative MTO heatmaps
- Mode-family chemical interpretation
```

### Fig. 7: Chemical Intervention

Panels:

```text
- Matched molecular edits
- Delta MTO vs Delta property
- Mode masking
- Fragment-level intervention
```

### Fig. 8: Experimental Transfer or Design Loop

Panels:

```text
- QM9S -> experimental spectra transfer
or
- MTO-guided molecular design and DFT validation
```

---

## 16. Execution Priority

### Priority 1: Minimum First Paper

These experiments are required for a credible first paper:

```text
P0. Data and code audit
P1. Tensor adapter and equivariance audit
P2. QM9S mu/alpha full benchmark
P3. Pooling/global-token/attention comparison
P4. 5-10 seed subspace stability
P5. Frozen probe
P6. Stage transfer
P7. Full ablation
P8. RDKit functional-group enrichment with matched controls
```

This can support a strong JACS / Nature Chemistry / Nature Computational Science style paper if results are solid.

### Priority 2: Strong Version

Add:

```text
P9. IR/Raman/UV-Vis full training
P10. Spectrum peak-level metrics
P11. Chemical intervention
P12. MTO-orbital relationship analysis
P13. Experimental NMR/IR small transfer
```

This supports the stronger claim:

> MTO-Net learns stable, reusable and chemically meaningful response modes.

### Priority 3: Nature Main-Journal Version

Add:

```text
P14. Cross-dataset generalization
P15. Experimental spectra connection
P16. DFT-validated MTO-guided design loop
P17. One strong chemical discovery case study
```

This supports:

> MTO-Net establishes a symmetry-constrained neural representation framework for chemistry.

---

## 17. Minimum Viable Version vs Nature Version

### 17.1 MVP Version

```text
1. Tensor MTO implementation audit
2. Equivariance audit
3. QM9S mu/alpha benchmark
4. MTO vs pooling/global-token/attention
5. 5-seed stability
6. Frozen probe mu <-> alpha
7. Full ablation
8. Functional-group enrichment
```

Supports:

> MTO-Net is a symmetry-preserving molecular response mode assembly layer.

### 17.2 Strong Version

```text
MVP +
IR/Raman/UV-Vis
10-seed stability
Stage transfer across mu/alpha/spectra
Chemical intervention
MTO-orbital controlled analysis
```

Supports:

> MTO-Net learns stable, reusable and chemically meaningful molecular response modes.

### 17.3 Nature Version

```text
Strong version +
Experimental spectra transfer
Cross-dataset generalization
DFT or experimental design loop
```

Supports:

> MTO-Net establishes a symmetry-constrained neural representation chemistry framework.

---

## 18. Decision Rules

Use the following decision tree after the first complete experiment cycle.

### Case A: Prediction improves, stability improves, transfer improves

Conclusion:

```text
Full story is supported.
Proceed toward Nature Chemistry / Nature Computational Science / Nature main-journal strengthened version.
```

### Case B: Prediction improves, but stability/transfer do not

Conclusion:

```text
MTO is a useful readout, but not yet a stable response representation.
Focus paper on architecture and benchmark; weaken interpretability claims.
```

### Case C: Prediction does not improve, but stability/transfer/interpretability improve

Conclusion:

```text
MTO is valuable as an interpretable representation layer.
Position paper around response representation, not SOTA accuracy.
```

### Case D: None improve

Conclusion:

```text
Architecture needs revision.
Check scalar-only leakage, tensor adapter, optimization collapse, K choice, gates and loss normalization.
```

---

## 19. Final Scientific Standard

The experiments must establish three statements:

1. **MTO-Net is symmetry-preserving.**
2. **MTO-Net replaces terminal pooling with explicit tensor mode assembly.**
3. **The assembled modes are stable, reusable and chemically meaningful molecular response subspaces.**

If these three statements are supported, the macro-narrative becomes a scientific conclusion rather than a rhetorical framing.

---

## 20. Reference Basis

These public sources support the dataset and related-work choices:

```text
- QM9S Figshare dataset: reports a QM9Spectra dataset constructed from about 130K organic molecules based on QM9.
- DetaNet / molecular spectra work: E(3)-equivariant self-attention model for predicting molecular spectra including IR, Raman, UV-Vis and NMR-related tasks.
- e3nn: Euclidean Neural Networks: describes TensorProduct and spherical harmonics as core E(3)-equivariant operations.
- OrbNet: uses symmetry-adapted atomic-orbital features for quantum chemistry learning, providing a useful contrast to MTO's learned response modes.
- QMe14S: a larger later spectral dataset that may serve as an optional cross-dataset extension.
```

The key distinction to preserve in writing:

> OrbNet uses orbital-derived features as input for quantum-chemistry prediction. MTO-Net learns molecule-level response modes from atom-centred equivariant tensor fields. MTOs are not quantum orbitals, but learned symmetry-constrained response coordinates.

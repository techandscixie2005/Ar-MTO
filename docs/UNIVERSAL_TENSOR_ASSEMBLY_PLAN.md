# Universal Tensor Assembly Plan (wt-12)

## Naming

Outside chemistry, the generic method is called **Tensor Mode Assembly (TMA)**, not MTO. "MTO" (Molecular Tensor Orbital) is the chemistry-specific framing because of the AO/MO analogy.

## Goal

Show that the TMA representation principle is broader than chemistry: local equivariant tensor fields can be assembled into global tensor modes for any 3D point-cloud problem with SO(3)-structured targets.

## Phase 1: Synthetic SO(3) Multipole Tasks

### Task Definition

Generate synthetic 3D point clouds with:
- **Inputs**: N points with 3D coordinates + scalar charges/masses/types.
- **Targets**:
  - Scalar total (sum of charges/masses) — tests l=0.
  - Vector dipole-like response — tests l=1.
  - Rank-2 quadrupole-like response — tests l=2.
  - Cancellation cases (equal positive/negative charges) — tests sign sensitivity.
  - Anisotropy cases — tests directional sensitivity.

### Baselines

Compare against:
1. Sum pooling + MLP
2. Mean pooling + MLP
3. Scalar attention pooling + MLP
4. Global token / Set Transformer readout
5. Tensor pooling without mode bank
6. Full TMA (mode bank + signed routing + CG coupling + invariant gates)

### Metrics

- Scalar MAE/RMSE
- Vector direction cosine similarity + magnitude error
- Rank-2 tensor Frobenius error, isotropic/anisotropic decomposition
- Mode usage entropy
- Generalization to unseen point counts (N → N')

## Phase 2: ModelNet40 Classification (Optional Later)

### Task

Rotated ModelNet40 point-cloud classification with SO(3) augmentation.

### Purpose

Test whether TMA improves classification under arbitrary 3D rotations compared to:
- PointNet / PointNet++
- DGCNN
- SE(3)-Transformers
- Standard equivariant networks without explicit mode assembly

### Metrics

- Classification accuracy (overall, per-class)
- Rotation robustness (accuracy vs rotation angle)
- Mode interpretability (what spatial patterns do modes capture?)

## Deliverables

1. `src/ar_mto/tma_generic.py` — Generic TMA module (not chemistry-specific).
2. `scripts/generate_synthetic_multipole.py` — Synthetic data generator.
3. `scripts/train_tma_synthetic.py` — Training script for synthetic tasks.
4. `outputs/tma_synthetic/` — Results, metrics, comparison plots.
5. `outputs/tma_synthetic/tma_synthetic_report.md` — Summary report.

## Rules

- Keep the generic TMA module separate from chemistry-specific MTO code.
- Chemistry remains the most elegant application because of the AO/MO analogy — the synthetic tasks are supporting evidence, not the main story.
- All TMA modules must pass the same equivariance tests as MTO modules.

# Ar-MTO Plans.md — Minimum Decisive MTO Path

作成日: 2026-06-20

team_validation_mode: manual-pass
team_validation_note: >
  User provided extremely detailed 8-stage specification. All Product/Architecture/Security/QA/Skeptic
  perspectives are covered by the user's own requirements list. CLAUDE.md already defines the full
  product contract. The plan is a direct translation of user-specified stages into executable tasks.

spec_delta: none
spec_skip_reason: >
  CLAUDE.md sections 1–12 define the full product contract (architecture, symmetry constraints,
  non-negotiable rules, dataset strategy, experiment roadmap). This plan fills remaining implementation
  gaps without changing the product contract. No new API, data model, or user-visible behavior beyond
  what CLAUDE.md prescribes.

lint_formatter_baseline: >
  No formatter/linter configured. Not blocking — Python code is clean and this is a research project.
  If style issues arise in training harness, add a `ruff format` task.

---

## Stage 1: Inspect Current State → Implementation Map

**Findings (pre-populated from inspection):**

All 7 source modules are present and well-structured:

| File | Lines | Status |
|------|-------|--------|
| `src/ar_mto/__init__.py` | 25 | Clean exports |
| `src/ar_mto/tensor_adapter.py` | 128 | Complete. Split S[128]→h0, T[1920]→h1/h2/h3. Exact reconstruct. channel_mix preserves orders. |
| `src/ar_mto/signed_routing.py` | 321 | Complete. Batch-aware softmax+L2/abs norm. tanh signs. Order-specific sign projections. route_stats. |
| `src/ar_mto/mto_core.py` | 278 | Complete. Per-molecule assembly loop. MTOModeAssembly + ScalarOnlyMTO. forward_with_masks. compute_valence_adaptive_k. |
| `src/ar_mto/cg_coupling.py` | 429 | Complete. CGCouplingMinimal (6 fixed paths, parity-correct). CGCoupling (full programmatic paths). Path table logging. |
| `src/ar_mto/tensor_gate.py` | 350 | Complete. TensorGate (sigmoid residuals). NoGate. ScalarOnlyGate. Per-l gate stats. |
| `src/ar_mto/readouts.py` | 395 | Complete. Scalar/Vector/Rank2Tensor/Spectral readouts. Mode-weight softmax for vector. Sph→Cart for rank2. |
| `src/ar_mto/mto_net.py` | 386 | Complete. MTOConfig + MTONet + make_mto_net. Full pipeline wiring. |
| `src/ar_mto/detanet_bridge.py` | 141 | Complete. Import/locate/make_latent_detanet. radius_edges (pure PyTorch, no pyg dependency). |

Tests are comprehensive but have gaps:

| Test File | Coverage |
|-----------|----------|
| `tests/test_detanet_import.py` | Import/locate/make tests |
| `tests/test_mto_equivariance.py` | MTO equivariance, shapes, batch isolation, mode masking, scalar-only, ValenceAdaptiveK |
| `tests/test_mto_forward_backward.py` | Full pipeline fwd/bwd, gradient sanity, scalar-only ablation, config variants, MTONet wrapper, checkpoint |
| `tests/test_signed_routing.py` | Shapes, properties (sign range, L2/abs norm, batched, deterministic, stats), invariance under rotation |
| `tests/test_cg_coupling.py` | CG Minimal and Full: shapes, batch isolation, equivariance, no-NaN, path tables, parity |
| `tests/test_tensor_gate.py` | TensorGate/NoGate/ScalarOnlyGate: equivariance, no-NaN, stats, mode masking, batch isolation |

**Gaps to fill:**

1. No dedicated `test_tensor_adapter.py` — split/reconstruct, Wigner-D on real DetaNet features
2. No `test_translation.py` — translation invariance/equivariance
3. No `test_permutation.py` — same-element atom permutation consistency
4. No `scripts/verify_mto_core.sh` — unified verification script
5. Missing a few existing tests for: adapter exactness, Wigner-D transform for h1/h2/h3 from real DetaNet

---

## Phase 1: Fix Missing Tests & Verification Script

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 1.1 | Add test_tensor_adapter.py: split/reconstruct exact, Wigner-D h1/h2/h3 from DetaNet | `pytest tests/test_tensor_adapter.py -v` passes | - | cc:完了 |
| 1.2 | Add test_translation.py: translation invariance for MTO, CG, gates | `pytest tests/test_translation.py -v` passes | - | cc:完了 [300e8ea] |
| 1.2b | **BLOCKER** Fix CGCouplingFull equivariance: diagnose root cause (dimension/parity/path/tolerance), fix or explicitly downgrade Full CG to experimental, ensure default MTO training path passes equivariance | `pytest tests/test_cg_coupling.py -v` all pass including CGCouplingFull; `pytest tests/test_translation.py -v` all pass; report at `outputs/reports/cg_full_equivariance_fix_report.md` | 1.2 | cc:完了 [a9a78e0] |
| 1.3 | Add test_permutation.py: same-element atom permutation consistency | `pytest tests/test_permutation.py -v` passes | 1.2b | cc:完了 [a7b579b] |
| 1.4 | Create scripts/verify_mto_core.sh running all mandatory tests | `bash scripts/verify_mto_core.sh` exits 0 locally | 1.1, 1.2, 1.2b, 1.3 | cc:完了 [0515b12] |
| 1.5 | Run existing test suite, fix any regressions | All 6 existing test files pass | 1.2b | cc:完了 [3dfc37e] |
| 1.6 | **BLOCKER** N16R4 server rules hardening: probe server, update global & project CLAUDE.md with verified module/env rules, create scripts/hpc_env.sh + scripts/probe_hpc_env.sh, forbid `/tmp` and hard-coded modules | Remote probe passes; `~/.claude/CLAUDE.md` updated with N16R4 rules; project `CLAUDE.md` updated; `scripts/hpc_env.sh` prints env and imports torch; `scripts/probe_hpc_env.sh` discovers modules safely; `outputs/reports/n16r4_server_rules_report.md` written; Phase 2 remains blocked until review | 1.5 | cc:完了 [31f70de] |

---

## Phase 2: Mu-Only Training Harness

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 2.1 | Create split files (train/val/test) from QM9S dataset on server if not present | Split files at MTO/data/qm9s/splits/ exist; or report missing and stop | 1.6 | cc:TODO |
| 2.2 | Implement training script: config-driven, full tensor MTO, dipole target, vector readout | `python scripts/train_mu.py --dry-run` parses config and initializes model | 2.1 | cc:TODO |
| 2.3 | Implement metric computation: vec MAE, norm MAE, RMSE, R², angular error | Metrics logged to metrics.json/csv each epoch | 2.2 | cc:TODO |
| 2.4 | Implement checkpointing: best.ckpt, last.ckpt, MTO cache, routing stats, mode stats | Reload best.ckpt → same predictions within 1e-5 | 2.2 | cc:TODO |
| 2.5 | Implement Slurm job submission wrapper for HPC | `bash scripts/run_mu_smoke.sh` submits a valid sbatch job | 2.2 | cc:TODO |
| 2.6 | Run verify_mto_core.sh on server environment before training | verify_mto_core.sh exits 0 on server | 1.6 | cc:TODO |

---

## Phase 3: Smoke Training (Staged)

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 3.1 | Smoke 1: 64 molecules, 2 epochs | No NaN/inf, finite loss, checkpoint saves & reloads | Phase 2 | cc:TODO |
| 3.2 | Smoke 2: 500 molecules, 5 epochs | No NaN/inf, loss decreases, metrics produced | 3.1 | cc:TODO |
| 3.3 | Smoke 3: 5000 molecules, 10 epochs | No NaN/inf, validation metrics reasonable | 3.2 | cc:TODO |
| 3.4 | Full QM9S, seed=0 | Complete run with all outputs saved | 3.3 | cc:TODO |
| 3.5 | Full QM9S, seeds=1,2,3,4 | All 5 seeds complete with all output artifacts | 3.4 | cc:TODO |

---

## Phase 4: Effective MTO Mode Analysis

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 4.1 | Compute mode activity, readout weights, routing entropy, signed cancellation, K_eff | `outputs/tables/mu_effective_modes_summary.csv` generated | 3.5 | cc:TODO |
| 4.2 | Generate mode activity distribution figure | `outputs/figures/mu_effective_modes/mode_activity_distribution.pdf` | 4.1 | cc:TODO |
| 4.3 | Generate effective K distribution figure | `outputs/figures/mu_effective_modes/effective_K_distribution.pdf` | 4.1 | cc:TODO |
| 4.4 | Compute & plot top-r masking curve (r=1,2,4,8,16,32) | `outputs/figures/mu_effective_modes/top_r_masking_curve.pdf` | 4.1 | cc:TODO |
| 4.5 | Compute & plot leave-one-mode-out importance | `outputs/figures/mu_effective_modes/leave_one_mode_out_importance.pdf` | 4.1 | cc:TODO |
| 4.6 | Write effective modes report | `outputs/reports/mu_effective_modes_report.md` | 4.2–4.5 | cc:TODO |

---

## Phase 5: Seed Stability Analysis

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 5.1 | Compute seed-to-seed subspace overlap matrix | `outputs/tables/mu_seed_stability_summary.csv` with overlap matrix | 3.5 | cc:TODO |
| 5.2 | Generate subspace overlap heatmap | `outputs/figures/mu_stability/seed_subspace_overlap_heatmap.pdf` | 5.1 | cc:TODO |
| 5.3 | Compute & plot principal angle distributions | `outputs/figures/mu_stability/principal_angle_distribution.pdf` | 5.1 | cc:TODO |
| 5.4 | Compare K_eff across seeds | `outputs/figures/mu_stability/keff_across_seeds.pdf` | 5.1 | cc:TODO |
| 5.5 | Plot good vs bad seed training curves | `outputs/figures/mu_stability/good_bad_seed_training_curves.pdf` | 5.1 | cc:TODO |
| 5.6 | Write seed stability report | `outputs/reports/mu_seed_stability_report.md` | 5.2–5.5 | cc:TODO |

---

## Phase 6: Final Decision Report

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 6.1 | Write minimal decision report answering all 10 questions | `outputs/reports/mto_mu_minimal_decision_report.md` | Phase 4, Phase 5 | cc:TODO |

---

## Notes

- **Dataset**: QM9S at `/data/home/scwc008/run/xxy/MTO/data/qm9s/qm9s.pt`. Split files may need creation.
- **Seeds**: 0, 1, 2, 3, 4
- **Run directory**: `/data/home/scwc008/run/xxy/MTO/runs/mto_mu_full/<timestamp>_<seed>/`
- **GPU preference**: H200 > H100 > A800 (Slurm only)
- **HPC connection**: `ssh bjhpc_xxy_1` only
- **HPC environment**: Loaded via `scripts/hpc_env.sh` (Task 1.6). Module/python env must be probed, never hard-coded. `/tmp` forbidden for project scripts; use `MTO/tmp/` on server.
- **Model config**: `configs/model/mto_full.yaml` is the canonical full tensor MTO config
- **Current branch**: `main` — all modified source files will be committed before training
- **Key constraint**: MTO is NOT scalar pooling. Never use scalar_only MTO as main model.
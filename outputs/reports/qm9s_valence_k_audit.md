# QM9S Valence Electron Count Audit — K_max Recommendation

**Date**: 2026-06-21
**Dataset**: QM9S subset_medium (5000 molecules)
**Server**: ln01 (N16R4)

## Method

For each molecule with atomic numbers Z_i:
- N_val = sum_i valence[Z_i]
- K_half = ceil(N_val / 2)

Valence counts used: standard neutral-atom valence electrons for main group and transition metals.

## subset_medium (5000 molecules)

| Statistic | N_val | K_half |
|-----------|-------|--------|
| Min | 8 | 4 |
| Max | 52 | 26 |
| Mean | 38.8 | 19.4 |
| Std | 4.2 | 2.1 |
| p50 | 40 | 20 |
| p90 | 44 | 22 |
| p95 | 44 | 22 |
| p99 | 46 | 23 |

### K_half distribution (selected bins)

| K | Count | Fraction |
|---|-------|----------|
| 4-15 | 196 | 3.9% |
| 16-18 | 1015 | 20.3% |
| 19 | 1202 | 24.0% |
| 20 | 1039 | 20.8% |
| 21 | 890 | 17.8% |
| 22 | 559 | 11.2% |
| 23-26 | 98 | 2.0% |

### Cap analysis

| K_max Cap | Molecules Capped | Fraction |
|-----------|-----------------|----------|
| 16 | 4587 | 91.74% |
| 20 | 1547 | 30.94% |
| 24 | 6 | 0.12% |
| 26 | 0 | 0.00% |

## Recommendation

**K_max = 32** for smoke training on subset_medium.

Rationale:
- K_half_max = 26 covers all 5000 molecules
- K_max = 32 provides a modest safety margin for the full QM9S dataset
- Padding from typical K ≈ 17-22 up to K_max = 32 is manageable
- Total mode capacity: 32 × 64 channels = 2048-dimensional mode bank per molecule
- For 5000 molecules × 10 epochs at batch_size=32: ~214k parameter model with K_max=32 (~5.5M params)

For full QM9S (~130k molecules), K_max = 32 or 48 depending on available GPU memory. A higher cap (48) provides more headroom for large molecules at the cost of larger mode banks.

## Limitations

- Valence counts use neutral-atom values; formal charged species (ions) would need adjustment
- Some transition metals in the full QM9S dataset may have unusual valence configurations
- The audit does not account for batch-level memory constraints (padded K_max × batch_size)
- Full QM9S stats pending (subset_medium is representative for small organic molecules)

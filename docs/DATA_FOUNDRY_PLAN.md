# Data Foundry Plan (wt-06)

## Status

QM9S is already present on the server. This worktree handles dataset registry, manifests, ingestion plans, and download scripts for all datasets beyond QM9S.

## Datasets

### Existing
- **QM9S**: Present on server. Verified by wt-01 audit.

### To Be Added

#### QMe14S
- **Purpose**: Broader chemistry (14 elements), functional-group OOD splits, spectral generalization.
- **Format**: Extended XYZ, HDF5, or NPZ (TBD by inspection).
- **Size**: TBD (estimate: 100K-150K molecules).
- **Splits**: Element OOD, functional-group OOD, random.

#### QM7-X
- **Purpose**: Conformer/non-equilibrium stability, response-subspace continuity under geometry changes.
- **Format**: Extended XYZ with conformer indexing.
- **Size**: ~7K molecules × ~100 conformers each.
- **Splits**: Molecule-level (conformers of same molecule stay together).

#### Experimental Spectra
- **Sources**: NIST Chemistry WebBook, SDBS, nmrshiftdb2, NMRexp.
- **Target scale**: 50-500 clean molecules for first case study.
- **Matching**: SMILES/InChIKey where possible.

#### Non-Chemical (for wt-12)
- **Synthetic SO(3) multipole**: Generated locally from 3D point clouds with scalar/vector/rank-2 targets.
- **ModelNet40** (optional later): Point-cloud classification.

## Deliverables

1. `data/registry.json` — All datasets with sources, URLs, sizes, checksums, split definitions.
2. `data/manifests/` — Per-dataset manifest files (filenames, splits, metadata).
3. `scripts/download_*.sh` — Download scripts for each dataset (no actual downloads in setup).
4. `scripts/checksum_verify.sh` — Checksum verification script.
5. `scripts/ingest_*.py` — Ingestion/conversion scripts for each dataset.
6. `data/splits/` — Split files (train/val/test indices).

## Rules

- Store only scripts, manifests, checksums, splits, and reports in Git.
- Treat acquired datasets as read-only.
- No large downloads in setup tasks.
- Do not modify, move, delete, or overwrite existing dataset files.
- All download scripts must verify checksums after download.
- All ingestion scripts must log shapes, dtypes, NaN/inf counts, and element sets.

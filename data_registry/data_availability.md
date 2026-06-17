# Data Availability Statement for MTO-Net

> This document is manuscript-facing. It describes where each dataset comes from,
> how to access it, what is included in the repository, and what must be
> obtained separately.

## Overview

The MTO-Net project uses a combination of:

1. **Public quantum chemical datasets** (QM9S, QMe14S, QM7-X) — downloaded in
   full and stored on our computation server.
2. **Public experimental NMR databases** (nmrshiftdb2, NMRexp) — downloaded or
   registered with links to official sources.
3. **Curated experimental spectra** from NIST and SDBS — accessed manually for
   case-study validation. Bulk data is not redistributed.
4. **Synthetic datasets** — generated in-project for method verification.

## Data Locations

### In This Repository (Git)

The following metadata and tooling files are included in the repository:

- `data_registry/dataset_registry.yaml` — Unified registry of all datasets
- `data_registry/download_manifest.json` — Download operation records
- `data_registry/target_table.csv` — Target specification for all datasets
- `data_registry/license_citation_access_notes.md` — License and access details
- `data_registry/data_availability.md` — This file
- `data_registry/dataset_ingestion_report.md` — Engineering ingestion report
- `configs/data_foundry/*.yaml` — Per-dataset configuration files
- `scripts/data_foundry/` — Download, generation, and audit scripts

### NOT in This Repository

Raw datasets, archives, HDF5 files, CSV data dumps, SD files, and generated
large data are **excluded from Git** (see `.gitignore`). They are stored on
the computation server and/or local staging storage.

## Dataset Access

### QM9S

- **Source:** `https://figshare.com/s/889262a4e999b5c9a5b3`
- **Status:** Fully downloaded and available on our computation server.
- **How to reproduce:** Download from the Figshare link above.
- **Size:** ~25 GiB compressed and extracted.

### QMe14S

- **Source:** `https://figshare.com/s/889262a4e999b5c9a5b3`
- **Status:** Download scripts available; pending full download.
- **How to reproduce:** Run `scripts/data_foundry/download_qme14s.py` or download
  manually from the Figshare link.
- **Paper:** DOI: 10.1021/acs.jpclett.5c00839

### QM7-X

- **Source:** `https://zenodo.org/records/3905361`
- **Status:** Partially downloaded (README.txt, createDB.py, 8000.xz/89MB).
  Remaining 7 XZ files (~9.5 GB) pending. Datasets are downloading locally
  due to HPC DNS issues with zenodo.org, then uploading via rsync.
- **How to reproduce:** Run `scripts/data_foundry/robust_downloader.py` or download
  manually from the Zenodo record.
- **Paper:** DOI: 10.1038/s41597-021-00812-2
- **License:** CC BY 4.0
- **Size:** ~10 GB compressed (8 XZ files), ~40+ GiB extracted

### nmrshiftdb2

- **Source:** `https://nmrshiftdb.nmr.uni-koeln.de/`
- **Status:** Public archive endpoint being located.
- **How to reproduce:** Access via the website or SourceForge project page.

### NMRexp (3.37M experimental NMR spectra)

- **Source:** `https://zenodo.org/records/17296666`
- **Status:** Small validation files downloaded. Main data files pending (HPC DNS issues
  with zenodo.org; downloading locally and uploading).
- **How to reproduce:** Run `scripts/data_foundry/robust_downloader.py` with the
  Zenodo file URLs, or download manually from the Zenodo record.
- **Paper:** DOI: 10.1038/s41597-025-06245-5
- **License:** CC BY 4.0
- **Size:** ~3.3 GB total
- **Nuclei:** 1H, 13C, 19F, 31P, 11B, 29Si

### NIST Chemistry WebBook

- **Source:** `https://webbook.nist.gov/`
- **Status:** Manual curation only. Cannot be bulk downloaded.
- **How to access:** Visit the NIST WebBook, search for compounds of interest,
  and download spectra individually.
- **Attribution:** "Data from NIST Standard Reference Database 69:
  NIST Chemistry WebBook, DOI: 10.18434/T4D303"

### SDBS / AIST

- **Source:** `https://sdbs.db.aist.go.jp/`
- **Status:** Manual curation only. Cannot be bulk downloaded.
- **How to access:** Visit the SDBS website, search for compounds of interest,
  and download spectra individually.
- **Attribution:** "SDBS: Spectral Database for Organic Compounds, AIST, Japan"

### Synthetic TMA

- **Source:** Generated in-project via `scripts/data_foundry/generate_synthetic_tma.py`
- **Status:** Smoke dataset generated; script available for larger generation.
- **How to reproduce:** Run the generation script.

## Reproducibility

To reproduce the full data environment:

```bash
# 1. Clone the repository
git clone git@github.com:techandscixie2005/Ar-MTO.git
cd Ar-MTO
git checkout feat/06-data-foundry

# 2. Download public datasets (requires server access or local fallback)
python scripts/data_foundry/download_qm7x.py
python scripts/data_foundry/download_qme14s.py

# 3. Generate synthetic TMA smoke dataset
python scripts/data_foundry/generate_synthetic_tma.py

# 4. Run dataset audit
python scripts/data_foundry/audit_external_datasets.py
python scripts/data_foundry/compute_checksums.py
python scripts/data_foundry/build_dataset_registry.py
```

## Contact

For questions about data availability, contact the repository maintainer.

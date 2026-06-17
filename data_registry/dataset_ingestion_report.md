# Dataset Ingestion Report — Data Foundry (wt-06)

> **Branch:** feat/06-data-foundry
> **Date:** 2026-06-17
> **Purpose:** Engineering report of all dataset download, registration, and ingestion
> operations for the MTO-Net data foundry task.

## 1. Summary

| Dataset      | Status            | Server Path                                                    | Files | Size       | Notes                                      |
|-------------|-------------------|----------------------------------------------------------------|-------|------------|---------------------------------------------|
| QM9S        | available_full    | /data/home/scwc008/run/xxy/MTO/data/qm9s                       | 13    | ~25 GB     | Pre-existing; registered only               |
| QMe14S      | pending_download  | /data/home/scwc008/run/xxy/MTO/data/external/qme14s            | 0     | TBD        | Figshare article ID unresolved              |
| QM7-X       | available_partial | /data/home/scwc008/run/xxy/MTO/data/external/qm7x              | 1/10  | 89 MB/10GB | 8000.xz only; rest pending local download   |
| nmrshiftdb2 | pending_download  | /data/home/scwc008/run/xxy/MTO/data/external/nmrshiftdb2       | 0     | TBD        | No public bulk endpoint confirmed           |
| NMRexp      | available_partial | /data/home/scwc008/run/xxy/MTO/data/external/nmrexp            | 7/10  | 590 KB/3.3GB| Small files on HPC; large files pending     |
| NIST        | manual_curation   | /data/home/scwc008/run/xxy/MTO/data/external/experimental_spectra/nist | 0 | N/A        | No bulk download permitted                  |
| SDBS        | manual_curation   | /data/home/scwc008/run/xxy/MTO/data/external/experimental_spectra/sdbs | 0 | N/A        | No bulk download permitted                  |
| synthetic   | generated         | /data/home/scwc008/run/xxy/MTO/data/external/synthetic_tma     | 5     | ~100 KB    | Smoke dataset generated; equivariance passed|

## 2. Per-Dataset Details

### 2.1 QM9S

- **Action:** Registered only. Data was already present on server.
- **Server path:** `/data/home/scwc008/run/xxy/MTO/data/qm9s`
- **Files found:** 13 files/directories including:
  - `qm9s.pt` (2.68 GiB) — PyTorch Geometric format
  - `qm9s_csv.zip` (3.46 GiB) — CSV format
  - `ir_boraden.csv` (8.13 GiB), `raman_boraden.csv` (8.18 GiB), `uv_boraden.csv` (1.43 GiB)
  - `ext_val.zip` (133 MiB), `ext_val_env.zip` (134 MiB)
  - `subset_smoke/`, `subset_medium/` — pre-made splits
  - Metadata files: `FIGSHARE_ALL_FILES_MANIFEST.json`, `figshare_article_24235333.json`, `readme.txt`, `atomic_energy_reference.txt`
- **Figshare article:** 24235333
- **License:** CC BY 4.0 (confirmed from article metadata)
- **Molecule count:** ~134,000
- **QMe14S investigation:** The same Figshare share link (`https://figshare.com/s/889262a4e999b5c9a5b3`) is reported to host both QM9S and QMe14S. However, the Figshare API only returned article 24235333 (QM9S). QMe14S may be a separate article under the same share that was not indexed or is not publicly accessible via the search API.

### 2.2 QMe14S

- **Attempts made:**
  1. Figshare share link API (`/v2/share/{hash}`) → 404
  2. Figshare article search API (multiple queries) → empty results or unrelated spam
  3. Figshare article ID probing (near QM9S range) → all 404
  4. Zenodo search → no results
  5. GitHub code search → no results
  6. arXiv paper fetch → no Figshare link in abstract
- **Current assessment:** The dataset is consistently reported at `https://figshare.com/s/889262a4e999b5c9a5b3` by multiple web searches. This share link currently hosts QM9S (article 24235333). QMe14S may need to be accessed via browser (JS-rendered page) to see the full article list. The Figshare public API may not index all articles under a share.
- **Recommendation:** Open the share link in a browser, locate the QMe14S article, and note its article ID. Alternatively, contact the authors (Mingzhi Yuan, Wei Hu) for direct access.
- **Known facts:** 186,102 molecules, 14 elements, 47 functional groups, B3LYP/TZVP, IR/Raman/NMR spectra, AIMD configurations. 15 files on Figshare (last modified 2025-03-19).

### 2.3 QM7-X

- **Source:** Zenodo record 3905361
- **Files mapped from Zenodo API (10 files, 9.62 GB compressed):**
  - `1000.xz` (0.72 GB, md5:b50c6a5d0a4493c274368cf22285503e)
  - `2000.xz` (1.04 GB, md5:4418a813daf5e0d44aa5a26544249ee6)
  - `3000.xz` (2.05 GB, md5:f7b5aac39a745f11436047c12d1eb24e)
  - `4000.xz` (1.46 GB, md5:26819601705ef8c14080fa7fc69decd4)
  - `5000.xz` (1.14 GB, md5:85ac444596b87812aaa9e48d203d0b70)
  - `6000.xz` (2.02 GB, md5:787fc4a9036af0e67c034a30adc54c07)
  - `7000.xz` (1.10 GB, md5:5ecce00a188410d06b747cb683d8d347)
  - `8000.xz` (0.09 GB, md5:c893ae88b8f5c32541c3f024fc1daa45)
  - `README.txt` (3 KB)
  - `createDB.py` (3 KB)
- **Download status:**
  - README.txt, createDB.py, 8000.xz: **Downloaded** (local: `/mnt/e/Ar-MTO-data-foundry/qm7x/`)
  - Remaining 7 XZ files: **Pending.** Multiple download methods attempted:
    - HPC wget: failed (DNS: "Could not resolve host: zenodo.org")
    - HPC curl: failed (same DNS issue)
    - Local curl parallel: failed (stuck at 0 bytes)
    - Local wget: failed (exit code 2)
  - HPC DNS diagnosis: `nslookup zenodo.org` returns SERVFAIL. The DNS server `127.0.0.53` (systemd-resolved) cannot resolve zenodo.org. This is a persistent issue, not intermittent.
  - Local downloads: One file (8000.xz, 89 MB) succeeded via local curl. Larger files hang at 0 bytes, possibly due to Zenodo rate limiting or WSL network issues.
- **Remaining work:** Use `robust_downloader.py` with small chunk sizes and long timeouts to download the remaining 7 files locally. Then rsync all to HPC. Estimate: ~2-3 hours for 9.5 GB.

### 2.4 nmrshiftdb2

- **Attempts made:**
  1. SourceForge project page check: project exists (`https://sourceforge.net/projects/nmrshiftdb2/`), but RSS feed returned empty
  2. nmrshiftdb2 homepage fetch: HTML saved as `sf_files.html` on HPC
  3. Known download paths probed: `/download`, `/data`, `/export` — all returned error responses
- **Current assessment:** No public bulk download endpoint was confirmed. SourceForge project may have file releases, but they were not accessible via API. The website provides individual compound lookup only.
- **Recommendation:** Contact nmrshiftdb2 maintainers (University of Cologne) for bulk data access. Register as pending until access is confirmed.
- **Alternative:** The NMRexp dataset (Zenodo 17296666) provides a larger, more recent experimental NMR resource.

### 2.5 NMRexp

- **Source:** Zenodo record 17296666 (DOI: 10.5281/zenodo.17296666)
- **License:** CC BY 4.0 (confirmed)
- **Total size:** ~3.3 GB (10 files)
- **Download status:**
  - **Downloaded on HPC (7 small files):**
    - `NMRexp_10to24_1_0811.py` (218 KB) — extraction script
    - `F_50_checked.csv` (20 KB) — 19F validation (50 records)
    - `hetero_200_checked.csv` (74 KB) — heteroatom validation (200 records)
    - `test_300_checked.csv` (254 KB) — test set (300 records)
    - `Si_50_checked.csv` (18 KB), `P_50_checked.csv` (19 KB), `B_50_checked.csv` (18 KB)
  - **Pending (3 large files):**
    - `NMRexp_10to24_1_1004.csv` (2.14 GB) — main CSV data
    - `NMRexp_10to24_1_1004.parquet` (661 MB) — Parquet format
    - `NMRexp_10to24_1_1004_sc_less_than_1.parquet` (528 MB) — subset
  - Local download of parquet attempted: got only 13 MB of 661 MB (partial). Same DNS/network issues as QM7-X.
- **Remaining work:** Download 3 large files locally with `robust_downloader.py`, then rsync to HPC.

### 2.6 NIST Chemistry WebBook

- **Action:** Registered only. No bulk download.
- **Access policy:** Public domain (US government). Manual download of individual spectra permitted. Bulk scraping prohibited.
- **Template needed:** Curation script for manually-selected spectra (placeholder created).

### 2.7 SDBS / AIST

- **Action:** Registered only. No bulk download.
- **Access policy:** Free for research. Bulk download and redistribution prohibited.
- **Template needed:** Curation script for manually-selected spectra (placeholder created).

### 2.8 Synthetic TMA

- **Action:** Smoke dataset generated and audited.
- **Server path:** `/data/home/scwc008/run/xxy/MTO/data/external/synthetic_tma/smoke/`
- **Files:**
  - `train.npz` — 100 systems
  - `val.npz` — 20 systems
  - `test.npz` — 20 systems
  - `metadata.json` — generation parameters
  - `equivariance_audit.json` — all tests passed
- **Equivariance audit results:**
  - vector_dipole (l=1): max error 1.43e-06 → PASS
  - rank2_quadrupole (l=2): max error 3.81e-06 → PASS
  - scalar_total (translation invariance): max error 0.0 → PASS
  - Rotation equivariance: scalar 4.77e-07, vector 9.54e-07, rank-2 3.81e-06 → PASS
- **Git note:** Smoke data is on HPC (not in git). Scripts are in `scripts/data_foundry/`.

## 3. Infrastructure

### 3.1 Directory Layout (Server)

```
/data/home/scwc008/run/xxy/MTO/data/
  qm9s/                          # Full dataset (~25 GB)
  external/
    qme14s/raw/                  # Empty (pending Figshare article ID)
    qm7x/raw/                    # README.txt, createDB.py, zenodo_record.json
    nmrshiftdb2/raw/             # sf_files.html (metadata only)
    nmrexp/raw/                  # 7 small validation files
    experimental_spectra/
      nist/                      # Empty (manual curation only)
      sdbs/                      # Empty (manual curation only)
      curated_cases/             # Empty (future)
    synthetic_tma/
      smoke/                     # train/test/val.npz + metadata + audit
      generated/                 # Empty (for larger generation)
      manifests/                 # Empty (for checksums/file lists)
```

### 3.2 Git-Tracked Files

All in `wt-06-data-foundry`:
- `data_registry/` — dataset_registry.yaml, download_manifest.json, target_table.csv,
  license_citation_access_notes.md, data_availability.md
- `scripts/data_foundry/` — download_*.sh, robust_downloader.py, generate_synthetic_tma.py,
  compute_checksums.py, audit_external_datasets.py, build_dataset_registry.py, make_splits.py
- `configs/data_foundry/` — per-dataset YAML configs

### 3.3 Excluded from Git

Raw data is excluded via `.gitignore` patterns:
- `data/`, `data/external/`, `*.zip`, `*.tar.*`, `*.hdf5`, `*.h5`, `*.npz`, `*.sdf`, `*.sd`,
  `*_boraden.csv`, `*.parquet`, large JSON/CSV dumps

## 4. Known Issues

1. **HPC Zenodo DNS:** The HPC DNS resolver (systemd-resolved at 127.0.0.53) cannot resolve
   `zenodo.org` (returns SERVFAIL). This affects all Zenodo downloads. API calls sometimes
   succeed (intermittent), but file downloads fail consistently.

2. **QMe14S Figshare API:** The Figshare public search API does not return QMe14S results.
   The share link `/s/889262a4e999b5c9a5b3` hosts QM9S but QMe14S article ID is unresolved.
   Browser access to the share page may reveal the QMe14S article.

3. **Local Zenodo large-file downloads:** Files >500 MB fail to start downloading from
   the local WSL environment. Smaller files (89 MB 8000.xz) succeed. This may be due to
   Zenodo rate limiting, WSL network stack issues, or server-side timeouts.

4. **nmrshiftdb2 public endpoint:** No confirmed public bulk download URL. The project
   exists on SourceForge but bulk file access is unclear.

## 5. Next Steps

1. **Download remaining QM7-X files:** Use `robust_downloader.py` locally with retry
   and small chunk size. Then `rsync` to HPC. Alternative: try a different network
   (non-WSL) for downloads.

2. **Download remaining NMRexp files:** Same approach as QM7-X.

3. **Resolve QMe14S:** Open `https://figshare.com/s/889262a4e999b5c9a5b3` in a browser,
   locate QMe14S article, note article ID, then download.

4. **Contact nmrshiftdb2:** Reach out to University of Cologne maintainers for bulk access
   or confirm SourceForge file availability.

5. **Create NIST/SDBS curation templates:** Placeholder scripts for manual spectrum ingestion.

6. **Generate larger synthetic TMA:** After scripts are verified, generate larger datasets
   for method validation.

7. **Compute checksums:** Run `compute_checksums.py` on all downloaded files after full
   download completion.

## 6. Verification

- All remote operations stayed within `/data/home/scwc008/run/xxy`
- No raw data committed to Git
- `.gitignore` covers all raw data patterns
- Equivariance audit passed for synthetic TMA smoke dataset

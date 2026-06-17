# License, Citation, and Access Notes for MTO-Net External Datasets

> **Last updated:** 2026-06-17
> **Branch:** feat/06-data-foundry
> **Purpose:** Document the legal and access status of every dataset used in MTO-Net.
> **Important:** Treat all acquired datasets as read-only after download.

---

## 1. QM9S

- **License:** Public dataset on Figshare. Exact license terms are in the Figshare article metadata
  (`figshare_article_24235333.json` in the data directory).
- **Citation:** "QM9S: A Comprehensive Spectral Dataset for Small Organic Molecules" (Figshare).
  If a journal paper exists, cite that instead.
- **Access:** Free download from Figshare. No registration required.
- **Usage restrictions:** Standard Figshare terms. Attribution required.
- **Redistribution:** Follow Figshare license terms. Do not redistribute raw files; point to original Figshare URL.
- **Commercial use:** Check Figshare license terms.

## 2. QMe14S

- **License:** Public dataset on Figshare. Check Figshare article metadata for exact terms.
- **Citation:** "QMe14S, A Comprehensive and Efficient Spectral Dataset for Small Organic Molecules",
  J. Phys. Chem. Lett. (2025), DOI: 10.1021/acs.jpclett.5c00839.
- **Access:** Free download from `https://figshare.com/s/889262a4e999b5c9a5b3`.
- **Usage restrictions:** Attribution required. Cite the paper.
- **Redistribution:** Point to original Figshare URL. Do not redistribute raw files.
- **Commercial use:** Check Figshare and journal terms.

## 3. QM7-X

- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
  Confirmed on Zenodo record `https://zenodo.org/records/3905361`.
- **Citation:** "QM7-X: A comprehensive dataset of quantum-mechanical properties spanning the
  chemical space of small organic molecules", Scientific Data 8, 43 (2021),
  DOI: 10.1038/s41597-021-00812-2.
- **Access:** Free download from Zenodo. No registration required.
- **Usage restrictions:** CC BY 4.0 — you may share and adapt, but must give appropriate credit.
- **Redistribution:** Permitted under CC BY 4.0 with attribution.
- **Commercial use:** Permitted under CC BY 4.0.

## 4. nmrshiftdb2

- **License:** Open access. Check `https://nmrshiftdb.nmr.uni-koeln.de/` and
  `https://sourceforge.net/projects/nmrshiftdb2/` for specific license terms.
- **Citation:** "nmrshiftdb2: A Database for NMR Chemical Shifts", University of Cologne.
  Cite the website and/or associated paper when publishing.
- **Access:** Public web interface and SourceForge data repository.
- **Usage restrictions:** Attribution to University of Cologne required.
  NMReDATA SD files available under their respective terms.
- **Redistribution:** Check SourceForge project license. Prefer pointing to original source.
- **Commercial use:** Check nmrshiftdb2 license terms.

## 5. NMRexp (3.37M experimental NMR spectra)

- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
  Confirmed on Zenodo record `https://zenodo.org/records/17296666`.
- **Citation:** "NMRexp: A database of 3.37 million experimental NMR spectra",
  Scientific Data 12, 1954 (2025), DOI: 10.1038/s41597-025-06245-5.
  Also cite the Zenodo record: DOI: 10.5281/zenodo.17296666.
- **Access:** Free download from `https://zenodo.org/records/17296666`. No registration required.
- **Usage restrictions:** CC BY 4.0 — you may share and adapt, but must give appropriate credit.
- **Redistribution:** Permitted under CC BY 4.0 with attribution.
- **Commercial use:** Permitted under CC BY 4.0.

## 6. NIST Chemistry WebBook

- **License:** US Government public domain (NIST Standard Reference Database Number 69).
- **Citation:** "NIST Chemistry WebBook, NIST Standard Reference Database Number 69",
  Eds. P.J. Linstrom and W.G. Mallard, National Institute of Standards and Technology,
  Gaithersburg MD, 20899, DOI: 10.18434/T4D303.
- **Access:** Free web access at `https://webbook.nist.gov/`. Individual compound lookups only.
- **Usage restrictions:**
  - **Bulk scraping is PROHIBITED.** NIST robots.txt and terms of use forbid automated bulk downloads.
  - Individual spectra may be downloaded manually for specific compounds of interest.
  - Each downloaded spectrum must be attributed to NIST.
  - Do not redistribute NIST data; point users to the NIST WebBook.
- **Redistribution:** NOT PERMITTED. Direct users to `https://webbook.nist.gov/`.
- **Commercial use:** NIST data is public domain but check NIST terms for any restrictions.

## 7. SDBS / AIST Spectral Database

- **License:** Free for research and educational use. Check `https://sdbs.db.aist.go.jp/`
  for full terms.
- **Citation:** "SDBS: Spectral Database for Organic Compounds", National Institute of
  Advanced Industrial Science and Technology (AIST), Japan. Cite the website URL
  and date of access.
- **Access:** Free web access at `https://sdbs.db.aist.go.jp/`. Individual compound lookups.
- **Usage restrictions:**
  - **Bulk downloading is PROHIBITED** by SDBS terms of use.
  - Individual spectra may be downloaded for research purposes.
  - Attribution to SDBS/AIST is required.
  - **Redistribution is RESTRICTED.** Do not redistribute downloaded spectra.
  - Commercial use may require separate permission from AIST.
- **Redistribution:** NOT PERMITTED. Direct users to `https://sdbs.db.aist.go.jp/`.
- **Commercial use:** Contact AIST for permission.

## 8. Synthetic TMA

- **License:** Generated within this project. No external license restrictions.
- **Citation:** Not applicable (synthetic/generated data). If used in a publication,
  cite this repository or describe the generation procedure.
- **Access:** Generated locally via `scripts/data_foundry/generate_synthetic_tma.py`.
- **Usage restrictions:** None.
- **Redistribution:** No restrictions.
- **Commercial use:** No restrictions.

---

## Summary

| Dataset      | License              | Bulk Download | Redistribution   | Commercial Use     |
|-------------|---------------------|---------------|------------------|--------------------|
| QM9S        | Figshare (check)    | Yes           | Point to source  | Check terms        |
| QMe14S      | Figshare (check)    | Yes           | Point to source  | Check terms        |
| QM7-X       | CC BY 4.0           | Yes           | With attribution | Yes (CC BY 4.0)    |
| nmrshiftdb2 | Open access          | Yes (public) | Point to source  | Check terms        |
| NMRexp      | CC BY 4.0           | Yes           | With attribution | Yes (CC BY 4.0)    |
| NIST        | US Public Domain    | **NO**       | **NOT PERMITTED**| Check NIST terms   |
| SDBS        | Free for research   | **NO**       | **NOT PERMITTED**| Contact AIST       |
| Synthetic   | Project-generated   | N/A          | No restrictions  | No restrictions    |

---

## Action Items

- [x] Verify QM9S Figshare license from `figshare_article_24235333.json` (CC BY 4.0)
- [ ] Verify QMe14S Figshare license after article ID confirmed
- [x] Locate NMRexp dataset and record license (CC BY 4.0, Zenodo 17296666)
- [ ] Confirm nmrshiftdb2 SourceForge project license
- [ ] Create curated-case input template for NIST manual downloads
- [ ] Create curated-case input template for SDBS manual downloads

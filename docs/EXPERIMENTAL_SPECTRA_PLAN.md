# Experimental Spectra Plan (wt-10)

## Role

Experimental spectra are a curated real-world validation layer, not the first benchmark. They test whether MTO modes learned on QM9S (computed) transfer meaningfully to real measured spectra.

## Target Scale

- **First case study**: 50-500 clean molecules.
- **Stretch**: 500-2000 molecules if the first batch is clean and informative.

## Sources

| Source | Molecule Types | Spectra | Notes |
|--------|---------------|---------|-------|
| NIST Chemistry WebBook | Small organics | IR, mass spec | Well-curated, downloadable |
| SDBS | Organics | IR, Raman, NMR, MS | Japanese database, comprehensive |
| nmrshiftdb2 | Organics | NMR (1H, 13C) | Open, community-curated |
| NMRexp | Organics | NMR | Experimental NMR data |

## Workflow

1. **Registry**: Build a molecule registry with SMILES, InChIKey, source, available spectra.
2. **Matching**: Match to QM9S/QMe14S molecules by SMILES/InChIKey where overlap exists.
3. **Preprocessing**: Align spectra grids, normalize intensities, document solvent/phase/instrument limitations.
4. **Prediction**: Run MTO-Net prediction on matched molecules.
5. **Comparison**:
   - Predicted vs experimental spectra overlay plots.
   - Peak position MAE.
   - Peak intensity correlation.
   - Cosine similarity.
   - MTO mode attribution on experimental spectra.
6. **Failure analysis**: Document cases where prediction and experiment diverge, with hypotheses.

## Limitations to Document

- Solvent effects not modeled by gas-phase QM9S.
- Phase differences (gas vs liquid vs solid).
- Instrument-specific broadening and calibration.
- Concentration and temperature effects.
- Impurities and baseline artifacts.
- Limited spectral range in some databases.

## Deliverables

1. `outputs/experimental_spectra/registry.json` — Curated molecule list with metadata.
2. `outputs/experimental_spectra/matched_molecules.csv` — Molecules matched to QM9S/QMe14S.
3. `outputs/experimental_spectra/preprocessed/` — Aligned and normalized spectra.
4. `outputs/experimental_spectra/comparisons/` — Overlay plots, metrics, attribution plots.
5. `outputs/experimental_spectra/failure_cases/` — Documented failure cases.
6. `outputs/experimental_spectra/case_study_report.md` — Summary report.

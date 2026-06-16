# Nature Worktree Roadmap

Phase-by-phase roadmap from Phase 0 (audit) through Phase 8 (Nature-level upgrade), mapped to worktrees wt-00 through wt-14.

## Completed Phases (wt-00 through wt-05)

| Phase | Worktree | Branch | Status |
|-------|----------|--------|--------|
| 0: Repository & data audit | wt-00, wt-01 | feat/00, feat/01 | Complete |
| 1: Tensor adapter & correctness | wt-02 | feat/02-tensor-audit | Complete |
| 2: Smoke training | wt-03 | feat/03-mto-core | Complete |
| 3: MTO core implementation | wt-03 | feat/03-mto-core | Complete |
| 4: Baselines & ablation | wt-04 | feat/04-baselines-ablation | Complete |
| 5: QM9S mainline training | wt-05 | feat/05-training-evidence | Complete |

## Next-Stage Phases (wt-06 through wt-14)

| Phase | Worktree | Branch | Depends On |
|-------|----------|--------|------------|
| 6: Data foundation | wt-06 | feat/06-data-foundry | wt-05 |
| 7: Stability & transfer | wt-07 | feat/07-stability-transfer | wt-05, wt-06 |
| 8: Chemical interrogation | wt-08 | feat/08-chemical-interrogation | wt-05 |
| 9: QM9S spectra | wt-09 | feat/09-qm9s-spectra | wt-05 |
| 10: Experimental spectra | wt-10 | feat/10-experimental-spectra | wt-06, wt-09 |
| 11: External generalization | wt-11 | feat/11-external-generalization | wt-06, wt-09 |
| 12: Universal tensor assembly | wt-12 | feat/12-universal-tensor-assembly | wt-03 (MTO core) |
| 13: Figures & reporting | wt-13 | feat/13-figures-reporting | wt-05 through wt-12 |
| 14: Final artifact audit | wt-14 | feat/14-final-artifact-audit | wt-05 through wt-13 |

## Dependency Graph

```
wt-00/01 ──> wt-02 ──> wt-03 ──> wt-04 ──> wt-05
                                                │
                    ┌─────────────────────────────┼──────────────────────────┐
                    ▼                            ▼                           ▼
                  wt-06                     wt-07/wt-08                  wt-09
                    │                            │                           │
                    ▼                            ▼                           ▼
                  wt-10                     (stability)              (qm9s spectra)
                    │
                    ▼
                  wt-11
                    │
                    ▼
                  wt-12 (can run in parallel)
                    │
                    ▼
                  wt-13 (figures, depends on all above)
                    │
                    ▼
                  wt-14 (final audit, depends on all above)
```

## Milestones

1. **Data ready**: wt-06 manifests and scripts committed, QM9S confirmed on server.
2. **Stability proven**: wt-07 subspace overlap, frozen probes, stage transfer all positive.
3. **Chemistry validated**: wt-08 SMARTS enrichment with controls, MMP, mode masking complete.
4. **Spectra working**: wt-09 QM9S spectra training done; wt-10 experimental spectra curated.
5. **Generalization shown**: wt-11 QM7-X and QMe14S results positive.
6. **Universality**: wt-12 TMA works on synthetic SO(3) tasks.
7. **Reported**: wt-13 all figures generated from source data.
8. **Audited**: wt-14 reproducibility confirmed, claim-evidence table complete.

## Recommended Next Task

Start wt-06-data-foundry: create dataset registry, download manifests, and ingestion/audit plans. No large downloads yet.

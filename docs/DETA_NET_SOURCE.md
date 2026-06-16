# DetaNet Source Import

## Source

- **URL**: https://github.com/techandscixie2005/DetaNet
- **License**: See DetaNet repository LICENSE file

## Import Details

- **Method**: Local clone then rsync to HPC (remote HPC cannot resolve github.com)
- **Local clone path**: `/home/xiangyu_xie/Ar-MTO/third_party/DetaNet`
- **Remote path**: `/data/home/scwc008/run/xxy/MTO/third_party/DetaNet`
- **Commit hash**: `4f92e643ab64651b91c4a1392cf389ddfd0d89f0`
- **Import date**: 2025-06-16

## Notes

- DetaNet is **not** vendored into the Ar-MTO Git repository. It is excluded via `.gitignore` (`third_party/DetaNet/`).
- The remote copy on the HPC server includes the full `.git` directory for provenance tracking.
- The remote `data/` directory (`/data/home/scwc008/run/xxy/MTO/data`) was **not** modified during this import.

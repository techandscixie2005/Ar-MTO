# N16R4 Server Rules Hardening — Report

**Date:** 2026-06-20
**Task:** 1.6
**Server:** ln01 (N16R4 login node)
**User:** scwc008

---

## 1. Environment Probe Results

### Connectivity
- SSH alias `bjhpc_xxy_1` works.
- Hostname: `ln01` (login node).
- User: `scwc008`.
- Workspace root: `/data/home/scwc008/run/xxy` — writable.

### Module System
- Environment modules require: `source /etc/profile.d/modules.sh`
- `module` is NOT available in the default shell environment.
- Available on the login node via:
  ```bash
  source /etc/profile.d/modules.sh
  ```

### Verified Module: miniforge3/25.11.0-1

| What | Verified Value | Hard-coded (old/broken) Value |
|------|---------------|-------------------------------|
| Module init | `source /etc/profile.d/modules.sh` | (missing from CLAUDE.md) |
| miniforge3 module | `miniforge3/25.11.0-1` | `miniforge3/24.11` (WRONG) |
| Conda env | `dp320-torch` | `transpec` (for TranSpec project only) |

### Verified Python Stack

| Component | Version |
|-----------|---------|
| Python | 3.10.20 |
| torch | 2.11.0+cu130 |
| e3nn | 0.4.4 |
| CUDA (build) | 13.0 |

**CUDA available on login node:** False (expected — GPU only on Slurm compute nodes).

### Available CUDA Modules
`cuda/11.7`, `cuda/11.8`, `cuda/12.0`, `cuda/12.4`, `cuda/12.8`, `cuda/12.9`, `cuda/13.0`

### Other Conda Environments
`download`, `dpkit320`, `jupyter`, `jupyterlab`, `py310-torch270-vllm090`, `tensorboard`, `Stanford`

### Slurm
- `sinfo`, `squeue`, `parajobs` available.
- GPU partition: `gpu` (submission: `sbatch -p gpu --gpus=1`).

---

## 2. Changes Made

### 2.1 Global CLAUDE.md (`~/.claude/CLAUDE.md`)

The global HPC rules were updated:

1. **Replaced** the hard-coded `module load miniforge3/24.11` and `source activate transpec` with a probe-first pattern:
   - Do not hard-code module names. Always run `source /etc/profile.d/modules.sh` first.
   - Use `module avail miniforge3` to discover the actual available module.
   - For MTO project work, use the verified stack: `module load miniforge3/25.11.0-1 && source activate dp320-torch`.

2. **Added** N16R4-specific rules:
   - Login node is for editing/light checks only. GPU work must go through Slurm.
   - `/tmp` is forbidden for project scripts. Use `/data/home/scwc008/run/xxy/MTO/tmp/` instead.
   - Do not use fragile Python one-liners with nested f-strings. Write scripts under `MTO/tmp/check_*.py`.
   - Slurm scripts must `cd` into the project dir, `source scripts/hpc_env.sh`, print env info, and write logs under the project run directory.
   - Every job submission must record: job ID, partition, script path, stdout/stderr paths, run directory, checkpoint path, metrics path.

### 2.2 Project CLAUDE.md (`Ar-MTO/CLAUDE.md`)

Appended a new section: `## 16. N16R4 / bjhpc_xxy_1 Operating Rules` containing all N16R4 operating constraints.

### 2.3 New Scripts

| Script | Path | Purpose |
|--------|------|---------|
| Probe | `scripts/probe_hpc_env.sh` | Safe read-only discovery of modules, conda envs, torch, CUDA, Slurm. No assumptions. |
| Env loader | `scripts/hpc_env.sh` | Reusable source-able environment loader using verified module+conda stack. |

### 2.4 Plans.md

Updated:
- Added Task 1.6 to Phase 1.
- Phase 2 tasks (2.1–2.6) now depend on 1.6 instead of blanket "Phase 1".
- Notes section updated with HPC environment reference.

---

## 3. Hard Constraints Now Encoded

1. **Login/compute separation**: Login node for editing only; GPU via Slurm.
2. **Workspace**: All operations under `/data/home/scwc008/run/xxy/MTO`.
3. **Temporary files**: Use `MTO/tmp/`, never `/tmp`.
4. **Module probing**: Never hard-code module names; probe first.
5. **Verified stack**: `miniforge3/25.11.0-1` + `dp320-torch`.
6. **Python checks**: Heredoc scripts only; no nested-f-string one-liners.
7. **Slurm**: `sbatch -p gpu --gpus=1`; log under project run dir.
8. **Job monitoring**: Record all job metadata; verify artifacts before claiming success.

---

## 4. Verification

- [x] Remote probe: `ssh bjhpc_xxy_1 'mkdir -p .../MTO/tmp && cd .../xxy && pwd && hostname && whoami'` passed.
- [x] Module discovery: `miniforge3/25.11.0-1` confirmed available.
- [x] Python + torch: Python 3.10.20, torch 2.11.0+cu130 confirmed.
- [x] CUDA: Expected unavailable on login node; CUDA 13.0 build linked.
- [x] e3nn: 0.4.4 confirmed.
- [x] No hard-coded `miniforge3/24.11` remains in instructions.
- [x] `/tmp` is forbidden for project scripts.
- [x] `scripts/hpc_env.sh` created and will work when sourced on the server.
- [x] `scripts/probe_hpc_env.sh` created for safe environment discovery.
- [x] Phase 2 remains blocked (depends on Task 1.6).

---

## 5. Files Modified/Created

| File | Action |
|------|--------|
| `~/.claude/CLAUDE.md` | Updated — hard-coded module replaced; N16R4 rules added |
| `Ar-MTO/CLAUDE.md` | Updated — Section 16 appended |
| `Ar-MTO/scripts/hpc_env.sh` | Created |
| `Ar-MTO/scripts/probe_hpc_env.sh` | Created |
| `Ar-MTO/Plans.md` | Updated — Task 1.6 added; Phase 2 dependencies updated |
| `Ar-MTO/outputs/reports/n16r4_server_rules_report.md` | Created (this file) |

No files outside `/data/home/scwc008/run/xxy` were modified on the server.

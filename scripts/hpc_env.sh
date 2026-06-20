#!/bin/bash
# hpc_env.sh — Reusable N16R4 environment loader for MTO project
#
# Sources the verified module stack and activates the working conda environment.
# This script is sourced (not executed) so that env variables persist.
#
# Usage: source scripts/hpc_env.sh
#
# Verified on: 2026-06-20
# Server:      ln01 (N16R4 login node)
# User:        scwc008

# --- mandatory: working directory ---
EXPECTED_ROOT="/data/home/scwc008/run/xxy/MTO"
if [ "$(pwd)" != "$EXPECTED_ROOT" ]; then
    echo "hpc_env.sh: WARNING — expected pwd=$EXPECTED_ROOT but pwd=$(pwd)" >&2
    echo "hpc_env.sh: attempting cd to $EXPECTED_ROOT" >&2
    cd "$EXPECTED_ROOT" || {
        echo "hpc_env.sh: ERROR — cannot cd to $EXPECTED_ROOT" >&2
        return 1
    }
fi

# --- module system ---
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
else
    echo "hpc_env.sh: ERROR — /etc/profile.d/modules.sh not found" >&2
    return 1
fi

# --- load verified module ---
# NOTE: miniforge3/25.11.0-1 is the verified working module as of 2026-06-20.
# If unavailable, run scripts/probe_hpc_env.sh to discover the current module.
module load miniforge3/25.11.0-1 2>/dev/null || {
    echo "hpc_env.sh: ERROR — miniforge3/25.11.0-1 unavailable." >&2
    echo "hpc_env.sh: Run scripts/probe_hpc_env.sh to discover current modules." >&2
    return 1
}

# --- activate verified conda environment ---
source activate dp320-torch 2>/dev/null || {
    echo "hpc_env.sh: ERROR — conda env dp320-torch not found." >&2
    echo "hpc_env.sh: Run scripts/probe_hpc_env.sh to discover available envs." >&2
    return 1
}

# --- print environment summary ---
echo "========================================"
echo "MTO HPC Environment (N16R4)"
echo "========================================"
echo "hostname:        $(hostname)"
echo "whoami:          $(whoami)"
echo "pwd:             $(pwd)"
echo "which python:    $(which python 2>/dev/null || echo 'NOT FOUND')"
echo "python --version: $(python --version 2>&1 || echo 'NOT FOUND')"
echo "torch version:   $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "CUDA available:  $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'NOT FOUND')"
echo "CUDA devices:    $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'NOT FOUND')"
echo "loaded modules:  $(module list 2>&1 | tail -1 || echo 'N/A')"
echo "========================================"

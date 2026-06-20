#!/bin/bash
# probe_hpc_env.sh — Safe read-only N16R4 environment probe
#
# Purpose: Discover available modules and Python environments without assuming
# any hard-coded module name or conda env. Prints findings for human review.
#
# Usage: bash scripts/probe_hpc_env.sh
# Must be run on N16R4 login node (ln01) via ssh bjhpc_xxy_1.
# This script is safe to run any time; it only reads environment state.

set -euo pipefail

echo "========================================"
echo "N16R4 Environment Probe"
echo "Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "========================================"

# ---- identity ----
echo ""
echo "--- Identity ---"
echo "hostname: $(hostname)"
echo "whoami:   $(whoami)"
echo "pwd:      $(pwd)"

# ---- module system init ----
echo ""
echo "--- Module System ---"
if [ -f /etc/profile.d/modules.sh ]; then
    echo "source /etc/profile.d/modules.sh  # found"
    source /etc/profile.d/modules.sh
    echo "module command: $(command -v module || echo 'still not found after sourcing')"
else
    echo "source /etc/profile.d/modules.sh  # NOT FOUND — cannot use modules"
fi

# ---- available miniforge modules ----
echo ""
echo "--- Available miniforge/miniforge3 modules ---"
if command -v module &>/dev/null; then
    module avail miniforge 2>&1 | head -20 || echo "(none)"
    echo "---"
    module avail miniforge3 2>&1 | head -20 || echo "(none)"
else
    echo "module command not available; skipping module discovery"
fi

# ---- available CUDA modules ----
echo ""
echo "--- Available CUDA modules ---"
if command -v module &>/dev/null; then
    module avail cuda 2>&1 | head -20 || echo "(none)"
else
    echo "module command not available; skipping CUDA discovery"
fi

# ---- currently loaded modules ----
echo ""
echo "--- Currently Loaded Modules ---"
if command -v module &>/dev/null; then
    module list 2>&1 || echo "(none loaded)"
else
    echo "module command not available"
fi

# ---- conda environments (requires module first) ----
echo ""
echo "--- Conda Environments ---"
if command -v module &>/dev/null && module load miniforge3 2>/dev/null; then
    MINIFORGE_MODULE=$(module avail miniforge3 2>&1 | grep -oP 'miniforge3/\S+' | head -1 || echo "")
    if [ -n "$MINIFORGE_MODULE" ]; then
        module load "$MINIFORGE_MODULE" 2>/dev/null || true
    fi
    conda env list 2>&1 || echo "conda not available after loading miniforge3"
else
    echo "Cannot load miniforge3 module; checking for conda in PATH..."
    command -v conda &>/dev/null && conda env list 2>&1 || echo "conda not found"
fi

# ---- probe each conda env for torch ----
echo ""
echo "--- Torch Availability per Conda Env ---"
for env_path in /data/apps/miniforge3/*/envs/*/bin/python; do
    if [ -x "$env_path" ]; then
        env_name=$(echo "$env_path" | sed 's|/bin/python||' | xargs basename)
        echo -n "  $env_name: "
        "$env_path" -c "
import torch
print(f'python {torch.__version__} cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}')
" 2>/dev/null || echo "torch import failed or not installed"
    fi
done

# ---- GPU visibility (login node — expected to be 0) ----
echo ""
echo "--- GPU Visibility ---"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi failed"
else
    echo "nvidia-smi not found (expected on login node)"
fi

# ---- Slurm status ----
echo ""
echo "--- Slurm Status ---"
command -v sinfo &>/dev/null && sinfo --version 2>&1 || echo "sinfo not found"
command -v squeue &>/dev/null && squeue -u "$(whoami)" 2>&1 | head -10 || echo "squeue not found"
command -v parajobs &>/dev/null && parajobs 2>&1 | head -10 || echo "parajobs not found"

# ---- Disk space ----
echo ""
echo "--- Disk Space (project area) ---"
df -h /data/home/scwc008/run/xxy 2>/dev/null || echo "cannot stat project area"

echo ""
echo "========================================"
echo "Probe complete."
echo "========================================"

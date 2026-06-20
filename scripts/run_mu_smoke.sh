#!/bin/bash
# run_mu_smoke.sh — Submit MTO-Net dipole smoke training to Slurm.
#
# Smoke tests: small datasets, few epochs, verify pipeline works.
# Does NOT run full QM9S training.
#
# Usage:
#   bash scripts/run_mu_smoke.sh              # default: 500 mol, 5 epochs
#   bash scripts/run_mu_smoke.sh medium       # 5000 mol, 10 epochs
#   bash scripts/run_mu_smoke.sh dry          # dry-run only (no GPU)

set -euo pipefail

PROJECT_ROOT="/data/home/scwc008/run/xxy/MTO"
cd "$PROJECT_ROOT"

MODE="${1:-tiny}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

case "$MODE" in
    dry)
        echo "=== DRY-RUN MODE (login node, no Slurm) ==="
        source /etc/profile.d/modules.sh
        module load miniforge3/25.11.0-1
        source activate dp320-torch
        echo "hostname: $(hostname)"
        echo "pwd: $(pwd)"
        echo "which python: $(which python)"
        echo "python --version: $(python --version)"
        python scripts/train_mu.py --config configs/model/mto_full.yaml --dry-run
        echo "Dry-run complete."
        exit 0
        ;;
    tiny)
        DATASET="data/qm9s/subset_smoke/qm9s.pt"
        SPLITS="data/qm9s/splits/smoke/splits.json"
        RUN_DIR="runs/mu_smoke_tiny_${TIMESTAMP}"
        EPOCHS=5
        BATCH_SIZE=32
        GPU_COUNT=1
        JOB_NAME="mto_mu_smoke_tiny"
        WALLTIME="00:30:00"
        ;;
    medium)
        DATASET="data/qm9s/subset_medium/qm9s.pt"
        SPLITS="data/qm9s/splits/medium/splits.json"
        RUN_DIR="runs/mu_smoke_medium_${TIMESTAMP}"
        EPOCHS=10
        BATCH_SIZE=64
        GPU_COUNT=1
        JOB_NAME="mto_mu_smoke_med"
        WALLTIME="01:00:00"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: bash scripts/run_mu_smoke.sh [dry|tiny|medium]"
        exit 1
        ;;
esac

# Write job script
JOB_SCRIPT="tmp/_mu_smoke_${MODE}_${TIMESTAMP}.sh"
cat > "$JOB_SCRIPT" << JOBEOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --gpus=${GPU_COUNT}
#SBATCH --partition=gpu_a800
#SBATCH --time=${WALLTIME}
#SBATCH --output=${RUN_DIR}/slurm_%j.out
#SBATCH --error=${RUN_DIR}/slurm_%j.err

cd "$PROJECT_ROOT"

# Environment
source /etc/profile.d/modules.sh
module load miniforge3/25.11.0-1
source activate dp320-torch
export PYTHONPATH="/data/home/scwc008/run/xxy/MTO/repo-main/src:/data/home/scwc008/run/xxy/MTO/repo-main/third_party/DetaNet:${PYTHONPATH:-}"

echo "============================================"
echo "MTO-Net Mu Smoke Training"
echo "============================================"
echo "Job ID:    \$SLURM_JOB_ID"
echo "hostname:  \$(hostname)"
echo "pwd:       \$(pwd)"
echo "which python: \$(which python)"
echo "python --version: \$(python --version)"
echo "torch version: \$(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: \$(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA devices: \$(python -c 'import torch; print(torch.cuda.device_count())')"
echo "MODE:      ${MODE}"
echo "EPOCHS:    ${EPOCHS}"
echo "DATASET:   ${DATASET}"
echo "RUN_DIR:   ${RUN_DIR}"
echo "============================================"

mkdir -p "${RUN_DIR}"

python scripts/train_mu.py \\
    --config configs/model/mto_full.yaml \\
    --dataset "${DATASET}" \\
    --splits "${SPLITS}" \\
    --run-dir "${RUN_DIR}" \\
    --epochs ${EPOCHS} \\
    --batch-size ${BATCH_SIZE} \\
    --seed 0

EXIT_CODE=\$?

echo ""
echo "============================================"
echo "Training exited with code: \$EXIT_CODE"
echo "Run directory: ${RUN_DIR}"
echo "============================================"

exit \$EXIT_CODE
JOBEOF

chmod +x "$JOB_SCRIPT"

echo "============================================"
echo "Submitting MTO-Net Mu Smoke Job (${MODE})"
echo "============================================"
echo "Mode:       ${MODE}"
echo "Dataset:    ${DATASET}"
echo "Run dir:    ${RUN_DIR}"
echo "Epochs:     ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Job script: ${JOB_SCRIPT}"
echo "============================================"

sbatch "$JOB_SCRIPT"

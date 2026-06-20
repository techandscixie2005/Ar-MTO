#!/bin/bash
# run_mu_full.sh — Submit MTO-Net full QM9S dipole training to Slurm.
#
# Runs full training on the complete QM9S dataset (~130k molecules).
# Supports multiple seeds for statistical analysis.
#
# Usage:
#   bash scripts/run_mu_full.sh              # seed=0, 200 epochs
#   bash scripts/run_mu_full.sh 1            # seed=1
#   bash scripts/run_mu_full.sh 0 5          # seeds 0-4 (5 runs)
#
# DO NOT use this for smoke testing — use run_mu_smoke.sh instead.

set -euo pipefail

PROJECT_ROOT="/data/home/scwc008/run/xxy/MTO"
cd "$PROJECT_ROOT"

SEED_START="${1:-0}"
SEED_END="${2:-$SEED_START}"

EPOCHS=200
BATCH_SIZE=64
LR="5e-4"
DATASET="data/qm9s/qm9s.pt"
SPLITS="data/qm9s/splits/full/splits.json"
GPU_COUNT=1
JOB_NAME="mto_mu_full"
WALLTIME="12:00:00"

for SEED in $(seq "$SEED_START" "$SEED_END"); do
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    RUN_DIR="runs/mu_full/${TIMESTAMP}_seed${SEED}"
    JOB_SCRIPT="tmp/_mu_full_seed${SEED}_${TIMESTAMP}.sh"

    cat > "$JOB_SCRIPT" << JOBEOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}_s${SEED}
#SBATCH --gpus=${GPU_COUNT}
#SBATCH --partition=gpu_a800
#SBATCH --time=${WALLTIME}
#SBATCH --output=${RUN_DIR}/slurm_%j.out
#SBATCH --error=${RUN_DIR}/slurm_%j.err

cd "$PROJECT_ROOT"

source /etc/profile.d/modules.sh
module load miniforge3/25.11.0-1
source activate dp320-torch
export PYTHONPATH="/data/home/scwc008/run/xxy/MTO/repo-main/src:/data/home/scwc008/run/xxy/MTO/repo-main/third_party/DetaNet:${PYTHONPATH:-}"

echo "============================================"
echo "MTO-Net Mu Full Training — Seed ${SEED}"
echo "============================================"
echo "Job ID:    \${SLURM_JOB_ID}"
echo "hostname:  \$(hostname)"
echo "pwd:       \$(pwd)"
echo "which python: \$(which python)"
echo "python --version: \$(python --version)"
echo "torch version: \$(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: \$(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA devices: \$(python -c 'import torch; print(torch.cuda.device_count())')"
echo "SEED:      ${SEED}"
echo "EPOCHS:    ${EPOCHS}"
echo "RUN_DIR:   ${RUN_DIR}"
echo "============================================"

mkdir -p "${RUN_DIR}"

python scripts/train_mu.py \
    --config configs/model/mto_full.yaml \
    --dataset "${DATASET}" \
    --splits "${SPLITS}" \
    --run-dir "${RUN_DIR}" \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --seed ${SEED}

EXIT_CODE=\$?

echo ""
echo "============================================"
echo "Training exited with code: \$EXIT_CODE"
echo "Run directory: ${RUN_DIR}"
echo "============================================"

exit \$EXIT_CODE
JOBEOF

    chmod +x "$JOB_SCRIPT"

    echo "Submitting full training (seed=${SEED})..."
    echo "  Run dir:    ${RUN_DIR}"
    echo "  Job script: ${JOB_SCRIPT}"

    sbatch "$JOB_SCRIPT"
done

echo ""
echo "Done. Check jobs with: squeue -u scwc008"

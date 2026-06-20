#!/usr/bin/env bash
# run_check_detanet_pyg_compat.sh — Submit the DetaNet/PyG compatibility check as a Slurm GPU job.
#
# Usage:
#   bash scripts/run_check_detanet_pyg_compat.sh              # submit with defaults
#   bash scripts/run_check_detanet_pyg_compat.sh --partition gpu --gpus 1
#   bash scripts/run_check_detanet_pyg_compat.sh --skip-slurm # run directly on current node
#
# This script:
#   1. Creates a self-contained Slurm job script
#   2. Submits it via sbatch
#   3. Reports the job ID and expected output paths
#
# The job script will:
#   - source scripts/hpc_env.sh
#   - run scripts/check_detanet_pyg_compat.py
#   - write JSON report to outputs/reports/detanet_pyg_compat_report.json
#   - exit with the compatibility script's exit code

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --- Defaults ---
PARTITION="gpu"
GPUS="1"
CPUS="4"
TIME="00:10:00"
SKIP_SLURM=""
JOB_NAME="mto-pyg-compat"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition) PARTITION="$2"; shift 2 ;;
        --gpus)      GPUS="$2"; shift 2 ;;
        --cpus)      CPUS="$2"; shift 2 ;;
        --time)      TIME="$2"; shift 2 ;;
        --skip-slurm) SKIP_SLURM="1"; shift ;;
        *)           echo "Unknown arg: $1"; exit 2 ;;
    esac
done

# --- Paths ---
CHECK_SCRIPT="${PROJECT_DIR}/scripts/check_detanet_pyg_compat.py"
DATASET="${PROJECT_DIR}/data/qm9s/qm9s.pt"
OUTPUT_JSON="${PROJECT_DIR}/outputs/reports/detanet_pyg_compat_report.json"
OUTPUT_LOG="${PROJECT_DIR}/outputs/reports/detanet_pyg_compat_job_%j.out"

mkdir -p "${PROJECT_DIR}/outputs/reports"
mkdir -p "${PROJECT_DIR}/tmp"

# --- Verify check script exists ---
if [ ! -f "$CHECK_SCRIPT" ]; then
    echo "ERROR: check script not found: $CHECK_SCRIPT"
    exit 1
fi

# --- Direct run mode (current node) ---
if [ "$SKIP_SLURM" = "1" ]; then
    echo "=== Direct run mode (no Slurm) ==="
    echo "Host: $(hostname)"
    echo "Date: $(date)"
    echo ""
    cd "$PROJECT_DIR"
    source scripts/hpc_env.sh
    echo ""
    python "$CHECK_SCRIPT" --data "$DATASET" --output "$OUTPUT_JSON"
    EXIT_CODE=$?
    echo ""
    echo "Exit code: $EXIT_CODE"
    echo "Report: $OUTPUT_JSON"
    exit $EXIT_CODE
fi

# --- Slurm job script ---
JOB_SCRIPT="${PROJECT_DIR}/tmp/_check_detanet_pyg_compat_job.sh"

cat > "$JOB_SCRIPT" << 'SLURM_EOF'
#!/usr/bin/env bash
#SBATCH --job-name=JOB_NAME_PLACEHOLDER
#SBATCH --partition=PARTITION_PLACEHOLDER
#SBATCH --gpus=GPUS_PLACEHOLDER
#SBATCH --cpus-per-task=CPUS_PLACEHOLDER
#SBATCH --time=TIME_PLACEHOLDER
#SBATCH --output=OUTPUT_LOG_PLACEHOLDER
#SBATCH --error=OUTPUT_LOG_PLACEHOLDER

set -euo pipefail

PROJECT_DIR="PROJECT_DIR_PLACEHOLDER"
CHECK_SCRIPT="CHECK_SCRIPT_PLACEHOLDER"
DATASET="DATASET_PLACEHOLDER"
OUTPUT_JSON="OUTPUT_JSON_PLACEHOLDER"

echo "=============================================="
echo " DetaNet/PyG Compatibility Check — Slurm Job"
echo " Job ID: ${SLURM_JOB_ID:-N/A}"
echo " Host: $(hostname)"
echo " Date: $(date)"
echo "=============================================="
echo ""

cd "$PROJECT_DIR"

echo "--- Environment setup ---"
echo "Sourcing hpc_env.sh ..."
source scripts/hpc_env.sh 2>&1
echo ""

echo "Python: $(which python)"
echo "Python version: $(python --version 2>&1)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo ""

echo "--- GPU info ---"
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
"
echo ""

echo "--- Running compatibility check ---"
python "$CHECK_SCRIPT" --data "$DATASET" --output "$OUTPUT_JSON"
EXIT_CODE=$?

echo ""
echo "=============================================="
echo " Job complete. Exit code: $EXIT_CODE"
echo " Report: $OUTPUT_JSON"
echo "=============================================="

exit $EXIT_CODE
SLURM_EOF

# --- Substitute placeholders ---
sed -i \
    -e "s|JOB_NAME_PLACEHOLDER|${JOB_NAME}|g" \
    -e "s|PARTITION_PLACEHOLDER|${PARTITION}|g" \
    -e "s|GPUS_PLACEHOLDER|${GPUS}|g" \
    -e "s|CPUS_PLACEHOLDER|${CPUS}|g" \
    -e "s|TIME_PLACEHOLDER|${TIME}|g" \
    -e "s|OUTPUT_LOG_PLACEHOLDER|${OUTPUT_LOG}|g" \
    -e "s|PROJECT_DIR_PLACEHOLDER|${PROJECT_DIR}|g" \
    -e "s|CHECK_SCRIPT_PLACEHOLDER|${CHECK_SCRIPT}|g" \
    -e "s|DATASET_PLACEHOLDER|${DATASET}|g" \
    -e "s|OUTPUT_JSON_PLACEHOLDER|${OUTPUT_JSON}|g" \
    "$JOB_SCRIPT"

echo "=== Slurm Job Script ==="
echo "Script: $JOB_SCRIPT"
echo "Partition: $PARTITION"
echo "GPUs: $GPUS"
echo "Time: $TIME"
echo "Output log: $OUTPUT_LOG"
echo "Report: $OUTPUT_JSON"
echo ""

echo "--- Submitting ---"
sbatch "$JOB_SCRIPT"

echo ""
echo "Monitor with:"
echo "  squeue -u scwc008"
echo "  parajobs"
echo ""
echo "After completion, read report:"
echo "  cat $OUTPUT_JSON"

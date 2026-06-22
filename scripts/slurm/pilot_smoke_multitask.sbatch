#!/bin/bash
#SBATCH --job-name=pilot_smoke_mt
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/pilot_2k_mu_uv/slurm/smoke_multitask_%j.out
#SBATCH --error=outputs/pilot_2k_mu_uv/slurm/smoke_multitask_%j.err

set -euo pipefail

cd /data/home/scwc008/run/xxy/MTO
source scripts/hpc_env.sh

echo "=== HOSTNAME: $(hostname) ==="
echo "=== PWD: $(pwd) ==="
echo "=== PYTHON: $(which python) ==="
echo "=== GIT: $(git rev-parse --short HEAD) ==="

SPLITS_DIR="outputs/splits"
RUN_DIR="outputs/pilot_2k_mu_uv/checkpoints/smoke_mt_mto_k8_s0"
mkdir -p "$RUN_DIR"

echo "=== Smoke Multitask Training: 64 mols, 5 epochs, MTO K=8 ==="
python scripts/train_pilot.py \
    --config configs/model/pilot/mto_full_multitask.yaml \
    --target-type multitask \
    --dataset data/qm9s/qm9s.pt \
    --train-split "$SPLITS_DIR/qm9s_2k_pilot_train.json" \
    --val-split "$SPLITS_DIR/qm9s_2k_pilot_val.json" \
    --test-split "$SPLITS_DIR/qm9s_2k_pilot_test.json" \
    --run-dir "$RUN_DIR" \
    --batch-size 64 \
    --epochs 5 \
    --seed 0

echo "=== SELF-VERIFY ==="
for f in "$RUN_DIR/best.ckpt" "$RUN_DIR/last.ckpt" "$RUN_DIR/test_metrics.json"; do
    [ -f "$f" ] && echo "  PASS: $f" || echo "  FAIL: $f MISSING"
done

echo "=== SMOKE MULTITASK COMPLETE ==="

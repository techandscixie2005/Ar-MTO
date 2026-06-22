#!/bin/bash
#SBATCH --job-name=pilot_smoke_mu
#SBATCH --partition=gpu_a800
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/pilot_2k_mu_uv/slurm/smoke_mu_%j.out
#SBATCH --error=outputs/pilot_2k_mu_uv/slurm/smoke_mu_%j.err

set -euo pipefail

cd /data/home/scwc008/run/xxy/MTO
source scripts/hpc_env.sh

echo "=== HOSTNAME: $(hostname) ==="
echo "=== PWD: $(pwd) ==="
echo "=== PYTHON: $(which python) ==="
echo "=== TORCH: $(python -c 'import torch; print(torch.__version__)') ==="
echo "=== CUDA AVAIL: $(python -c 'import torch; print(torch.cuda.is_available())') ==="
echo "=== GIT: $(git rev-parse --short HEAD) ==="

SPLITS_DIR="outputs/splits"
RUN_DIR="outputs/pilot_2k_mu_uv/checkpoints/smoke_mu_mto_k8_s0"
mkdir -p "$RUN_DIR"

echo "=== Smoke μ Training: 64 mols, 5 epochs, MTO K=8 ==="
python scripts/train_pilot.py \
    --config configs/model/mto_full.yaml \
    --target-type mu \
    --dataset data/qm9s/qm9s.pt \
    --train-split "$SPLITS_DIR/qm9s_2k_pilot_train.json" \
    --val-split "$SPLITS_DIR/qm9s_2k_pilot_val.json" \
    --test-split "$SPLITS_DIR/qm9s_2k_pilot_test.json" \
    --run-dir "$RUN_DIR" \
    --batch-size 64 \
    --epochs 5 \
    --seed 0

echo "=== SELF-VERIFY: Checking artifacts ==="
for f in "$RUN_DIR/best.ckpt" "$RUN_DIR/last.ckpt" "$RUN_DIR/test_metrics.json" \
         "$RUN_DIR/config.yaml" "$RUN_DIR/run_config.json" "$RUN_DIR/mto_cache_test.pt"; do
    if [ -f "$f" ]; then
        echo "  PASS: $f"
    else
        echo "  FAIL: $f MISSING"
    fi
done

# Check for NaN/inf in test metrics
python -c "
import json
with open('$RUN_DIR/test_metrics.json') as f:
    m = json.load(f)
for k, v in m.items():
    import math
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        print(f'  NAN/INF in {k}: {v}')
        exit(1)
print('  All metrics finite')
"

echo "=== SMOKE μ COMPLETE ==="

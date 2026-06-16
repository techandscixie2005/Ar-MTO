#!/usr/bin/env python3
"""Minimal smoke test for DetaNet forward pass within Ar-MTO.

This script exercises the DetaNet import bridge and runs a forward pass
with synthetic molecular input. No dataset required.

Usage:
    python scripts/smoke_detanet.py
"""

import sys
import os
from pathlib import Path

# Ensure src/ is on the path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


def main():
    import torch
    from ar_mto.detanet_bridge import (
        make_latent_detanet,
        get_detanet_path,
        run_latent_forward,
    )

    print("=" * 60)
    print("Ar-MTO DetaNet Smoke Test")
    print("=" * 60)

    # 1. Environment
    print(f"\n[1/5] Environment")
    print(f"  Python: {sys.version}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  DetaNet path: {get_detanet_path()}")

    # 2. Model instantiation
    print(f"\n[2/5] Building latent DetaNet (num_block=2, maxl=3)")
    model = make_latent_detanet(num_block=2, device="cpu")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Output type: {model.out_type}")
    print(f"  Scalar out size: {model.scalar_outsize}")
    print(f"  Tensor vdim: {model.vdim}")

    # 3. Synthetic molecule
    print(f"\n[3/5] Creating synthetic H2O molecule")
    z = torch.tensor([8, 1, 1], dtype=torch.long)
    pos = torch.tensor(
        [
            [0.0000, 0.0000, 0.1173],
            [0.0000, 0.7572, -0.4692],
            [0.0000, -0.7572, -0.4692],
        ],
        dtype=torch.float32,
    )
    print(f"  Atoms: {z.tolist()}")
    print(f"  Positions shape: {pos.shape}")

    # 4. Forward pass
    print(f"\n[4/5] Running forward pass")
    model.eval()
    with torch.no_grad():
        S, T = run_latent_forward(model, z=z, pos=pos)

    print(f"  Scalar features S: shape={list(S.shape)}, "
          f"mean={S.mean().item():.6f}, std={S.std().item():.6f}")
    print(f"  Tensor features T: shape={list(T.shape)}, "
          f"mean={T.mean().item():.6f}, std={T.std().item():.6f}")

    # NaN/inf check
    s_ok = not torch.isnan(S).any() and not torch.isinf(S).any()
    t_ok = not torch.isnan(T).any() and not torch.isinf(T).any()
    print(f"  S clean: {s_ok}")
    print(f"  T clean: {t_ok}")

    if not (s_ok and t_ok):
        print("FAIL: NaN or inf detected in output")
        return 1

    # 5. Batch test
    print(f"\n[5/5] Batched inference (H2O + CH4)")
    z_ch4 = torch.tensor([6, 1, 1, 1, 1], dtype=torch.long)
    pos_ch4 = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000],
            [0.6287, 0.6287, 0.6287],
            [-0.6287, -0.6287, 0.6287],
            [-0.6287, 0.6287, -0.6287],
            [0.6287, -0.6287, -0.6287],
        ],
        dtype=torch.float32,
    )
    z_batch = torch.cat([z, z_ch4])
    pos_batch = torch.cat([pos, pos_ch4])
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)

    with torch.no_grad():
        S_batch, T_batch = run_latent_forward(model, z=z_batch, pos=pos_batch, batch=batch)

    print(f"  Batched S: shape={list(S_batch.shape)}")
    print(f"  Batched T: shape={list(T_batch.shape)}")
    assert S_batch.shape[0] == 8, f"Expected 8 atoms, got {S_batch.shape[0]}"
    assert not torch.isnan(S_batch).any()
    assert not torch.isnan(T_batch).any()

    print("\n" + "=" * 60)
    print("Smoke test PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

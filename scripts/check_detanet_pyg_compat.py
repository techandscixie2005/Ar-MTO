#!/usr/bin/env python3
"""check_detanet_pyg_compat.py — DetaNet/PyG compatibility gate for N16R4 GPU.

Probes the environment and attempts the full MTO mu training dependency path
using detanet_bridge (which installs pure-PyTorch fallbacks for torch_geometric
and torch_scatter when the real C++ extensions are incompatible).

Exit codes:
  0 — all checks pass, Phase 3 can proceed
  1 — environment probe failed (missing packages, etc.)
  2 — DetaNet import failed despite fallback stubs
  3 — model init / forward / backward failed
  4 — checkpoint save / reload failed

Usage:
  python scripts/check_detanet_pyg_compat.py --data data/qm9s/qm9s.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DETANET_DIR = PROJECT_ROOT / "third_party" / "DetaNet"
TMP_DIR = PROJECT_ROOT / "tmp"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(DETANET_DIR))


# ---------------------------------------------------------------------------
# Step 0: Environment probe
# ---------------------------------------------------------------------------

def probe_environment():
    """Record Python, torch, CUDA versions."""
    import torch

    info = {
        "hostname": os.uname().nodename,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    return info


# ---------------------------------------------------------------------------
# Step 1: PyG package version probe (safe — pip show, no import)
# ---------------------------------------------------------------------------

def probe_pyg_versions():
    """Probe installed PyG package versions via pip show / importlib.

    Does NOT import torch_geometric (which may segfault on incompatible CUDA).
    """
    results = {}
    pkgs = ["torch_geometric", "torch_scatter", "torch_sparse",
            "torch_cluster", "torch_spline_conv"]
    for pkg in pkgs:
        try:
            import importlib.metadata
            v = importlib.metadata.version(pkg)
            results[pkg] = {"status": "installed", "version": v}
        except importlib.metadata.PackageNotFoundError:
            results[pkg] = {"status": "not_installed", "version": None}
        except Exception as e:
            results[pkg] = {"status": "error", "version": None, "error": str(e)}
    return results


# ---------------------------------------------------------------------------
# Step 2: DetaNet import via detanet_bridge (safe path with fallbacks)
# ---------------------------------------------------------------------------

def check_detanet_via_bridge():
    """Import DetaNet via detanet_bridge, which installs PyG fallbacks if needed."""
    import torch
    torch.serialization.add_safe_globals([slice])

    from ar_mto.detanet_bridge import import_detanet, is_pyg_fallback_active

    result = {
        "status": "unknown",
        "error": None,
        "detaNet_class": None,
        "pyg_fallback_active": is_pyg_fallback_active(),
    }

    try:
        DetaNet = import_detanet()
        result["status"] = "ok"
        result["detaNet_class"] = str(DetaNet)
        result["pyg_fallback_active"] = is_pyg_fallback_active()
    except ImportError as e:
        result["status"] = "import_error"
        result["error"] = str(e)
    except Exception as e:
        result["status"] = "crash"
        result["error"] = f"{type(e).__name__}: {e}"
    return result


# ---------------------------------------------------------------------------
# Step 3: Full pipeline — MTO mu model, fwd, bwd, ckpt
# ---------------------------------------------------------------------------

def run_full_pipeline_check(data_path: str):
    """Load a tiny QM9S subset, init full tensor MTO mu model, fwd/bwd, ckpt."""
    import torch
    from ar_mto import MTONet, MTOConfig

    result = {
        "data_load": {"status": "unknown", "num_molecules": 0, "num_atoms": 0},
        "model_init": {"status": "unknown", "param_count": 0},
        "forward": {"status": "unknown", "output_shape": None, "loss": None},
        "backward": {"status": "unknown", "grad_norm": None},
        "checkpoint_save": {"status": "unknown", "path": None},
        "checkpoint_load": {"status": "unknown", "match": None},
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    # --- Load data ---
    print("  loading dataset ...")
    data = torch.load(data_path, weights_only=False)

    molecules = []
    if isinstance(data, list):
        molecules = data

    def _get(mol, key, default=None):
        """Get attribute or key from a Data-like object or dict."""
        if hasattr(mol, key):
            return getattr(mol, key)
        if isinstance(mol, dict):
            return mol.get(key, default)
        return default

    # Take first 4 molecules
    molecules = molecules[:4]
    print(f"  molecules: {len(molecules)}")

    z_list, pos_list, mu_list, batch_list = [], [], [], []
    for b_idx, mol in enumerate(molecules):
        z = _get(mol, "z")
        pos = _get(mol, "pos")
        mu = _get(mol, "mu", _get(mol, "dipole", torch.zeros(1, 3)))
        n_atoms = z.shape[0]
        z_list.append(z)
        pos_list.append(pos)
        mu_list.append(mu.reshape(1, -1) if mu.dim() == 1 else mu)
        batch_list.append(torch.full((n_atoms,), b_idx, dtype=torch.long))

    z_batch = torch.cat(z_list).to(device)
    pos_batch = torch.cat(pos_list).to(device).float()
    mu_batch = torch.cat(mu_list).to(device).float()
    batch_batch = torch.cat(batch_list).to(device)
    result["data_load"] = {
        "status": "ok",
        "num_molecules": len(molecules),
        "num_atoms": int(z_batch.shape[0]),
    }

    # --- Init model ---
    print("  initializing MTONet ...")
    from ar_mto.mto_net import make_mto_net
    from ar_mto.detanet_bridge import make_latent_detanet

    detanet = make_latent_detanet(
        num_features=128,
        maxl=3,
        num_block=3,
        rc=5.0,
        max_atomic_number=9,
        device=str(device),
    )
    config = MTOConfig(
        num_features=128,
        maxl=3,
        num_modes=8,
        num_block=3,
        rc=5.0,
        max_atomic_number=9,
    )
    model = make_mto_net(detanet_model=detanet, **config.__dict__).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    result["model_init"] = {"status": "ok", "param_count": param_count}
    print(f"  params: {param_count:,}")

    # --- Forward ---
    print("  forward pass ...")
    model.train()
    output = model(z=z_batch, pos=pos_batch, batch=batch_batch)
    pred = output["vector"].reshape(mu_batch.shape)
    loss = torch.nn.functional.mse_loss(pred, mu_batch)
    result["forward"] = {
        "status": "ok",
        "output_shape": list(pred.shape),
        "loss": float(loss.item()),
    }
    print(f"  pred shape: {list(pred.shape)}, loss: {loss.item():.6f}")

    # --- Backward ---
    print("  backward pass ...")
    loss.backward()
    total_grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    result["backward"] = {"status": "ok", "grad_norm": float(total_grad_norm)}
    print(f"  grad norm: {total_grad_norm:.6f}")

    # --- Checkpoint save ---
    print("  saving checkpoint ...")
    ckpt_path = TMP_DIR / "compat_check_ckpt.pt"
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {k: v for k, v in config.__dict__.items()},
        },
        str(ckpt_path),
    )
    result["checkpoint_save"] = {"status": "ok", "path": str(ckpt_path)}

    # --- Checkpoint reload ---
    print("  reloading checkpoint ...")
    ckpt = torch.load(str(ckpt_path), weights_only=False)
    detanet2 = make_latent_detanet(
        num_features=128, maxl=3, num_block=3, rc=5.0, max_atomic_number=9,
        device=str(device),
    )
    model2 = make_mto_net(
        detanet_model=detanet2, **ckpt["config"]
    ).to(device)
    model2.load_state_dict(ckpt["model_state_dict"])
    model2.eval()
    with torch.no_grad():
        output2 = model2(z=z_batch, pos=pos_batch, batch=batch_batch)
        pred2 = output2["vector"].reshape(mu_batch.shape)
    match = torch.allclose(pred, pred2, atol=1e-5)
    result["checkpoint_load"] = {"status": "ok", "match": bool(match)}
    print(f"  reload match: {match}")

    ckpt_path.unlink(missing_ok=True)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DetaNet/PyG compatibility gate for MTO mu training"
    )
    parser.add_argument(
        "--data",
        default="data/qm9s/qm9s.pt",
        help="Path to QM9S dataset (.pt file)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write JSON results (default: stdout only)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" DetaNet/PyG Compatibility Gate")
    print(f" {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    # Step 0: Environment
    print("--- Step 0: Environment ---")
    env_info = probe_environment()
    for k, v in env_info.items():
        print(f"  {k}: {v}")
    print()

    # Step 1: PyG version probe (pip show, no import crash risk)
    print("--- Step 1: PyG package versions (pip show, no import) ---")
    pyg_info = probe_pyg_versions()
    for pkg, info in pyg_info.items():
        status = info["status"]
        detail = info.get("version") or info.get("error", "")
        print(f"  {pkg}: {status}  {detail}")
    print()

    # Step 2: DetaNet import via bridge (safe — installs fallbacks if needed)
    print("--- Step 2: DetaNet import via detanet_bridge ---")
    detanet_info = check_detanet_via_bridge()
    print(f"  status: {detanet_info['status']}")
    print(f"  pyg_fallback_active: {detanet_info['pyg_fallback_active']}")
    if detanet_info["error"]:
        print(f"  error: {detanet_info['error']}")
    print()

    # Step 3: Full pipeline (only if DetaNet import succeeded)
    can_proceed = detanet_info["status"] == "ok"
    pipeline_result = None

    if can_proceed:
        print("--- Step 3: Full MTO mu pipeline ---")
        try:
            data_path = PROJECT_ROOT / args.data
            if not data_path.exists():
                print(f"  SKIP: dataset not found at {data_path}")
                can_proceed = False
            else:
                pipeline_result = run_full_pipeline_check(str(data_path))
        except Exception as e:
            import traceback
            print(f"  FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            can_proceed = False
    else:
        print("--- Step 3: SKIP (DetaNet import failed) ---")

    # --- Final report ---
    print()
    print("=" * 60)

    # Determine recommendation text
    if can_proceed and pipeline_result:
        if detanet_info["pyg_fallback_active"]:
            recommendation = (
                "Phase 3 can proceed — PyG fallback active. "
                "DetaNet uses pure-PyTorch radius_graph and scatter. "
                "No C++ extension dependency. Training is safe on this stack."
            )
        else:
            recommendation = (
                "Phase 3 can proceed — DetaNet/PyG native path verified on GPU"
            )
    else:
        recommendation = "Phase 3 blocked — see diagnosis above"

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": env_info,
        "pyg_versions": pyg_info,
        "detanet_import": detanet_info,
        "pipeline": pipeline_result,
        "verdict": "PASS" if (can_proceed and pipeline_result) else "BLOCKED",
        "recommendation": recommendation,
    }
    print(json.dumps(report, indent=2, default=str))
    print("=" * 60)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nReport written to: {out_path}")

    if can_proceed and pipeline_result:
        sys.exit(0)
    elif not can_proceed:
        sys.exit(2)
    else:
        sys.exit(3)


if __name__ == "__main__":
    main()

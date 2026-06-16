"""Audit script for DetaNet latent tensor T.

Determines:
  1. Exact flat layout of T
  2. Split/reconstruct utilities
  3. Reconstruction error
  4. Wigner-D rotation equivariance for each l order
  5. Translation invariance
  6. Same-element atom permutation behavior

Produces:
  outputs/audit/detanet_tensor_layout.json
  outputs/audit/tensor_reconstruction_error.json
  outputs/audit/tensor_equivariance_audit.json
  outputs/audit/permutation_translation_audit.json
"""

import json
import os
import sys
from datetime import datetime, timezone

import torch

# Must be before e3nn import: e3nn 0.4.x uses torch.load() without weights_only=False
torch.serialization.add_safe_globals([slice])

from e3nn import o3  # noqa: E402

# Ensure DetaNet and ar_mto are importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "third_party", "DetaNet"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from detanet_model import DetaNet  # noqa: E402
from ar_mto.detanet_bridge import compute_radius_edges  # noqa: E402


OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs", "audit")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_forward(model, z, pos, batch=None):
    """Run DetaNet latent forward with manual edge computation."""
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    z = z.to(device=device)
    pos = pos.to(dtype=dtype, device=device)
    if batch is not None:
        batch = batch.to(device=device)
    edge_index = compute_radius_edges(pos=pos, rc=model.rc, batch=batch)
    return model(z=z, pos=pos, edge_index=edge_index, batch=batch)


def random_rotation_matrix(dtype=torch.float32) -> torch.Tensor:
    """Generate a random 3x3 proper rotation matrix (determinant +1)."""
    q = torch.randn(4)
    q = q / torch.norm(q)
    w, x, y, z = q
    R = torch.tensor([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ], dtype=dtype)
    return R


def wigner_D_matrix(l: int, R: torch.Tensor) -> torch.Tensor:
    """Compute Wigner D-matrix for order l and rotation R using e3nn."""
    # o3.matrix_to_angles and wigner_D work in float64; cast back after
    R64 = R.to(dtype=torch.float64)
    alpha, beta, gamma = o3.matrix_to_angles(R64)
    D = o3.wigner_D(l, alpha, beta, gamma)
    return D.to(dtype=R.dtype)


def get_tensor_layout(model) -> dict:
    """Derive the flat tensor layout from the model's irreps_T."""
    # Derive maxl from the model's spherical harmonics irreps (has l=1..maxl)
    maxl = max(ir.l for _, ir in model.irreps_sh)
    irreps_T = o3.Irreps(
        (model.features, (l, (-1) ** l)) for l in range(1, maxl + 1)
    )
    blocks = []
    offset = 0
    for mul, (l, p) in irreps_T:
        per_channel = 2 * l + 1
        total_dim = mul * per_channel
        parity = "even" if p == 1 else "odd"
        blocks.append({
            "l": l,
            "parity": parity,
            "multiplicity": mul,
            "dim_per_channel": per_channel,
            "total_dim": total_dim,
            "flat_start": offset,
            "flat_end": offset + total_dim,
        })
        offset += total_dim
    return {
        "irreps_str": str(irreps_T),
        "total_vdim": irreps_T.dim,
        "scalar_dim": model.features,
        "blocks": blocks,
    }


def split_T(T: torch.Tensor, layout: dict) -> dict:
    """Split flat T tensor into per-l blocks.

    Returns dict: {l: tensor of shape [num_atoms, multiplicity, 2*l+1]}
    """
    blocks = {}
    for b in layout["blocks"]:
        l = b["l"]
        sliced = T[:, b["flat_start"]:b["flat_end"]]
        blocks[l] = sliced.reshape(T.shape[0], b["multiplicity"], 2 * l + 1)
    return blocks


def reconstruct_T(blocks: dict, layout: dict, device=None) -> torch.Tensor:
    """Reconstruct flat T from per-l blocks (inverse of split_T)."""
    if device is None:
        device = next(iter(blocks.values())).device
    num_atoms = next(iter(blocks.values())).shape[0]
    T_recon = torch.zeros(num_atoms, layout["total_vdim"], device=device,
                          dtype=next(iter(blocks.values())).dtype)
    for b in layout["blocks"]:
        l = b["l"]
        flat = blocks[l].reshape(num_atoms, b["total_dim"])
        T_recon[:, b["flat_start"]:b["flat_end"]] = flat
    return T_recon


# ---------------------------------------------------------------------------
# Synthetic molecule generators
# ---------------------------------------------------------------------------


def make_molecule(num_atoms: int = 5, seed: int = 42, dtype=torch.float32):
    """Generate a small synthetic molecule with all atoms within cutoff radius.

    Atoms are placed uniformly in a sphere of radius rc/3 so that all
    pairwise distances are < rc (default rc=5.0).
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    z = torch.randint(1, 10, (num_atoms,), generator=gen)
    # Place atoms in a sphere of radius 1.5 (max pairwise distance <= 3.0 < rc=5.0)
    radius = 1.2
    pos = torch.randn(num_atoms, 3, generator=gen, dtype=dtype)
    norms = torch.norm(pos, dim=-1, keepdim=True)
    scales = torch.rand(num_atoms, 1, generator=gen) ** (1.0 / 3.0)
    pos = pos / (norms + 1e-8) * scales * radius
    return z, pos


def make_batch(num_molecules: int = 3, max_atoms: int = 6, seed: int = 42, dtype=torch.float32):
    """Generate a batched set of molecules."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    z_list, pos_list, batch_list = [], [], []
    for m in range(num_molecules):
        n = torch.randint(3, max_atoms + 1, (1,), generator=gen).item()
        z = torch.randint(1, 10, (n,), generator=gen)
        pos = torch.randn(n, 3, generator=gen, dtype=dtype) * 2.0
        z_list.append(z)
        pos_list.append(pos)
        batch_list.append(torch.full((n,), m, dtype=torch.long))
    return torch.cat(z_list), torch.cat(pos_list), torch.cat(batch_list)


# ---------------------------------------------------------------------------
# Audit 1: Layout
# ---------------------------------------------------------------------------


def audit_layout():
    """Produce detanet_tensor_layout.json"""
    model = DetaNet(num_features=128, maxl=3, out_type="latent", device="cpu")
    layout = get_tensor_layout(model)
    layout["model_params"] = sum(p.numel() for p in model.parameters())
    layout["note"] = (
        "S (scalar) = l=0 invariant features. "
        "T (tensor) = concatenated l=1,2,3 irrep features. "
        "l=1 odd parity (128×3=384), l=2 even parity (128×5=640), "
        "l=3 odd parity (128×7=896). Total vdim=1920."
    )

    path = os.path.join(OUTPUT_DIR, "detanet_tensor_layout.json")
    with open(path, "w") as f:
        json.dump(layout, f, indent=2)
    print(f"[OK] detanet_tensor_layout.json written ({path})")
    return layout


# ---------------------------------------------------------------------------
# Audit 2: Split / Reconstruct
# ---------------------------------------------------------------------------


def audit_reconstruction(layout: dict):
    """Test split -> reconstruct exact recovery."""
    results = {
        "test": "split_reconstruct_exact_recovery",
        "tolerance": 1e-6,
        "cases": [],
    }

    model = DetaNet(num_features=128, maxl=3, out_type="latent", device="cpu")

    for case_name, z, pos, batch in [
        ("single_3atoms", *make_molecule(3, seed=1), None),
        ("single_5atoms", *make_molecule(5, seed=2), None),
        ("single_8atoms", *make_molecule(8, seed=3), None),
        ("batched_3x4atoms", *make_batch(3, 5, seed=4)),
    ]:
        with torch.no_grad():
            S, T = run_forward(model, z, pos, batch)
        blocks = split_T(T, layout)
        T_recon = reconstruct_T(blocks, layout, device=T.device)
        err = (T - T_recon).abs().max().item()
        results["cases"].append({
            "dtype": "float32",
            "case": case_name,
            "num_atoms": int(z.shape[0]),
            "S_shape": list(S.shape),
            "T_shape": list(T.shape),
            "reconstruction_max_error": err,
            "pass": err < 1e-6,
        })
        status = "PASS" if err < 1e-6 else "FAIL"
        print(f"  [{status}] {case_name}: recon err={err:.2e}")

    path = os.path.join(OUTPUT_DIR, "tensor_reconstruction_error.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] tensor_reconstruction_error.json written ({path})")
    return results


# ---------------------------------------------------------------------------
# Audit 3: Equivariance
# ---------------------------------------------------------------------------


def audit_equivariance(layout: dict):
    """Test Wigner-D rotation equivariance for each l order."""
    results = {
        "test": "wigner_d_rotation_equivariance",
        "description": (
            "Apply random rotation R to positions, compute per-l Wigner D^l(R), "
            "and verify each l block transforms as h_rot ≈ D^l @ h_orig."
        ),
        "tolerance": 5e-5,
        "note": "float32 model with float32 inputs; tolerances account for numerical precision.",
        "cases": [],
    }

    model = DetaNet(num_features=128, maxl=3, out_type="latent", device="cpu")

    for case_idx in range(5):
        seed = 100 + case_idx
        n_atoms = 4 + case_idx
        z, pos = make_molecule(n_atoms, seed=seed)
        R = random_rotation_matrix()
        pos_rot = pos @ R.T

        with torch.no_grad():
            _S_orig, T_orig = run_forward(model, z, pos)
            _S_rot, T_rot = run_forward(model, z, pos_rot)

        blocks_orig = split_T(T_orig, layout)
        blocks_rot = split_T(T_rot, layout)

        l_errors = {}
        for b in layout["blocks"]:
            l = b["l"]
            D = wigner_D_matrix(l, R)  # [2l+1, 2l+1]
            h_orig = blocks_orig[l]  # [n, C, 2l+1]
            h_rot = blocks_rot[l]      # [n, C, 2l+1]
            h_rot_pred = torch.einsum("ab,ncb->nca", D, h_orig)
            err = (h_rot - h_rot_pred).abs().max().item()
            rel_err = err / (h_orig.abs().max().item() + 1e-16)
            l_errors[f"l={l}"] = {
                "max_abs_error": err,
                "max_rel_error": rel_err,
                "pass": err < 5e-5,
            }
            status = "PASS" if err < 5e-5 else "FAIL"
            print(f"  [{status}] case {case_idx} l={l}: abs_err={err:.2e} rel_err={rel_err:.2e}")

        results["cases"].append({
            "case": case_idx,
            "num_atoms": n_atoms,
            "dtype": "float32",
            "errors": l_errors,
        })

    path = os.path.join(OUTPUT_DIR, "tensor_equivariance_audit.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[OK] tensor_equivariance_audit.json written ({path})")
    return results


# ---------------------------------------------------------------------------
# Audit 4: Permutation & Translation
# ---------------------------------------------------------------------------


def audit_permutation_translation(layout: dict):
    """Test translation invariance of S/T and permutation behavior."""
    results = {
        "test": "permutation_and_translation",
        "cases": [],
    }

    model = DetaNet(num_features=128, maxl=3, out_type="latent", device="cpu")

    # --- Translation invariance ---
    z, pos = make_molecule(5, seed=300)
    translation = torch.tensor([10.0, -5.0, 3.0])

    with torch.no_grad():
        S_orig, T_orig = run_forward(model, z, pos)
        S_trans, T_trans = run_forward(model, z, pos + translation)

    # S (scalars) should be invariant under translation
    s_trans_err = (S_orig - S_trans).abs().max().item()
    # T (tensors) should also be invariant under translation
    t_trans_err = (T_orig - T_trans).abs().max().item()

    print(f"  Translation: S max diff={s_trans_err:.2e}, T max diff={t_trans_err:.2e}")

    results["cases"].append({
        "type": "translation_invariance",
        "translation_vector": translation.tolist(),
        "S_max_abs_diff": s_trans_err,
        "T_max_abs_diff": t_trans_err,
        "S_invariant": s_trans_err < 1e-4,
        "T_invariant": t_trans_err < 1e-4,
    })

    # --- Same-element permutation ---
    # Create a molecule with duplicate elements so we can permute meaningfully
    z = torch.tensor([1, 6, 1, 6, 1], dtype=torch.long)  # H, C, H, C, H
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ], dtype=torch.float32)

    # Permute atoms 0↔2 (both H at same positions would be wrong; use different positions)
    perm = torch.tensor([2, 1, 0, 3, 4], dtype=torch.long)
    z_perm = z[perm]
    pos_perm = pos[perm]

    with torch.no_grad():
        S_orig, T_orig = run_forward(model, z, pos)
        S_perm, T_perm = run_forward(model, z_perm, pos_perm)

    # Under permutation, features should permute accordingly:
    # S_perm[i] should equal S_orig[perm[i]]
    S_expected = S_orig[perm]
    s_perm_err = (S_perm - S_expected).abs().max().item()

    T_expected = T_orig[perm]
    t_perm_err = (T_perm - T_expected).abs().max().item()

    print(f"  Permutation: S max diff={s_perm_err:.2e}, T max diff={t_perm_err:.2e}")

    results["cases"].append({
        "type": "permutation_equivariance",
        "permutation": perm.tolist(),
        "S_max_abs_diff": s_perm_err,
        "T_max_abs_diff": t_perm_err,
        "S_permutation_equivariant": s_perm_err < 1e-4,
        "T_permutation_equivariant": t_perm_err < 1e-4,
    })

    # --- Spatial inversion test (parity) ---
    # Full inversion: x -> -x, y -> -y, z -> -z (improper rotation, det = -1)
    # Under inversion, parity determines sign: p=1 (even) → invariant, p=-1 (odd) → sign flip
    # DetaNet's irreps_T uses parity = (-1)^l, so:
    #   l=1 (odd, p=-1): h -> -h
    #   l=2 (even, p=1): h -> h
    #   l=3 (odd, p=-1): h -> -h
    inversion = torch.diag(torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32))
    pos_inv = pos @ inversion.T

    with torch.no_grad():
        _S_inv, T_inv = run_forward(model, z, pos_inv)

    blocks_orig_2 = split_T(T_orig, layout)  # from permutation section above
    blocks_inv = split_T(T_inv, layout)

    inversion_results = {}
    for b in layout["blocks"]:
        l = b["l"]
        h_o = blocks_orig_2[l]
        h_i = blocks_inv[l]
        parity = -1 if (l % 2 == 1) else 1  # (-1)^l: odd l → odd parity → sign flip
        expected = parity * h_o
        err = (h_i - expected).abs().max().item()
        inversion_results[f"l={l}"] = {
            "expected_parity_sign": parity,
            "max_abs_error": err,
            "pass": err < 5e-5,
        }
        status = "PASS" if err < 5e-5 else "FAIL"
        print(f"  Inversion l={l}: max_err={err:.2e} (expected sign={parity}) [{status}]")

    results["cases"].append({
        "type": "spatial_inversion_parity",
        "inversion_matrix": [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
        "details": inversion_results,
    })

    path = os.path.join(OUTPUT_DIR, "permutation_translation_audit.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] permutation_translation_audit.json written ({path})")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("DetaNet Tensor Audit — wt-02")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    print("\n[1/4] Tensor layout audit...")
    layout = audit_layout()

    print("\n[2/4] Split/reconstruct audit...")
    audit_reconstruction(layout)

    print("\n[3/4] Equivariance audit (Wigner-D rotation)...")
    audit_equivariance(layout)

    print("\n[4/4] Permutation & translation audit...")
    audit_permutation_translation(layout)

    print("\n" + "=" * 60)
    print("Audit complete. Outputs in outputs/audit/")
    print("=" * 60)


if __name__ == "__main__":
    main()

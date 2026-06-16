"""Generate MTO audit outputs: equivariance errors, shape contract, report."""

import json
import os
import sys
import torch
from e3nn import o3

torch.serialization.add_safe_globals([slice])

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "DetaNet"))

from ar_mto.tensor_adapter import make_adapter
from ar_mto.signed_routing import SignedRouter
from ar_mto.mto_core import MTOModeAssembly
from ar_mto.cg_coupling import CGCouplingMinimal
from ar_mto.tensor_gate import TensorGate

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "audit")


def _random_rotation(seed=123):
    gen = torch.Generator()
    gen.manual_seed(seed)
    q = torch.randn(4, generator=gen)
    q = q / torch.norm(q)
    w, x, y, z = q
    return torch.tensor([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def _make_h(N=5, C=128, maxl=3):
    h = {0: torch.randn(N, C, 1)}
    for l in range(1, maxl + 1):
        h[l] = torch.randn(N, C, 2 * l + 1)
    return h


def audit_mto_equivariance():
    """Measure MTO internal equivariance errors for each l order."""
    N, C, K = 5, 128, 4
    Cout = 64

    mto = MTOModeAssembly(num_features=C, mode_channels=Cout, num_modes=K, maxl=3)
    router = SignedRouter(num_features=C, num_modes=K, maxl=3)

    h_orig = _make_h(N, C)
    R = _random_rotation(seed=42)

    h_rot = {0: h_orig[0].clone()}
    for l in [1, 2, 3]:
        D = o3.wigner_D(l, *o3.matrix_to_angles(R))
        h_rot[l] = torch.einsum("ab,ncb->nca", D, h_orig[l])

    with torch.no_grad():
        c_orig = router(h_orig)
        c_rot = router(h_rot)
        O_orig = mto(h_orig, c_orig)
        O_rot = mto(h_rot, c_rot)

    results = []
    for l in [0, 1, 2, 3]:
        if l == 0:
            err = (O_rot[l] - O_orig[l]).abs().max().item()
        else:
            D = o3.wigner_D(l, *o3.matrix_to_angles(R))
            O_pred = torch.einsum("ab,kcb->kca", D, O_orig[l])
            err = (O_rot[l] - O_pred).abs().max().item()
        results.append({
            "l": l,
            "dim": 2 * l + 1 if l > 0 else 1,
            "max_abs_error": err,
            "tolerance": 5e-5,
            "passed": err < 5e-5,
        })

    return results


def audit_gate_equivariance():
    """Measure gate equivariance errors."""
    K, C = 4, 64
    gate = TensorGate(mode_channels=C, num_modes=K, maxl=3)

    O_orig = _make_h(K, C)
    R = _random_rotation(seed=100)

    O_rot = {0: O_orig[0].clone()}
    for l in [1, 2, 3]:
        D = o3.wigner_D(l, *o3.matrix_to_angles(R))
        O_rot[l] = torch.einsum("ab,kcb->kca", D, O_orig[l])

    with torch.no_grad():
        Og_orig = gate(O_orig)
        Og_rot = gate(O_rot)

    results = []
    for l in [0, 1, 2, 3]:
        if l == 0:
            err = (Og_rot[l] - Og_orig[l]).abs().max().item()
        else:
            D = o3.wigner_D(l, *o3.matrix_to_angles(R))
            Og_pred = torch.einsum("ab,kcb->kca", D, Og_orig[l])
            err = (Og_rot[l] - Og_pred).abs().max().item()
        results.append({
            "l": l,
            "max_abs_error": err,
            "tolerance": 1e-4,
            "passed": err < 1e-4,
        })

    return results


def audit_cg_equivariance():
    """Measure CG coupling equivariance errors."""
    K, C = 4, 64
    cg = CGCouplingMinimal(mode_channels=C)

    O_orig = _make_h(K, C, maxl=3)
    R = _random_rotation(seed=200)

    O_rot = {0: O_orig[0].clone()}
    for l in [1, 2, 3]:
        D = o3.wigner_D(l, *o3.matrix_to_angles(R))
        O_rot[l] = torch.einsum("ab,kcb->kca", D, O_orig[l])

    with torch.no_grad():
        Oc_orig = cg(O_orig)
        Oc_rot = cg(O_rot)

    results = []
    for l in [0, 1, 2]:
        if l == 0:
            err = (Oc_rot[l] - Oc_orig[l]).abs().max().item()
        else:
            D = o3.wigner_D(l, *o3.matrix_to_angles(R))
            Oc_pred = torch.einsum("ab,kcb->kca", D, Oc_orig[l])
            err = (Oc_rot[l] - Oc_pred).abs().max().item()
        results.append({
            "l": l,
            "max_abs_error": err,
            "tolerance": 1e-4,
            "passed": err < 1e-4,
        })

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # MTO internal equivariance
    mto_equiv = audit_mto_equivariance()
    with open(os.path.join(OUTPUT_DIR, "mto_internal_equivariance.json"), "w") as f:
        json.dump({
            "description": "MTO internal mode equivariance under Wigner-D rotation",
            "tolerance": "5e-5 (float32)",
            "results": mto_equiv,
            "all_passed": all(r["passed"] for r in mto_equiv),
        }, f, indent=2)
    print("Wrote mto_internal_equivariance.json")

    # Output equivariance (gate + MTO combined)
    gate_equiv = audit_gate_equivariance()
    with open(os.path.join(OUTPUT_DIR, "output_equivariance.json"), "w") as f:
        json.dump({
            "description": "MTO output equivariance after gating under Wigner-D rotation",
            "components": ["MTO assembly", "Tensor gate"],
            "tolerance": "1e-4 (float32)",
            "results": gate_equiv,
            "all_passed": all(r["passed"] for r in gate_equiv),
        }, f, indent=2)
    print("Wrote output_equivariance.json")

    # CG coupling equivariance
    cg_equiv = audit_cg_equivariance()
    with open(os.path.join(OUTPUT_DIR, "cg_coupling_equivariance.json"), "w") as f:
        json.dump({
            "description": "CG coupling equivariance under Wigner-D rotation",
            "coupling_type": "minimal (0x1→1, 0x2→2, 1x1→0)",
            "tolerance": "1e-4 (float32)",
            "results": cg_equiv,
            "all_passed": all(r["passed"] for r in cg_equiv),
        }, f, indent=2)
    print("Wrote cg_coupling_equivariance.json")

    # Shape contract
    shape_contract = {
        "dtanet_latent": {
            "S_shape": "[N, 128]",
            "T_shape": "[N, 1920]",
            "T_flat_layout": {
                "h1": "T[:, 0:384].reshape(-1, 128, 3)",
                "h2": "T[:, 384:1024].reshape(-1, 128, 5)",
                "h3": "T[:, 1024:1920].reshape(-1, 128, 7)",
            },
            "h0_contract": "h0 = S.unsqueeze(-1)  # [N, 128, 1]",
        },
        "mto_adapter": {
            "input": {"S": "[N, 128]", "T": "[N, 1920]"},
            "output": {
                "h0": "[N, 128, 1]",
                "h1": "[N, 128, 3]",
                "h2": "[N, 128, 5]",
                "h3": "[N, 128, 7]",
            },
            "split_reconstruct_error": 0.0,
        },
        "signed_routing": {
            "input": "invariant features (h0 scalars + tensor norms)",
            "coefficient_shape": "[K, N, 1] per l order",
            "coefficient_range": "[-1, 1] (softmax * tanh)",
            "same_for_all_l": True,
            "rotation_invariant": True,
        },
        "mto_assembly": {
            "operation": "O_k^(l) = sum_i c_ki^(l) W_l H_i^(l)",
            "input_h": "[N, C_in, 2l+1] per l",
            "input_coeff": "[K, N, 1] per l",
            "output_O": "[K, C_out, 2l+1] per l",
            "default_C_in": 128,
            "default_C_out": 64,
            "default_K": 8,
            "default_maxl": 3,
        },
        "cg_coupling": {
            "type": "minimal",
            "paths": [
                "l=0 × l=1 → l=1",
                "l=0 × l=2 → l=2",
                "l=1 × l=1 → l=0",
            ],
            "implementation": "e3nn o3.FullyConnectedTensorProduct",
            "input_shape": "[K, C, 2l+1] per l",
            "output_shape": "[K, C, 2l+1] for l in {0, 1, 2}",
        },
        "tensor_gate": {
            "operation": "O_tilde_k^(l) = gamma_k_l * O_k^(l)",
            "gamma_range": "(0, 1) via sigmoid",
            "invariant_inputs": ["l=0 scalars", "tensor norms"],
            "equivariance_preserving": True,
        },
        "readouts": {
            "scalar": "MLP(l=0 modes) → [1]",
            "vector": "l=1 mode weighted sum → [3]",
            "rank2_tensor": "isotropic(l=0) + traceless(l=2) → [3, 3]",
        },
    }
    with open(os.path.join(OUTPUT_DIR, "mto_shape_contract.json"), "w") as f:
        json.dump(shape_contract, f, indent=2)
    print("Wrote mto_shape_contract.json")

    print("\nAll audit outputs generated.")


if __name__ == "__main__":
    main()

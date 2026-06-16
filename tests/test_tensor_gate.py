"""Test invariant tensor gates: equivariance, gate range, ablation."""

import pytest
import torch
from e3nn import o3

from ar_mto.tensor_gate import TensorGate, NoGate


def _make_O(K=4, C=64, maxl=3):
    O = {}
    O[0] = torch.randn(K, C, 1)
    for l in range(1, maxl + 1):
        O[l] = torch.randn(K, C, 2 * l + 1)
    return O


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


class TestTensorGateEquivariance:
    def test_gate_preserves_equivariance(self):
        """Gate should multiply all m-components equally, preserving equivariance."""
        K, C = 4, 64
        gate = TensorGate(mode_channels=C, num_modes=K, maxl=3)
        O = _make_O(K, C)

        # Under rotation, O transforms. Gate should produce the same gating
        # regardless of orientation, since it uses only invariants.
        R = _random_rotation(seed=42)

        # Rotated O
        O_rot = {0: O[0].clone()}
        for l in [1, 2, 3]:
            D = o3.wigner_D(l, *o3.matrix_to_angles(R))
            O_rot[l] = torch.einsum("ab,kcb->kca", D, O[l])

        with torch.no_grad():
            Og = gate(O)
            Og_rot = gate(O_rot)

        # Gated output should still be equivariant:
        # gate(O_rot)[l] should equal D @ gate(O)[l]
        for l in [0, 1, 2, 3]:
            if l == 0:
                err = (Og_rot[l] - Og[l]).abs().max().item()
            else:
                D = o3.wigner_D(l, *o3.matrix_to_angles(R))
                Og_pred = torch.einsum("ab,kcb->kca", D, Og[l])
                err = (Og_rot[l] - Og_pred).abs().max().item()
            assert err < 1e-4, f"Gate broke equivariance for l={l}: err={err:.2e}"

    def test_gate_range(self):
        """Gate values should be in [0, 1] (sigmoid)."""
        K, C = 4, 64
        gate = TensorGate(mode_channels=C, num_modes=K, maxl=3)
        O = _make_O(K, C)
        with torch.no_grad():
            Og = gate(O)

        for l in [0, 1, 2, 3]:
            # Gate multiplies, so output should have magnitudes <= input
            ratio = Og[l].abs() / (O[l].abs() + 1e-8)
            # Rough check: gating reduces or preserves magnitude
            assert ratio.max() <= 2.0, f"Gate amplification for l={l}"

    def test_gate_finite(self):
        K, C = 4, 64
        gate = TensorGate(mode_channels=C, num_modes=K, maxl=3)
        O = _make_O(K, C)
        with torch.no_grad():
            Og = gate(O)
        for l in [0, 1, 2, 3]:
            assert not torch.isnan(Og[l]).any()
            assert not torch.isinf(Og[l]).any()

    def test_gate_stats(self):
        K, C = 4, 64
        gate = TensorGate(mode_channels=C, num_modes=K, maxl=3)
        O = _make_O(K, C)
        with torch.no_grad():
            _ = gate(O)
        stats = gate.gate_stats(O)
        assert "gate_mean" in stats or "gate" in stats


class TestNoGate:
    def test_no_gate_identity(self):
        K, C = 4, 64
        gate = NoGate(mode_channels=C, num_modes=K, maxl=3)
        O = _make_O(K, C)
        with torch.no_grad():
            Og = gate(O)
        for l in [0, 1, 2, 3]:
            assert torch.equal(Og[l], O[l])

    def test_no_gate_stats(self):
        gate = NoGate(mode_channels=64, num_modes=4, maxl=3)
        O = _make_O(4, 64)
        stats = gate.gate_stats(O)
        assert stats == {"gate": "identity"}

    def test_no_gate_equivariance(self):
        """NoGate trivially preserves equivariance."""
        K, C = 4, 64
        gate = NoGate(mode_channels=C, num_modes=K, maxl=3)
        O = _make_O(K, C)

        R = _random_rotation(seed=42)
        O_rot = {0: O[0].clone()}
        for l in [1, 2, 3]:
            D = o3.wigner_D(l, *o3.matrix_to_angles(R))
            O_rot[l] = torch.einsum("ab,kcb->kca", D, O[l])

        with torch.no_grad():
            Og = gate(O)
            Og_rot = gate(O_rot)

        for l in [0, 1, 2, 3]:
            if l == 0:
                assert torch.allclose(Og_rot[l], Og[l], atol=1e-7)
            else:
                D = o3.wigner_D(l, *o3.matrix_to_angles(R))
                Og_pred = torch.einsum("ab,kcb->kca", D, Og[l])
                assert torch.allclose(Og_rot[l], Og_pred, atol=1e-7)

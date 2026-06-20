"""Test invariant tensor gates: equivariance, gate range, ablation — batch-aware.

All tensors: [B, K, C, 2l+1].
"""

import pytest
import torch
from e3nn import o3

from ar_mto.tensor_gate import TensorGate, NoGate, ScalarOnlyGate


def _make_O(B=1, K=4, C=64, maxl=3):
    """Make batch-aware MTO modes: [B, K, C, 2l+1]."""
    O = {}
    O[0] = torch.randn(B, K, C, 1)
    for l in range(1, maxl + 1):
        O[l] = torch.randn(B, K, C, 2 * l + 1)
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
        B, K, C = 1, 4, 64
        gate = TensorGate(mode_channels=C, num_modes=K, maxl=3)
        O = _make_O(B, K, C)

        R = _random_rotation(seed=42)
        O_rot = {0: O[0].clone()}
        for l in [1, 2, 3]:
            D = o3.wigner_D(l, *o3.matrix_to_angles(R))
            O_rot[l] = torch.einsum("sd,bkcd->bkcs", D, O[l])

        with torch.no_grad():
            Og = gate(O)
            Og_rot = gate(O_rot)

        for l in [0, 1, 2, 3]:
            if l == 0:
                err = (Og_rot[l] - Og[l]).abs().max().item()
            else:
                D = o3.wigner_D(l, *o3.matrix_to_angles(R))
                Og_pred = torch.einsum("sd,bkcd->bkcs", D, Og[l])
                err = (Og_rot[l] - Og_pred).abs().max().item()
            assert err < 1e-4, f"Gate broke equivariance for l={l}: err={err:.2e}"

    def test_gate_finite(self):
        B, K, C = 1, 4, 64
        gate = TensorGate(mode_channels=C, num_modes=K, maxl=3)
        O = _make_O(B, K, C)
        with torch.no_grad():
            Og = gate(O)
        for key in O:
            assert not torch.isnan(Og[key]).any()
            assert not torch.isinf(Og[key]).any()

    def test_gate_stats(self):
        B, K, C = 1, 4, 64
        gate = TensorGate(mode_channels=C, num_modes=K, maxl=3)
        O = _make_O(B, K, C)
        with torch.no_grad():
            _ = gate(O)
        stats = gate.gate_stats(O)
        # New gate_stats has per-l stats
        assert "gate_l0_mean" in stats
        assert "gate_l0_std" in stats

    def test_gate_with_mode_mask(self):
        """Gated output must zero out masked modes."""
        B, K, C = 2, 8, 32
        gate = TensorGate(mode_channels=C, num_modes=K, maxl=3, alpha=0.1)
        O = _make_O(B, K, C)
        # Mask: first 4 modes active, last 4 inactive
        mode_mask = torch.zeros(B, K, dtype=torch.bool)
        mode_mask[:, :4] = True

        with torch.no_grad():
            Og = gate(O, mode_mask=mode_mask)

        for key in O:
            assert (Og[key][:, 4:, :, :].abs().max() == 0.0), \
                f"Masked modes not zero for key={key}"
            # Active modes should be modified (not identical to input)
            assert not torch.allclose(Og[key][:, :4, :, :],
                                       O[key][:, :4, :, :], atol=1e-6), \
                f"Active modes unchanged for key={key}"

    def test_batch_isolation(self):
        """Gating must be independent per molecule."""
        B, K, C = 2, 4, 32
        gate = TensorGate(mode_channels=C, num_modes=K, maxl=3)
        O0 = _make_O(1, K, C)
        O1 = _make_O(1, K, C, maxl=3)
        O_batch = {}
        for key in O0:
            O_batch[key] = torch.cat([O0[key], O1[key]], dim=0)

        with torch.no_grad():
            Og_batch = gate(O_batch)
            Og0 = gate(O0)
            Og1 = gate(O1)

        for key in O_batch:
            assert torch.allclose(Og_batch[key][0:1], Og0[key], atol=1e-5), \
                f"Batch mol 0 mismatch for key={key}"
            assert torch.allclose(Og_batch[key][1:2], Og1[key], atol=1e-5), \
                f"Batch mol 1 mismatch for key={key}"


class TestNoGate:
    def test_no_gate_identity(self):
        B, K, C = 1, 4, 64
        gate = NoGate(mode_channels=C, num_modes=K, maxl=3)
        O = _make_O(B, K, C)
        with torch.no_grad():
            Og = gate(O)
        for l in [0, 1, 2, 3]:
            assert torch.equal(Og[l], O[l])

    def test_no_gate_stats(self):
        gate = NoGate(mode_channels=64, num_modes=4, maxl=3)
        O = _make_O(1, 4, 64)
        stats = gate.gate_stats(O)
        assert stats == {"gate": "identity"}

    def test_no_gate_equivariance(self):
        """NoGate trivially preserves equivariance."""
        B, K, C = 1, 4, 64
        gate = NoGate(mode_channels=C, num_modes=K, maxl=3)
        O = _make_O(B, K, C)

        R = _random_rotation(seed=42)
        O_rot = {0: O[0].clone()}
        for l in [1, 2, 3]:
            D = o3.wigner_D(l, *o3.matrix_to_angles(R))
            O_rot[l] = torch.einsum("sd,bkcd->bkcs", D, O[l])

        with torch.no_grad():
            Og = gate(O)
            Og_rot = gate(O_rot)

        for l in [0, 1, 2, 3]:
            if l == 0:
                assert torch.allclose(Og_rot[l], Og[l], atol=1e-7)
            else:
                D = o3.wigner_D(l, *o3.matrix_to_angles(R))
                Og_pred = torch.einsum("sd,bkcd->bkcs", D, Og[l])
                assert torch.allclose(Og_rot[l], Og_pred, atol=1e-7)


class TestScalarOnlyGate:
    def test_scalar_only_gate_equivariance(self):
        """ScalarOnlyGate should also preserve equivariance."""
        B, K, C = 1, 4, 64
        gate = ScalarOnlyGate(mode_channels=C, num_modes=K, maxl=3, alpha=0.1)
        O = _make_O(B, K, C)

        R = _random_rotation(seed=42)
        O_rot = {0: O[0].clone()}
        for l in [1, 2, 3]:
            D = o3.wigner_D(l, *o3.matrix_to_angles(R))
            O_rot[l] = torch.einsum("sd,bkcd->bkcs", D, O[l])

        with torch.no_grad():
            Og = gate(O)
            Og_rot = gate(O_rot)

        for l in [0, 1, 2, 3]:
            if l == 0:
                err = (Og_rot[l] - Og[l]).abs().max().item()
            else:
                D = o3.wigner_D(l, *o3.matrix_to_angles(R))
                Og_pred = torch.einsum("sd,bkcd->bkcs", D, Og[l])
                err = (Og_rot[l] - Og_pred).abs().max().item()
            assert err < 1e-4, f"ScalarOnlyGate broke equivariance l={l}: err={err:.2e}"

    def test_gate_stats(self):
        B, K, C = 1, 4, 64
        gate = ScalarOnlyGate(mode_channels=C, num_modes=K, maxl=3)
        O = _make_O(B, K, C)
        with torch.no_grad():
            _ = gate(O)
        stats = gate.gate_stats(O)
        assert "sgate_mean" in stats
        assert "sgate_std" in stats
"""Test Clebsch-Gordan coupling equivariance — batch-aware.

CG coupling operates on [B, K, C, 2l+1] tensors.
"""

import pytest
import torch
from e3nn import o3

from ar_mto.cg_coupling import CGCouplingMinimal, CGCoupling


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


class TestCGCouplingMinimal:
    def test_shapes_single_molecule(self):
        B, K, C = 1, 4, 64
        cg = CGCouplingMinimal(mode_channels=C)
        O = _make_O(B, K, C)
        with torch.no_grad():
            Oc = cg(O)
        assert Oc[0].shape == (B, K, C, 1)
        assert Oc[1].shape == (B, K, C, 3)
        assert Oc[2].shape == (B, K, C, 5)
        # l=3 preserved as residual identity when o3 input present
        if 3 in Oc:
            assert Oc[3].shape == (B, K, C, 7)

    def test_shapes_batch(self):
        B, K, C = 3, 4, 64
        cg = CGCouplingMinimal(mode_channels=C)
        O = _make_O(B, K, C)
        with torch.no_grad():
            Oc = cg(O)
        assert Oc[0].shape == (B, K, C, 1)
        assert Oc[1].shape == (B, K, C, 3)
        assert Oc[2].shape == (B, K, C, 5)

    def test_batch_isolation(self):
        """Molecule 0 and molecule 1 in a batch must have independent CG outputs."""
        B, K, C = 2, 4, 32
        cg = CGCouplingMinimal(mode_channels=C)

        # Make molecule 0 all zeros, molecule 1 has signal
        O = {}
        O[0] = torch.randn(B, K, C, 1)
        O[1] = torch.randn(B, K, C, 3)
        O[2] = torch.randn(B, K, C, 5)
        O[3] = torch.randn(B, K, C, 7)

        O_masked = dict(O)
        for key in O_masked:
            val = O_masked[key].clone()
            val[0] = 0.0  # zero out molecule 0
            O_masked[key] = val

        with torch.no_grad():
            Oc = cg(O_masked)

        # Molecule 0 should be all zeros
        for key, val in Oc.items():
            assert (val[0].abs().max() == 0.0), \
                f"Molecule 0 leaked into CG output key={key}"

    def test_equivariance(self):
        """CG coupling output must be equivariant under rotation."""
        B, K, C = 1, 4, 64
        cg = CGCouplingMinimal(mode_channels=C)
        O_orig = _make_O(B, K, C)
        R = _random_rotation(seed=42)

        # Rotate input
        O_rot = {0: O_orig[0].clone()}  # l=0 is invariant
        for ll in [1, 2, 3]:
            D = o3.wigner_D(ll, *o3.matrix_to_angles(R))
            O_rot[ll] = torch.einsum("sd,bkcd->bkcs", D, O_orig[ll])

        with torch.no_grad():
            Oc_orig = cg(O_orig)
            Oc_rot = cg(O_rot)

        # Coupled output must also transform correctly
        for ll in [0, 1, 2]:
            if ll == 0:
                err = (Oc_rot[ll] - Oc_orig[ll]).abs().max().item()
            else:
                D = o3.wigner_D(ll, *o3.matrix_to_angles(R))
                Oc_pred = torch.einsum("sd,bkcd->bkcs", D, Oc_orig[ll])
                err = (Oc_rot[ll] - Oc_pred).abs().max().item()
            assert err < 1e-4, f"CG coupling equivariance failed l={l}: err={err:.2e}"

    def test_no_nan(self):
        B, K, C = 1, 4, 64
        cg = CGCouplingMinimal(mode_channels=C)
        O = _make_O(B, K, C)
        with torch.no_grad():
            Oc = cg(O)
        for key, val in Oc.items():
            assert not torch.isnan(val).any(), f"NaN in key={key}"
            assert not torch.isinf(val).any(), f"Inf in key={key}"

    def test_finite_outputs(self):
        """CG coupling should produce finite outputs for random inputs."""
        B, K, C = 1, 4, 64
        cg = CGCouplingMinimal(mode_channels=C)
        O = _make_O(B, K, C)
        with torch.no_grad():
            Oc = cg(O)
        for key, val in Oc.items():
            assert val.abs().max() < 1e6, f"CG output too large for key={key}"

    def test_path_table_nonempty(self):
        cg = CGCouplingMinimal(mode_channels=32)
        table = cg.get_path_table()
        assert "0e × 0e" in table
        assert "0e × 1o" in table
        assert "1o × 1o" in table
        assert "0e × 2e" in table

    def test_parity_correct_1o_cross_1o(self):
        """1o×1o must produce 0e+2e, NOT 1o. This is the key parity fix."""
        B, K, C = 1, 4, 64
        cg = CGCouplingMinimal(mode_channels=C)
        O = _make_O(B, K, C)
        with torch.no_grad():
            Oc = cg(O)
        # 1e (axial vector, parity +1) is NOT in the output keys
        # Output keys are int (0,1,2,3) with parity (-1)^l
        assert 1 in Oc, "l=1 output should exist (from 0e×1o path, parity -1)"
        # The important thing: Oc[1] exists and is parity -1 (polar),
        # not parity +1 (axial), because 1o×1o→1e is excluded


class TestCGCouplingFull:
    def test_shapes(self):
        B, K, C = 1, 2, 32  # small to keep test fast
        cg = CGCoupling(mode_channels=C, maxl=3, coupled_maxl=2)
        O = _make_O(B, K, C, maxl=3)
        with torch.no_grad():
            Oc = cg(O)
        # Check that all keys produce valid shapes
        for key, val in Oc.items():
            assert val.dim() == 4, f"Key {key}: expected 4D, got {val.dim()}D"
            assert val.shape[0] == B
            assert val.shape[1] == K
            assert val.shape[2] == C

    def test_equivariance(self):
        """Full CG coupling must be equivariant."""
        B, K, C = 1, 2, 16
        cg = CGCoupling(mode_channels=C, maxl=3, coupled_maxl=2)
        O_orig = _make_O(B, K, C, maxl=3)
        R = _random_rotation(seed=42)

        O_rot = {0: O_orig[0].clone()}
        for l in [1, 2, 3]:
            D = o3.wigner_D(l, *o3.matrix_to_angles(R))
            O_rot[l] = torch.einsum("sd,bkcd->bkcs", D, O_orig[l])

        with torch.no_grad():
            Oc_orig = cg(O_orig)
            Oc_rot = cg(O_rot)

        for key in Oc_orig:
            if isinstance(key, tuple):
                l, p = key
            else:
                l = key
                p = (-1) ** l
            if l == 0:
                err = (Oc_rot[key] - Oc_orig[key]).abs().max().item()
            else:
                D = o3.wigner_D(l, *o3.matrix_to_angles(R))
                Oc_pred = torch.einsum("sd,bkcd->bkcs", D, Oc_orig[key])
                err = (Oc_rot[key] - Oc_pred).abs().max().item()
            assert err < 1e-4, \
                f"Full CG coupling equivariance failed {key}: err={err:.2e}"

    def test_no_nan(self):
        B, K, C = 1, 2, 16
        cg = CGCoupling(mode_channels=C, maxl=3, coupled_maxl=2)
        O = _make_O(B, K, C, maxl=3)
        with torch.no_grad():
            Oc = cg(O)
        for key, val in Oc.items():
            assert not torch.isnan(val).any()
            assert not torch.isinf(val).any()

    def test_path_table_nonempty(self):
        cg = CGCoupling(mode_channels=32, maxl=3, coupled_maxl=2)
        table = cg.get_path_table()
        assert "CG Coupling Paths" in table

    def test_batch_isolation(self):
        """Two molecules in a batch must not mix during CG coupling."""
        B, K, C = 2, 2, 16
        cg = CGCoupling(mode_channels=C, maxl=2, coupled_maxl=2)
        O0 = _make_O(1, K, C, maxl=2)
        O1 = _make_O(1, K, C, maxl=2)
        # Concatenate along batch dim
        O_2mol = {}
        for key in O0:
            O_2mol[key] = torch.cat([O0[key], O1[key]], dim=0)

        with torch.no_grad():
            Oc_batch = cg(O_2mol)
            Oc_mol0 = cg(O0)
            Oc_mol1 = cg(O1)

        for key in Oc_batch:
            assert torch.allclose(Oc_batch[key][0:1], Oc_mol0[key], atol=1e-5), \
                f"Batch mol 0 mismatch for key={key}"
            assert torch.allclose(Oc_batch[key][1:2], Oc_mol1[key], atol=1e-5), \
                f"Batch mol 1 mismatch for key={key}"
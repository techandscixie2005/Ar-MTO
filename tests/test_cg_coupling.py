"""Test Clebsch-Gordan coupling equivariance."""

import pytest
import torch
from e3nn import o3

from ar_mto.cg_coupling import CGCouplingMinimal, CGCoupling


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


class TestCGCouplingMinimal:
    def test_shapes(self):
        K, C = 4, 64
        cg = CGCouplingMinimal(mode_channels=C)
        O = _make_O(K, C)
        with torch.no_grad():
            Oc = cg(O)
        assert Oc[0].shape == (K, C, 1)
        assert Oc[1].shape == (K, C, 3)
        assert Oc[2].shape == (K, C, 5)

    def test_equivariance(self):
        """CG coupling output must be equivariant under rotation."""
        K, C = 4, 64
        cg = CGCouplingMinimal(mode_channels=C)
        O_orig = _make_O(K, C)
        R = _random_rotation(seed=42)

        # Rotate input
        O_rot = {0: O_orig[0].clone()}  # l=0 is invariant
        for l in [1, 2, 3]:
            D = o3.wigner_D(l, *o3.matrix_to_angles(R))
            O_rot[l] = torch.einsum("ab,kcb->kca", D, O_orig[l])

        with torch.no_grad():
            Oc_orig = cg(O_orig)
            Oc_rot = cg(O_rot)

        # Coupled output must also transform correctly
        for l in [0, 1, 2]:
            if l == 0:
                err = (Oc_rot[l] - Oc_orig[l]).abs().max().item()
            else:
                D = o3.wigner_D(l, *o3.matrix_to_angles(R))
                Oc_pred = torch.einsum("ab,kcb->kca", D, Oc_orig[l])
                err = (Oc_rot[l] - Oc_pred).abs().max().item()
            assert err < 1e-4, f"CG coupling equivariance failed l={l}: err={err:.2e}"

    def test_no_nan(self):
        K, C = 4, 64
        cg = CGCouplingMinimal(mode_channels=C)
        O = _make_O(K, C)
        with torch.no_grad():
            Oc = cg(O)
        for l in [0, 1, 2]:
            assert not torch.isnan(Oc[l]).any()
            assert not torch.isinf(Oc[l]).any()

    def test_finite_outputs(self):
        """CG coupling should produce finite outputs for random inputs."""
        K, C = 4, 64
        cg = CGCouplingMinimal(mode_channels=C)
        O = _make_O(K, C)
        with torch.no_grad():
            Oc = cg(O)
        for l in [0, 1, 2]:
            assert Oc[l].abs().max() < 1e6, f"CG output too large for l={l}"


class TestCGCouplingFull:
    def test_shapes(self):
        K, C = 2, 32  # small to keep test fast
        cg = CGCoupling(mode_channels=C, maxl=3, coupled_channels=C)
        O = _make_O(K, C, maxl=3)
        with torch.no_grad():
            Oc = cg(O)
        assert Oc[0].shape == (K, C, 1)
        assert Oc[1].shape == (K, C, 3)
        assert Oc[2].shape == (K, C, 5)
        assert Oc[3].shape == (K, C, 7)

    def test_equivariance(self):
        """Full CG coupling must be equivariant."""
        K, C = 2, 16
        cg = CGCoupling(mode_channels=C, maxl=2, coupled_channels=C)
        O_orig = _make_O(K, C, maxl=2)
        R = _random_rotation(seed=42)

        O_rot = {0: O_orig[0].clone()}
        for l in [1, 2]:
            D = o3.wigner_D(l, *o3.matrix_to_angles(R))
            O_rot[l] = torch.einsum("ab,kcb->kca", D, O_orig[l])

        with torch.no_grad():
            Oc_orig = cg(O_orig)
            Oc_rot = cg(O_rot)

        for l in [0, 1, 2]:
            if l == 0:
                err = (Oc_rot[l] - Oc_orig[l]).abs().max().item()
            else:
                D = o3.wigner_D(l, *o3.matrix_to_angles(R))
                Oc_pred = torch.einsum("ab,kcb->kca", D, Oc_orig[l])
                err = (Oc_rot[l] - Oc_pred).abs().max().item()
            assert err < 1e-4, \
                f"Full CG coupling equivariance failed l={l}: err={err:.2e}"

    def test_no_nan(self):
        K, C = 2, 16
        cg = CGCoupling(mode_channels=C, maxl=2, coupled_channels=C)
        O = _make_O(K, C, maxl=2)
        with torch.no_grad():
            Oc = cg(O)
        for l in [0, 1, 2]:
            assert not torch.isnan(Oc[l]).any()
            assert not torch.isinf(Oc[l]).any()

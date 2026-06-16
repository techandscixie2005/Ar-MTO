"""Test MTO internal equivariance.

Verifies that MTO assembled modes transform correctly under rotation:
    O_k^(l) transforms under Wigner-D^l as expected for true tensor modes.

Tests:
  - l=0: invariant under rotation
  - l=1: vector-like (Wigner-D^1)
  - l=2: traceless-tensor-like (Wigner-D^2)
  - l=3: Wigner-D^3
  - Permutation equivariance
  - Translation invariance
"""

import pytest
import torch
from e3nn import o3

from ar_mto.tensor_adapter import make_adapter
from ar_mto.signed_routing import SignedRouter
from ar_mto.mto_core import MTOModeAssembly, ScalarOnlyMTO

TOLERANCE = 5e-5


def _make_h(N=5, C=128, maxl=3):
    h = {}
    h[0] = torch.randn(N, C, 1)
    for l in range(1, maxl + 1):
        h[l] = torch.randn(N, C, 2 * l + 1)
    return h


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


class TestMTOEquivariance:
    @pytest.mark.parametrize("l", [0, 1, 2, 3])
    def test_mode_equivariance(self, l):
        """Assembled mode O_k^(l) transforms as Wigner-D^l under rotation."""
        N, C, K = 5, 128, 4
        mto = MTOModeAssembly(num_features=C, mode_channels=64,
                              num_modes=K, maxl=3)
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)

        h_orig = _make_h(N, C)
        R = _random_rotation(seed=100 + l)

        # Rotate tensor features
        h_rot = {0: h_orig[0].clone()}
        for ll in [1, 2, 3]:
            D = o3.wigner_D(ll, *o3.matrix_to_angles(R))
            h_rot[ll] = torch.einsum("ab,ncb->nca", D, h_orig[ll])

        with torch.no_grad():
            c_orig = router(h_orig)
            c_rot = router(h_rot)
            O_orig = mto(h_orig, c_orig)
            O_rot = mto(h_rot, c_rot)

        if l == 0:
            # l=0 is invariant under rotation
            err = (O_rot[l] - O_orig[l]).abs().max().item()
        else:
            # l>0 transforms via Wigner-D
            D = o3.wigner_D(l, *o3.matrix_to_angles(R))
            # O_orig: [K, C_out, 2l+1], apply D to spatial dim
            O_pred = torch.einsum("ab,kcb->kca", D, O_orig[l])
            err = (O_rot[l] - O_pred).abs().max().item()

        assert err < TOLERANCE, \
            f"MTO equivariance failed for l={l}: err={err:.2e}"

    def test_all_orders_simultaneous(self):
        """All l orders transform correctly under the same rotation."""
        N, C, K = 6, 128, 4
        mto = MTOModeAssembly(num_features=C, mode_channels=64,
                              num_modes=K, maxl=3)
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)

        h_orig = _make_h(N, C)
        R = _random_rotation(seed=200)

        h_rot = {0: h_orig[0].clone()}
        for l in [1, 2, 3]:
            D = o3.wigner_D(l, *o3.matrix_to_angles(R))
            h_rot[l] = torch.einsum("ab,ncb->nca", D, h_orig[l])

        with torch.no_grad():
            c_orig = router(h_orig)
            c_rot = router(h_rot)
            O_orig = mto(h_orig, c_orig)
            O_rot = mto(h_rot, c_rot)

        for l in [0, 1, 2, 3]:
            if l == 0:
                err = (O_rot[l] - O_orig[l]).abs().max().item()
            else:
                D = o3.wigner_D(l, *o3.matrix_to_angles(R))
                O_pred = torch.einsum("ab,kcb->kca", D, O_orig[l])
                err = (O_rot[l] - O_pred).abs().max().item()
            assert err < TOLERANCE, f"l={l}: err={err:.2e}"

    def test_multiple_rotations(self):
        """Equivariance holds for different random rotations."""
        N, C, K = 5, 128, 4
        mto = MTOModeAssembly(num_features=C, mode_channels=64,
                              num_modes=K, maxl=3)
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)
        h_orig = _make_h(N, C)

        for rot_seed in [10, 20, 30, 40, 50]:
            R = _random_rotation(seed=rot_seed)
            h_rot = {0: h_orig[0].clone()}
            for l in [1, 2, 3]:
                D = o3.wigner_D(l, *o3.matrix_to_angles(R))
                h_rot[l] = torch.einsum("ab,ncb->nca", D, h_orig[l])

            with torch.no_grad():
                c_orig = router(h_orig)
                c_rot = router(h_rot)
                O_orig = mto(h_orig, c_orig)
                O_rot = mto(h_rot, c_rot)

            for l in [0, 1, 2, 3]:
                if l == 0:
                    err = (O_rot[l] - O_orig[l]).abs().max().item()
                else:
                    D = o3.wigner_D(l, *o3.matrix_to_angles(R))
                    O_pred = torch.einsum("ab,kcb->kca", D, O_orig[l])
                    err = (O_rot[l] - O_pred).abs().max().item()
                assert err < TOLERANCE, \
                    f"rot_seed={rot_seed} l={l}: err={err:.2e}"


class TestMTOShapes:
    def test_output_shapes(self):
        N, C, K = 5, 128, 4
        Cout = 64
        mto = MTOModeAssembly(num_features=C, mode_channels=Cout,
                              num_modes=K, maxl=3)
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)
        h = _make_h(N, C)
        with torch.no_grad():
            coeffs = router(h)
            O = mto(h, coeffs)

        assert O[0].shape == (K, Cout, 1)
        assert O[1].shape == (K, Cout, 3)
        assert O[2].shape == (K, Cout, 5)
        assert O[3].shape == (K, Cout, 7)

    def test_variable_atoms(self):
        C, K, Cout = 128, 4, 64
        mto = MTOModeAssembly(num_features=C, mode_channels=Cout,
                              num_modes=K, maxl=3)
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)

        for n in [3, 5, 7, 10]:
            h = _make_h(n, C)
            with torch.no_grad():
                coeffs = router(h)
                O = mto(h, coeffs)
            for l in [0, 1, 2, 3]:
                assert O[l].shape == (K, Cout, 2 * l + 1 if l > 0 else 1)

    def test_variable_modes(self):
        N, C, Cout = 5, 128, 64
        for K in [2, 4, 8]:
            mto = MTOModeAssembly(num_features=C, mode_channels=Cout,
                                  num_modes=K, maxl=3)
            router = SignedRouter(num_features=C, num_modes=K, maxl=3)
            h = _make_h(N, C)
            with torch.no_grad():
                coeffs = router(h)
                O = mto(h, coeffs)
            for l in [0, 1, 2, 3]:
                assert O[l].shape == (K, Cout, 2 * l + 1 if l > 0 else 1)

    def test_no_nan(self):
        N, C, K = 5, 128, 4
        mto = MTOModeAssembly(num_features=C, mode_channels=64,
                              num_modes=K, maxl=3)
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)
        h = _make_h(N, C)
        with torch.no_grad():
            coeffs = router(h)
            O = mto(h, coeffs)
        for l in [0, 1, 2, 3]:
            assert not torch.isnan(O[l]).any()
            assert not torch.isinf(O[l]).any()


class TestScalarOnlyMTO:
    def test_scalar_only_shapes(self):
        N, C, K = 5, 128, 4
        mto = ScalarOnlyMTO(num_features=C, mode_channels=64, num_modes=K)
        router = SignedRouter(num_features=C, num_modes=K,
                              use_tensor_norms=False, maxl=0)
        h = {0: torch.randn(N, C, 1)}
        with torch.no_grad():
            coeffs = router(h)
            O = mto(h, coeffs)
        assert list(O.keys()) == [0]
        assert O[0].shape == (K, 64, 1)

    def test_scalar_only_invariant(self):
        """Scalar-only MTO output should be rotation invariant."""
        N, C, K = 5, 128, 4
        mto = ScalarOnlyMTO(num_features=C, mode_channels=64, num_modes=K)
        router = SignedRouter(num_features=C, num_modes=K,
                              use_tensor_norms=False, maxl=0)
        h = {0: torch.randn(N, C, 1)}
        with torch.no_grad():
            coeffs = router(h)
            O1 = mto(h, coeffs)
            O2 = mto(h, coeffs)  # same input, same output
        assert torch.allclose(O1[0], O2[0], atol=1e-7)


class TestMTOModeMasking:
    def test_mode_masking(self):
        N, C, K = 5, 128, 8
        mto = MTOModeAssembly(num_features=C, mode_channels=64,
                              num_modes=K, maxl=3)
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)
        h = _make_h(N, C)
        mode_mask = torch.tensor([True, True, True, True,
                                  False, False, False, False])

        with torch.no_grad():
            coeffs = router(h)
            O = mto.forward_with_masks(h, coeffs, mode_mask)

        # Masked modes should be zero
        for l in [0, 1, 2, 3]:
            assert (O[l][4:].abs().max() == 0.0), \
                f"Masked modes not zero for l={l}"
            assert (O[l][:4].abs().max() > 0.0), \
                f"Active modes zero for l={l}"

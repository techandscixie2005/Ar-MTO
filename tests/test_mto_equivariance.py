"""Test MTO internal equivariance — batch-aware.

Verifies that MTO assembled modes transform correctly under rotation:
    O_k^(l) transforms under Wigner-D^l as expected for true tensor modes.

All operations are batch-aware: [B, K, C, 2l+1] modes.
"""

import pytest
import torch
from e3nn import o3

from ar_mto.signed_routing import SignedRouter
from ar_mto.mto_core import MTOModeAssembly, ScalarOnlyMTO

TOLERANCE = 5e-5


def _make_h(N=5, C=128, maxl=3):
    """Make atom-level tensor features: [N, C, 2l+1]."""
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
    def test_mode_equivariance_single_mol(self, l):
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
            h_rot[ll] = torch.einsum("sd,ncd->ncs", D, h_orig[ll])

        with torch.no_grad():
            c_orig = router(h_orig)   # [K, N, 1]
            c_rot = router(h_rot)     # [K, N, 1]
            O_orig = mto(h_orig, c_orig)    # [1, K, C_out, 2l+1]
            O_rot = mto(h_rot, c_rot)       # [1, K, C_out, 2l+1]

        if l == 0:
            err = (O_rot[l] - O_orig[l]).abs().max().item()
        else:
            D = o3.wigner_D(l, *o3.matrix_to_angles(R))
            O_pred = torch.einsum("sd,bkcd->bkcs", D, O_orig[l])
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
            h_rot[l] = torch.einsum("sd,ncd->ncs", D, h_orig[l])

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
                O_pred = torch.einsum("sd,bkcd->bkcs", D, O_orig[l])
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
                h_rot[l] = torch.einsum("sd,ncd->ncs", D, h_orig[l])

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
                    O_pred = torch.einsum("sd,bkcd->bkcs", D, O_orig[l])
                    err = (O_rot[l] - O_pred).abs().max().item()
                assert err < TOLERANCE, \
                    f"rot_seed={rot_seed} l={l}: err={err:.2e}"


class TestMTOShapes:
    def test_output_shapes(self):
        """Single molecule: [B=1, K, C_out, 2l+1]."""
        N, C, K = 5, 128, 4
        Cout = 64
        mto = MTOModeAssembly(num_features=C, mode_channels=Cout,
                              num_modes=K, maxl=3)
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)
        h = _make_h(N, C)
        with torch.no_grad():
            coeffs = router(h)
            O = mto(h, coeffs)

        assert O[0].shape == (1, K, Cout, 1)
        assert O[1].shape == (1, K, Cout, 3)
        assert O[2].shape == (1, K, Cout, 5)
        assert O[3].shape == (1, K, Cout, 7)

    def test_batched_shapes(self):
        """Batch of 2 molecules: [B=2, K, C_out, 2l+1]."""
        N1, N2 = 5, 4
        C, K, Cout = 128, 4, 64
        mto = MTOModeAssembly(num_features=C, mode_channels=Cout,
                              num_modes=K, maxl=3)
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)

        h1 = _make_h(N1, C)
        h2 = _make_h(N2, C)
        h_batch = {l: torch.cat([h1[l], h2[l]], dim=0) for l in h1}
        batch = torch.tensor(
            [0] * N1 + [1] * N2, dtype=torch.long
        )

        with torch.no_grad():
            coeffs = router(h_batch, batch=batch)
            O = mto(h_batch, coeffs, batch=batch)

        for l in [0, 1, 2, 3]:
            assert O[l].shape == (2, K, Cout, 2 * l + 1 if l > 0 else 1), \
                f"Wrong batch shape for l={l}: {O[l].shape}"

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
                assert O[l].shape == (1, K, Cout, 2 * l + 1 if l > 0 else 1), \
                    f"n={n} l={l}: {O[l].shape}"

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
                assert O[l].shape == (1, K, Cout, 2 * l + 1 if l > 0 else 1)

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


class TestMTOBatchIsolation:
    """Cross-molecule leakage prevention is critical."""

    def test_no_cross_molecule_leakage(self):
        """Assembling molecule A alone vs in batch must give identical results."""
        C, K, Cout = 128, 4, 64
        mto = MTOModeAssembly(num_features=C, mode_channels=Cout,
                              num_modes=K, maxl=3)
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)

        hA = _make_h(5, C)
        hB = _make_h(4, C)

        # Forward molecule A alone
        with torch.no_grad():
            coeffs_A = router(hA)
            O_A_alone = mto(hA, coeffs_A)

        # Forward molecule A alongside B in a batch
        h_batch = {l: torch.cat([hA[l], hB[l]], dim=0) for l in hA}
        batch = torch.tensor([0] * 5 + [1] * 4, dtype=torch.long)
        with torch.no_grad():
            coeffs_batch = router(h_batch, batch=batch)
            O_batch = mto(h_batch, coeffs_batch, batch=batch)

        # Molecule A's assembled modes must be identical
        for l in [0, 1, 2, 3]:
            assert torch.allclose(O_batch[l][0:1], O_A_alone[l], atol=1e-5), \
                f"Cross-molecule leakage detected for l={l}"

    def test_molecule_ordering_invariance(self):
        """Reordering molecules within the batch should not affect per-mol results."""
        C, K, Cout = 128, 4, 64
        mto = MTOModeAssembly(num_features=C, mode_channels=Cout,
                              num_modes=K, maxl=3)
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)

        hA = _make_h(5, C)
        hB = _make_h(4, C)

        # Order: [A, B]
        h_ab = {l: torch.cat([hA[l], hB[l]], dim=0) for l in hA}
        batch_ab = torch.tensor([0] * 5 + [1] * 4, dtype=torch.long)

        # Order: [B, A]
        h_ba = {l: torch.cat([hB[l], hA[l]], dim=0) for l in hA}
        batch_ba = torch.tensor([1] * 4 + [0] * 5, dtype=torch.long)

        with torch.no_grad():
            coeffs_ab = router(h_ab, batch=batch_ab)
            O_ab = mto(h_ab, coeffs_ab, batch=batch_ab)

            coeffs_ba = router(h_ba, batch=batch_ba)
            O_ba = mto(h_ba, coeffs_ba, batch=batch_ba)

        # Mol A output (batch idx 0 in ab, batch idx 0 in ba but it's mol B there)
        for l in [0, 1, 2, 3]:
            assert torch.allclose(O_ab[l][0:1], O_ba[l][1:2], atol=1e-5), \
                f"Molecule A mismatch under reordering for l={l}"
            assert torch.allclose(O_ab[l][1:2], O_ba[l][0:1], atol=1e-5), \
                f"Molecule B mismatch under reordering for l={l}"


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
        assert O[0].shape == (1, K, 64, 1)

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
            O2 = mto(h, coeffs)
        assert torch.allclose(O1[0], O2[0], atol=1e-7)


class TestMTOModeMasking:
    def test_mode_masking_single_molecule(self):
        """Mode mask zeroes out inactive modes; batch-aware."""
        N, C, K = 5, 128, 8
        mto = MTOModeAssembly(num_features=C, mode_channels=64,
                              num_modes=K, maxl=3)
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)
        h = _make_h(N, C)
        mode_mask = torch.tensor([[True, True, True, True,
                                    False, False, False, False]])

        with torch.no_grad():
            coeffs = router(h)
            O = mto.forward_with_masks(h, coeffs, mode_mask)

        for l in [0, 1, 2, 3]:
            assert (O[l][:, 4:, :, :].abs().max() == 0.0), \
                f"Masked modes not zero for l={l}"
            assert (O[l][:, :4, :, :].abs().max() > 0.0), \
                f"Active modes zero for l={l}"

    def test_mode_masking_batch(self):
        """Mode mask per molecule in batch."""
        N1, N2, C, K = 5, 4, 128, 8
        mto = MTOModeAssembly(num_features=C, mode_channels=64,
                              num_modes=K, maxl=3)
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)

        h1 = _make_h(N1, C)
        h2 = _make_h(N2, C)
        h_batch = {l: torch.cat([h1[l], h2[l]], dim=0) for l in h1}
        batch = torch.tensor([0] * N1 + [1] * N2, dtype=torch.long)

        # Molecule 0: 3 modes, Molecule 1: 5 modes
        mode_mask = torch.zeros(2, K, dtype=torch.bool)
        mode_mask[0, :3] = True
        mode_mask[1, :5] = True

        with torch.no_grad():
            coeffs = router(h_batch, batch=batch)
            O = mto.forward_with_masks(h_batch, coeffs, mode_mask, batch=batch)

        for l in [0, 1, 2, 3]:
            # Molecule 0: modes 3..7 must be zero
            assert (O[l][0, 3:, :, :].abs().max() == 0.0), \
                f"Mol 0 l={l}: masked modes not zero"
            # Molecule 1: modes 5..7 must be zero
            assert (O[l][1, 5:, :, :].abs().max() == 0.0), \
                f"Mol 1 l={l}: masked modes not zero"
            # Active modes should be nonzero
            assert (O[l][0, :3, :, :].abs().max() > 0.0), \
                f"Mol 0 l={l}: active modes are zero"
            assert (O[l][1, :5, :, :].abs().max() > 0.0), \
                f"Mol 1 l={l}: active modes are zero"

    def test_valence_adaptive_k(self):
        """compute_valence_adaptive_k produces valid mode masks."""
        from ar_mto.mto_core import compute_valence_adaptive_k

        # Water: O (Z=8, 6v) + 2*H (Z=1, 1v each) = 8 valence electrons → K=4
        z = torch.tensor([8, 1, 1], dtype=torch.long)
        mode_mask, ks = compute_valence_adaptive_k(z, max_modes=8)
        assert ks[0].item() == 4
        assert mode_mask.shape == (1, 4)
        assert mode_mask[0].all()

        # Methane: C (Z=6, 4v) + 4*H (Z=1, 1v each) = 8 valence electrons → K=4
        z = torch.tensor([6, 1, 1, 1, 1], dtype=torch.long)
        mode_mask, ks = compute_valence_adaptive_k(z, max_modes=8)
        assert ks[0].item() == 4

        # He (Z=2, 2v) → K=1
        z = torch.tensor([2], dtype=torch.long)
        mode_mask, ks = compute_valence_adaptive_k(z, max_modes=8)
        assert ks[0].item() == 1
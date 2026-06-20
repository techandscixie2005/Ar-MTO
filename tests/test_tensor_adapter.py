"""Test TensorAdapter with real DetaNet features: split/reconstruct exactness,
Wigner-D equivariance for h1/h2/h3, parity, translation, and permutation.

All tests use the TensorAdapter class to consume real DetaNet (S,T) outputs,
verifying the adapter preserves exact reconstruction and tensor transformation
properties end-to-end.
"""

import pytest
import torch

torch.serialization.add_safe_globals([slice])

from e3nn import o3

from ar_mto.tensor_adapter import TensorAdapter, make_adapter
from ar_mto.detanet_bridge import (
    make_latent_detanet,
    run_latent_forward,
)

TOLERANCE = 5e-5
DETANET_CACHE: dict = {}  # module-scope cache to avoid re-creating model per test


def _get_model(num_block=2):
    key = num_block
    if key not in DETANET_CACHE:
        DETANET_CACHE[key] = make_latent_detanet(num_block=num_block, device="cpu")
    return DETANET_CACHE[key]


def _make_molecule(num_atoms=5, seed=42):
    gen = torch.Generator()
    gen.manual_seed(seed)
    z = torch.randint(1, 10, (num_atoms,), generator=gen)
    radius = 1.2
    pos = torch.randn(num_atoms, 3, generator=gen)
    norms = torch.norm(pos, dim=-1, keepdim=True)
    scales = torch.rand(num_atoms, 1, generator=gen) ** (1.0 / 3.0)
    pos = pos / (norms + 1e-8) * scales * radius
    return z, pos


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


def _wigner_D(l, R):
    R64 = R.to(dtype=torch.float64)
    alpha, beta, gamma = o3.matrix_to_angles(R64)
    D = o3.wigner_D(l, alpha, beta, gamma)
    return D.to(dtype=R.dtype)


class TestAdapterSplitReconstructDetaNet:
    """Split/reconstruct exactness using real DetaNet features through TensorAdapter."""

    @pytest.mark.parametrize("n_atoms", [3, 4, 5, 6, 8, 10])
    def test_exact_reconstruction(self, n_atoms):
        model = _get_model()
        z, pos = _make_molecule(n_atoms, seed=n_atoms * 10)
        adapter = make_adapter()

        with torch.no_grad():
            S, T = run_latent_forward(model, z=z, pos=pos)

        h = adapter(S, T)
        S_recon, T_recon = adapter.reconstruct(h)

        assert torch.equal(S, S_recon), "S reconstruction must be exact"
        assert torch.equal(T, T_recon), "T reconstruction must be exact"

    def test_reconstruction_zero_error(self):
        """Reconstruction is pure slicing/reshaping — must be exact (err==0)."""
        model = _get_model()
        z, pos = _make_molecule(5)
        adapter = make_adapter()

        with torch.no_grad():
            S, T = run_latent_forward(model, z=z, pos=pos)

        h = adapter(S, T)
        S_recon, T_recon = adapter.reconstruct(h)

        assert (S - S_recon).abs().max().item() == 0.0
        assert (T - T_recon).abs().max().item() == 0.0

    def test_h0_is_S_unsqueezed(self):
        """h[0] must be S.unsqueeze(-1), not derived from T."""
        model = _get_model()
        z, pos = _make_molecule(5)
        adapter = make_adapter()

        with torch.no_grad():
            S, T = run_latent_forward(model, z=z, pos=pos)

        h = adapter(S, T)
        assert torch.equal(h[0].squeeze(-1), S)

    def test_h_shapes_from_detanet(self):
        """Each l-block has the correct [N, C, 2l+1] shape."""
        model = _get_model()
        z, pos = _make_molecule(7)
        adapter = make_adapter()

        with torch.no_grad():
            S, T = run_latent_forward(model, z=z, pos=pos)

        h = adapter(S, T)
        C = adapter.num_features
        N = 7
        assert h[0].shape == (N, C, 1)
        assert h[1].shape == (N, C, 3)
        assert h[2].shape == (N, C, 5)
        assert h[3].shape == (N, C, 7)

    def test_different_maxl(self):
        for maxl in [1, 2, 3]:
            m = make_latent_detanet(maxl=maxl, num_block=1, device="cpu")
            z, pos = _make_molecule(4, seed=99)
            with torch.no_grad():
                S, T = run_latent_forward(m, z=z, pos=pos)
            adapter = make_adapter(maxl=maxl)
            h = adapter(S, T)
            S_r, T_r = adapter.reconstruct(h)
            assert torch.equal(S, S_r)
            assert torch.equal(T, T_r)
            for l in range(1, maxl + 1):
                assert h[l].shape == (4, 128, 2 * l + 1)


class TestTensorAdapterWignerD:
    """Wigner-D equivariance through the TensorAdapter pipeline."""

    @pytest.mark.parametrize("l", [1, 2, 3])
    def test_wigner_d_per_order(self, l):
        model = _get_model()
        z, pos = _make_molecule(5, seed=42)
        R = _random_rotation(seed=100)
        adapter = make_adapter()

        with torch.no_grad():
            S_orig, T_orig = run_latent_forward(model, z=z, pos=pos)
            S_rot, T_rot = run_latent_forward(model, z=z, pos=pos @ R.T)

        h_orig = adapter(S_orig, T_orig)
        h_rot = adapter(S_rot, T_rot)

        D = _wigner_D(l, R)
        h_rot_pred = torch.einsum("ab,ncb->nca", D, h_orig[l])

        err = (h_rot[l] - h_rot_pred).abs().max().item()
        assert err < TOLERANCE, f"Wigner-D equivariance failed for l={l}: err={err:.2e}"

    def test_all_orders_simultaneous(self):
        model = _get_model()
        z, pos = _make_molecule(6, seed=200)
        R = _random_rotation(seed=300)
        adapter = make_adapter()

        with torch.no_grad():
            S_orig, T_orig = run_latent_forward(model, z=z, pos=pos)
            S_rot, T_rot = run_latent_forward(model, z=z, pos=pos @ R.T)

        h_orig = adapter(S_orig, T_orig)
        h_rot = adapter(S_rot, T_rot)

        for l in [1, 2, 3]:
            D = _wigner_D(l, R)
            h_pred = torch.einsum("ab,ncb->nca", D, h_orig[l])
            err = (h_rot[l] - h_pred).abs().max().item()
            assert err < TOLERANCE, f"l={l} failed: err={err:.2e}"

    def test_multiple_rotations(self):
        model = _get_model()
        z, pos = _make_molecule(5, seed=42)
        adapter = make_adapter()

        for rot_seed in [10, 20, 30, 40, 50]:
            R = _random_rotation(seed=rot_seed)
            with torch.no_grad():
                S_orig, T_orig = run_latent_forward(model, z=z, pos=pos)
                S_rot, T_rot = run_latent_forward(model, z=z, pos=pos @ R.T)
            h_orig = adapter(S_orig, T_orig)
            h_rot = adapter(S_rot, T_rot)
            for l in [1, 2, 3]:
                D = _wigner_D(l, R)
                h_pred = torch.einsum("ab,ncb->nca", D, h_orig[l])
                err = (h_rot[l] - h_pred).abs().max().item()
                assert err < TOLERANCE, f"rot_seed={rot_seed} l={l}: err={err:.2e}"

    def test_large_molecule(self):
        """Wigner-D holds for a 20-atom molecule."""
        model = _get_model()
        z, pos = _make_molecule(20, seed=500)
        R = _random_rotation(seed=600)
        adapter = make_adapter()

        with torch.no_grad():
            S_orig, T_orig = run_latent_forward(model, z=z, pos=pos)
            S_rot, T_rot = run_latent_forward(model, z=z, pos=pos @ R.T)

        h_orig = adapter(S_orig, T_orig)
        h_rot = adapter(S_rot, T_rot)

        for l in [1, 2, 3]:
            D = _wigner_D(l, R)
            h_pred = torch.einsum("ab,ncb->nca", D, h_orig[l])
            err = (h_rot[l] - h_pred).abs().max().item()
            assert err < TOLERANCE, f"l={l} large mol failed: err={err:.2e}"


class TestTensorAdapterParity:
    """Parity (spatial inversion) through the TensorAdapter."""

    def test_inversion_odd_even(self):
        model = _get_model()
        z, pos = _make_molecule(5, seed=42)
        inversion = torch.diag(torch.tensor([-1.0, -1.0, -1.0]))
        adapter = make_adapter()

        with torch.no_grad():
            S_orig, T_orig = run_latent_forward(model, z=z, pos=pos)
            S_inv, T_inv = run_latent_forward(model, z=z, pos=pos @ inversion.T)

        h_orig = adapter(S_orig, T_orig)
        h_inv = adapter(S_inv, T_inv)

        for l in [1, 2, 3]:
            parity = -1 if l % 2 == 1 else 1
            expected = parity * h_orig[l]
            err = (h_inv[l] - expected).abs().max().item()
            assert err < TOLERANCE, f"Inversion parity failed for l={l}: err={err:.2e}"


class TestTensorAdapterTranslation:
    """Translation invariance through the TensorAdapter."""

    def test_translation_invariance(self):
        model = _get_model()
        z, pos = _make_molecule(5, seed=42)
        translation = torch.tensor([10.0, -5.0, 3.0])
        adapter = make_adapter()

        with torch.no_grad():
            S_orig, T_orig = run_latent_forward(model, z=z, pos=pos)
            S_trans, T_trans = run_latent_forward(model, z=z, pos=pos + translation)

        h_orig = adapter(S_orig, T_orig)
        h_trans = adapter(S_trans, T_trans)

        for l in [0, 1, 2, 3]:
            err = (h_trans[l] - h_orig[l]).abs().max().item()
            assert err < TOLERANCE, f"Translation violated for l={l}: err={err:.2e}"

    def test_moderate_translation(self):
        """Translation invariance holds for moderate shifts (~50 Å).

        Very large translations (>> 1e3 Å) can degrade numerical precision in
        the radial Bessel basis due to float32 computation, even though
        pairwise distances are unchanged. This is a numerical artifact of the
        basis function implementation, not a violation of translation symmetry.
        """
        model = _get_model()
        z, pos = _make_molecule(8, seed=77)
        translation = torch.tensor([53.0, -27.0, 31.0])
        adapter = make_adapter()

        with torch.no_grad():
            S_orig, T_orig = run_latent_forward(model, z=z, pos=pos)
            S_trans, T_trans = run_latent_forward(model, z=z, pos=pos + translation)

        h_orig = adapter(S_orig, T_orig)
        h_trans = adapter(S_trans, T_trans)

        for l in [0, 1, 2, 3]:
            err = (h_trans[l] - h_orig[l]).abs().max().item()
            assert err < TOLERANCE, f"Translation invariance failed l={l}: err={err:.2e}"


class TestTensorAdapterPermutation:
    """Atom permutation consistency through the TensorAdapter."""

    def test_permutation_equivariance(self):
        model = _get_model()
        z = torch.tensor([1, 6, 1, 6, 1], dtype=torch.long)
        pos = torch.tensor([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
        ], dtype=torch.float32)
        perm = torch.tensor([2, 1, 0, 3, 4], dtype=torch.long)
        adapter = make_adapter()

        with torch.no_grad():
            S_orig, T_orig = run_latent_forward(model, z=z, pos=pos)
            S_perm, T_perm = run_latent_forward(model, z=z[perm], pos=pos[perm])

        h_orig = adapter(S_orig, T_orig)
        h_perm = adapter(S_perm, T_perm)

        for l in [0, 1, 2, 3]:
            err = (h_perm[l] - h_orig[l][perm]).abs().max().item()
            assert err < TOLERANCE, f"Permutation equivariance failed for l={l}: err={err:.2e}"

    def test_same_element_permutation(self):
        """Permuting atoms of the same element type must give identical result for those atoms."""
        model = _get_model()
        z = torch.tensor([6, 6, 6, 8, 8], dtype=torch.long)
        pos = torch.randn(5, 3)
        perm = torch.tensor([2, 1, 0, 4, 3], dtype=torch.long)  # swap all-carbons, all-oxygens
        adapter = make_adapter()

        with torch.no_grad():
            S_orig, T_orig = run_latent_forward(model, z=z, pos=pos)
            S_perm, T_perm = run_latent_forward(model, z=z[perm], pos=pos[perm])

        h_orig = adapter(S_orig, T_orig)
        h_perm = adapter(S_perm, T_perm)

        for l in [0, 1, 2, 3]:
            err = (h_perm[l] - h_orig[l][perm]).abs().max().item()
            assert err < TOLERANCE, f"Same-element perm failed l={l}: err={err:.2e}"


class TestTensorAdapterRoundtrip:
    """Roundtrip: adapter(S,T) → reconstruct → adapter again yields same dict."""

    def test_roundtrip_idempotent(self):
        model = _get_model()
        z, pos = _make_molecule(5)
        adapter = make_adapter()

        with torch.no_grad():
            S, T = run_latent_forward(model, z=z, pos=pos)

        h1 = adapter(S, T)
        S_r, T_r = adapter.reconstruct(h1)
        h2 = adapter(S_r, T_r)

        for l in [0, 1, 2, 3]:
            assert torch.equal(h1[l], h2[l]), f"Roundtrip mismatch l={l}"
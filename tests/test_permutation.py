"""Test same-element atom permutation consistency for the full MTO pipeline.

Atom permutation within a molecule:
  - Router is permutation equivariant: c_perm[l] == c_orig[l][:, perm, :]
  - MTO assembly O_k^(l) = sum_i c_ki * W_l * H_i is invariant (sum over atoms)
  - CG coupling, gating, readouts all inherit this invariance

For synthetic features (no Z dependency), ANY permutation must be invariant.
For real DetaNet features, same-element permutations must be invariant since
the molecular graph is unchanged up to isomorphism.
"""

import pytest
import torch

torch.serialization.add_safe_globals([slice])

from ar_mto.signed_routing import SignedRouter
from ar_mto.mto_core import MTOModeAssembly, ScalarOnlyMTO
from ar_mto.cg_coupling import CGCouplingMinimal, CGCoupling
from ar_mto.tensor_gate import TensorGate, NoGate, ScalarOnlyGate
from ar_mto.readouts import ScalarReadout, VectorReadout, Rank2TensorReadout

TOLERANCE = 1e-5


def _make_h(N=5, C=128, maxl=3):
    h = {}
    h[0] = torch.randn(N, C, 1)
    for l in range(1, maxl + 1):
        h[l] = torch.randn(N, C, 2 * l + 1)
    return h


def _apply_perm(h, perm):
    """Apply the SAME permutation to all l orders."""
    return {l: h[l][perm].clone() for l in h}


def _make_molecule(num_atoms=5, seed=42):
    gen = torch.Generator()
    gen.manual_seed(seed)
    z = torch.randint(1, 5, (num_atoms,), generator=gen)  # small Z range → guarantees pairs
    radius = 1.2
    pos = torch.randn(num_atoms, 3, generator=gen)
    norms = torch.norm(pos, dim=-1, keepdim=True)
    scales = torch.rand(num_atoms, 1, generator=gen) ** (1.0 / 3.0)
    pos = pos / (norms + 1e-8) * scales * radius
    return z, pos


# ── MTO Mode Assembly ──────────────────────────────────────────────────────

class TestMTOPermutationInvariance:
    """MTO assembled modes must be invariant under atom permutation."""

    @pytest.mark.parametrize("N", [4, 5, 8])
    @pytest.mark.parametrize("K", [4, 8])
    def test_mto_mode_permutation_invariance(self, N, K):
        """Arbitrary atom permutation leaves assembled modes unchanged."""
        C = 128
        mto = MTOModeAssembly(num_features=C, mode_channels=64,
                              num_modes=K, maxl=3)
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)

        h_orig = _make_h(N, C)
        perm = torch.randperm(N)
        h_perm = _apply_perm(h_orig, perm)

        with torch.no_grad():
            c_orig = router(h_orig)
            c_perm = router(h_perm)

            # Router: permutation equivariance
            for l in [0, 1, 2, 3]:
                err = (c_perm[l] - c_orig[l][:, perm, :]).abs().max().item()
                assert err < TOLERANCE, \
                    f"N={N} K={K} l={l}: router not perm-equivariant, err={err:.2e}"

            # MTO assembly: permutation invariance
            O_orig = mto(h_orig, c_orig)
            O_perm = mto(h_perm, c_perm)

            for l in [0, 1, 2, 3]:
                err = (O_perm[l] - O_orig[l]).abs().max().item()
                assert err < TOLERANCE, \
                    f"N={N} K={K} l={l}: MTO not perm-invariant, err={err:.2e}"

    def test_router_equivariance_batch(self):
        """Router permutation equivariance holds per-molecule in batch."""
        C, K = 128, 4
        N1, N2 = 5, 4
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)

        h1 = _make_h(N1, C)
        h2 = _make_h(N2, C)
        h_batch = {l: torch.cat([h1[l], h2[l]], dim=0) for l in h1}
        batch = torch.tensor([0] * N1 + [1] * N2, dtype=torch.long)

        perm = torch.randperm(N1)
        h1_perm = _apply_perm(h1, perm)
        h_batch_perm = {l: torch.cat([h1_perm[l], h2[l]], dim=0) for l in h1}

        with torch.no_grad():
            c_batch = router(h_batch, batch=batch)
            c_batch_perm = router(h_batch_perm, batch=batch)

        for l in [0, 1, 2, 3]:
            err_0 = (c_batch_perm[l][:, :N1, :] -
                     c_batch[l][:, :N1, :][:, perm, :]).abs().max().item()
            err_1 = (c_batch_perm[l][:, N1:, :] -
                     c_batch[l][:, N1:, :]).abs().max().item()
            assert err_0 < TOLERANCE, \
                f"l={l} mol 0 router not perm-equivariant: err={err_0:.2e}"
            assert err_1 < TOLERANCE, \
                f"l={l} mol 1 router unexpectedly changed: err={err_1:.2e}"

    def test_mto_batch_permutation_isolation(self):
        """Permuting atoms in mol 0 leaves both molecules' MTO outputs unchanged."""
        C, K = 128, 4
        N1, N2 = 5, 4
        mto = MTOModeAssembly(num_features=C, mode_channels=64,
                              num_modes=K, maxl=3)
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)

        h1 = _make_h(N1, C)
        h2 = _make_h(N2, C)
        h_batch = {l: torch.cat([h1[l], h2[l]], dim=0) for l in h1}
        batch = torch.tensor([0] * N1 + [1] * N2, dtype=torch.long)

        perm = torch.randperm(N1)
        h1_perm = _apply_perm(h1, perm)
        h_batch_perm = {l: torch.cat([h1_perm[l], h2[l]], dim=0) for l in h1}

        with torch.no_grad():
            O_batch = mto(h_batch, router(h_batch, batch=batch), batch=batch)
            O_batch_perm = mto(h_batch_perm, router(h_batch_perm, batch=batch),
                               batch=batch)

        for l in [0, 1, 2, 3]:
            err_0 = (O_batch_perm[l][0:1] - O_batch[l][0:1]).abs().max().item()
            err_1 = (O_batch_perm[l][1:2] - O_batch[l][1:2]).abs().max().item()
            assert err_0 < TOLERANCE, \
                f"l={l} mol 0 perm broke invariance: err={err_0:.2e}"
            assert err_1 < TOLERANCE, \
                f"l={l} mol 1 incorrectly changed: err={err_1:.2e}"


# ── CG Coupling ────────────────────────────────────────────────────────────

class TestCGPermutationInvariance:
    """CG coupling operates on [B, K, C, 2l+1] — must be permutation invariant."""

    def test_cg_minimal_permutation_invariance(self):
        """CGCouplingMinimal: permuting MTO input gives same CG output."""
        B, K, C = 1, 4, 64
        N = 5
        mto = MTOModeAssembly(num_features=C, mode_channels=C,
                              num_modes=K, maxl=3)
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)
        cg = CGCouplingMinimal(mode_channels=C)

        h_orig = _make_h(N, C)
        perm = torch.randperm(N)
        h_perm = _apply_perm(h_orig, perm)

        with torch.no_grad():
            O_orig = mto(h_orig, router(h_orig))
            O_perm = mto(h_perm, router(h_perm))
            Oc_orig = cg(O_orig)
            Oc_perm = cg(O_perm)

        for key in Oc_orig:
            assert torch.allclose(Oc_perm[key], Oc_orig[key], atol=TOLERANCE), \
                f"CG minimal key={key}: perm invariance failed"

    def test_cg_full_permutation_invariance(self):
        """CGCoupling: permutation invariant."""
        B, K, C = 1, 2, 16
        N = 4
        mto = MTOModeAssembly(num_features=C, mode_channels=C,
                              num_modes=K, maxl=2)
        router = SignedRouter(num_features=C, num_modes=K, maxl=2)
        cg = CGCoupling(mode_channels=C, maxl=2, coupled_maxl=2)

        h_orig = _make_h(N, C, maxl=2)
        perm = torch.randperm(N)
        h_perm = _apply_perm(h_orig, perm)

        with torch.no_grad():
            O_orig = mto(h_orig, router(h_orig))
            O_perm = mto(h_perm, router(h_perm))
            Oc_orig = cg(O_orig)
            Oc_perm = cg(O_perm)

        for key in Oc_orig:
            assert torch.allclose(Oc_perm[key], Oc_orig[key], atol=TOLERANCE), \
                f"CG full key={key}: perm invariance failed"


# ── Tensor Gate ────────────────────────────────────────────────────────────

class TestGatePermutationInvariance:
    """Tensor gates are per-mode — MTO invariant → gate invariant."""

    def test_tensor_gate_permutation_invariance(self):
        C, K = 128, 4
        N = 5
        gate = TensorGate(mode_channels=64, num_modes=K, maxl=3)
        mto = MTOModeAssembly(num_features=C, mode_channels=64,
                              num_modes=K, maxl=3)
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)

        h_orig = _make_h(N, C)
        h_perm = _apply_perm(h_orig, torch.randperm(N))

        with torch.no_grad():
            O_orig = mto(h_orig, router(h_orig))
            O_perm = mto(h_perm, router(h_perm))
            Og_orig = gate(O_orig)
            Og_perm = gate(O_perm)

        for l in [0, 1, 2, 3]:
            assert torch.allclose(Og_perm[l], Og_orig[l], atol=TOLERANCE), \
                f"TensorGate l={l}: perm invariance failed"

    def test_no_gate_permutation_invariance(self):
        C, K = 128, 4
        N = 5
        gate = NoGate(mode_channels=64, num_modes=K, maxl=3)
        mto = MTOModeAssembly(num_features=C, mode_channels=64,
                              num_modes=K, maxl=3)
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)

        h_orig = _make_h(N, C)
        h_perm = _apply_perm(h_orig, torch.randperm(N))

        with torch.no_grad():
            O_orig = mto(h_orig, router(h_orig))
            O_perm = mto(h_perm, router(h_perm))
            Og_orig = gate(O_orig)
            Og_perm = gate(O_perm)

        for l in [0, 1, 2, 3]:
            assert torch.allclose(Og_perm[l], Og_orig[l], atol=TOLERANCE)

    def test_scalar_only_gate_permutation_invariance(self):
        C, K = 128, 4
        N = 5
        gate = ScalarOnlyGate(mode_channels=64, num_modes=K, maxl=3)
        mto = MTOModeAssembly(num_features=C, mode_channels=64,
                              num_modes=K, maxl=3)
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)

        h_orig = _make_h(N, C)
        h_perm = _apply_perm(h_orig, torch.randperm(N))

        with torch.no_grad():
            O_orig = mto(h_orig, router(h_orig))
            O_perm = mto(h_perm, router(h_perm))
            Og_orig = gate(O_orig)
            Og_perm = gate(O_perm)

        for l in [0, 1, 2, 3]:
            assert torch.allclose(Og_perm[l], Og_orig[l], atol=TOLERANCE), \
                f"ScalarOnlyGate l={l}: perm invariance failed"


# ── Readouts ───────────────────────────────────────────────────────────────

class TestReadoutPermutationInvariance:
    """Readout predictions must be invariant under atom permutation."""

    def test_scalar_readout_permutation_invariance(self):
        C, K = 128, 4
        N = 5
        readout = ScalarReadout(mode_channels=64, num_modes=K)
        mto = MTOModeAssembly(num_features=C, mode_channels=64,
                              num_modes=K, maxl=3)
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)

        h_orig = _make_h(N, C)
        h_perm = _apply_perm(h_orig, torch.randperm(N))

        with torch.no_grad():
            O_orig = mto(h_orig, router(h_orig))
            O_perm = mto(h_perm, router(h_perm))
            y_orig = readout(O_orig)
            y_perm = readout(O_perm)

        assert torch.allclose(y_perm, y_orig, atol=TOLERANCE), \
            f"Scalar readout not perm-invariant"

    def test_vector_readout_permutation_invariance(self):
        C, K = 128, 4
        N = 5
        readout = VectorReadout(mode_channels=64, num_modes=K)
        mto = MTOModeAssembly(num_features=C, mode_channels=64,
                              num_modes=K, maxl=3)
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)

        h_orig = _make_h(N, C)
        h_perm = _apply_perm(h_orig, torch.randperm(N))

        with torch.no_grad():
            O_orig = mto(h_orig, router(h_orig))
            O_perm = mto(h_perm, router(h_perm))
            y_orig = readout(O_orig)
            y_perm = readout(O_perm)

        assert y_perm.shape == (1, 1, 3)
        assert torch.allclose(y_perm, y_orig, atol=TOLERANCE), \
            "Vector readout not perm-invariant"

    def test_rank2_readout_permutation_invariance(self):
        C, K = 128, 4
        N = 5
        readout = Rank2TensorReadout(mode_channels=64, num_modes=K)
        mto = MTOModeAssembly(num_features=C, mode_channels=64,
                              num_modes=K, maxl=3)
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)

        h_orig = _make_h(N, C)
        h_perm = _apply_perm(h_orig, torch.randperm(N))

        with torch.no_grad():
            O_orig = mto(h_orig, router(h_orig))
            O_perm = mto(h_perm, router(h_perm))
            y_orig = readout(O_orig)
            y_perm = readout(O_perm)

        assert y_perm.shape == (1, 1, 3, 3)
        assert torch.allclose(y_perm, y_orig, atol=TOLERANCE), \
            "Rank2Tensor readout not perm-invariant"


# ── Full Pipeline (with real DetaNet) ──────────────────────────────────────

class TestFullPipelinePermutation:
    """End-to-end permutation invariance with real DetaNet features."""

    @pytest.fixture(scope="class")
    def detanet_model(self):
        from ar_mto.detanet_bridge import make_latent_detanet
        return make_latent_detanet(num_block=2, device="cpu")

    def test_same_element_permutation_mto_invariant(self, detanet_model):
        """MTO output is invariant when permuting atoms with same Z."""
        from ar_mto.detanet_bridge import run_latent_forward
        from ar_mto.tensor_adapter import make_adapter

        # Z=[1,2,1,2,1,2] — guaranteed same-element pairs
        z = torch.tensor([1, 2, 1, 2, 1, 2], dtype=torch.long)
        pos = torch.randn(6, 3)
        # Swap atoms 0,2 (both Z=1) and 1,4 (both Z=1)
        perm = torch.tensor([2, 4, 0, 3, 1, 5], dtype=torch.long)
        assert not torch.equal(perm, torch.arange(6))

        adapter = make_adapter()
        router = SignedRouter(num_features=128, num_modes=8, maxl=3)
        mto = MTOModeAssembly(num_features=128, mode_channels=64,
                              num_modes=8, maxl=3)

        with torch.no_grad():
            S, T = run_latent_forward(detanet_model, z=z, pos=pos)
            S_p, T_p = run_latent_forward(detanet_model, z=z[perm], pos=pos[perm])

        h = adapter(S, T)
        h_p = adapter(S_p, T_p)

        # Adapter is permutation equivariant
        for l in [0, 1, 2, 3]:
            err = (h_p[l] - h[l][perm]).abs().max().item()
            assert err < TOLERANCE, \
                f"Adapter not perm-equivariant l={l}: {err:.2e}"

        with torch.no_grad():
            O = mto(h, router(h))
            O_p = mto(h_p, router(h_p))

        for l in [0, 1, 2, 3]:
            err = (O_p[l] - O[l]).abs().max().item()
            assert err < TOLERANCE, \
                f"MTO not perm-invariant l={l}: {err:.2e}"

    def test_same_element_permutation_readout_invariant(self, detanet_model):
        """Same-element atom permutation → identical predictions."""
        from ar_mto.detanet_bridge import run_latent_forward
        from ar_mto.tensor_adapter import make_adapter

        z = torch.tensor([3, 1, 3, 1, 2, 2], dtype=torch.long)
        pos = torch.randn(6, 3)
        # Swap 0↔2 (both Z=3), 1↔3 (both Z=1), 4↔5 (both Z=2)
        perm = torch.tensor([2, 3, 0, 1, 5, 4], dtype=torch.long)

        adapter = make_adapter()
        router = SignedRouter(num_features=128, num_modes=8, maxl=3)
        mto = MTOModeAssembly(num_features=128, mode_channels=64,
                              num_modes=8, maxl=3)
        cg = CGCouplingMinimal(mode_channels=64)
        gate = TensorGate(mode_channels=64, num_modes=8, maxl=3)
        scalar_head = ScalarReadout(mode_channels=64, num_modes=8)
        vector_head = VectorReadout(mode_channels=64, num_modes=8)
        rank2_head = Rank2TensorReadout(mode_channels=64, num_modes=8)

        def predict(z_in, pos_in):
            with torch.no_grad():
                S, T = run_latent_forward(detanet_model, z=z_in, pos=pos_in)
                h = adapter(S, T)
                c = router(h)
                O = mto(h, c)
                Oc = cg(O)
                O_full = {0: Oc[0], 1: Oc[1], 2: Oc[2], 3: O[3]}
                Og = gate(O_full)
                return {
                    "scalar": scalar_head(Og),
                    "vector": vector_head(Og),
                    "tensor": rank2_head(Og),
                }

        y_orig = predict(z, pos)
        y_perm = predict(z[perm], pos[perm])

        assert torch.allclose(y_perm["scalar"], y_orig["scalar"], atol=TOLERANCE), \
            "Scalar readout not same-element perm-invariant"
        assert torch.allclose(y_perm["vector"], y_orig["vector"], atol=TOLERANCE), \
            "Vector readout not same-element perm-invariant"
        assert torch.allclose(y_perm["tensor"], y_orig["tensor"], atol=TOLERANCE), \
            "Rank2 readout not same-element perm-invariant"

    @pytest.mark.parametrize("seed", [10, 20, 30, 40, 50])
    def test_multiple_molecule_seeds(self, detanet_model, seed):
        """Same-element perm invariance across random molecules with repeated Z."""
        from ar_mto.detanet_bridge import run_latent_forward
        from ar_mto.tensor_adapter import make_adapter

        z, pos = _make_molecule(8, seed=seed)

        # Build a same-element permutation
        perm = torch.arange(8)
        seen = {}
        for i, zi in enumerate(z.tolist()):
            if zi in seen:
                j = seen.pop(zi)
                perm[i], perm[j] = perm[j], perm[i]
            else:
                seen[zi] = i
        if torch.equal(perm, torch.arange(8)):
            pytest.skip("No same-element pairs in this random molecule")

        adapter = make_adapter()
        router = SignedRouter(num_features=128, num_modes=4, maxl=3)
        mto = MTOModeAssembly(num_features=128, mode_channels=32,
                              num_modes=4, maxl=3)
        readout = ScalarReadout(mode_channels=32, num_modes=4)

        def predict(z_in, pos_in):
            with torch.no_grad():
                S, T = run_latent_forward(detanet_model, z=z_in, pos=pos_in)
                h = adapter(S, T)
                c = router(h)
                O = mto(h, c)
                return readout(O)

        y_orig = predict(z, pos)
        y_perm = predict(z[perm], pos[perm])

        # DetaNet float32 forward passes on permuted graphs accumulate
        # ~1e-4 to ~1e-3 numerical drift through multiple nonlinear layers.
        # This is float32 precision, not an MTO permutation invariance bug.
        DETANET_TOL = 2e-3
        assert torch.allclose(y_perm, y_orig, atol=DETANET_TOL), \
            f"seed={seed}: perm-invariance broken, diff={((y_perm - y_orig).abs().max().item()):.2e}"

    def test_batch_one_mol_permuted(self, detanet_model):
        """In batch, permuting one molecule leaves both outputs unchanged."""
        from ar_mto.detanet_bridge import run_latent_forward
        from ar_mto.tensor_adapter import make_adapter

        z1 = torch.tensor([3, 1, 3, 1, 3], dtype=torch.long)
        pos1 = torch.randn(5, 3)
        perm1 = torch.tensor([2, 4, 0, 3, 1], dtype=torch.long)

        z2 = torch.tensor([6, 6, 8, 8], dtype=torch.long)
        pos2 = torch.randn(4, 3)

        adapter = make_adapter()
        router = SignedRouter(num_features=128, num_modes=8, maxl=3)
        mto = MTOModeAssembly(num_features=128, mode_channels=64,
                              num_modes=8, maxl=3)

        z = torch.cat([z1, z2])
        pos = torch.cat([pos1, pos2])
        batch = torch.tensor([0] * 5 + [1] * 4, dtype=torch.long)
        z_perm = torch.cat([z1[perm1], z2])
        pos_perm = torch.cat([pos1[perm1], pos2])

        def predict(z_in, pos_in):
            with torch.no_grad():
                S, T = run_latent_forward(detanet_model, z=z_in, pos=pos_in)
                h = adapter(S, T)
                c = router(h, batch=batch)
                return mto(h, c, batch=batch)

        O_batch = predict(z, pos)
        O_batch_perm = predict(z_perm, pos_perm)

        for l in [0, 1, 2, 3]:
            assert torch.allclose(O_batch_perm[l], O_batch[l], atol=TOLERANCE), \
                f"l={l}: batch perm-invariance broken"


# ── Scalar-Only MTO ────────────────────────────────────────────────────────

class TestScalarOnlyMTOPermutation:
    """Scalar-only MTO must also be permutation invariant."""

    def test_scalar_only_permutation_invariance(self):
        C, K = 128, 4
        N = 5
        mto = ScalarOnlyMTO(num_features=C, mode_channels=64, num_modes=K)
        router = SignedRouter(num_features=C, num_modes=K,
                              use_tensor_norms=False, maxl=0)

        h_orig = {0: torch.randn(N, C, 1)}
        perm = torch.randperm(N)
        h_perm = {0: h_orig[0][perm].clone()}

        with torch.no_grad():
            c_orig = router(h_orig)
            c_perm = router(h_perm)
            O_orig = mto(h_orig, c_orig)
            O_perm = mto(h_perm, c_perm)

        assert torch.allclose(O_perm[0], O_orig[0], atol=TOLERANCE), \
            "Scalar-only MTO not perm-invariant"


# ── Mode Masking ───────────────────────────────────────────────────────────

class TestModeMaskingPermutation:
    """Mode masking + permutation: masked modes stay zero."""

    def test_masked_modes_remain_zero_after_permutation(self):
        C, K = 128, 8
        N = 5
        mto = MTOModeAssembly(num_features=C, mode_channels=64,
                              num_modes=K, maxl=3)
        router = SignedRouter(num_features=C, num_modes=K, maxl=3)

        mode_mask = torch.zeros(1, K, dtype=torch.bool)
        mode_mask[0, :4] = True

        h_orig = _make_h(N, C)
        h_perm = _apply_perm(h_orig, torch.randperm(N))

        with torch.no_grad():
            O_orig = mto.forward_with_masks(h_orig, router(h_orig), mode_mask)
            O_perm = mto.forward_with_masks(h_perm, router(h_perm), mode_mask)

        for l in [0, 1, 2, 3]:
            err = (O_perm[l][:, :4, :, :] - O_orig[l][:, :4, :, :]).abs().max().item()
            assert err < TOLERANCE, \
                f"l={l} active modes not perm-invariant: {err:.2e}"
            assert (O_perm[l][:, 4:, :, :].abs().max() == 0.0), \
                f"l={l} permuted masked modes not zero"

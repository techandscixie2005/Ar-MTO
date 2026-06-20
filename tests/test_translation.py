"""Test translation invariance through the full MTO pipeline.

DetaNet features are translation-invariant (position → pairwise distances only).
All downstream modules (SignedRouter, MTOModeAssembly, CGCoupling, TensorGate)
consume only tensor features, never raw positions. Therefore the entire pipeline
must be invariant under rigid spatial translations.

This file verifies translation invariance for each module individually and for
the combined MTO → CG → Gate pipeline.
"""

import pytest
import torch

torch.serialization.add_safe_globals([slice])

from ar_mto.tensor_adapter import make_adapter
from ar_mto.detanet_bridge import (
    make_latent_detanet,
    run_latent_forward,
)
from ar_mto.signed_routing import SignedRouter
from ar_mto.mto_core import MTOModeAssembly
from ar_mto.cg_coupling import CGCouplingMinimal, CGCoupling
from ar_mto.tensor_gate import TensorGate, NoGate, ScalarOnlyGate

TOLERANCE = 5e-5

# Module-scoped model cache to avoid re-instantiation
_DETANET_CACHE: dict = {}
_ADAPTER_CACHE: dict = {}


def _get_model(num_block=2):
    if num_block not in _DETANET_CACHE:
        _DETANET_CACHE[num_block] = make_latent_detanet(num_block=num_block, device="cpu")
    return _DETANET_CACHE[num_block]


def _get_adapter():
    if "default" not in _ADAPTER_CACHE:
        _ADAPTER_CACHE["default"] = make_adapter()
    return _ADAPTER_CACHE["default"]


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


def _get_features(z, pos, model=None, adapter=None):
    if model is None:
        model = _get_model()
    if adapter is None:
        adapter = _get_adapter()
    with torch.no_grad():
        S, T = run_latent_forward(model, z=z, pos=pos)
    return adapter(S, T)


TRANSLATIONS = [
    torch.tensor([0.0, 0.0, 0.0]),
    torch.tensor([5.0, 0.0, 0.0]),
    torch.tensor([0.0, 7.0, 0.0]),
    torch.tensor([-3.0, 2.0, -8.0]),
    torch.tensor([25.0, -13.0, 42.0]),
]


class TestDetaNetTranslationInvariance:
    """DetaNet features must be invariant under rigid translation."""

    @pytest.mark.parametrize("t_idx", range(len(TRANSLATIONS)))
    def test_single_translation(self, t_idx):
        model = _get_model()
        adapter = _get_adapter()
        z, pos = _make_molecule(6, seed=42)
        translation = TRANSLATIONS[t_idx]

        h_orig = _get_features(z, pos, model, adapter)
        h_trans = _get_features(z, pos + translation, model, adapter)

        for l in [0, 1, 2, 3]:
            err = (h_trans[l] - h_orig[l]).abs().max().item()
            assert err < TOLERANCE, (
                f"Translation [{translation.tolist()}] violated l={l}: err={err:.2e}"
            )

    def test_variable_atoms(self):
        model = _get_model()
        adapter = _get_adapter()
        translation = torch.tensor([10.0, 20.0, -15.0])

        for n in [3, 4, 5, 6, 8]:
            z, pos = _make_molecule(n, seed=n * 10)
            h_orig = _get_features(z, pos, model, adapter)
            h_trans = _get_features(z, pos + translation, model, adapter)
            for l in [0, 1, 2, 3]:
                err = (h_trans[l] - h_orig[l]).abs().max().item()
                assert err < TOLERANCE, (
                    f"Translation failed n={n} l={l}: err={err:.2e}"
                )

    def test_batched_molecules(self):
        model = _get_model()
        adapter = _get_adapter()
        z1, pos1 = _make_molecule(4, seed=10)
        z2, pos2 = _make_molecule(3, seed=20)
        z = torch.cat([z1, z2])
        pos = torch.cat([pos1, pos2])
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
        translation = torch.tensor([7.0, -3.0, 12.0])

        with torch.no_grad():
            S_o, T_o = run_latent_forward(model, z=z, pos=pos, batch=batch)
            S_t, T_t = run_latent_forward(model, z=z, pos=pos + translation, batch=batch)

        h_orig = adapter(S_o, T_o)
        h_trans = adapter(S_t, T_t)

        for l in [0, 1, 2, 3]:
            err = (h_trans[l] - h_orig[l]).abs().max().item()
            assert err < TOLERANCE, f"Batched translation l={l}: err={err:.2e}"


class TestSignedRouterTranslation:
    """SignedRouter only consumes invariant (scalar) inputs → translation-invariant."""

    def test_routing_translation_invariant(self):
        model = _get_model()
        adapter = _get_adapter()
        z, pos = _make_molecule(5, seed=42)
        translation = torch.tensor([15.0, -7.0, 3.0])

        h_orig = _get_features(z, pos, model, adapter)
        h_trans = _get_features(z, pos + translation, model, adapter)

        router = SignedRouter(num_features=128, num_modes=8, maxl=3, normalization="l2")
        router.eval()

        with torch.no_grad():
            coeffs_orig = router(h_orig)
            coeffs_trans = router(h_trans)

        for l in coeffs_orig:
            err = (coeffs_trans[l] - coeffs_orig[l]).abs().max().item()
            assert err < TOLERANCE, f"Routing coeffs l={l}: err={err:.2e}"

        # route_stats should also match
        stats_orig = router.route_stats(coeffs_orig)
        stats_trans = router.route_stats(coeffs_trans)
        for key in stats_orig:
            v_orig = stats_orig[key]
            v_trans = stats_trans[key]
            if isinstance(v_orig, (int, float)):
                assert abs(v_orig - v_trans) < TOLERANCE, \
                    f"Route stat {key}: {v_orig:.6f} vs {v_trans:.6f}"

    def test_routing_sign_conservation(self):
        """Sign patterns must be deterministic irrespective of translation."""
        model = _get_model()
        adapter = _get_adapter()
        z, pos = _make_molecule(6, seed=100)
        translation = torch.tensor([-10.0, 5.0, 20.0])

        h_orig = _get_features(z, pos, model, adapter)
        h_trans = _get_features(z, pos + translation, model, adapter)

        router = SignedRouter(num_features=128, num_modes=8, maxl=3, normalization="l2")
        router.eval()

        with torch.no_grad():
            coeffs_orig = router(h_orig)
            coeffs_trans = router(h_trans)

        # Full coefficients must match (same inputs → same output)
        # The sign information is baked into coeffs via tanh(sign_hidden) * attn
        for l in coeffs_orig:
            err = (coeffs_trans[l] - coeffs_orig[l]).abs().max().item()
            assert err < TOLERANCE, f"Sign mismatch l={l}: err={err:.2e}"

        # Route stats also match
        stats_orig = router.route_stats(coeffs_orig)
        stats_trans = router.route_stats(coeffs_trans)
        for key in stats_orig:
            v_orig = stats_orig[key]
            v_trans = stats_trans[key]
            if isinstance(v_orig, (int, float)):
                assert abs(v_orig - v_trans) < TOLERANCE, \
                    f"Route stat {key}: {v_orig:.6f} vs {v_trans:.6f}"


class TestMTOTranslation:
    """MTOModeAssembly consumes h and coeffs, not positions → translation-invariant."""

    def test_mto_assembly_translation(self):
        model = _get_model()
        adapter = _get_adapter()
        z, pos = _make_molecule(6, seed=42)
        translation = torch.tensor([12.0, -8.0, 5.0])

        h_orig = _get_features(z, pos, model, adapter)
        h_trans = _get_features(z, pos + translation, model, adapter)

        router = SignedRouter(num_features=128, num_modes=8, maxl=3, normalization="l2")
        router.eval()
        mto = MTOModeAssembly(num_features=128, mode_channels=64, num_modes=8, maxl=3)
        mto.eval()

        with torch.no_grad():
            coeffs_orig = router(h_orig)
            coeffs_trans = router(h_trans)
            O_orig = mto(h_orig, coeffs_orig)
            O_trans = mto(h_trans, coeffs_trans)

        for l in O_orig:
            err = (O_trans[l] - O_orig[l]).abs().max().item()
            assert err < TOLERANCE, f"MTO output l={l}: err={err:.2e}"

    def test_mto_batched_translation(self):
        model = _get_model()
        adapter = _get_adapter()
        z1, pos1 = _make_molecule(4, seed=10)
        z2, pos2 = _make_molecule(3, seed=20)
        z = torch.cat([z1, z2])
        pos = torch.cat([pos1, pos2])
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
        translation = torch.tensor([5.0, 5.0, 5.0])

        with torch.no_grad():
            S_o, T_o = run_latent_forward(model, z=z, pos=pos, batch=batch)
            S_t, T_t = run_latent_forward(model, z=z, pos=pos + translation, batch=batch)

        h_orig = adapter(S_o, T_o)
        h_trans = adapter(S_t, T_t)

        router = SignedRouter(num_features=128, num_modes=8, maxl=3, normalization="l2")
        router.eval()
        mto = MTOModeAssembly(num_features=128, mode_channels=64, num_modes=8, maxl=3)
        mto.eval()

        with torch.no_grad():
            coeffs_orig = router(h_orig)
            coeffs_trans = router(h_trans)
            O_orig = mto(h_orig, coeffs_orig, batch=batch)
            O_trans = mto(h_trans, coeffs_trans, batch=batch)

        for l in O_orig:
            err = (O_trans[l] - O_orig[l]).abs().max().item()
            assert err < TOLERANCE, f"Batched MTO l={l}: err={err:.2e}"


class TestCGCouplingTranslation:
    """CG coupling only consumes O dict → translation-invariant."""

    def test_cg_minimal_translation(self):
        model = _get_model()
        adapter = _get_adapter()
        z, pos = _make_molecule(6, seed=42)
        translation = torch.tensor([8.0, -4.0, 11.0])

        h_orig = _get_features(z, pos, model, adapter)
        h_trans = _get_features(z, pos + translation, model, adapter)

        router = SignedRouter(num_features=128, num_modes=8, maxl=3, normalization="l2")
        router.eval()
        mto = MTOModeAssembly(num_features=128, mode_channels=64, num_modes=8, maxl=3)
        mto.eval()
        cg = CGCouplingMinimal(mode_channels=64)
        cg.eval()

        with torch.no_grad():
            coeffs_orig = router(h_orig)
            coeffs_trans = router(h_trans)
            O_orig = mto(h_orig, coeffs_orig)
            O_trans = mto(h_trans, coeffs_trans)
            C_orig = cg(O_orig)
            C_trans = cg(O_trans)

        for l in C_orig:
            err = (C_trans[l] - C_orig[l]).abs().max().item()
            assert err < TOLERANCE, f"CG Minimal output l={l}: err={err:.2e}"

    def test_cg_full_translation(self):
        """Full CGCoupling with scalar-conditioned weights."""
        model = _get_model()
        adapter = _get_adapter()
        z, pos = _make_molecule(5, seed=77)
        translation = torch.tensor([-5.0, 3.0, 9.0])

        h_orig = _get_features(z, pos, model, adapter)
        h_trans = _get_features(z, pos + translation, model, adapter)

        router = SignedRouter(num_features=128, num_modes=8, maxl=3, normalization="l2")
        router.eval()
        mto = MTOModeAssembly(num_features=128, mode_channels=64, num_modes=8, maxl=3)
        mto.eval()
        cg = CGCoupling(mode_channels=64, maxl=3, coupled_maxl=2)
        cg.eval()

        with torch.no_grad():
            coeffs_orig = router(h_orig)
            coeffs_trans = router(h_trans)
            O_orig = mto(h_orig, coeffs_orig)
            O_trans = mto(h_trans, coeffs_trans)
            C_orig = cg(O_orig)
            C_trans = cg(O_trans)

        for key in C_orig:
            err = (C_trans[key] - C_orig[key]).abs().max().item()
            assert err < TOLERANCE, (
                f"CG Full output key={key}: err={err:.2e}"
            )


class TestTensorGateTranslation:
    """Tensor gates only consume O dict → translation-invariant."""

    @pytest.mark.parametrize("gate_cls", [TensorGate, NoGate, ScalarOnlyGate])
    def test_gate_translation(self, gate_cls):
        model = _get_model()
        adapter = _get_adapter()
        z, pos = _make_molecule(6, seed=42)
        translation = torch.tensor([3.0, -9.0, 14.0])

        h_orig = _get_features(z, pos, model, adapter)
        h_trans = _get_features(z, pos + translation, model, adapter)

        router = SignedRouter(num_features=128, num_modes=8, maxl=3, normalization="l2")
        router.eval()
        mto = MTOModeAssembly(num_features=128, mode_channels=64, num_modes=8, maxl=3)
        mto.eval()
        gate = gate_cls(mode_channels=64, num_modes=8, maxl=3)
        gate.eval()

        with torch.no_grad():
            coeffs_orig = router(h_orig)
            coeffs_trans = router(h_trans)
            O_orig = mto(h_orig, coeffs_orig)
            O_trans = mto(h_trans, coeffs_trans)
            G_orig = gate(O_orig)
            G_trans = gate(O_trans)

        for l in G_orig:
            err = (G_trans[l] - G_orig[l]).abs().max().item()
            assert err < TOLERANCE, f"{gate_cls.__name__} output l={l}: err={err:.2e}"

    def test_gate_stats_reproducible(self):
        """Gate statistics must be reproducible under translation."""
        model = _get_model()
        adapter = _get_adapter()
        z, pos = _make_molecule(6, seed=42)
        translation = torch.tensor([20.0, -10.0, 30.0])

        h_orig = _get_features(z, pos, model, adapter)
        h_trans = _get_features(z, pos + translation, model, adapter)

        router = SignedRouter(num_features=128, num_modes=8, maxl=3, normalization="l2")
        router.eval()
        mto = MTOModeAssembly(num_features=128, mode_channels=64, num_modes=8, maxl=3)
        mto.eval()
        gate = TensorGate(mode_channels=64, num_modes=8, maxl=3)
        gate.eval()

        with torch.no_grad():
            coeffs_orig = router(h_orig)
            coeffs_trans = router(h_trans)
            O_orig = mto(h_orig, coeffs_orig)
            O_trans = mto(h_trans, coeffs_trans)
            gate(O_orig)
            stats_orig = gate.gate_stats(O_orig)
            gate(O_trans)
            stats_trans = gate.gate_stats(O_trans)

        for key in stats_orig:
            v_orig = stats_orig[key]
            v_trans = stats_trans[key]
            if isinstance(v_orig, (int, float)):
                assert abs(v_orig - v_trans) < TOLERANCE, (
                    f"Gate stat {key} differs: {v_orig:.6f} vs {v_trans:.6f}"
                )


class TestFullPipelineTranslation:
    """End-to-end: DetaNet → Adapter → Router → MTO → CG → Gate."""

    def test_full_pipeline_single_molecule(self):
        model = _get_model()
        adapter = _get_adapter()
        z, pos = _make_molecule(6, seed=42)
        translation = torch.tensor([7.0, -2.0, 13.0])

        router = SignedRouter(num_features=128, num_modes=8, maxl=3, normalization="l2")
        router.eval()
        mto = MTOModeAssembly(num_features=128, mode_channels=64, num_modes=8, maxl=3)
        mto.eval()
        cg = CGCouplingMinimal(mode_channels=64)
        cg.eval()
        gate = TensorGate(mode_channels=64, num_modes=8, maxl=3)
        gate.eval()

        with torch.no_grad():
            h_orig = _get_features(z, pos, model, adapter)
            coeffs_orig = router(h_orig)
            O_orig = mto(h_orig, coeffs_orig)
            C_orig = cg(O_orig)
            G_orig = gate(C_orig)

            h_trans = _get_features(z, pos + translation, model, adapter)
            coeffs_trans = router(h_trans)
            O_trans = mto(h_trans, coeffs_trans)
            C_trans = cg(O_trans)
            G_trans = gate(C_trans)

        for l in G_orig:
            err = (G_trans[l] - G_orig[l]).abs().max().item()
            assert err < TOLERANCE, (
                f"Full pipeline output l={l}: err={err:.2e}"
            )

    def test_full_pipeline_multiple_translations(self):
        model = _get_model()
        adapter = _get_adapter()
        z, pos = _make_molecule(7, seed=999)

        router = SignedRouter(num_features=128, num_modes=8, maxl=3, normalization="l2")
        router.eval()
        mto = MTOModeAssembly(num_features=128, mode_channels=64, num_modes=8, maxl=3)
        mto.eval()
        cg = CGCouplingMinimal(mode_channels=64)
        cg.eval()
        gate = TensorGate(mode_channels=64, num_modes=8, maxl=3)
        gate.eval()

        for t_idx in range(5):
            translation = TRANSLATIONS[t_idx]
            with torch.no_grad():
                h = _get_features(z, pos + translation, model, adapter)
                coeffs = router(h)
                O = mto(h, coeffs)
                C = cg(O)
                G = gate(C)

            # Verify against the zero-translation baseline
            with torch.no_grad():
                h0 = _get_features(z, pos, model, adapter)
                coeffs0 = router(h0)
                O0 = mto(h0, coeffs0)
                C0 = cg(O0)
                G0 = gate(C0)

            for l in G0:
                err = (G[l] - G0[l]).abs().max().item()
                assert err < TOLERANCE, (
                    f"Pipeline translation[{t_idx}] l={l}: err={err:.2e}"
                )
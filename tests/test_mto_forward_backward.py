"""Smoke test: full MTO-Net forward/backward with DetaNet tensor features.

Tests:
  - Full tensor MTO forward pass
  - Backward pass (gradient flow)
  - No NaN/Inf in outputs or gradients
  - Shapes are valid throughout pipeline
  - Scalar-only MTO ablation
  - Config variants (no CG, no gate, no signed routing)
"""

import pytest
import torch

torch.serialization.add_safe_globals([slice])

from ar_mto.detanet_bridge import make_latent_detanet, run_latent_forward
from ar_mto.tensor_adapter import make_adapter
from ar_mto.signed_routing import SignedRouter
from ar_mto.mto_core import MTOModeAssembly, ScalarOnlyMTO
from ar_mto.cg_coupling import CGCouplingMinimal
from ar_mto.tensor_gate import TensorGate, NoGate
from ar_mto.readouts import ScalarReadout, VectorReadout, Rank2TensorReadout, SpectralReadout
from ar_mto.mto_net import MTOConfig, MTONet, make_mto_net


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


class TestMTOForwardBackward:
    """Full pipeline forward/backward smoke tests."""

    @pytest.fixture(scope="class")
    def detanet_model(self):
        return make_latent_detanet(num_block=2, device="cpu")

    def test_full_pipeline_forward(self, detanet_model):
        """End-to-end forward: DetaNet → adapter → routing → MTO → CG → gate → readouts."""
        z, pos = _make_molecule(5)
        with torch.no_grad():
            S, T = run_latent_forward(detanet_model, z=z, pos=pos)

        adapter = make_adapter()
        h = adapter(S, T)

        router = SignedRouter(num_features=128, num_modes=8, maxl=3)
        coeffs = router(h)

        mto = MTOModeAssembly(num_features=128, mode_channels=64,
                              num_modes=8, maxl=3)
        O = mto(h, coeffs)  # [1, K, C, 2l+1]

        cg = CGCouplingMinimal(mode_channels=64)
        O_coupled = cg(O)
        # Merge: CG handles l=0,1,2; keep original l=3
        O_full = {0: O_coupled[0], 1: O_coupled[1], 2: O_coupled[2], 3: O[3]}

        gate = TensorGate(mode_channels=64, num_modes=8, maxl=3)
        O_full = gate(O_full)

        scalar_readout = ScalarReadout(mode_channels=64, num_modes=8)
        y = scalar_readout(O_full)

        assert y.shape == (1, 1), f"Scalar output shape: {y.shape}"
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

        for key, val in O_full.items():
            assert not torch.isnan(val).any(), f"NaN in O[{key}]"
            assert not torch.isinf(val).any(), f"Inf in O[{key}]"

    def test_full_pipeline_backward(self, detanet_model):
        """Gradients flow through the entire pipeline."""
        z, pos = _make_molecule(5, seed=100)
        z = z.clone()
        pos = pos.clone().requires_grad_(False)

        adapter = make_adapter()
        router = SignedRouter(num_features=128, num_modes=4, maxl=3)
        mto = MTOModeAssembly(num_features=128, mode_channels=32,
                              num_modes=4, maxl=3)
        gate = TensorGate(mode_channels=32, num_modes=4, maxl=3)
        readout = ScalarReadout(mode_channels=32, num_modes=4)

        mto_params = list(mto.parameters())
        gate_params = list(gate.parameters())
        router_params = list(router.parameters())

        S, T = run_latent_forward(detanet_model, z=z, pos=pos)

        S = S.detach().requires_grad_(True)
        T = T.detach().requires_grad_(True)

        h = adapter(S, T)
        coeffs = router(h)
        O = mto(h, coeffs)
        O = gate(O)
        y = readout(O)

        loss = y.sum()
        loss.backward()

        for name, params in [("MTO", mto_params), ("Gate", gate_params),
                              ("Router", router_params)]:
            for p in params:
                if p.grad is not None:
                    assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"
                    assert not torch.isinf(p.grad).any(), f"Inf grad in {name}"

        assert S.grad is not None
        assert T.grad is not None
        assert not torch.isnan(S.grad).any()
        assert not torch.isinf(S.grad).any()

    def test_gradient_magnitude_sane(self, detanet_model):
        """Gradients should not explode."""
        z, pos = _make_molecule(5, seed=200)
        z = z.clone()

        mto = MTOModeAssembly(num_features=128, mode_channels=32,
                              num_modes=4, maxl=3)
        router = SignedRouter(num_features=128, num_modes=4, maxl=3)
        gate = TensorGate(mode_channels=32, num_modes=4, maxl=3)
        readout = ScalarReadout(mode_channels=32, num_modes=4)

        S, T = run_latent_forward(detanet_model, z=z, pos=pos)
        S = S.detach().requires_grad_(True)
        T = T.detach().requires_grad_(True)

        adapter = make_adapter()
        h = adapter(S, T)
        coeffs = router(h)
        O = mto(h, coeffs)
        O = gate(O)
        y = readout(O)

        loss = y.sum()
        loss.backward()

        for p in mto.parameters():
            if p.grad is not None:
                grad_norm = p.grad.norm().item()
                assert grad_norm < 100.0, \
                    f"Gradient too large: {grad_norm}"

    def test_no_nan_during_training_simulated(self, detanet_model):
        """Simulate a few "training steps" checking for NaN."""
        mto = MTOModeAssembly(num_features=128, mode_channels=32,
                              num_modes=4, maxl=3)
        router = SignedRouter(num_features=128, num_modes=4, maxl=3)
        gate = TensorGate(mode_channels=32, num_modes=4, maxl=3)
        readout = ScalarReadout(mode_channels=32, num_modes=4)

        opt = torch.optim.Adam(
            list(mto.parameters()) + list(router.parameters()) +
            list(gate.parameters()) + list(readout.parameters()),
            lr=1e-3,
        )

        for step in range(5):
            opt.zero_grad()
            z, pos = _make_molecule(5, seed=step)
            S, T = run_latent_forward(detanet_model, z=z, pos=pos)
            S = S.detach()
            T = T.detach()

            adapter = make_adapter()
            h = adapter(S, T)
            coeffs = router(h)
            O = mto(h, coeffs)
            O = gate(O)
            y = readout(O)

            loss = y.sum()
            loss.backward()
            opt.step()

            assert not torch.isnan(y)
            assert not torch.isinf(y)
            for p in mto.parameters():
                assert not torch.isnan(p).any(), f"NaN in MTO params at step {step}"


class TestScalarOnlyAblation:
    """Scalar-only MTO as ablation baseline."""

    @pytest.fixture(scope="class")
    def detanet_model(self):
        return make_latent_detanet(num_block=2, device="cpu")

    def test_scalar_only_forward(self, detanet_model):
        """Scalar-only MTO uses only l=0 features."""
        z, pos = _make_molecule(5)
        with torch.no_grad():
            S, T = run_latent_forward(detanet_model, z=z, pos=pos)

        adapter = make_adapter()
        h = adapter(S, T)

        router = SignedRouter(num_features=128, num_modes=8,
                              use_tensor_norms=False, maxl=0)
        coeffs = router({0: h[0]})

        mto = ScalarOnlyMTO(num_features=128, mode_channels=64, num_modes=8)
        O = mto({0: h[0]}, coeffs)

        gate = TensorGate(mode_channels=64, num_modes=8, maxl=0,
                          use_tensor_info=False)
        O = gate(O)

        readout = ScalarReadout(mode_channels=64, num_modes=8)
        y = readout(O)

        assert y.shape == (1, 1)
        assert not torch.isnan(y).any()
        assert list(O.keys()) == [0]

    def test_scalar_only_gradient_flow(self, detanet_model):
        """Scalar-only MTO supports gradient flow."""
        z, pos = _make_molecule(5)
        S, T = run_latent_forward(detanet_model, z=z, pos=pos)

        adapter = make_adapter()
        h = adapter(S, T)

        router = SignedRouter(num_features=128, num_modes=4,
                              use_tensor_norms=False, maxl=0)
        mto = ScalarOnlyMTO(num_features=128, mode_channels=32, num_modes=4)
        readout = ScalarReadout(mode_channels=32, num_modes=4)

        h0 = h[0].detach().requires_grad_(True)
        coeffs = router({0: h0})
        O = mto({0: h0}, coeffs)
        y = readout(O)

        loss = y.sum()
        loss.backward()

        assert h0.grad is not None
        assert not torch.isnan(h0.grad).any()


class TestConfigVariants:
    """Test different config ablation variants."""

    @pytest.fixture(scope="class")
    def detanet_model(self):
        return make_latent_detanet(num_block=2, device="cpu")

    def test_no_cg_variant(self, detanet_model):
        """MTO without CG coupling should still work."""
        z, pos = _make_molecule(5)
        with torch.no_grad():
            S, T = run_latent_forward(detanet_model, z=z, pos=pos)

        adapter = make_adapter()
        h = adapter(S, T)

        router = SignedRouter(num_features=128, num_modes=4, maxl=3)
        coeffs = router(h)

        mto = MTOModeAssembly(num_features=128, mode_channels=32,
                              num_modes=4, maxl=3)
        O = mto(h, coeffs)

        gate = TensorGate(mode_channels=32, num_modes=4, maxl=3)
        O = gate(O)

        readout = ScalarReadout(mode_channels=32, num_modes=4)
        y = readout(O)

        assert not torch.isnan(y).any()

    def test_no_gate_variant(self, detanet_model):
        """MTO without tensor gate should still work."""
        z, pos = _make_molecule(5)
        with torch.no_grad():
            S, T = run_latent_forward(detanet_model, z=z, pos=pos)

        adapter = make_adapter()
        h = adapter(S, T)

        router = SignedRouter(num_features=128, num_modes=4, maxl=3)
        coeffs = router(h)

        mto = MTOModeAssembly(num_features=128, mode_channels=32,
                              num_modes=4, maxl=3)
        O = mto(h, coeffs)

        gate = NoGate(mode_channels=32, num_modes=4, maxl=3)
        O = gate(O)

        readout = ScalarReadout(mode_channels=32, num_modes=4)
        y = readout(O)

        assert not torch.isnan(y).any()

    def test_no_signed_routing_variant(self, detanet_model):
        """MTO without signed routing (uniform weights) should still work."""
        z, pos = _make_molecule(5)
        with torch.no_grad():
            S, T = run_latent_forward(detanet_model, z=z, pos=pos)

        adapter = make_adapter()
        h = adapter(S, T)

        N = h[0].shape[0]
        K = 4
        uniform_coeff = torch.ones(K, N, 1) / N

        mto = MTOModeAssembly(num_features=128, mode_channels=32,
                              num_modes=K, maxl=3)
        coeffs = {l: uniform_coeff for l in range(4)}
        O = mto(h, coeffs)

        readout = ScalarReadout(mode_channels=32, num_modes=K)
        y = readout(O)

        assert not torch.isnan(y).any()


class TestFullMTONet:
    """End-to-end tests using the MTONet wrapper."""

    @pytest.fixture(scope="class")
    def detanet_model(self):
        return make_latent_detanet(num_block=2, device="cpu")

    def test_make_mto_net_full(self, detanet_model):
        """Full MTONet forward with all readout heads."""
        model = make_mto_net(
            detanet_model=detanet_model,
            num_features=128,
            num_modes=8,
            mode_channels=64,
            maxl=3,
            active_heads=["scalar", "vector", "rank2", "spectral"],
            spectral_bins=100,
        )

        z, pos = _make_molecule(5)
        with torch.no_grad():
            result = model(z=z, pos=pos)

        assert result["scalar"].shape == (1, 1)
        assert result["vector"].shape == (1, 1, 3)
        assert result["tensor"].shape == (1, 1, 3, 3)
        assert result["spectrum"].shape == (1, 100)

        for v in result.values():
            if isinstance(v, torch.Tensor):
                assert not torch.isnan(v).any()

    def test_make_mto_net_vector_head(self, detanet_model):
        """MTONet with only vector head."""
        model = make_mto_net(
            detanet_model=detanet_model,
            num_features=128, num_modes=4, mode_channels=32, maxl=3,
            active_heads=["vector"],
        )
        z, pos = _make_molecule(5)
        with torch.no_grad():
            result = model(z=z, pos=pos)
        assert result["vector"].shape == (1, 1, 3)
        assert "scalar" not in result
        assert "tensor" not in result
        assert "spectrum" not in result

    def test_make_mto_net_dim_selection(self):
        """MTONet respects active_heads configuration."""
        model = make_mto_net(
            num_features=128, num_modes=4, mode_channels=32, maxl=3,
            active_heads=["scalar", "rank2"],
        )
        z, pos = _make_molecule(3)
        with torch.no_grad():
            result = model(z=z, pos=pos)
        assert "scalar" in result
        assert "tensor" in result
        assert "vector" not in result
        assert "spectrum" not in result

    def test_forward_with_diagnostics(self, detanet_model):
        """MTONet returns diagnostics on request."""
        model = make_mto_net(
            detanet_model=detanet_model,
            num_features=128, num_modes=4, mode_channels=32, maxl=3,
            active_heads=["scalar"],
        )
        z, pos = _make_molecule(5)
        with torch.no_grad():
            result = model(z=z, pos=pos, return_diagnostics=True)
        diag = result["diagnostics"]
        assert "route_stats" in diag
        assert "gate_stats" in diag
        assert "gate_l0_mean" in diag["gate_stats"]

    def test_forward_return_modes(self, detanet_model):
        """MTONet can return assembled modes."""
        model = make_mto_net(
            detanet_model=detanet_model,
            num_features=128, num_modes=4, mode_channels=32, maxl=3,
            active_heads=["scalar"],
        )
        z, pos = _make_molecule(5)
        with torch.no_grad():
            result = model(z=z, pos=pos, return_modes=True)
        assert "modes" in result
        O = result["modes"]
        for l in [0, 1, 2, 3]:
            assert l in O, f"Missing l={l} in returned modes"
            assert O[l].shape[0] == 1, f"Batch dim mismatch for l={l}"

    def test_valence_adaptive_k_forward(self, detanet_model):
        """MTONet forward_with_adaptive_k produces valid outputs."""
        model = make_mto_net(
            detanet_model=detanet_model,
            num_features=128, num_modes=16, mode_channels=32, maxl=3,
            k_policy="valence_adaptive",
            active_heads=["scalar"],
        )
        z, pos = _make_molecule(5)
        with torch.no_grad():
            result = model.forward_with_adaptive_k(z=z, pos=pos)
        assert result["scalar"].shape[0] == 1

    def test_batch_forward(self, detanet_model):
        """MTONet handles batched molecules."""
        model = make_mto_net(
            detanet_model=detanet_model,
            num_features=128, num_modes=8, mode_channels=32, maxl=3,
            active_heads=["scalar", "vector", "rank2", "spectral"],
            spectral_bins=100,
        )
        z1, pos1 = _make_molecule(5, seed=10)
        z2, pos2 = _make_molecule(4, seed=20)

        z = torch.cat([z1, z2])
        pos = torch.cat([pos1, pos2])
        batch = torch.tensor([0] * 5 + [1] * 4, dtype=torch.long)

        with torch.no_grad():
            result = model(z=z, pos=pos, batch=batch)

        assert result["scalar"].shape == (2, 1)
        assert result["vector"].shape == (2, 1, 3)
        assert result["tensor"].shape == (2, 1, 3, 3)
        assert result["spectrum"].shape == (2, 100)

    def test_batch_isolation(self, detanet_model):
        """Molecule A alone vs in batch must give same predictions."""
        model = make_mto_net(
            detanet_model=detanet_model,
            num_features=128, num_modes=8, mode_channels=32, maxl=3,
            active_heads=["scalar", "vector", "rank2", "spectral"],
            spectral_bins=100,
        )

        zA, posA = _make_molecule(5, seed=10)
        zB, posB = _make_molecule(4, seed=20)

        # Forward A alone
        with torch.no_grad():
            result_A = model(z=zA, pos=posA)

        # Forward A alongside B
        z = torch.cat([zA, zB])
        pos = torch.cat([posA, posB])
        batch = torch.tensor([0] * 5 + [1] * 4, dtype=torch.long)
        with torch.no_grad():
            result_batch = model(z=z, pos=pos, batch=batch)

        for key in ["scalar", "vector", "tensor", "spectrum"]:
            assert torch.allclose(result_batch[key][0:1], result_A[key], atol=1e-4), \
                f"Batch isolation failed for {key}"

    def test_checkpoint_save_load(self, detanet_model, tmp_path):
        """MTONet checkpoint save/load preserves inference results."""
        model = make_mto_net(
            detanet_model=detanet_model,
            num_features=128, num_modes=4, mode_channels=32, maxl=3,
            active_heads=["scalar", "vector", "rank2", "spectral"],
            spectral_bins=100,
        )
        model.eval()
        z, pos = _make_molecule(5)

        with torch.no_grad():
            result_before = model(z=z, pos=pos)

        # Save
        ckpt_path = tmp_path / "model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": model.config.to_dict(),
        }, ckpt_path)

        # Load into a fresh model
        loaded = make_mto_net(
            detanet_model=detanet_model,
            **model.config.to_dict(),
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        loaded.load_state_dict(ckpt["model_state_dict"])
        loaded.eval()

        with torch.no_grad():
            result_after = loaded(z=z, pos=pos)

        for key in ["scalar", "vector", "tensor", "spectrum"]:
            assert torch.allclose(result_before[key], result_after[key], atol=1e-6), \
                f"Checkpoint mismatch for {key}"

    def test_gradient_flow_all_heads(self, detanet_model):
        """Gradients flow through all readout heads."""
        model = make_mto_net(
            detanet_model=detanet_model,
            num_features=128, num_modes=4, mode_channels=32, maxl=3,
            active_heads=["scalar", "vector", "rank2", "spectral"],
            spectral_bins=100,
        )
        z, pos = _make_molecule(5)

        result = model(z=z, pos=pos)
        loss = (
            result["scalar"].sum()
            + result["vector"].sum()
            + result["tensor"].sum()
            + result["spectrum"].sum()
        )
        loss.backward()

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params_with_grad = sum(
            1 for p in model.parameters()
            if p.requires_grad and p.grad is not None
        )
        assert params_with_grad > 0, "No parameters received gradients"
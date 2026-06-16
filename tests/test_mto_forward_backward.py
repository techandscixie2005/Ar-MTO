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
from ar_mto.readouts import ScalarReadout


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
        """End-to-end forward pass: DetaNet → adapter → routing → MTO → readout."""
        z, pos = _make_molecule(5)
        with torch.no_grad():
            S, T = run_latent_forward(detanet_model, z=z, pos=pos)

        adapter = make_adapter()
        h = adapter(S, T)

        router = SignedRouter(num_features=128, num_modes=8, maxl=3)
        coeffs = router(h)

        mto = MTOModeAssembly(num_features=128, mode_channels=64,
                              num_modes=8, maxl=3)
        O = mto(h, coeffs)

        cg = CGCouplingMinimal(mode_channels=64)
        O_coupled = cg(O)
        # Merge: CG coupling handles l=0,1,2; keep original l=3
        O_full = {0: O_coupled[0], 1: O_coupled[1], 2: O_coupled[2], 3: O[3]}

        gate = TensorGate(mode_channels=64, num_modes=8, maxl=3)
        O_full = gate(O_full)

        scalar_readout = ScalarReadout(mode_channels=64, num_modes=8)
        y = scalar_readout(O_full)

        # Check shapes
        assert y.numel() == 1, f"Scalar output has wrong shape: {y.shape}"

        # Check no NaN/Inf
        for l in [0, 1, 2, 3]:
            assert not torch.isnan(O_full[l]).any(), f"NaN in O[{l}]"
            assert not torch.isinf(O_full[l]).any(), f"Inf in O[{l}]"
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_full_pipeline_backward(self, detanet_model):
        """Gradients flow through the entire pipeline."""
        z, pos = _make_molecule(5, seed=100)
        z = z.clone()
        pos = pos.clone().requires_grad_(False)

        # Make all MTO params trainable
        adapter = make_adapter()
        router = SignedRouter(num_features=128, num_modes=4, maxl=3)
        mto = MTOModeAssembly(num_features=128, mode_channels=32,
                              num_modes=4, maxl=3)
        gate = TensorGate(mode_channels=32, num_modes=4, maxl=3)
        readout = ScalarReadout(mode_channels=32, num_modes=4)

        # Track gradient flow through MTO params
        mto_params = list(mto.parameters())
        gate_params = list(gate.parameters())
        router_params = list(router.parameters())

        S, T = run_latent_forward(detanet_model, z=z, pos=pos)

        # Detach S,T from DetaNet (we're testing MTO grad flow specifically)
        S = S.detach().requires_grad_(True)
        T = T.detach().requires_grad_(True)

        h = adapter(S, T)
        coeffs = router(h)
        O = mto(h, coeffs)
        O = gate(O)
        y = readout(O)

        loss = y.sum()
        loss.backward()

        # Check gradients exist and are finite
        for name, params in [("MTO", mto_params), ("Gate", gate_params),
                              ("Router", router_params)]:
            for p in params:
                if p.grad is not None:
                    assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"
                    assert not torch.isinf(p.grad).any(), f"Inf grad in {name}"

        # Input gradients should exist
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

        assert y.numel() == 1
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

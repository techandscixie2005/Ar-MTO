"""Test DetaNet forward pass with synthetic molecular inputs.

Uses tiny synthetic molecules (no real data required) to verify:
- Forward pass runs without error
- Latent output returns both scalar (S) and tensor (T) features
- Shapes are consistent with DetaNet architecture
- No NaN or inf in outputs

All forward passes use run_latent_forward() to remain compatible with
environments where pyg-lib / torch_geometric.radius_graph is unavailable.
"""

import pytest
import torch
from ar_mto.detanet_bridge import make_latent_detanet, run_latent_forward


@pytest.fixture(scope="module")
def latent_model():
    return make_latent_detanet(num_block=2, device="cpu")


def _water_molecule():
    """Return (z, pos) for a synthetic H2O-like molecule."""
    z = torch.tensor([8, 1, 1], dtype=torch.long)  # O, H, H
    pos = torch.tensor(
        [
            [0.0000, 0.0000, 0.1173],
            [0.0000, 0.7572, -0.4692],
            [0.0000, -0.7572, -0.4692],
        ],
        dtype=torch.float32,
    )
    return z, pos


def _methane_molecule():
    """Return (z, pos) for a synthetic CH4-like molecule."""
    z = torch.tensor([6, 1, 1, 1, 1], dtype=torch.long)  # C, H, H, H, H
    pos = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000],
            [0.6287, 0.6287, 0.6287],
            [-0.6287, -0.6287, 0.6287],
            [-0.6287, 0.6287, -0.6287],
            [0.6287, -0.6287, -0.6287],
        ],
        dtype=torch.float32,
    )
    return z, pos


class TestLatentForward:
    """Forward pass tests for DetaNet in latent mode."""

    def test_single_molecule_forward(self, latent_model):
        z, pos = _water_molecule()
        S, T = run_latent_forward(latent_model, z=z, pos=pos)

        assert S.shape == (3, 128), f"Scalar shape mismatch: {S.shape}"
        assert S.dtype == torch.float32
        assert not torch.isnan(S).any(), "NaN in scalar output"
        assert not torch.isinf(S).any(), "Inf in scalar output"

        vdim = latent_model.vdim
        assert T.shape == (3, vdim), f"Tensor shape mismatch: {T.shape}"
        assert T.dtype == torch.float32
        assert not torch.isnan(T).any(), "NaN in tensor output"
        assert not torch.isinf(T).any(), "Inf in tensor output"

    def test_batched_molecules_forward(self, latent_model):
        z1, pos1 = _water_molecule()
        z2, pos2 = _methane_molecule()

        z = torch.cat([z1, z2])
        pos = torch.cat([pos1, pos2])
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)

        S, T = run_latent_forward(latent_model, z=z, pos=pos, batch=batch)

        assert S.shape == (8, 128)
        assert T.shape[0] == 8
        assert T.shape[1] == latent_model.vdim
        assert not torch.isnan(S).any()
        assert not torch.isnan(T).any()

    def test_single_atom_not_supported(self, latent_model):
        """DetaNet requires >= 2 atoms to form an edge graph.

        Single-atom molecules are fundamentally unsupported by the GNN
        architecture - there are no interatomic edges to propagate messages.
        This is a known limitation, not a bug.
        """
        z = torch.tensor([6], dtype=torch.long)
        pos = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        with pytest.raises(RuntimeError):
            run_latent_forward(latent_model, z=z, pos=pos)

    def test_dimer_minimal(self, latent_model):
        """Two-atom dimer should work fine."""
        z = torch.tensor([6, 6], dtype=torch.long)
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float32)
        S, T = run_latent_forward(latent_model, z=z, pos=pos)
        assert S.shape == (2, 128)
        assert T.shape[0] == 2

    def test_large_molecule(self, latent_model):
        """30-atom synthetic chain."""
        n = 30
        z = torch.ones(n, dtype=torch.long) * 6  # all carbon
        pos = torch.randn(n, 3, dtype=torch.float32)
        S, T = run_latent_forward(latent_model, z=z, pos=pos)
        assert S.shape == (n, 128)
        assert T.shape[0] == n

    def test_deterministic(self, latent_model):
        """Same input should produce identical output in eval mode."""
        latent_model.eval()
        z, pos = _water_molecule()
        with torch.no_grad():
            S1, T1 = run_latent_forward(latent_model, z=z, pos=pos)
            S2, T2 = run_latent_forward(latent_model, z=z, pos=pos)

        assert torch.allclose(S1, S2, atol=1e-7)
        assert torch.allclose(T1, T2, atol=1e-7)

    def test_maxl_configs(self):
        """Test different maxl values produce expected tensor dimensions."""
        z, pos = _water_molecule()

        for maxl in [1, 2, 3]:
            model = make_latent_detanet(maxl=maxl, num_block=1, device="cpu")
            S, T = run_latent_forward(model, z=z, pos=pos)
            assert S.shape == (3, 128)

            expected_vdim = 128 * sum(2 * l + 1 for l in range(1, maxl + 1))
            assert T.shape == (3, expected_vdim), (
                f"maxl={maxl}: expected vdim={expected_vdim}, got {T.shape[1]}"
            )


class TestEdgeCases:
    """Edge case handling."""

    def test_no_batch_single_molecule(self, latent_model):
        """Single molecule without batch tensor should work."""
        z, pos = _water_molecule()
        S, T = run_latent_forward(latent_model, z=z, pos=pos)
        assert S.shape == (3, 128)

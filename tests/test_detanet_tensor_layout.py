"""Test DetaNet latent tensor layout — verify S/T shapes and channel structure."""

import sys
import os
import pytest

import torch

torch.serialization.add_safe_globals([slice])

from e3nn import o3  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "DetaNet"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detanet_model import DetaNet  # noqa: E402
from ar_mto.detanet_bridge import compute_radius_edges  # noqa: E402


def _run_forward(model, z, pos, batch=None):
    edge_index = compute_radius_edges(pos=pos, rc=model.rc, batch=batch)
    return model(z=z, pos=pos, edge_index=edge_index, batch=batch)


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


def get_tensor_layout(model):
    maxl = max(ir.l for _, ir in model.irreps_sh)
    irreps_T = o3.Irreps((model.features, (l, (-1) ** l)) for l in range(1, maxl + 1))
    blocks = []
    offset = 0
    for mul, (l, p) in irreps_T:
        total_dim = mul * (2 * l + 1)
        blocks.append({"l": l, "parity": "even" if p == 1 else "odd",
                        "multiplicity": mul, "total_dim": total_dim,
                        "flat_start": offset, "flat_end": offset + total_dim})
        offset += total_dim
    return {"irreps_str": str(irreps_T), "total_vdim": irreps_T.dim,
            "scalar_dim": model.features, "blocks": blocks}


class TestTensorLayout:
    def test_model_instantiation(self):
        model = DetaNet(num_features=128, maxl=3, out_type="latent", device="cpu")
        assert model.features == 128
        assert model.vdim == 1920
        assert model.out_type == "latent"

    def test_irreps_string(self):
        layout = get_tensor_layout(DetaNet(num_features=128, maxl=3, out_type="latent", device="cpu"))
        assert layout["irreps_str"] == "128x1o+128x2e+128x3o"
        assert layout["total_vdim"] == 1920
        assert layout["scalar_dim"] == 128

    def test_block_structure(self):
        layout = get_tensor_layout(DetaNet(num_features=128, maxl=3, out_type="latent", device="cpu"))
        blocks = layout["blocks"]
        assert len(blocks) == 3

        assert blocks[0]["l"] == 1
        assert blocks[0]["parity"] == "odd"
        assert blocks[0]["multiplicity"] == 128
        assert blocks[0]["total_dim"] == 384
        assert blocks[0]["flat_start"] == 0
        assert blocks[0]["flat_end"] == 384

        assert blocks[1]["l"] == 2
        assert blocks[1]["parity"] == "even"
        assert blocks[1]["multiplicity"] == 128
        assert blocks[1]["total_dim"] == 640
        assert blocks[1]["flat_start"] == 384
        assert blocks[1]["flat_end"] == 1024

        assert blocks[2]["l"] == 3
        assert blocks[2]["parity"] == "odd"
        assert blocks[2]["multiplicity"] == 128
        assert blocks[2]["total_dim"] == 896
        assert blocks[2]["flat_start"] == 1024
        assert blocks[2]["flat_end"] == 1920

    def test_forward_shapes(self):
        model = DetaNet(num_features=128, maxl=3, out_type="latent", device="cpu")
        z, pos = _make_molecule(5)
        with torch.no_grad():
            S, T = _run_forward(model, z, pos)
        assert S.shape == (5, 128), f"Expected S shape (5, 128), got {S.shape}"
        assert T.shape == (5, 1920), f"Expected T shape (5, 1920), got {T.shape}"

    def test_forward_shapes_variable_atoms(self):
        model = DetaNet(num_features=128, maxl=3, out_type="latent", device="cpu")
        for n in [3, 4, 6, 8]:
            z, pos = _make_molecule(n, seed=n)
            with torch.no_grad():
                S, T = _run_forward(model, z, pos)
            assert S.shape == (n, 128)
            assert T.shape == (n, 1920)

    def test_no_nan(self):
        model = DetaNet(num_features=128, maxl=3, out_type="latent", device="cpu")
        z, pos = _make_molecule(5)
        with torch.no_grad():
            S, T = _run_forward(model, z, pos)
        assert not torch.isnan(S).any()
        assert not torch.isinf(S).any()
        assert not torch.isnan(T).any()
        assert not torch.isinf(T).any()

    def test_l_0_not_in_T(self):
        """S (l=0) scalars live in S, not in T's irrep blocks."""
        model = DetaNet(num_features=128, maxl=3, out_type="latent", device="cpu")
        layout = get_tensor_layout(model)
        for b in layout["blocks"]:
            assert b["l"] >= 1, "T should not contain l=0 irreps (those are in S)"

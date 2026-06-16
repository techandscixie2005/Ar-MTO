"""Test T split/reconstruct: exact recovery and dimension correctness."""

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


LAYOUT = None


def _get_layout():
    global LAYOUT
    if LAYOUT is None:
        model = DetaNet(num_features=128, maxl=3, out_type="latent", device="cpu", summation=False, scale=None)
        maxl = max(ir.l for _, ir in model.irreps_sh)
        irreps_T = o3.Irreps((model.features, (l, (-1) ** l)) for l in range(1, maxl + 1))
        blocks = []
        offset = 0
        for mul, (l, p) in irreps_T:
            dim = mul * (2 * l + 1)
            blocks.append({"l": l, "multiplicity": mul, "total_dim": dim,
                           "flat_start": offset, "flat_end": offset + dim})
            offset += dim
        LAYOUT = {"total_vdim": irreps_T.dim, "blocks": blocks}
    return LAYOUT


def split_T(T):
    layout = _get_layout()
    blocks = {}
    for b in layout["blocks"]:
        l = b["l"]
        sliced = T[:, b["flat_start"]:b["flat_end"]]
        blocks[l] = sliced.reshape(T.shape[0], b["multiplicity"], 2 * l + 1)
    return blocks


def reconstruct_T(blocks):
    layout = _get_layout()
    num_atoms = next(iter(blocks.values())).shape[0]
    T = torch.zeros(num_atoms, layout["total_vdim"], dtype=next(iter(blocks.values())).dtype)
    for b in layout["blocks"]:
        l = b["l"]
        T[:, b["flat_start"]:b["flat_end"]] = blocks[l].reshape(num_atoms, b["total_dim"])
    return T


class TestSplitReconstruct:
    def test_split_dimensions(self):
        model = DetaNet(num_features=128, maxl=3, out_type="latent", device="cpu", summation=False, scale=None)
        z, pos = _make_molecule(5)
        with torch.no_grad():
            _S, T = _run_forward(model, z, pos)
        blocks = split_T(T)
        assert blocks[1].shape == (5, 128, 3), f"h1 shape wrong: {blocks[1].shape}"
        assert blocks[2].shape == (5, 128, 5), f"h2 shape wrong: {blocks[2].shape}"
        assert blocks[3].shape == (5, 128, 7), f"h3 shape wrong: {blocks[3].shape}"

    @pytest.mark.parametrize("n_atoms", [3, 4, 5, 6, 8])
    def test_exact_reconstruction(self, n_atoms):
        model = DetaNet(num_features=128, maxl=3, out_type="latent", device="cpu", summation=False, scale=None)
        z, pos = _make_molecule(n_atoms, seed=n_atoms * 10)
        with torch.no_grad():
            _S, T = _run_forward(model, z, pos)
        blocks = split_T(T)
        T_recon = reconstruct_T(blocks)
        assert torch.allclose(T, T_recon, atol=1e-7), \
            f"Reconstruction failed for {n_atoms} atoms"

    def test_reconstruction_exact_zero_error(self):
        """Reconstruction should be exact (identity slices)."""
        model = DetaNet(num_features=128, maxl=3, out_type="latent", device="cpu", summation=False, scale=None)
        z, pos = _make_molecule(5)
        with torch.no_grad():
            _S, T = _run_forward(model, z, pos)
        blocks = split_T(T)
        T_recon = reconstruct_T(blocks)
        err = (T - T_recon).abs().max().item()
        assert err == 0.0, f"Reconstruction should be exact, got err={err}"

    def test_variable_atom_counts(self):
        """Split/reconstruct should work for any number of atoms."""
        model = DetaNet(num_features=128, maxl=3, out_type="latent", device="cpu", summation=False, scale=None)
        for n in [3, 4, 6, 7, 8, 10]:
            z, pos = _make_molecule(n, seed=n)
            with torch.no_grad():
                _S, T = _run_forward(model, z, pos)
            blocks = split_T(T)
            T_recon = reconstruct_T(blocks)
            assert (T - T_recon).abs().max().item() == 0.0

"""Test Wigner-D rotation equivariance and parity of DetaNet tensor features."""

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


TOLERANCE = 5e-5


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


def _split_T(T):
    # Layout: 128x1o (384) + 128x2e (640) + 128x3o (896)
    return {
        1: T[:, 0:384].reshape(-1, 128, 3),
        2: T[:, 384:1024].reshape(-1, 128, 5),
        3: T[:, 1024:1920].reshape(-1, 128, 7),
    }


class TestRotationEquivariance:
    @pytest.mark.parametrize("l", [1, 2, 3])
    def test_wigner_d_equivariance(self, l):
        model = DetaNet(num_features=128, maxl=3, out_type="latent", device="cpu", summation=False, scale=None)
        z, pos = _make_molecule(5, seed=42)
        R = _random_rotation(seed=100)
        pos_rot = pos @ R.T

        with torch.no_grad():
            _, T_orig = _run_forward(model, z, pos)
            _, T_rot = _run_forward(model, z, pos_rot)

        blocks_orig = _split_T(T_orig)
        blocks_rot = _split_T(T_rot)

        D = _wigner_D(l, R)
        h_orig = blocks_orig[l]  # [n, C, 2l+1]
        h_rot = blocks_rot[l]    # [n, C, 2l+1]
        h_rot_pred = torch.einsum("ab,ncb->nca", D, h_orig)

        err = (h_rot - h_rot_pred).abs().max().item()
        assert err < TOLERANCE, f"Wigner-D equivariance failed for l={l}: err={err:.2e}"

    def test_all_orders_simultaneous(self):
        """All l blocks transform correctly under the same rotation."""
        model = DetaNet(num_features=128, maxl=3, out_type="latent", device="cpu", summation=False, scale=None)
        z, pos = _make_molecule(6, seed=200)
        R = _random_rotation(seed=300)

        with torch.no_grad():
            _, T_orig = _run_forward(model, z, pos)
            _, T_rot = _run_forward(model, z, pos @ R.T)

        blocks_orig = _split_T(T_orig)
        blocks_rot = _split_T(T_rot)

        for l in [1, 2, 3]:
            D = _wigner_D(l, R)
            h_pred = torch.einsum("ab,ncb->nca", D, blocks_orig[l])
            err = (blocks_rot[l] - h_pred).abs().max().item()
            assert err < TOLERANCE, f"l={l} failed: err={err:.2e}"

    def test_multiple_rotations(self):
        """Equivariance holds across different random rotations."""
        model = DetaNet(num_features=128, maxl=3, out_type="latent", device="cpu", summation=False, scale=None)
        z, pos = _make_molecule(5, seed=42)

        for rot_seed in [10, 20, 30, 40, 50]:
            R = _random_rotation(seed=rot_seed)
            with torch.no_grad():
                _, T_orig = _run_forward(model, z, pos)
                _, T_rot = _run_forward(model, z, pos @ R.T)
            for l in [1, 2, 3]:
                D = _wigner_D(l, R)
                blocks = _split_T(T_orig), _split_T(T_rot)
                h_pred = torch.einsum("ab,ncb->nca", D, blocks[0][l])
                err = (blocks[1][l] - h_pred).abs().max().item()
                assert err < TOLERANCE, f"rot_seed={rot_seed} l={l}: err={err:.2e}"


class TestParity:
    def test_spatial_inversion_parity(self):
        """Under full inversion, odd-l blocks flip sign, even-l blocks are invariant."""
        model = DetaNet(num_features=128, maxl=3, out_type="latent", device="cpu", summation=False, scale=None)
        z, pos = _make_molecule(5, seed=42)
        inversion = torch.diag(torch.tensor([-1.0, -1.0, -1.0]))

        with torch.no_grad():
            _, T_orig = _run_forward(model, z, pos)
            _, T_inv = _run_forward(model, z, pos @ inversion.T)

        blocks_orig = _split_T(T_orig)
        blocks_inv = _split_T(T_inv)

        for l in [1, 2, 3]:
            parity = -1 if l % 2 == 1 else 1
            expected = parity * blocks_orig[l]
            err = (blocks_inv[l] - expected).abs().max().item()
            assert err < TOLERANCE, f"Inversion parity failed for l={l}: err={err:.2e}"


class TestTranslationInvariance:
    def test_translation_invariance(self):
        model = DetaNet(num_features=128, maxl=3, out_type="latent", device="cpu", summation=False, scale=None)
        z, pos = _make_molecule(5, seed=42)
        translation = torch.tensor([10.0, -5.0, 3.0])

        with torch.no_grad():
            S_orig, T_orig = _run_forward(model, z, pos)
            S_trans, T_trans = _run_forward(model, z, pos + translation)

        assert (S_orig - S_trans).abs().max().item() < TOLERANCE
        assert (T_orig - T_trans).abs().max().item() < TOLERANCE


class TestPermutationEquivariance:
    def test_permutation_equivariance(self):
        model = DetaNet(num_features=128, maxl=3, out_type="latent", device="cpu", summation=False, scale=None)
        z = torch.tensor([1, 6, 1, 6, 1], dtype=torch.long)
        pos = torch.tensor([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
        ], dtype=torch.float32)
        perm = torch.tensor([2, 1, 0, 3, 4], dtype=torch.long)

        with torch.no_grad():
            S_orig, T_orig = _run_forward(model, z, pos)
            S_perm, T_perm = _run_forward(model, z[perm], pos[perm])

        assert (S_perm - S_orig[perm]).abs().max().item() < TOLERANCE
        assert (T_perm - T_orig[perm]).abs().max().item() < TOLERANCE

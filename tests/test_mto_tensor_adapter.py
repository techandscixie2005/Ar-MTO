"""Test MTO tensor adapter: shapes, split/reconstruct, channel mixing."""

import pytest
import torch

from ar_mto.tensor_adapter import TensorAdapter, make_adapter


class TestTensorAdapterShapes:
    def test_make_adapter_default(self):
        adapter = make_adapter()
        assert adapter.num_features == 128
        assert adapter.maxl == 3
        assert adapter.vdim == 1920  # 384+640+896

    def test_make_adapter_maxl1(self):
        adapter = make_adapter(maxl=1)
        assert adapter.vdim == 384  # 128*3

    def test_make_adapter_maxl2(self):
        adapter = make_adapter(maxl=2)
        assert adapter.vdim == 1024  # 384+640

    def test_irreps_string(self):
        adapter = make_adapter()
        assert str(adapter.irreps_T) == "128x1o+128x2e+128x3o"

    def test_blocks_structure(self):
        adapter = make_adapter()
        blocks = adapter.blocks
        assert len(blocks) == 3

        assert blocks[0]["l"] == 1
        assert blocks[0]["parity"] == "odd"
        assert blocks[0]["multiplicity"] == 128
        assert blocks[0]["total_dim"] == 384
        assert blocks[0]["flat_start"] == 0
        assert blocks[0]["flat_end"] == 384

        assert blocks[1]["l"] == 2
        assert blocks[1]["parity"] == "even"
        assert blocks[1]["total_dim"] == 640

        assert blocks[2]["l"] == 3
        assert blocks[2]["parity"] == "odd"
        assert blocks[2]["total_dim"] == 896

    def test_forward_shapes(self):
        adapter = make_adapter()
        N = 5
        S = torch.randn(N, 128)
        T = torch.randn(N, 1920)
        h = adapter(S, T)

        assert h[0].shape == (N, 128, 1)
        assert h[1].shape == (N, 128, 3)
        assert h[2].shape == (N, 128, 5)
        assert h[3].shape == (N, 128, 7)

    def test_forward_variable_atoms(self):
        adapter = make_adapter()
        for n in [3, 4, 6, 8, 10]:
            S = torch.randn(n, 128)
            T = torch.randn(n, 1920)
            h = adapter(S, T)
            for l in [0, 1, 2, 3]:
                assert h[l].shape[0] == n


class TestSplitReconstruct:
    def test_exact_reconstruction(self):
        adapter = make_adapter()
        N = 5
        S_orig = torch.randn(N, 128)
        T_orig = torch.randn(N, 1920)

        h = adapter(S_orig, T_orig)
        S_recon, T_recon = adapter.reconstruct(h)

        assert torch.equal(S_orig, S_recon)
        assert torch.equal(T_orig, T_recon)

    def test_reconstruction_variable_atoms(self):
        adapter = make_adapter()
        for n in [3, 4, 6, 8, 10]:
            S = torch.randn(n, 128)
            T = torch.randn(n, 1920)
            h = adapter(S, T)
            S_r, T_r = adapter.reconstruct(h)
            assert torch.equal(S, S_r)
            assert torch.equal(T, T_r)

    def test_reconstruction_maxl1(self):
        adapter = make_adapter(maxl=1)
        N = 5
        S = torch.randn(N, 128)
        T = torch.randn(N, 384)
        h = adapter(S, T)
        S_r, T_r = adapter.reconstruct(h)
        assert torch.equal(S, S_r)
        assert torch.equal(T, T_r)

    def test_reconstruction_maxl2(self):
        adapter = make_adapter(maxl=2)
        N = 5
        S = torch.randn(N, 128)
        T = torch.randn(N, 1024)
        h = adapter(S, T)
        S_r, T_r = adapter.reconstruct(h)
        assert torch.equal(S, S_r)
        assert torch.equal(T, T_r)

    def test_h0_from_S(self):
        """h0 should be exactly S unsqueezed, not from T."""
        adapter = make_adapter()
        N = 5
        S = torch.randn(N, 128)
        T = torch.randn(N, 1920)
        h = adapter(S, T)
        assert torch.equal(h[0].squeeze(-1), S)


class TestChannelMixing:
    def test_channel_mix_shapes(self):
        adapter = make_adapter()
        N = 5
        S = torch.randn(N, 128)
        T = torch.randn(N, 1920)
        h = adapter(S, T)

        out_channels = 64
        weights = {
            l: torch.randn(out_channels, 128) * 0.1
            for l in [0, 1, 2, 3]
        }
        h_mixed = adapter.channel_mix(h, weights)

        assert h_mixed[0].shape == (N, out_channels, 1)
        assert h_mixed[1].shape == (N, out_channels, 3)
        assert h_mixed[2].shape == (N, out_channels, 5)
        assert h_mixed[3].shape == (N, out_channels, 7)

    def test_channel_mix_preserves_spatial(self):
        """Channel mixing should not mix m-components across atoms."""
        adapter = make_adapter()
        N = 5
        S = torch.randn(N, 128)
        T = torch.randn(N, 1920)
        h = adapter(S, T)

        out_channels = 32
        weights = {l: torch.randn(out_channels, 128) * 0.1 for l in [0, 1, 2, 3]}
        h_mixed = adapter.channel_mix(h, weights)

        # Each atom should be independently transformed
        for l in [0, 1, 2, 3]:
            assert h_mixed[l].shape[0] == N

    def test_partial_mix(self):
        """Only mix specified orders."""
        adapter = make_adapter()
        N = 3
        S = torch.randn(N, 128)
        T = torch.randn(N, 1920)
        h = adapter(S, T)

        weights = {1: torch.randn(32, 128) * 0.1}
        h_mixed = adapter.channel_mix(h, weights)

        # l=1 is mixed
        assert h_mixed[1].shape == (N, 32, 3)
        # l=0,2,3 unchanged
        assert h_mixed[0].shape == (N, 128, 1)
        assert h_mixed[2].shape == (N, 128, 5)
        assert h_mixed[3].shape == (N, 128, 7)

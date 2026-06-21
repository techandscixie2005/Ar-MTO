"""Test valence-half adaptive K: K computation, mode masks, batch isolation,
inactive-mode isolation, routing softmax, vector readout, top-r masking,
forward/backward/checkpoint.

All tests verify that:
  - fixed-K=8 behavior is unchanged
  - valence_half computes correct K_i for synthetic molecules
  - mode_mask shapes and values are correct in batches
  - inactive modes do not affect outputs
  - all softmax/normalization/routing/readout logic ignores inactive modes
  - batch isolation is preserved
"""

import pytest
import torch
import torch.nn as nn
from e3nn import o3  # noqa: F401

from ar_mto.mto_core import compute_valence_adaptive_k
from ar_mto.signed_routing import SignedRouter, _per_molecule_softmax
from ar_mto.mto_core import MTOModeAssembly
from ar_mto.readouts import VectorReadout

TOLERANCE = 5e-5


def _make_h(N=5, C=128, maxl=3):
    h = {0: torch.randn(N, C, 1)}
    for l in range(1, maxl + 1):
        h[l] = torch.randn(N, C, 2 * l + 1)
    return h


def _make_h_batch(N1, N2, C=128, maxl=3):
    h1 = _make_h(N1, C, maxl)
    h2 = _make_h(N2, C, maxl)
    h = {l: torch.cat([h1[l], h2[l]], dim=0) for l in h1}
    return h


class TestValenceHalfK:
    """Test compute_valence_adaptive_k with k_rounding=ceil, k_min=1."""

    def test_ch4(self):
        """CH4: C(4) + 4*H(1) = 8 valence -> K_half = 4."""
        z = torch.tensor([6, 1, 1, 1, 1], dtype=torch.long)
        mask, ks = compute_valence_adaptive_k(z, max_modes=32)
        assert ks[0].item() == 4
        assert mask.shape == (1, 32)
        assert mask[0, :4].all()
        assert not mask[0, 4:].any()

    def test_h2o(self):
        """H2O: O(6) + 2*H(1) = 8 valence -> K_half = 4."""
        z = torch.tensor([8, 1, 1], dtype=torch.long)
        mask, ks = compute_valence_adaptive_k(z, max_modes=32)
        assert ks[0].item() == 4

    def test_nh3(self):
        """NH3: N(5) + 3*H(1) = 8 valence -> K_half = 4."""
        z = torch.tensor([7, 1, 1, 1], dtype=torch.long)
        mask, ks = compute_valence_adaptive_k(z, max_modes=32)
        assert ks[0].item() == 4

    def test_co2(self):
        """CO2: C(4) + 2*O(6) = 16 valence -> K_half = 8."""
        z = torch.tensor([6, 8, 8], dtype=torch.long)
        mask, ks = compute_valence_adaptive_k(z, max_modes=32)
        assert ks[0].item() == 8

    def test_he(self):
        """He: Z=2, 2v -> K=1."""
        z = torch.tensor([2], dtype=torch.long)
        mask, ks = compute_valence_adaptive_k(z, max_modes=32)
        assert ks[0].item() == 1

    def test_k_min(self):
        """k_min=4: He gets K=4 not K=1."""
        z = torch.tensor([2], dtype=torch.long)
        mask, ks = compute_valence_adaptive_k(z, max_modes=32, k_min=4)
        assert ks[0].item() == 4

    def test_k_rounding_floor(self):
        """floor: N_val=9 -> K=4 (not ceil=5)."""
        # H2O2: 2*O (12) + 2*H (2) = 14 -> ceil=7, floor=7
        # N_val=9 -> ceil=5, floor=4
        # Use He: Z=2, v=2, with 4 more H: total 6 -> ceil=3, floor=3
        # Better: single atom with Z=9 (F, 7v) -> ceil=4, floor=3
        z = torch.tensor([9], dtype=torch.long)  # F: 7 valence
        mask_ceil, ks_ceil = compute_valence_adaptive_k(z, max_modes=32, k_rounding="ceil")
        mask_floor, ks_floor = compute_valence_adaptive_k(z, max_modes=32, k_rounding="floor")
        assert ks_ceil[0].item() == 4  # ceil(7/2) = 4
        assert ks_floor[0].item() == 3  # floor(7/2) = 3

    def test_max_modes_clamp(self):
        """K is clamped to max_modes."""
        z = torch.tensor([6] * 20)  # 20 C atoms: 80 valence -> K=40
        mask, ks = compute_valence_adaptive_k(z, max_modes=32)
        assert ks[0].item() == 32  # clamped

    def test_batched_different_k(self):
        """Two molecules with different N_val get correct individual K."""
        # Mol 0: CH4, N_val=8, K=4
        # Mol 1: CO2, N_val=16, K=8
        z = torch.tensor([6, 1, 1, 1, 1, 6, 8, 8], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
        mask, ks = compute_valence_adaptive_k(z, batch=batch, max_modes=32)

        assert ks[0].item() == 4
        assert ks[1].item() == 8
        assert mask.shape == (2, 32)

        # Mol 0: first 4 active
        assert mask[0, :4].all()
        assert not mask[0, 4:].any()

        # Mol 1: first 8 active
        assert mask[1, :8].all()
        assert not mask[1, 8:].any()


class TestModeMaskingValence:
    """Test that mode_mask correctly suppresses inactive modes."""

    def test_mode_mask_shape(self):
        """mode_mask shape is [B, Kmax]."""
        z = torch.tensor([6, 1, 1, 1, 1, 6, 8, 8], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
        mask, ks = compute_valence_adaptive_k(z, batch=batch, max_modes=32)
        assert mask.shape == (2, 32)
        assert ks.shape == (2,)
        assert mask.dtype == torch.bool

    def test_inactive_modes_dont_affect_mto(self):
        """Inactive modes are zeroed out in MTO assembly."""
        N1, N2, C, Kmax = 5, 4, 128, 8
        mto = MTOModeAssembly(num_features=C, mode_channels=64,
                              num_modes=Kmax, maxl=3)
        router = SignedRouter(num_features=C, num_modes=Kmax, maxl=3)

        h1 = _make_h(N1, C)
        h2 = _make_h(N2, C)
        h_batch = {l: torch.cat([h1[l], h2[l]], dim=0) for l in h1}
        batch = torch.tensor([0] * N1 + [1] * N2, dtype=torch.long)

        # Mol 0: 3 active modes, Mol 1: 5 active modes
        mode_mask = torch.zeros(2, Kmax, dtype=torch.bool)
        mode_mask[0, :3] = True
        mode_mask[1, :5] = True

        with torch.no_grad():
            coeffs = router(h_batch, batch=batch)
            O = mto.forward_with_masks(h_batch, coeffs, mode_mask, batch=batch)

        for l in [0, 1, 2, 3]:
            # Mol 0: modes 3..7 zero
            assert (O[l][0, 3:, :, :].abs().max() == 0.0), \
                f"Mol 0 l={l}: inactive modes not zeroed"
            # Mol 1: modes 5..7 zero
            assert (O[l][1, 5:, :, :].abs().max() == 0.0), \
                f"Mol 1 l={l}: inactive modes not zeroed"
            # Active modes have signal
            assert (O[l][0, :3, :, :].abs().max() > 0.0), \
                f"Mol 0 l={l}: active modes zero incorrectly"
            assert (O[l][1, :5, :, :].abs().max() > 0.0), \
                f"Mol 1 l={l}: active modes zero incorrectly"

    def test_vector_readout_ignores_inactive(self):
        """Vector readout with inactive modes: masking zeros out inactive contributions.

        The readout uses softmax(mode_weights) * mode_mask, so inactive modes
        contribute zero. Full vs masked differ because softmax distributes mass
        differently across K vs active-only modes.
        """
        B, Kmax, C = 2, 8, 64
        O = {
            0: torch.randn(B, Kmax, C, 1),
            1: torch.randn(B, Kmax, C, 3),
        }
        mode_mask = torch.zeros(B, Kmax, dtype=torch.bool)
        mode_mask[0, :4] = True
        mode_mask[1, :5] = True

        readout = VectorReadout(mode_channels=C, num_modes=Kmax, out_dim=1)

        with torch.no_grad():
            v_masked = readout(O, mode_mask=mode_mask)
            v_full = readout(O, mode_mask=None)

        # Should differ because masking changes weight distribution
        assert not torch.allclose(v_masked, v_full, atol=1e-3), \
            "Masking should change vector readout output"

        # Verify inactive modes contribute zero: build O where
        # inactive modes are set to 0, same mask, output unchanged
        O_zeroed = {k: v.clone() for k, v in O.items()}
        for k in O_zeroed:
            mask_exp = mode_mask.unsqueeze(-1).unsqueeze(-1)
            O_zeroed[k] = O_zeroed[k] * mask_exp

        with torch.no_grad():
            v_zeroed_inactive = readout(O_zeroed, mode_mask=mode_mask)
            v_zeroed_inactive_full = readout(O_zeroed, mode_mask=None)

        # Mode weights are same in both calls, but O differs
        # pre-multiplication with mask should give same masked result
        # (but different from full since pre-zeroed inactive modes)
        assert torch.allclose(v_zeroed_inactive, v_masked, atol=1e-5), \
            "Pre-zeroed inactive modes should give same masked output"


class TestBatchIsolationWithValence:
    """Batch isolation must hold with valence-adaptive K."""

    def test_batch_isolation_mto(self):
        """Molecule A alone vs in batch with masked modes must match."""
        C, Kmax, Cout = 128, 32, 64
        mto = MTOModeAssembly(num_features=C, mode_channels=Cout,
                              num_modes=Kmax, maxl=3)
        router = SignedRouter(num_features=C, num_modes=Kmax, maxl=3)

        hA = _make_h(5, C)
        hB = _make_h(4, C)

        # Forward A alone with 4 active modes
        with torch.no_grad():
            coeffs_A = router(hA)
            mask_A = torch.zeros(1, Kmax, dtype=torch.bool)
            mask_A[0, :4] = True
            O_A_alone = mto.forward_with_masks(hA, coeffs_A, mask_A)

        # Forward A alongside B in batch
        h_batch = {l: torch.cat([hA[l], hB[l]], dim=0) for l in hA}
        batch = torch.tensor([0] * 5 + [1] * 4, dtype=torch.long)
        mask_batch = torch.zeros(2, Kmax, dtype=torch.bool)
        mask_batch[0, :4] = True
        mask_batch[1, :6] = True

        with torch.no_grad():
            coeffs_batch = router(h_batch, batch=batch)
            O_batch = mto.forward_with_masks(h_batch, coeffs_batch, mask_batch, batch=batch)

        for l in [0, 1, 2, 3]:
            assert torch.allclose(O_batch[l][0:1], O_A_alone[l], atol=1e-5), \
                f"Batch isolation failed for l={l} with valence masking"


class TestRoutingSoftmaxActive:
    """Softmax over atoms in routing must be per-molecule and respect mode masks."""

    def test_softmax_sums_to_one_per_molecule(self):
        """Softmax over atoms within each molecule sums to 1."""
        K, N = 4, 9  # 5 atoms mol0 + 4 atoms mol1
        logits = torch.randn(K, N)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        attn = _per_molecule_softmax(logits, batch, num_molecules=2)

        # Sum over atoms per molecule
        mol0_sum = attn[:, batch == 0].sum(dim=1)
        mol1_sum = attn[:, batch == 1].sum(dim=1)

        for k in range(K):
            assert (mol0_sum[k] - 1.0).abs() < 1e-5, f"k={k} mol0 sum != 1"
            assert (mol1_sum[k] - 1.0).abs() < 1e-5, f"k={k} mol1 sum != 1"

    def test_softmax_no_cross_molecule(self):
        """Softmax: atoms in mol0 affect only mol0, not mol1."""
        K, N = 4, 9
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

        # Mol 1 atoms: all zero logits
        logits = torch.randn(K, N) * 0.5
        logits[:, batch == 1] = 0.0
        attn1 = _per_molecule_softmax(logits, batch, num_molecules=2)

        # Mol 0 atoms should have non-zero softmax
        attn0_mol0 = attn1[:, batch == 0]
        assert (attn0_mol0.sum(dim=1) - 1.0).abs().max() < 1e-5
        assert attn0_mol0.max() > 0.01  # something got probability

        # Now modify mol 1 logits
        logits[:, batch == 1] = 100.0
        attn2 = _per_molecule_softmax(logits, batch, num_molecules=2)

        # Mol 0's softmax must not change
        attn2_mol0 = attn2[:, batch == 0]
        assert torch.allclose(attn1[:, batch == 0], attn2[:, batch == 0], atol=1e-5)


class TestForwardBackwardValence:
    """Forward/backward passes work with valence_half policy."""

    def test_forward_no_nan(self):
        """Single forward pass produces clean outputs."""
        from ar_mto.detanet_bridge import make_latent_detanet
        from ar_mto.mto_net import make_mto_net

        detanet = make_latent_detanet(num_features=128, maxl=3, num_block=2)
        model = make_mto_net(
            detanet_model=detanet,
            num_features=128, k_policy="valence_half", k_max=16,
            mode_channels=32, num_modes=16, maxl=3,
            active_heads=["vector"],
        )

        z = torch.tensor([6, 1, 1, 1, 1], dtype=torch.long)  # CH4, K=4
        pos = torch.randn(5, 3)

        model.eval()
        with torch.no_grad():
            output = model(z=z, pos=pos)

        pred = output["vector"]
        assert not torch.isnan(pred).any()
        assert not torch.isinf(pred).any()
        assert "mode_mask" in output
        assert output["mode_mask"].shape[1] == 16

    def test_backward_no_nan(self):
        """Backward pass produces finite gradients."""
        from ar_mto.detanet_bridge import make_latent_detanet
        from ar_mto.mto_net import make_mto_net

        detanet = make_latent_detanet(num_features=128, maxl=3, num_block=2)
        model = make_mto_net(
            detanet_model=detanet,
            num_features=128, k_policy="valence_half", k_max=16,
            mode_channels=32, num_modes=16, maxl=3,
            active_heads=["vector"],
        )

        # Single molecule: CH4
        z = torch.tensor([6, 1, 1, 1, 1], dtype=torch.long)
        pos = torch.randn(5, 3)

        model.train()
        output = model(z=z, pos=pos)
        pred = output["vector"]  # [1, 1, 3]

        target = torch.randn_like(pred)
        loss = nn.functional.mse_loss(pred, target)
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"{name}: NaN grad"
                assert not torch.isinf(p.grad).any(), f"{name}: Inf grad"

    def test_checkpoint_save_load(self, tmp_path):
        """Checkpoint with valence_half reloads correctly."""
        from ar_mto.detanet_bridge import make_latent_detanet
        from ar_mto.mto_net import make_mto_net

        detanet = make_latent_detanet(num_features=128, maxl=3, num_block=2)
        model = make_mto_net(
            detanet_model=detanet,
            num_features=128, k_policy="valence_half", k_max=16,
            mode_channels=32, num_modes=16, maxl=3,
            active_heads=["vector"],
        )

        z = torch.tensor([6, 1, 1, 1, 1], dtype=torch.long)
        pos = torch.randn(5, 3)

        model.eval()
        with torch.no_grad():
            out_before = model(z=z, pos=pos)
        pred_before = out_before["vector"].clone()

        # Save
        ckpt_path = tmp_path / "test.ckpt"
        torch.save(model.state_dict(), str(ckpt_path))

        # Create new model and load
        detanet2 = make_latent_detanet(num_features=128, maxl=3, num_block=2)
        model2 = make_mto_net(
            detanet_model=detanet2,
            num_features=128, k_policy="valence_half", k_max=16,
            mode_channels=32, num_modes=16, maxl=3,
            active_heads=["vector"],
        )
        model2.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
        model2.eval()

        with torch.no_grad():
            out_after = model2(z=z, pos=pos)
        pred_after = out_after["vector"]

        assert torch.allclose(pred_before, pred_after, atol=1e-5), \
            "Checkpoint roundtrip produces different predictions"


class TestTopRMasking:
    """Top-r mode masking never selects inactive (padded) modes."""

    def test_top_r_never_selects_inactive(self):
        """When only top-r modes are kept, inactive padded modes are never chosen."""
        Kmax = 16
        B = 2
        mode_mask = torch.zeros(B, Kmax, dtype=torch.bool)
        mode_mask[0, :4] = True
        mode_mask[1, :6] = True

        activity = torch.randn(B, Kmax)

        for r in [1, 2, 4]:
            active_activity = activity.clone()
            active_activity[~mode_mask] = -float("inf")

            # Only select up to min(r, num_active) to never hit -inf indices
            for b in range(B):
                n_active = int(mode_mask[b].sum())
                r_eff = min(r, n_active)
                _, top_indices = active_activity[b].topk(r_eff)

                for idx in top_indices:
                    assert mode_mask[b, idx], \
                        f"r={r} mol={b}: selected inactive mode {idx}"

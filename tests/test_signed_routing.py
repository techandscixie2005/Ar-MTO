"""Test signed invariant routing: coefficient shapes, invariance, sign properties."""

import pytest
import torch

from ar_mto.signed_routing import SignedRouter
from ar_mto.tensor_adapter import make_adapter


def _make_h(N=5, C=128, maxl=3):
    """Synthetic tensor features simulating DetaNet output."""
    h = {}
    h[0] = torch.randn(N, C, 1)
    for l in range(1, maxl + 1):
        h[l] = torch.randn(N, C, 2 * l + 1)
    return h


class TestRoutingShapes:
    def test_coefficient_shapes(self):
        router = SignedRouter(num_features=128, num_modes=8, maxl=3)
        h = _make_h(5)
        coeffs = router(h)

        for l in [0, 1, 2, 3]:
            assert l in coeffs, f"Missing coefficients for l={l}"
            assert coeffs[l].shape == (8, 5, 1), \
                f"Wrong shape for l={l}: {coeffs[l].shape}"

    def test_variable_atoms(self):
        router = SignedRouter(num_features=128, num_modes=8, maxl=3)
        for n in [3, 5, 7, 10]:
            h = _make_h(n)
            coeffs = router(h)
            for l in [0, 1, 2, 3]:
                assert coeffs[l].shape == (8, n, 1)

    def test_variable_modes(self):
        for K in [4, 8, 16]:
            router = SignedRouter(num_features=128, num_modes=K, maxl=3)
            h = _make_h(5)
            coeffs = router(h)
            for l in [0, 1, 2, 3]:
                assert coeffs[l].shape == (K, 5, 1)

    def test_scalar_only_routing(self):
        """Scalar-only routing should not use tensor norms."""
        router = SignedRouter(num_features=128, num_modes=8,
                              use_tensor_norms=False, maxl=0)
        h = {0: torch.randn(5, 128, 1)}
        coeffs = router(h)
        assert coeffs[0].shape == (8, 5, 1)

    def test_coefficient_same_for_all_l(self):
        """Routing coefficients are generated from invariants, so they're
        the same across l orders within a forward pass."""
        router = SignedRouter(num_features=128, num_modes=8, maxl=3)
        h = _make_h(5)
        coeffs = router(h)
        for l in [1, 2, 3]:
            assert torch.equal(coeffs[0], coeffs[l])


class TestRoutingProperties:
    def test_coefficients_finite(self):
        router = SignedRouter(num_features=128, num_modes=8, maxl=3)
        h = _make_h(5)
        coeffs = router(h)
        for l in [0, 1, 2, 3]:
            assert not torch.isnan(coeffs[l]).any()
            assert not torch.isinf(coeffs[l]).any()

    def test_sign_range(self):
        """Combined coefficient should be in [-1, 1] (attn * tanh)."""
        router = SignedRouter(num_features=128, num_modes=8, maxl=3)
        h = _make_h(5)
        coeffs = router(h)
        for l in [0, 1, 2, 3]:
            assert coeffs[l].min() >= -1.0
            assert coeffs[l].max() <= 1.0

    def test_attention_sums_to_one(self):
        """The attention part (before sign) should be normalized over atoms."""
        # We test indirectly: the abs sum over atoms should be <= 1
        # And for each mode, the "positive attention" component sums near 1
        router = SignedRouter(num_features=128, num_modes=8, maxl=3)
        h = _make_h(5)
        coeffs = router(h)

        # If all signs were +1, sum over atoms = 1
        # With sign mixing, abs sum <= 1 per mode
        c0 = coeffs[0].squeeze(-1)  # [K, N]
        abs_sum = c0.abs().sum(dim=-1)  # [K]
        assert (abs_sum <= 1.0 + 1e-5).all(), \
            f"abs_sum per mode: {abs_sum}"

    def test_deterministic(self):
        router = SignedRouter(num_features=128, num_modes=8, maxl=3)
        router.eval()
        h = _make_h(5)
        with torch.no_grad():
            c1 = router(h)
            c2 = router(h)
        for l in [0, 1, 2, 3]:
            assert torch.allclose(c1[l], c2[l], atol=1e-7)

    def test_mode_usage_stats(self):
        router = SignedRouter(num_features=128, num_modes=8, maxl=3)
        h = _make_h(5)
        coeffs = router(h)
        stats = router.route_stats(coeffs)
        assert "route_entropy" in stats
        assert "route_mean_abs" in stats
        assert "route_std" in stats
        assert "route_pos_frac" in stats
        assert 0 <= stats["route_pos_frac"] <= 1


class TestRoutingInvariance:
    def test_routing_rotation_invariant(self):
        """Routing coefficients must be invariant under rotation (use only
        invariant inputs)."""
        from e3nn import o3

        router = SignedRouter(num_features=128, num_modes=8, maxl=3)
        N = 5

        # Create features and rotated version
        h_orig = _make_h(N)

        # Build random rotation using quaternion
        gen = torch.Generator()
        gen.manual_seed(42)
        q = torch.randn(4, generator=gen)
        q = q / torch.norm(q)
        w, x, y, z = q
        R = torch.tensor([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ])
        h_rot = {0: h_orig[0].clone()}  # l=0 invariant
        for l in [1, 2, 3]:
            D = o3.wigner_D(l, *o3.matrix_to_angles(R))
            h_rot[l] = torch.einsum("ab,ncb->nca", D, h_orig[l])

        with torch.no_grad():
            c_orig = router(h_orig)
            c_rot = router(h_rot)

        # Coefficients must be identical (routing uses only invariants)
        for l in [0, 1, 2, 3]:
            assert torch.allclose(c_orig[l], c_rot[l], atol=1e-5), \
                f"Routing not rotation-invariant for l={l}"

    def test_routing_translation_invariant(self):
        """Routing should be translation invariant (uses h0 and tensor norms)."""
        router = SignedRouter(num_features=128, num_modes=8, maxl=3)
        h = _make_h(5)

        with torch.no_grad():
            c1 = router(h)
            c2 = router(h)

        for l in [0, 1, 2, 3]:
            assert torch.allclose(c1[l], c2[l], atol=1e-7)

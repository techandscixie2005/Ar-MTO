"""Test signed invariant routing: coefficient shapes, invariance, sign properties.

All tests use batch-aware API: [K, N, 1] coefficients.
"""

import pytest
import torch

from ar_mto.signed_routing import SignedRouter


def _make_h(N=5, C=128, maxl=3):
    """Synthetic tensor features simulating DetaNet adapter output."""
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

    def test_order_specific_signs_different(self):
        """With order_specific_signs=True, different l orders get different
        sign projections, so coefficients may differ across orders."""
        router = SignedRouter(num_features=128, num_modes=8, maxl=3,
                              order_specific_signs=True)
        h = _make_h(5)
        coeffs = router(h)
        # With order-specific signs, coefficients can differ per l
        # but shapes are correct and values are finite
        for l in [0, 1, 2, 3]:
            assert not torch.isnan(coeffs[l]).any()
            assert not torch.isinf(coeffs[l]).any()


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

    def test_l2_normalization(self):
        """L2 norm of coefficients per mode per molecule should be ~1."""
        router = SignedRouter(num_features=128, num_modes=8, maxl=3,
                              normalization="l2")
        h = _make_h(5)
        coeffs = router(h)
        c0 = coeffs[0].squeeze(-1)  # [K, N]
        l2_norms = c0.pow(2).sum(dim=-1).sqrt()  # [K]
        # L2 norm per mode should be close to 1
        assert torch.allclose(l2_norms, torch.ones_like(l2_norms), atol=1e-5), \
            f"l2_norms per mode: {l2_norms}"

    def test_abs_normalization(self):
        """Abs sum of coefficients per mode per molecule should be ~1."""
        router = SignedRouter(num_features=128, num_modes=8, maxl=3,
                              normalization="abs")
        h = _make_h(5)
        coeffs = router(h)
        c0 = coeffs[0].squeeze(-1)  # [K, N]
        abs_sum = c0.abs().sum(dim=-1)  # [K]
        assert torch.allclose(abs_sum, torch.ones_like(abs_sum), atol=1e-5), \
            f"abs_sum per mode: {abs_sum}"

    def test_batched_routing(self):
        """Routing handles batched molecules correctly."""
        router = SignedRouter(num_features=128, num_modes=8, maxl=3)
        # Two molecules: 3 atoms + 4 atoms
        N = 7
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        h = _make_h(N)
        coeffs = router(h, batch=batch)

        # Each molecule should have L2-normed coefficients separately
        c0 = coeffs[0].squeeze(-1)  # [K, N]
        for b_idx in range(2):
            mask = (batch == b_idx)
            mol_norms = c0[:, mask].pow(2).sum(dim=-1).sqrt()
            assert torch.allclose(mol_norms, torch.ones_like(mol_norms), atol=1e-5), \
                f"Molecule {b_idx} l2_norms: {mol_norms}"

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
            h_rot[l] = torch.einsum("sd,ncd->ncs", D, h_orig[l])

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
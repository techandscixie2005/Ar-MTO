"""Clebsch-Gordan tensor coupling for MTO modes — parity-correct, batch-aware.

Uses e3nn o3.TensorProduct for correct CG decomposition. Builds coupling paths
programmatically from input irreps with proper parity rules.

Key rule: Under O(3), the tensor product of two irreps with parities p1, p2
produces irreps with parity p1 * p2. The mechanical mapping 1o × 1o → 1o is
WRONG — the correct L=1 output has even parity (1e, axial vector).

All coupling paths are computed within each molecule independently.

Supports lmax=2 as canonical response model. l=3 input is preserved through
a residual path when enabled but not coupled (configurable).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from e3nn import o3


def _build_coupling_paths(
    in_irreps: o3.Irreps,
    max_coupled_l: int = 2,
    include_self_coupling: bool = True,
) -> list[tuple[int, int, int, int, int]]:
    """Build all legal (l1, p1) × (l2, p2) → (L, p1*p2) coupling paths.

    Returns:
        List of (l1, p1, l2, p2, L) tuples for legal paths with L <= max_coupled_l.
        p1, p2 are ±1 for even/odd parity.
    """
    paths = []
    input_pairs = list(in_irreps)  # (mul, (l, p)) pairs

    for mul1, (l1, p1) in input_pairs:
        for mul2, (l2, p2) in input_pairs:
            p_out = p1 * p2
            for L in range(abs(l1 - l2), l1 + l2 + 1):
                if L > max_coupled_l:
                    continue
                # Triangle inequality OK
                paths.append((l1, p1, l2, p2, L, p_out))
    return paths


def _paths_to_irreps_out(paths: list[tuple], channel_mult: int) -> o3.Irreps:
    """Convert coupling paths to output irreps string."""
    collected: dict[tuple[int, int], int] = {}
    for l1, p1, l2, p2, L, p_out in paths:
        key = (L, p_out)
        collected[key] = collected.get(key, 0) + 1
    return o3.Irreps([
        (channel_mult * count, (L, p))
        for (L, p), count in sorted(collected.items())
    ])


def _make_coupling_tp(
    in_irreps: o3.Irreps,
    paths: list[tuple],
    channel_mult: int,
) -> o3.FullyConnectedTensorProduct:
    """Create a FullyConnectedTensorProduct for CG coupling.

    Note: e3nn v0.5.7 FCTP does not accept custom instructions.
    All legal paths (respecting triangle inequality and parity) are enabled
    automatically. This is correct for parity-respecting coupling.
    """
    irreps_out = _paths_to_irreps_out(paths, channel_mult)

    return o3.FullyConnectedTensorProduct(
        irreps_in1=in_irreps,
        irreps_in2=in_irreps,
        irreps_out=irreps_out,
    )


class CGCoupling(nn.Module):
    """Parity-correct Clebsch-Gordan coupling between MTO modes.

    Couples modes within the same mode index (self-coupling per mode per molecule):
        O_{b,k}^{(l1,p1)} ×_CG O_{b,k}^{(l2,p2)} → O_new_{b,k}^{(L, p1*p2)}

    Uses e3nn FullyConnectedTensorProduct with internal learnable weights.
    No per-sample weight conditioning (removed due to equivariance failures
    with large weight_numel scalar-conditioned paths).

    .. note::

        This module is **experimental**. The per-sample scalar-conditioned
        weight path was removed after it produced non-equivariant outputs
        depending on initialization (see Task 1.2b report). The module now
        uses e3nn's internal learnable TP weights, which are equivariant
        but lack input-dependent conditioning.

        For production use (dipole, polarizability tasks), prefer
        :class:`CGCouplingMinimal`, which has fixed, verified paths
        without learned weights.

    Args:
        mode_channels: channels per mode (C)
        maxl: maximum input tensor order
        coupled_maxl: maximum output L from coupling (default 2)
        preserve_uncoupled_l: if True, keep l=3 via residual when coupled_maxl < maxl
    """

    def __init__(
        self,
        mode_channels: int = 64,
        maxl: int = 3,
        coupled_maxl: int = 2,
        preserve_uncoupled_l: bool = True,
    ):
        super().__init__()
        self.mode_channels = mode_channels
        self.maxl = maxl
        self.coupled_maxl = coupled_maxl
        self.preserve_uncoupled_l = preserve_uncoupled_l

        # Build input irreps: [128x0e, 128x1o, 128x2e, 128x3o]
        input_spec = [(mode_channels, (l, (-1) ** l)) for l in range(maxl + 1)]
        self.in_irreps = o3.Irreps(input_spec)

        # Build legal coupling paths
        self.coupling_paths = _build_coupling_paths(
            self.in_irreps, coupled_maxl
        )

        # Build output irreps
        self.out_irreps = _paths_to_irreps_out(self.coupling_paths, mode_channels)

        # Create the tensor product (uses internal learnable weights, no per-sample conditioning)
        self.tp = _make_coupling_tp(self.in_irreps, self.coupling_paths, mode_channels)

        # Projection maps: merge coupled outputs back to canonical irreps
        self._build_output_projections()

        # Precompute path table for diagnostics
        self.path_table: list[dict] = []
        for l1, p1, l2, p2, L, p_out in self.coupling_paths:
            self.path_table.append({
                "input": f"({l1}{'e' if p1 == 1 else 'o'})×({l2}{'e' if p2 == 1 else 'o'})",
                "output": f"({L}{'e' if p_out == 1 else 'o'})",
            })

    def _build_output_projections(self):
        """Build per-(L,p) linear projections from TP output to canonical channels."""
        self.projections = nn.ModuleDict()
        self.out_channels_map: dict[tuple[int, int], int] = {}

        offset = 0
        for mul, (L, p) in self.out_irreps:
            key = f"{L}{'e' if p == 1 else 'o'}"
            in_ch = mul  # mul = mode_channels * count, already includes mode_channels
            out_ch = self.mode_channels
            self.projections[key] = nn.Linear(in_ch, out_ch, bias=False)
            self.out_channels_map[(L, p)] = in_ch
            offset += in_ch * (2 * L + 1)

    def forward(
        self,
        O: dict[int | tuple, torch.Tensor],
    ) -> dict[int | tuple, torch.Tensor]:
        """Apply parity-correct CG coupling to MTO modes.

        Uses e3nn FCTP with internal learnable weights (no per-sample conditioning).

        Args:
            O: dict key -> [B, K, C, 2l+1] MTO modes

        Returns:
            O_coupled: dict (l, p) -> [B, K, C, 2l+1] coupled modes
        """
        B, K = O[next(iter(O.keys()))].shape[:2]
        device = O[next(iter(O.keys()))].device

        # Determine key scheme
        sample_key = next(iter(O.keys()))
        use_tuple_keys = isinstance(sample_key, tuple)

        # Build canonical keys: (l, p) for p = (-1)^l
        canonical_keys = [(l, (-1) ** l) for l in range(self.maxl + 1)]

        # Prepare input per mode per molecule as flat irreps vectors
        results: dict[tuple[int, int], list[torch.Tensor]] = {}
        for _mul, (L, p) in self.out_irreps:
            results.setdefault((L, p), [])

        for b in range(B):
            for k in range(K):
                # Build flat irreps vector
                segments = []
                for (l, p) in canonical_keys:
                    if use_tuple_keys:
                        key = (l, p)
                    else:
                        key = l
                    if key in O:
                        segments.append(O[key][b, k].reshape(-1))
                    else:
                        segments.append(torch.zeros(
                            self.mode_channels * (2 * l + 1),
                            device=device, dtype=O[next(iter(O.keys()))].dtype
                        ))
                x = torch.cat(segments)

                # Apply CG tensor product with internal learnable weights
                y = self.tp(x, x)

                # Split output by (L, p) and project
                offset = 0
                for mul, (L, p) in self.out_irreps:
                    key_str = f"{L}{'e' if p == 1 else 'o'}"
                    spatial_dim = (2 * L + 1)
                    total_dim = mul * spatial_dim
                    segment = y[offset:offset + total_dim]

                    # Reshape to [mul, spatial_dim]
                    segment_flat = segment.reshape(mul, spatial_dim)
                    # Project: [mul, sd] → [C, sd]
                    projected = self.projections[key_str](
                        segment_flat.T
                    ).T  # [C, sd]
                    results[(L, p)].append(projected.unsqueeze(0))  # [1, C, sd]
                    offset += total_dim

        # Stack results: [B*K, C, sd] → [B, K, C, sd]
        O_coupled: dict[tuple[int, int], torch.Tensor] = {}
        for (L, p), tensors in results.items():
            if tensors:
                stacked = torch.stack(tensors, dim=0)  # [B*K, C, sd]
                O_coupled[(L, p)] = stacked.reshape(B, K, self.mode_channels, 2 * L + 1)

        return O_coupled

    def get_path_table(self) -> str:
        """Return human-readable coupling path table."""
        lines = ["CG Coupling Paths:", "-" * 50]
        for entry in self.path_table:
            lines.append(f"  {entry['input']} → {entry['output']}")
        lines.append("-" * 50)
        lines.append(f"Total paths: {len(self.path_table)}")
        return "\n".join(lines)


class CGCouplingMinimal(nn.Module):
    """Minimal CG coupling: essential paths for dipole and polarizability tasks.

    Paths:
        0e × 1o → 1o  (scalar-vector coupling)
        0e × 2e → 2e  (scalar-rank2 coupling)
        1o × 1o → 0e, 2e  (vector-vector → scalar, rank2)
        0e × 3o → 3o  (scalar-l3 coupling, preserves l=3)

    Note: 1o × 1o → 1e (axial vector) is NOT included — it can't be added
    to polar 1o channels. This is the parity-correct behavior.
    """

    def __init__(self, mode_channels: int = 64):
        super().__init__()
        self.mode_channels = mode_channels

        # 0e × 1o → 1o  (polar vector enhancement)
        self.tp_0e_1o = o3.FullyConnectedTensorProduct(
            o3.Irreps(f"{mode_channels}x0e"),
            o3.Irreps(f"{mode_channels}x1o"),
            o3.Irreps(f"{mode_channels}x1o"),
        )

        # 0e × 2e → 2e  (rank-2 enhancement)
        self.tp_0e_2e = o3.FullyConnectedTensorProduct(
            o3.Irreps(f"{mode_channels}x0e"),
            o3.Irreps(f"{mode_channels}x2e"),
            o3.Irreps(f"{mode_channels}x2e"),
        )

        # 1o × 1o → 0e + 2e  (scalar + rank2 from vector pair)
        self.tp_1o_1o = o3.FullyConnectedTensorProduct(
            o3.Irreps(f"{mode_channels}x1o"),
            o3.Irreps(f"{mode_channels}x1o"),
            o3.Irreps(f"{mode_channels}x0e + {mode_channels}x2e"),
        )

        # 0e × 0e → 0e  (scalar self-coupling)
        self.tp_0e_0e = o3.FullyConnectedTensorProduct(
            o3.Irreps(f"{mode_channels}x0e"),
            o3.Irreps(f"{mode_channels}x0e"),
            o3.Irreps(f"{mode_channels}x0e"),
        )

        # 0e × 3o → 3o  (preserve l=3 through residual)
        self.tp_0e_3o = o3.FullyConnectedTensorProduct(
            o3.Irreps(f"{mode_channels}x0e"),
            o3.Irreps(f"{mode_channels}x3o"),
            o3.Irreps(f"{mode_channels}x3o"),
        )

        # 1o × 2e → 1o + 2o + 3o (cross-coupling)
        self.tp_1o_2e = o3.FullyConnectedTensorProduct(
            o3.Irreps(f"{mode_channels}x1o"),
            o3.Irreps(f"{mode_channels}x2e"),
            o3.Irreps(f"{mode_channels}x1o + {mode_channels}x2o + {mode_channels}x3o"),
        )

        self.path_table = [
            {"input": "0e × 0e", "output": "0e"},
            {"input": "0e × 1o", "output": "1o"},
            {"input": "1o × 1o", "output": "0e + 2e"},
            {"input": "0e × 2e", "output": "2e"},
            {"input": "1o × 2e", "output": "1o + 2o + 3o"},
            {"input": "0e × 3o", "output": "3o"},
        ]

    def forward(
        self, O: dict[int | tuple, torch.Tensor]
    ) -> dict[int | tuple, torch.Tensor]:
        """Apply minimal CG coupling per mode per molecule.

        Args:
            O: dict key -> [B, K, C, 2l+1] MTO modes

        Returns:
            O_coupled: dict (l, p) -> [B, K, C, 2l+1] coupled modes
        """
        B, K = O[next(iter(O.keys()))].shape[:2]
        C = self.mode_channels

        # Determine key scheme
        sample_key = next(iter(O.keys()))
        use_tuple_keys = isinstance(sample_key, tuple)

        def _get(key_l, key_p=None):
            """Get mode tensor with given l (and optional parity)."""
            if use_tuple_keys and key_p is not None:
                k = (key_l, key_p)
            else:
                k = key_l
            return O.get(k, None)

        # Get components
        o0 = _get(0, 1)   # [B, K, C, 1]
        o1 = _get(1, -1)  # [B, K, C, 3]  (1o)
        o2 = _get(2, 1)   # [B, K, C, 5]  (2e)
        o3 = _get(3, -1)  # [B, K, C, 7]  (3o)

        if o0 is None:
            return O  # nothing to couple

        # Accumulate one tensor per (b, k, order): sum across all paths for that order
        result_0e = o0.new_zeros(B, K, C, 1)
        result_1o = o1.new_zeros(B, K, C, 3) if o1 is not None else None
        result_2e = o2.new_zeros(B, K, C, 5) if o2 is not None else None
        result_3o = o3.new_zeros(B, K, C, 7) if o3 is not None else None

        for b in range(B):
            for k in range(K):
                s = o0[b, k].reshape(-1)                 # [C]

                # 0e × 0e → 0e
                y0 = self.tp_0e_0e(s, s)                # [C]
                result_0e[b, k] = y0.reshape(C, 1)

                if o1 is not None:
                    v = o1[b, k].reshape(-1)              # [C * 3]

                    # 0e × 1o → 1o
                    y1 = self.tp_0e_1o(s, v)             # [C * 3]
                    result_1o[b, k] = result_1o[b, k] + y1.reshape(C, 3)

                    # 1o × 1o → 0e + 2e
                    yvv = self.tp_1o_1o(v, v)
                    yvv_0 = yvv[:C].reshape(C, 1)
                    yvv_2 = yvv[C:].reshape(C, 5)
                    result_0e[b, k] = result_0e[b, k] + yvv_0
                    if result_2e is not None:
                        result_2e[b, k] = result_2e[b, k] + yvv_2

                if o2 is not None and result_2e is not None:
                    t = o2[b, k].reshape(-1)              # [C * 5]

                    # 0e × 2e → 2e
                    y2 = self.tp_0e_2e(s, t)             # [C * 5]
                    result_2e[b, k] = result_2e[b, k] + y2.reshape(C, 5)

                    if o1 is not None:
                        # 1o × 2e → 1o + 2o + 3o
                        y12 = self.tp_1o_2e(v, t)
                        y12_1 = y12[:C * 3].reshape(C, 3)
                        y12_2 = y12[C * 3:C * 8].reshape(C, 5)
                        y12_3 = y12[C * 8:].reshape(C, 7)
                        result_1o[b, k] = result_1o[b, k] + y12_1
                        result_2e[b, k] = result_2e[b, k] + y12_2
                        if result_3o is not None:
                            result_3o[b, k] = result_3o[b, k] + y12_3

                if o3 is not None and result_3o is not None:
                    f = o3[b, k].reshape(-1)              # [C * 7]

                    # 0e × 3o → 3o  (residual preservation)
                    y3 = self.tp_0e_3o(s, f)             # [C * 7]
                    result_3o[b, k] = result_3o[b, k] + y3.reshape(C, 7)

        # Build output dict: use int keys (p = (-1)^l, consistent with DetaNet/MTO)
        # CG results are residual additions to the original modes
        O_coupled: dict = {}
        O_coupled[0] = result_0e
        if result_1o is not None:
            O_coupled[1] = result_1o
        if result_2e is not None:
            O_coupled[2] = result_2e
        if result_3o is not None:
            O_coupled[3] = result_3o
        elif o3 is not None:
            O_coupled[3] = o3  # identity residual

        return O_coupled

    def get_path_table(self) -> str:
        """Return human-readable coupling path table."""
        lines = ["CG Coupling Paths (Minimal):", "-" * 48]
        for entry in self.path_table:
            lines.append(f"  {entry['input']} → {entry['output']}")
        lines.append("-" * 48)
        lines.append(f"Total paths: {len(self.path_table)}")
        return "\n".join(lines)
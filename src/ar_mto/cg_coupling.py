"""Clebsch-Gordan tensor coupling for MTO modes.

Implements representation-legal cross-order interactions:

    O_new^(L) = sum_{|l1-l2| <= L <= l1+l2} O_k^(l1) x_CG O_j^(l2)

Uses e3nn o3.TensorProduct for correct CG decomposition. Each coupling path
is weighted by learned scalar coefficients generated from invariant features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from e3nn import o3


class CGCoupling(nn.Module):
    """Clebsch-Gordan tensor coupling between MTO modes.

    Couples modes within the same mode index (self-coupling):
        O_k^(l1) x_CG O_k^(l2) → O_new^(L)

    Args:
        mode_channels: channels per mode
        maxl: maximum tensor order
        coupled_channels: output channels per coupled order
    """

    def __init__(
        self,
        mode_channels: int = 64,
        maxl: int = 3,
        coupled_channels: int = 64,
    ):
        super().__init__()
        self.mode_channels = mode_channels
        self.maxl = maxl
        self.coupled_channels = coupled_channels

        # Build irreps for input modes
        irreps_in = o3.Irreps(
            (mode_channels, (l, (-1) ** l)) for l in range(maxl + 1)
        )

        # Self-coupling: irreps_in x irreps_in → irreps_out
        # Use o3.FullyConnectedTensorProduct for learnable coupling
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_in,
            irreps_out=irreps_in,
        )

        # Scalar path weights (invariant conditioning)
        self.scalar_condition = nn.Sequential(
            nn.Linear(mode_channels, mode_channels),
            nn.SiLU(),
            nn.Linear(mode_channels, self.tp.weight_numel),
        )

    def forward(
        self, O: dict[int, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        """Apply CG coupling to MTO modes.

        Args:
            O: dict l -> [K, C, 2l+1] MTO modes (per mode)

        Returns:
            O_coupled: dict l -> [K, C_coupled, 2l+1] coupled modes
        """
        K = O[0].shape[0]
        device = O[0].device

        results = {}
        for k in range(K):
            # Build irreps tensor for this mode
            mode_irreps = []
            for l in range(self.maxl + 1):
                if l in O:
                    mode_irreps.append(O[l][k])  # [C, 2l+1]

            # Concatenate into e3nn format: [C_total]
            x = torch.cat([m.reshape(-1) for m in mode_irreps])

            # Scalar condition from l=0 features
            scalar_feat = O[0][k].squeeze(-1)  # [C]
            weights = self.scalar_condition(scalar_feat)  # [weight_numel]

            # Apply CG tensor product
            y = self.tp(x, x, weight=weights)

            # Split output back into per-l tensors
            offset = 0
            out_l = {}
            for l in range(self.maxl + 1):
                dim = self.coupled_channels * (2 * l + 1)
                out_l[l] = y[offset:offset + dim].reshape(
                    self.coupled_channels, 2 * l + 1
                )
                offset += dim

            results[k] = out_l

        # Repack from {k: {l: tensor}} to {l: [K, C, 2l+1]}
        O_coupled = {}
        for l in range(self.maxl + 1):
            O_coupled[l] = torch.stack(
                [results[k][l] for k in range(K)], dim=0
            )
        return O_coupled


class CGCouplingMinimal(nn.Module):
    """Minimal CG coupling: couples l=0+l=1→l=1 and l=0+l=2→l=2 paths only.

    Used when full coupling across all orders is too expensive for smoke tests.
    """

    def __init__(self, mode_channels: int = 64):
        super().__init__()
        self.mode_channels = mode_channels

        # l=0 x l=1 → l=1
        self.tp_1 = o3.FullyConnectedTensorProduct(
            o3.Irreps(f"{mode_channels}x0e"),
            o3.Irreps(f"{mode_channels}x1o"),
            o3.Irreps(f"{mode_channels}x1o"),
        )
        # l=0 x l=2 → l=2
        self.tp_2 = o3.FullyConnectedTensorProduct(
            o3.Irreps(f"{mode_channels}x0e"),
            o3.Irreps(f"{mode_channels}x2e"),
            o3.Irreps(f"{mode_channels}x2e"),
        )
        # l=1 x l=1 → l=0
        self.tp_10 = o3.FullyConnectedTensorProduct(
            o3.Irreps(f"{mode_channels}x1o"),
            o3.Irreps(f"{mode_channels}x1o"),
            o3.Irreps(f"{mode_channels}x0e"),
        )

    def forward(
        self, O: dict[int, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        """Apply minimal CG coupling paths."""
        K = O[0].shape[0]
        device = O[0].device

        out_0, out_1, out_2 = [], [], []

        for k in range(K):
            s = O[0][k].reshape(-1)          # [C]
            v = O[1][k].reshape(-1)          # [C*3]
            t = O[2][k].reshape(-1)          # [C*5]

            # l=0 × l=1 → l=1
            y1 = self.tp_1(s, v)
            out_1.append(y1.reshape(self.mode_channels, 3))

            # l=0 × l=2 → l=2
            y2 = self.tp_2(s, t)
            out_2.append(y2.reshape(self.mode_channels, 5))

            # l=1 × l=1 → l=0
            y0 = self.tp_10(v, v)
            out_0.append(y0.reshape(self.mode_channels, 1))

        return {
            0: torch.stack(out_0, dim=0),
            1: torch.stack(out_1, dim=0),
            2: torch.stack(out_2, dim=0),
        }

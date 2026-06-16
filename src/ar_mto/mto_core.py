"""MTO core: molecular tensor mode assembly.

Implements the central MTO operation:

    O_k^(l) = sum_i c_ki^(l) * (W_l @ H_i^(l))

where:
  - H_i^(l): atom i tensor feature at order l, shape [N, C_in, 2l+1]
  - W_l: l-wise channel mixing, shape [C_out, C_in]
  - c_ki^(l): signed invariant routing coefficient, shape [K, N, 1]
  - O_k^(l): assembled molecular tensor mode k, shape [K, C_out, 2l+1]

Supports both full tensor MTO and scalar_only_mto ablation via config.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MTOModeAssembly(nn.Module):
    """Assemble molecular tensor modes from atom-level equivariant features.

    Args:
        num_features: input feature dimension (C_in)
        mode_channels: output channels per mode (C_out)
        num_modes: number of molecular modes K
        maxl: maximum tensor order
        scalar_only: if True, only assemble l=0 modes (ablation baseline)
    """

    def __init__(
        self,
        num_features: int = 128,
        mode_channels: int = 64,
        num_modes: int = 8,
        maxl: int = 3,
        scalar_only: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.mode_channels = mode_channels
        self.num_modes = num_modes
        self.maxl = maxl
        self.scalar_only = scalar_only

        # l-wise channel mixing weights W_l: [C_out, C_in] per l
        self.W = nn.ParameterDict()
        orders = [0] if scalar_only else list(range(maxl + 1))
        for l in orders:
            self.W[str(l)] = nn.Parameter(
                torch.randn(mode_channels, num_features) * 0.02
            )

        # Bias per mode per order
        self.bias = nn.ParameterDict()
        for l in orders:
            self.bias[str(l)] = nn.Parameter(
                torch.zeros(num_modes, mode_channels, 2 * l + 1 if l > 0 else 1)
            )

    def _active_orders(self) -> list[int]:
        if self.scalar_only:
            return [0]
        return list(range(self.maxl + 1))

    def forward(
        self,
        h: dict[int, torch.Tensor],
        coeffs: dict[int, torch.Tensor],
    ) -> dict[int, torch.Tensor]:
        """Assemble molecular tensor modes.

        Args:
            h: dict l -> [N, C_in, 2l+1] atom-level tensor features
            coeffs: dict l -> [K, N, 1] signed routing coefficients

        Returns:
            O: dict l -> [K, C_out, 2l+1] molecular tensor modes
        """
        O = {}
        for l in self._active_orders():
            h_l = h[l]                      # [N, C_in, 2l+1]
            c_l = coeffs.get(l, coeffs[0])  # [K, N, 1]
            W_l = self.W[str(l)]            # [C_out, C_in]

            # Channel mix: [N, C_in, 2l+1] → [N, C_out, 2l+1]
            mixed = torch.einsum("oc,ncm->nom", W_l, h_l)

            # Mode assembly: sum_i c_ki * mixed_i
            # c_l: [K, N, 1], mixed: [N, C_out, 2l+1]
            # → O_l: [K, C_out, 2l+1]
            O_l = torch.einsum("knm,ncm->kcm", c_l, mixed)

            # Add per-mode bias
            O[l] = O_l + self.bias[str(l)]

        return O

    def forward_with_masks(
        self,
        h: dict[int, torch.Tensor],
        coeffs: dict[int, torch.Tensor],
        mode_mask: torch.Tensor | None = None,
    ) -> dict[int, torch.Tensor]:
        """Assemble with optional mode masking (for valence-adaptive K)."""
        O = self.forward(h, coeffs)
        if mode_mask is not None:
            # mode_mask: [K] boolean, True = active
            mask = mode_mask.view(-1, 1, 1).to(O[0].device)
            O = {l: o_l * mask for l, o_l in O.items()}
        return O


class ScalarOnlyMTO(MTOModeAssembly):
    """Scalar-only MTO ablation: assembles only l=0 modes from scalar features."""

    def __init__(self, num_features: int = 128, mode_channels: int = 64,
                 num_modes: int = 8):
        super().__init__(
            num_features=num_features,
            mode_channels=mode_channels,
            num_modes=num_modes,
            maxl=0,
            scalar_only=True,
        )

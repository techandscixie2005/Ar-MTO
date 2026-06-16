"""Tensor adapter for DetaNet latent features.

Consumes DetaNet latent output (S, T) and produces typed tensor irreps:

    h0: [N, C, 1]   — l=0 scalar
    h1: [N, C, 3]   — l=1 vector (odd parity)
    h2: [N, C, 5]   — l=2 traceless tensor (even parity)
    h3: [N, C, 7]   — l=3 tensor (odd parity)

where C = num_features (default 128). All operations preserve irrep structure:
channel mixing only within the same l, no flattening across orders.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from e3nn import o3


class TensorAdapter(nn.Module):
    """Split DetaNet flat T tensor into per-l irrep blocks and reconstruct.

    T layout (num_features=128, maxl=3):
        l=1: T[:, 0:384]      → [N, 128, 3]   (128x1o)
        l=2: T[:, 384:1024]   → [N, 128, 5]   (128x2e)
        l=3: T[:, 1024:1920]  → [N, 128, 7]   (128x3o)
    """

    def __init__(self, num_features: int = 128, maxl: int = 3):
        super().__init__()
        self.num_features = num_features
        self.maxl = maxl

        blocks = []
        offset = 0
        for l in range(1, maxl + 1):
            parity = (-1) ** l
            multiplicity = num_features
            dim = multiplicity * (2 * l + 1)
            blocks.append({
                "l": l,
                "parity": "even" if parity == 1 else "odd",
                "multiplicity": multiplicity,
                "total_dim": dim,
                "flat_start": offset,
                "flat_end": offset + dim,
            })
            offset += dim

        self.vdim = offset  # total flat T dimension
        self._blocks = blocks
        self._irreps_T = o3.Irreps(
            (num_features, (l, (-1) ** l)) for l in range(1, maxl + 1)
        )

    @property
    def irreps_T(self) -> o3.Irreps:
        return self._irreps_T

    @property
    def blocks(self) -> list[dict]:
        return self._blocks

    def forward(self, S: torch.Tensor, T: torch.Tensor) -> dict[int, torch.Tensor]:
        """Split (S, T) into typed tensor dict.

        Args:
            S: scalar features [N, num_features]
            T: flat tensor features [N, vdim]

        Returns:
            dict mapping l -> tensor of shape [N, num_features, 2*l+1]
            0: [N, num_features, 1]
            1: [N, num_features, 3]
            2: [N, num_features, 5]
            3: [N, num_features, 7]
        """
        h = {0: S.unsqueeze(-1)}  # [N, C, 1]
        for b in self._blocks:
            l = b["l"]
            sliced = T[:, b["flat_start"]:b["flat_end"]]
            h[l] = sliced.reshape(T.shape[0], self.num_features, 2 * l + 1)
        return h

    def reconstruct(self, h: dict[int, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct flat (S, T) from typed tensor dict.

        Args:
            h: dict mapping l -> tensor [N, num_features, 2*l+1]

        Returns:
            S: scalar features [N, num_features]
            T: flat tensor features [N, vdim]
        """
        N = h[0].shape[0]
        S = h[0].squeeze(-1)  # [N, C]
        T = torch.zeros(N, self.vdim, dtype=S.dtype, device=S.device)
        for b in self._blocks:
            l = b["l"]
            T[:, b["flat_start"]:b["flat_end"]] = h[l].reshape(N, b["total_dim"])
        return S, T

    def channel_mix(self, h: dict[int, torch.Tensor],
                    weights: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Apply l-wise channel mixing: h_l_mixed = W_l @ h_l.

        Args:
            h: dict l -> [N, C_in, 2l+1]
            weights: dict l -> [C_out, C_in] linear map per l

        Returns:
            dict l -> [N, C_out, 2l+1]
        """
        out = {}
        for l, h_l in h.items():
            if l in weights:
                w = weights[l]  # [C_out, C_in]
                # h_l: [N, C_in, 2l+1] → w @ h_l → [N, C_out, 2l+1]
                out[l] = torch.einsum("oc,ncm->nom", w, h_l)
            else:
                out[l] = h_l
        return out


def make_adapter(num_features: int = 128, maxl: int = 3) -> TensorAdapter:
    """Create a TensorAdapter with the standard DetaNet T layout."""
    return TensorAdapter(num_features=num_features, maxl=maxl)

"""Signed invariant routing for MTO mode assembly.

Generates routing coefficients c_ki^(l) from invariant (scalar) information only.
Each coefficient is the product of a positive normalized attention weight and a
signed modulation term:

    c_ki^(l) = attn_ki^(l) * sign_ki^(l)

where attn is softmax-normalized over atoms and sign = tanh(raw_sign).

All inputs to the routing network are invariant under rotation:
  - h0 scalar features (l=0)
  - tensor norms from h1/h2/h3
  - learned mode embeddings
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SignedRouter(nn.Module):
    """Generate signed invariant routing coefficients for MTO mode assembly.

    Args:
        num_features: scalar feature dimension (C)
        num_modes: number of molecular tensor modes K
        hidden_dim: hidden dimension for routing MLPs
        use_tensor_norms: include l=1,2,3 tensor norms as routing input
        maxl: highest tensor order available (for norms)
    """

    def __init__(
        self,
        num_features: int = 128,
        num_modes: int = 8,
        hidden_dim: int = 64,
        use_tensor_norms: bool = True,
        maxl: int = 3,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_modes = num_modes
        self.use_tensor_norms = use_tensor_norms
        self.maxl = maxl

        # Mode embeddings: one learned vector per mode
        self.mode_embed = nn.Parameter(torch.randn(num_modes, hidden_dim) * 0.02)

        # Invariant feature dimension
        inv_dim = num_features  # h0 scalars per atom
        if use_tensor_norms:
            inv_dim += maxl  # one norm per l>0

        # Per-atom feature → hidden
        self.atom_net = nn.Sequential(
            nn.Linear(inv_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Mode × atom interaction → attention logits
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.sign_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Final projections per mode
        self.attn_out = nn.Linear(hidden_dim, 1, bias=False)
        self.sign_out = nn.Linear(hidden_dim, 1, bias=False)

    def _invariant_features(
        self, h: dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """Build per-atom invariant feature vector.

        Args:
            h: dict l -> tensor [N, C, 2l+1]

        Returns:
            inv: [N, inv_dim] scalar per-atom features
        """
        # h0 scalars: [N, C, 1] → [N, C]
        feats = [h[0].squeeze(-1)]

        if self.use_tensor_norms:
            for l in range(1, self.maxl + 1):
                if l in h:
                    # tensor norm per channel then mean: [N, C, 2l+1] → [N]
                    norms = torch.norm(h[l], dim=-1).mean(dim=-1)  # [N]
                    feats.append(norms.unsqueeze(-1))  # [N, 1]

        return torch.cat(feats, dim=-1)  # [N, inv_dim]

    def forward(
        self, h: dict[int, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        """Compute signed routing coefficients for each tensor order.

        Args:
            h: dict l -> tensor [N, C, 2l+1]

        Returns:
            coeffs: dict l -> tensor [K, N, 1]
                Signed routing coefficient for each mode k, atom i, order l.
        """
        N = h[0].shape[0]
        K = self.num_modes
        device = h[0].device

        # Per-atom invariant features: [N, inv_dim]
        inv = self._invariant_features(h)
        atom_hidden = self.atom_net(inv)  # [N, H]

        # Mode embeddings: [K, H]
        mode = self.mode_embed.to(device)

        # Interaction: mode × atom
        # attn_logits = mode @ attn_proj(atom)^T → [K, N]
        attn_logits = torch.einsum("kh,nh->kn", mode, self.attn_proj(atom_hidden))
        # sign_raw = mode @ sign_proj(atom)^T → [K, N]
        sign_raw = torch.einsum("kh,nh->kn", mode, self.sign_proj(atom_hidden))

        # Normalize attention over atoms (positive, sums to 1)
        attn = F.softmax(attn_logits, dim=-1)  # [K, N]

        # Signed modulation in [-1, 1]
        sign = torch.tanh(sign_raw)  # [K, N]

        # Combined coefficient
        coeff = attn * sign  # [K, N]
        coeff = coeff.unsqueeze(-1)  # [K, N, 1]

        # Same routing for all l orders (scalar generated, invariant)
        max_l = max(h.keys())
        return {l: coeff for l in range(max_l + 1)}

    def route_stats(self, coeffs: dict[int, torch.Tensor]) -> dict:
        """Compute routing statistics for logging and diagnostics."""
        c0 = coeffs[0]  # [K, N, 1]
        return {
            "route_entropy": -(
                F.softmax(c0.squeeze(-1), dim=-1)
                * F.log_softmax(c0.squeeze(-1), dim=-1)
            )
            .sum(dim=-1)
            .mean()
            .item(),
            "route_mean_abs": c0.abs().mean().item(),
            "route_std": c0.std().item(),
            "route_pos_frac": (c0 > 0).float().mean().item(),
        }

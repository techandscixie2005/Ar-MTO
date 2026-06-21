"""Signed invariant routing for MTO mode assembly — batch-aware, order-specific signs.

Generates routing coefficients c_ki^(l,p) from invariant (scalar) information only.
Each coefficient is the product of a molecule-normalized positive attention weight
and an order-specific signed modulation term:

    c_ki^(l,p) = attn_ki^(l,p) * sign_ki^(l,p)

L2 normalization (canonical default):
    attn_ki is normalized such that sum_i (attn_ki)^2 = 1 per mode within each molecule

Abs-value normalization (configurable alternative):
    sum_i |c_ki| = 1 per mode within each molecule

All inputs to the routing network are invariant under rotation:
  - h0 scalar features (l=0)
  - tensor norms from h1/h2/h3
  - learned mode embeddings
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPSILON = 1e-8


def _per_molecule_segment_sum(
    x: torch.Tensor, batch: torch.Tensor, num_molecules: int
) -> torch.Tensor:
    """Sum x over atoms within each molecule.

    Args:
        x: [K, N, ...]  values per mode per atom
        batch: [N] molecule index per atom
        num_molecules: B

    Returns:
        [K, B, ...]  sum per mode per molecule
    """
    out = x.new_zeros(x.shape[0], num_molecules, *x.shape[2:])
    for b_idx in range(num_molecules):
        mask = (batch == b_idx)
        out[:, b_idx] = x[:, mask].sum(dim=1)
    return out


def _per_molecule_softmax(
    logits: torch.Tensor, batch: torch.Tensor, num_molecules: int
) -> torch.Tensor:
    """Softmax over atoms within each molecule separately.

    Args:
        logits: [K, N]  logits per mode per atom
        batch: [N] molecule index per atom
        num_molecules: B

    Returns:
        [K, N]  normalized attention (sums to 1 per mode per molecule)
    """
    out = logits.new_zeros(logits.shape[0], logits.shape[1])
    max_vals = logits.new_full((logits.shape[0], num_molecules), -float("inf"))
    for b_idx in range(num_molecules):
        mask = (batch == b_idx)
        if mask.any():
            vals = logits[:, mask]
            max_vals[:, b_idx] = vals.max(dim=1).values
            out[:, mask] = vals - max_vals[:, b_idx, None]
            out[:, mask] = out[:, mask].exp()
            denom = out[:, mask].sum(dim=1, keepdim=True) + _EPSILON
            out[:, mask] = out[:, mask] / denom
    return out


def _per_molecule_l2_norm(
    x: torch.Tensor, batch: torch.Tensor, num_molecules: int
) -> torch.Tensor:
    """L2-normalize coefficients within each molecule: sum_i c_ki^2 = 1.

    Args:
        x: [K, N]  raw or signed coefficients per mode per atom
        batch: [N] molecule index per atom
        num_molecules: B

    Returns:
        [K, N]  L2-normalized coefficients
    """
    out = x.new_zeros(x.shape[0], x.shape[1])
    for b_idx in range(num_molecules):
        mask = (batch == b_idx)
        if mask.any():
            vals = x[:, mask]
            l2 = vals.pow(2).sum(dim=1, keepdim=True).sqrt()
            l2 = l2 + _EPSILON
            out[:, mask] = vals / l2
    return out


def _per_molecule_abs_norm(
    x: torch.Tensor, batch: torch.Tensor, num_molecules: int
) -> torch.Tensor:
    """Abs-value normalize within each molecule: sum_i |c_ki| = 1.

    Args:
        x: [K, N]  signed coefficients per mode per atom
        batch: [N] molecule index per atom
        num_molecules: B

    Returns:
        [K, N]  abs-normalized coefficients
    """
    out = x.new_zeros(x.shape[0], x.shape[1])
    for b_idx in range(num_molecules):
        mask = (batch == b_idx)
        if mask.any():
            vals = x[:, mask]
            denom = vals.abs().sum(dim=1, keepdim=True) + _EPSILON
            out[:, mask] = vals / denom
    return out


class SignedRouter(nn.Module):
    """Generate order-specific signed routing coefficients for MTO mode assembly.

    Args:
        num_features: scalar feature dimension (C)
        num_modes: number of molecular tensor modes Kmax
        hidden_dim: hidden dimension for routing MLPs
        use_tensor_norms: include l=1,2,3 tensor norms as routing input
        maxl: highest tensor order available (for norms)
        normalization: "l2" (canonical) or "abs" (manuscript alternative)
        order_specific_signs: if True, generate separate sign for each l order
    """

    def __init__(
        self,
        num_features: int = 128,
        num_modes: int = 8,
        hidden_dim: int = 64,
        use_tensor_norms: bool = True,
        maxl: int = 3,
        normalization: str = "l2",
        order_specific_signs: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_modes = num_modes
        self.use_tensor_norms = use_tensor_norms
        self.maxl = maxl
        self.normalization = normalization
        self.order_specific_signs = order_specific_signs

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

        # Attention projection
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Sign projections: one per l order if order_specific, else shared
        if order_specific_signs:
            self.sign_proj_list = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim, bias=False)
                for _ in range(maxl + 1)
            ])
        else:
            self.sign_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def _invariant_features(
        self, h: dict[int | tuple, torch.Tensor]
    ) -> torch.Tensor:
        """Build per-atom invariant feature vector.

        Args:
            h: dict keyed by l or (l, p) -> tensor [N, C, 2l+1]

        Returns:
            inv: [N, inv_dim] scalar per-atom features
        """
        # Get l=0 scalars: [N, C, 1] or [N, C] → [N, C]
        key0 = 0
        if 0 not in h and (0, 1) in h:
            key0 = (0, 1)
        h0 = h[key0]
        if h0.dim() == 3:
            h0 = h0.squeeze(-1)
        feats = [h0]

        if self.use_tensor_norms:
            for l in range(1, self.maxl + 1):
                key = None
                if l in h:
                    key = l
                elif (l, 1) in h:
                    key = (l, 1)
                elif (l, -1) in h:
                    key = (l, -1)
                if key is not None:
                    # tensor norm per channel then mean: [N, C, 2l+1] → [N]
                    norms = torch.norm(h[key], dim=-1).mean(dim=-1)
                    feats.append(norms.unsqueeze(-1))
                else:
                    feats.append(torch.zeros(
                        h0.shape[0], 1, dtype=h0.dtype, device=h0.device
                    ))

        return torch.cat(feats, dim=-1)

    def forward(
        self,
        h: dict[int | tuple, torch.Tensor],
        batch: torch.Tensor | None = None,
    ) -> dict[int | tuple, torch.Tensor]:
        """Compute order-specific signed routing coefficients.

        Args:
            h: dict keyed by l or (l, p) -> tensor [N, C, 2l+1]
            batch: [N] molecule index per atom, None for single molecule

        Returns:
            coeffs: dict with same keys as h -> tensor [K, N, 1]
                Signed routing coefficient for each mode k, atom i, order (l,p).
        """
        N = next(iter(h.values())).shape[0]
        K = self.num_modes
        device = next(iter(h.values())).device

        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)
        num_molecules = int(batch.max().item()) + 1

        # Per-atom invariant features: [N, inv_dim]
        inv = self._invariant_features(h)
        atom_hidden = self.atom_net(inv)  # [N, H]

        # Mode embeddings: [K, H]
        mode = self.mode_embed.to(device)

        # Attention: shared across orders
        attn_logits = torch.einsum("kh,nh->kn", mode, self.attn_proj(atom_hidden))
        attn = _per_molecule_softmax(attn_logits, batch, num_molecules)  # [K, N]

        # Generate coefficients per order
        coeffs = {}
        for key in h.keys():
            l = key if isinstance(key, int) else key[0]

            # Order-specific sign
            if self.order_specific_signs:
                sign_hidden = self.sign_proj_list[l](atom_hidden)
            else:
                sign_hidden = self.sign_proj(atom_hidden)
            sign_raw = torch.einsum("kh,nh->kn", mode, sign_hidden)  # [K, N]
            sign = torch.tanh(sign_raw)

            # Raw product
            raw = attn * sign  # [K, N]

            # Per-molecule normalization
            if self.normalization == "l2":
                normed = _per_molecule_l2_norm(raw, batch, num_molecules)
            elif self.normalization == "abs":
                normed = _per_molecule_abs_norm(raw, batch, num_molecules)
            else:
                normed = raw

            coeffs[key] = normed.unsqueeze(-1)  # [K, N, 1]

        return coeffs

    def route_stats(
        self, coeffs: dict[int | tuple, torch.Tensor],
        mode_mask: torch.Tensor | None = None,
    ) -> dict:
        """Compute routing statistics for logging and diagnostics.

        Args:
            coeffs: dict key -> [K, N, 1] routing coefficients
            mode_mask: [B, K] boolean, True = active. Stats computed only over active.

        Returns:
            dict of routing statistics
        """
        c0 = next(iter(coeffs.values())).squeeze(-1)  # [K, N]
        if mode_mask is not None:
            # Only use active mode coefficients
            # coeffs: [K, N] → need mask over K dimension
            mask = mode_mask.to(c0.device)  # [B, K]
            # Per-molecule, per-mode: weight by whether mode is active
            # For simplicity, flatten and filter
            c0_active = c0[mask.any(dim=0)]  # modes that are active in at least one molecule
            if c0_active.numel() > 0:
                c0 = c0_active
        return {
            "route_entropy": -(
                F.softmax(c0, dim=-1) * F.log_softmax(c0 + _EPSILON, dim=-1)
            )
            .sum(dim=-1)
            .mean()
            .item(),
            "route_mean_abs": c0.abs().mean().item(),
            "route_std": c0.std().item(),
            "route_pos_frac": (c0 > 0).float().mean().item(),
            "route_l2_per_mode": c0.pow(2).sum(dim=-1).sqrt().mean().item(),
            "route_abs_sum_per_mode": c0.abs().sum(dim=-1).mean().item(),
        }

    def _compute_active_entropy(
        self,
        coeffs: dict[int | tuple, torch.Tensor],
        mode_mask: torch.Tensor | None = None,
    ) -> float:
        """Compute entropy only over active modes."""
        c0 = next(iter(coeffs.values())).squeeze(-1)  # [K, N]
        if mode_mask is not None:
            c0 = c0[mode_mask]  # only active modes
        return -(
            F.softmax(c0, dim=-1) * F.log_softmax(c0 + _EPSILON, dim=-1)
        ).sum(dim=-1).mean().item()
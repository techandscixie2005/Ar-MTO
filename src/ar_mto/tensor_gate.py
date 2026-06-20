"""Invariant tensor gates for MTO modes — batch-aware, residual mode.

Implements representation-preserving nonlinearity:

    gamma_{b,k,l} = MLP(invariant_features_{b,k})
    O_new = O + alpha * gamma * linear(O)

where invariant_features includes:
  - l=0 scalar channels
  - tensor norms from l>0 orders
  - scalar contractions (l1 ⊗ l2 → 0e) from legal tensor products

This preserves equivariance: gamma is a scalar that multiplies all m-components
of a given (b, k, l) mode pair equally.

Supports three modes:
  - no_gate: identity (ablation)
  - scalar_only_gate: uses only l=0 features
  - tensor_information_gate: uses scalars + norms + contractions (canonical)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from e3nn import o3


def _resolve_key(O: dict, l: int, p: int):
    """Find a key in O matching order l and parity p. Returns None if not found."""
    if l in O:
        return l
    key = (l, p)
    if key in O:
        return key
    return None


def _get_tensor(O: dict, l: int, p: int) -> torch.Tensor | None:
    """Get mode tensor for order (l, p), returning None if absent."""
    if l in O:
        return O[l]
    key = (l, p)
    if key in O:
        return O[key]
    return None


class TensorGate(nn.Module):
    """Invariant gate with residual update for each (molecule, mode, order) triplet.

    Gate = sigmoid(base_gate + tensor_modulation) ∈ (0, 1)
    Residual: O_new = O + alpha * gate * linear_projection(O)

    Args:
        mode_channels: channels per mode (C)
        num_modes: maximum modes Kmax
        maxl: maximum tensor order
        hidden_dim: hidden dimension for gate MLP
        use_tensor_info: include tensor norms/contractions in gate input
        alpha: residual strength (default 0.1 for stable initialization)
    """

    def __init__(
        self,
        mode_channels: int = 64,
        num_modes: int = 8,
        maxl: int = 3,
        hidden_dim: int = 64,
        use_tensor_info: bool = True,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.mode_channels = mode_channels
        self.num_modes = num_modes
        self.maxl = maxl
        self.use_tensor_info = use_tensor_info
        self.alpha = alpha

        # Gate input dimension: C (l=0 scalars per mode) + maxl (norms)
        gate_in_dim = mode_channels + (maxl if use_tensor_info else 0)

        # Per-(mode, order) gate scalar
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, maxl + 1),  # one gate per l order
        )

        # l-wise linear residual projections: [C, C]
        self.residual_proj = nn.ParameterDict()
        for l in range(maxl + 1):
            self.residual_proj[str(l)] = nn.Parameter(
                torch.eye(mode_channels) * 0.01
            )

        # Scalar contraction extractors: for each l>0 pair, compute l⊗l → 0e norm
        self._build_contraction_pairs()

        # Stored gate values for stats
        self._stored_gates: dict[str, torch.Tensor] = {}

    def _build_contraction_pairs(self):
        """Build (l1, l2) pairs for scalar contraction computation."""
        self.contraction_pairs = []
        for l in range(1, self.maxl + 1):
            # Self-contraction: l ⊗ l → 0e
            p = (-1) ** l
            self.contraction_pairs.append((l, p, l, p, 0, 1))  # same parity → even

    @staticmethod
    def _tensor_norm(O_l: torch.Tensor) -> torch.Tensor:
        """Compute invariant norm: ||O||_F per (B, K).

        Args:
            O_l: [B, K, C, 2l+1]

        Returns:
            norm: [B, K] Frobenius norm averaged over channels
        """
        return torch.norm(O_l, dim=-1).mean(dim=-1)  # [B, K]

    @staticmethod
    def _scalar_contraction(
        O_l1: torch.Tensor, O_l2: torch.Tensor
    ) -> torch.Tensor:
        """Compute invariant scalar contraction: trace(O_l1^T @ O_l2) via einsum.

        For l1 == l2: sum over m of dot product over C.

        Args:
            O_l1, O_l2: [B, K, C, 2l+1]

        Returns:
            contraction: [B, K]
        """
        # dot over C and m dims: [B, K, C, sd] * [B, K, C, sd] → [B, K]
        return torch.einsum("bkcm,bkcm->bk", O_l1, O_l2) / O_l1.shape[-1]

    def _invariant_features(
        self, O: dict[int | tuple, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Extract invariant features from MTO modes.

        Args:
            O: dict key -> [B, K, C, 2l+1] MTO modes

        Returns:
            l0_feat: [B, K, C] scalar features from l=0
            tensor_info: [B, K, maxl] tensor norms + contractions, or None
        """
        # l=0 scalars — handle both int and tuple keys
        key0 = _resolve_key(O, 0, 1)
        if key0 is None:
            key0 = next(iter(O.keys()))
        l0_feat = O[key0].squeeze(-1)  # [B, K, C]

        tensor_info = None
        if self.use_tensor_info:
            info_parts = []
            # Get tensors by order — try both key forms
            o1 = _get_tensor(O, 1, -1)
            o2 = _get_tensor(O, 2, 1)
            o3 = _get_tensor(O, 3, -1)

            for l in range(1, self.maxl + 1):
                o_l = {1: o1, 2: o2, 3: o3}.get(l)
                if o_l is not None:
                    info_parts.append(self._tensor_norm(o_l).unsqueeze(-1))
                else:
                    info_parts.append(torch.zeros(
                        l0_feat.shape[0], l0_feat.shape[1], 1,
                        device=l0_feat.device, dtype=l0_feat.dtype
                    ))

            tensor_info = torch.cat(info_parts, dim=-1)  # [B, K, maxl]

        return l0_feat, tensor_info

    def forward(
        self,
        O: dict[int | tuple, torch.Tensor],
        mode_mask: torch.Tensor | None = None,
    ) -> dict[int | tuple, torch.Tensor]:
        """Apply invariant gates with residual updates.

        Args:
            O: dict key -> [B, K, C, 2l+1] MTO modes
            mode_mask: [B, K] boolean, True = active mode

        Returns:
            O_gated: dict key -> [B, K, C, 2l+1] gated modes
        """
        B, K = O[next(iter(O.keys()))].shape[:2]
        device = O[next(iter(O.keys()))].device

        l0_feat, tensor_info = self._invariant_features(O)

        # Build gate input
        gate_in = l0_feat  # [B, K, C]
        if self.use_tensor_info and tensor_info is not None:
            gate_in = torch.cat([gate_in, tensor_info], dim=-1)  # [B, K, C+maxl]

        # Gate values: [B, K, maxl+1]
        gate_vals = torch.sigmoid(self.gate_net(gate_in))

        # Store for stats
        self._stored_gates: dict[str, torch.Tensor] = {
            "raw": gate_vals.detach(),
        }

        # Apply residual gating: O_new = O + alpha * gate * W_res(O)
        O_gated = {}
        for key, o_l in O.items():
            l = key if isinstance(key, int) else key[0]

            g_l = gate_vals[:, :, l].unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
            W_res = self.residual_proj[str(l)]  # [C, C]

            # Residual projection: [B, K, C, sd] → [B, K, C, sd]
            residual = torch.einsum("oc,bkcd->bkod", W_res, o_l)

            # Gated residual update
            O_gated[key] = o_l + self.alpha * g_l * residual

            # Zero out masked modes
            if mode_mask is not None:
                mask = mode_mask.to(device)
                if mask.shape[1] < K:
                    pad = torch.zeros(
                        mask.shape[0], K - mask.shape[1],
                        dtype=mask.dtype, device=mask.device,
                    )
                    mask = torch.cat([mask, pad], dim=1)
                m = mask.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
                O_gated[key] = O_gated[key] * m

        return O_gated

    def gate_stats(self, O: dict[int | tuple, torch.Tensor]) -> dict:
        """Return gate statistics from last forward pass."""
        if not hasattr(self, "_stored_gates") or not self._stored_gates:
            gates = torch.ones(1, 1, self.maxl + 1)  # fallback
        else:
            gates = self._stored_gates["raw"]

        result = {}
        for l in range(self.maxl + 1):
            g_l = gates[:, :, l]
            result[f"gate_l{l}_mean"] = g_l.mean().item()
            result[f"gate_l{l}_std"] = g_l.std().item()
            result[f"gate_l{l}_min"] = g_l.min().item()
            result[f"gate_l{l}_max"] = g_l.max().item()
            result[f"gate_l{l}_saturation"] = (
                ((g_l < 0.05) | (g_l > 0.95)).float().mean().item()
            )
        return result


class NoGate(nn.Module):
    """Identity gate: O_new = O (for ablation)."""

    def __init__(
        self,
        mode_channels: int = 64,
        num_modes: int = 8,
        maxl: int = 3,
    ):
        super().__init__()
        self.mode_channels = mode_channels
        self.num_modes = num_modes
        self.maxl = maxl
        self.use_tensor_info = False
        self._stored_gates = {}

    def forward(
        self,
        O: dict[int | tuple, torch.Tensor],
        mode_mask: torch.Tensor | None = None,
    ) -> dict[int | tuple, torch.Tensor]:
        return O

    def gate_stats(self, O: dict[int | tuple, torch.Tensor]) -> dict:
        return {"gate": "identity"}


class ScalarOnlyGate(nn.Module):
    """Gate using only l=0 scalar features (ablation between NoGate and full)."""

    def __init__(
        self,
        mode_channels: int = 64,
        num_modes: int = 8,
        maxl: int = 3,
        hidden_dim: int = 64,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.mode_channels = mode_channels
        self.num_modes = num_modes
        self.maxl = maxl
        self.use_tensor_info = False
        self.alpha = alpha
        self._stored_gates = {}

        self.gate_net = nn.Sequential(
            nn.Linear(mode_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, maxl + 1),
        )

        self.residual_proj = nn.ParameterDict()
        for l in range(maxl + 1):
            self.residual_proj[str(l)] = nn.Parameter(
                torch.eye(mode_channels) * 0.01
            )

    def forward(
        self,
        O: dict[int | tuple, torch.Tensor],
        mode_mask: torch.Tensor | None = None,
    ) -> dict[int | tuple, torch.Tensor]:
        B, K = O[next(iter(O.keys()))].shape[:2]
        device = O[next(iter(O.keys()))].device

        key0 = 0 if 0 in O else (0, 1)
        l0_feat = O[key0].squeeze(-1)  # [B, K, C]

        gate_vals = torch.sigmoid(self.gate_net(l0_feat))  # [B, K, maxl+1]
        self._stored_gates = {"raw": gate_vals.detach()}

        O_gated = {}
        for key, o_l in O.items():
            l = key if isinstance(key, int) else key[0]
            g_l = gate_vals[:, :, l].unsqueeze(-1).unsqueeze(-1)
            W_res = self.residual_proj[str(l)]
            residual = torch.einsum("oc,bkcd->bkod", W_res, o_l)
            O_gated[key] = o_l + self.alpha * g_l * residual

            if mode_mask is not None:
                mask = mode_mask.to(device).unsqueeze(-1).unsqueeze(-1)
                O_gated[key] = O_gated[key] * mask

        return O_gated

    def gate_stats(self, O: dict[int | tuple, torch.Tensor]) -> dict:
        if not self._stored_gates:
            return {"scalar_only_gate": "no_stats"}
        gates = self._stored_gates["raw"]
        return {
            "sgate_mean": gates.mean().item(),
            "sgate_std": gates.std().item(),
        }
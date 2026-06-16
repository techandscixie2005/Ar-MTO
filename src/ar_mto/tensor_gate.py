"""Invariant tensor gates for MTO modes.

Implements representation-preserving nonlinearity:

    gamma_k_l = MLP(invariant_features)
    O_tilde_k^(l) = gamma_k_l * O_k^(l)

where invariant_features includes:
  - l=0 scalar features
  - tensor norms from l>0 orders
  - scalar contractions (l1·l2 invariants)

This preserves equivariance: gamma is a scalar that multiplies all m-components
of a given (k, l) mode equally.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TensorGate(nn.Module):
    """Invariant gate that multiplies each mode-order pair by a learned scalar.

    Args:
        mode_channels: channels per mode
        num_modes: number of molecular modes K
        maxl: maximum tensor order
        hidden_dim: hidden dimension for gate MLP
        use_tensor_info: include tensor norms/contractions in gate input
    """

    def __init__(
        self,
        mode_channels: int = 64,
        num_modes: int = 8,
        maxl: int = 3,
        hidden_dim: int = 64,
        use_tensor_info: bool = True,
    ):
        super().__init__()
        self.mode_channels = mode_channels
        self.num_modes = num_modes
        self.maxl = maxl
        self.use_tensor_info = use_tensor_info

        # Gate input: per-mode scalar features + optional tensor info
        gate_in_dim = mode_channels  # l=0 scalar per mode

        # Per-mode l=0 → gate scalar per (mode, order)
        self.gate_net = nn.Sequential(
            nn.Linear(mode_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, maxl + 1),  # one gate per order
            nn.Sigmoid(),  # gate in [0, 1] — multiplicative, non-negative
        )

        # Optional tensor-norm pathway
        if use_tensor_info:
            self.tensor_norm_net = nn.Sequential(
                nn.Linear(maxl, hidden_dim // 2),  # norms for l=1..maxl
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, maxl + 1),
                nn.Tanh(),  # modulation in [-1, 1]
            )
            self.gate_bias = nn.Parameter(torch.ones(num_modes, maxl + 1))

    def _invariant_features(
        self, O: dict[int, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Extract invariant features from MTO modes.

        Args:
            O: dict l -> [K, C, 2l+1] MTO modes

        Returns:
            l0_feat: [K, C] scalar features from l=0
            tensor_norms: [K, maxl] tensor norms per mode, or None
        """
        l0_feat = O[0].squeeze(-1)  # [K, C]

        tensor_norms = None
        if self.use_tensor_info:
            norms_list = []
            for l in range(1, self.maxl + 1):
                if l in O:
                    # [K, C, 2l+1] → norm over spatial dim → [K, C] → mean → [K]
                    n = torch.norm(O[l], dim=-1).mean(dim=-1)  # [K]
                    norms_list.append(n)
                else:
                    norms_list.append(torch.zeros(
                        O[0].shape[0], device=O[0].device
                    ))
            tensor_norms = torch.stack(norms_list, dim=-1)  # [K, maxl]

        return l0_feat, tensor_norms

    def forward(
        self, O: dict[int, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        """Apply invariant gates to MTO modes.

        Args:
            O: dict l -> [K, C, 2l+1] MTO modes

        Returns:
            O_gated: dict l -> [K, C, 2l+1] gated modes
        """
        K = O[0].shape[0]
        device = O[0].device

        l0_feat, tensor_norms = self._invariant_features(O)

        # Base gate from scalar features: [K, maxl+1]
        gates = self.gate_net(l0_feat)

        # Optional tensor-norm modulation
        if self.use_tensor_info and tensor_norms is not None:
            modulation = self.tensor_norm_net(tensor_norms)  # [K, maxl+1]
            gates = gates + modulation * 0.1  # mild modulation
            gates = gates + self.gate_bias.to(device)
            gates = torch.sigmoid(gates)  # re-normalize

        # Apply gate per mode per order
        O_gated = {}
        for l in range(self.maxl + 1):
            if l in O:
                g_l = gates[:, l].view(K, 1, 1)  # [K, 1, 1]
                O_gated[l] = O[l] * g_l

        return O_gated

    def gate_stats(self, O: dict[int, torch.Tensor]) -> dict:
        """Return gate statistics for logging."""
        _, gates = self._invariant_features(O)
        # Use forward pass to get actual gates
        l0_feat, tensor_norms = self._invariant_features(O)
        gates = self.gate_net(l0_feat)
        return {
            "gate_mean": gates.mean().item(),
            "gate_std": gates.std().item(),
            "gate_min": gates.min().item(),
            "gate_max": gates.max().item(),
        }


class NoGate(TensorGate):
    """Identity gate: O_tilde = O (for ablation)."""

    def __init__(self, mode_channels: int = 64, num_modes: int = 8,
                 maxl: int = 3):
        super().__init__(mode_channels, num_modes, maxl)
        self.maxl = maxl

    def forward(
        self, O: dict[int, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        return O

    def gate_stats(self, O: dict[int, torch.Tensor]) -> dict:
        return {"gate": "identity"}

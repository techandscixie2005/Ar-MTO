"""MTO core: batch-aware molecular tensor mode assembly.

Implements the central MTO operation per molecule within a batch:

    O_{b,k}^{(l,p)} = sum_{i in mol b} c_{b,k,i}^{(l,p)} * (W_l @ H_i^{(l,p)})

where:
  - H_i^{(l,p)}: atom i tensor feature at order (l,p), shape [N, C_in, 2*l+1]
  - W_(l,p): (l,p)-wise channel mixing, shape [C_out, C_in]
  - c_{b,k,i}^{(l,p)}: signed invariant routing coefficient, shape [Kmax, N, 1]
  - O_{b,k}^{(l,p)}: assembled molecular tensor mode k, shape [B, Kmax, C_out, 2*l+1]
  - mode_mask: [B, Kmax] boolean, True = active mode for that molecule

No cross-molecule aggregation. Every molecule has its own Kmax mode bank.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _per_molecule_assembly(
    h_l: torch.Tensor,
    coeff_l: torch.Tensor,
    W_l: torch.Tensor,
    batch: torch.Tensor,
    num_molecules: int,
    K: int,
    C_out: int,
    spatial_dim: int,
) -> torch.Tensor:
    """Assemble MTO modes per molecule, producing [B, K, C_out, spatial_dim].

    Args:
        h_l: [N, C_in, spatial_dim] atom features
        coeff_l: [K, N, 1] routing coefficients
        W_l: [C_out, C_in] channel mix matrix
        batch: [N] molecule index per atom
        num_molecules: B
        K: number of modes (Kmax)
        C_out: output channels
        spatial_dim: 2*l+1

    Returns:
        O_l: [B, K, C_out, spatial_dim]
    """
    # Channel mix: [N, C_in, sd] → [N, C_out, sd]
    mixed = torch.einsum("oc,ncd->nod", W_l, h_l)

    # Initialize output
    O_l = mixed.new_zeros(num_molecules, K, C_out, spatial_dim)

    for b_idx in range(num_molecules):
        mol_mask = (batch == b_idx)
        if not mol_mask.any():
            continue
        mol_mixed = mixed[mol_mask]           # [n_mol, C_out, sd]
        mol_coeff = coeff_l[:, mol_mask, :]   # [K, n_mol, 1]

        # sum_i c_{k,i} * mixed_i → [K, C_out, sd]
        assembled = torch.einsum("knm,ncm->kcm", mol_coeff, mol_mixed)
        O_l[b_idx] = assembled

    return O_l


class MTOModeAssembly(nn.Module):
    """Assemble molecular tensor modes from atom-level equivariant features.

    Supports both full tensor MTO and scalar_only_mto ablation via config.

    Args:
        num_features: input feature dimension (C_in)
        mode_channels: output channels per mode (C_out)
        num_modes: maximum number of molecular modes Kmax
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
        # Keyed by order l (0, 1, 2, 3)
        self.W = nn.ParameterDict()
        orders = [0] if scalar_only else list(range(maxl + 1))
        for l in orders:
            self.W[str(l)] = nn.Parameter(
                torch.randn(mode_channels, num_features) * 0.02
            )

        # Per-mode bias per order: [1, Kmax, C_out, 2*l+1]
        self.bias = nn.ParameterDict()
        for l in orders:
            self.bias[str(l)] = nn.Parameter(
                torch.zeros(1, num_modes, mode_channels, 2 * l + 1 if l > 0 else 1)
            )

    def _active_orders(self) -> list[int]:
        if self.scalar_only:
            return [0]
        return list(range(self.maxl + 1))

    def forward(
        self,
        h: dict[int | tuple, torch.Tensor],
        coeffs: dict[int | tuple, torch.Tensor],
        batch: torch.Tensor | None = None,
    ) -> dict[int | tuple, torch.Tensor]:
        """Assemble molecular tensor modes per molecule.

        Args:
            h: dict key -> [N, C_in, 2l+1] atom-level tensor features
            coeffs: dict key -> [Kmax, N, 1] signed routing coefficients
            batch: [N] molecule index per atom, None for single molecule

        Returns:
            O: dict key -> [B, Kmax, C_out, 2l+1] molecular tensor modes
        """
        N = next(iter(h.values())).shape[0]
        device = next(iter(h.values())).device

        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)
        num_molecules = int(batch.max().item()) + 1
        K = self.num_modes
        C_out = self.mode_channels

        O = {}
        for key in h.keys():
            l = key if isinstance(key, int) else key[0]
            if not self.scalar_only and l > self.maxl:
                continue
            if self.scalar_only and l > 0:
                continue

            h_l = h[key]                                  # [N, C_in, 2l+1]
            coeff_l = coeffs.get(key, coeffs[list(coeffs.keys())[0]])  # [K, N, 1]
            W_l = self.W[str(l)]                          # [C_out, C_in]
            spatial_dim = h_l.shape[-1]                   # 2*l+1

            O_l = _per_molecule_assembly(
                h_l=h_l, coeff_l=coeff_l, W_l=W_l,
                batch=batch, num_molecules=num_molecules,
                K=K, C_out=C_out, spatial_dim=spatial_dim,
            )

            # Add bias: [B, K, C_out, sd]
            bias = self.bias[str(l)]  # [1, K, C_out, sd]
            O[key] = O_l + bias

        return O

    def forward_with_masks(
        self,
        h: dict[int | tuple, torch.Tensor],
        coeffs: dict[int | tuple, torch.Tensor],
        mode_mask: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> dict[int | tuple, torch.Tensor]:
        """Assemble with optional mode masking (for valence-adaptive K).

        Args:
            h: dict key -> [N, C_in, 2l+1]
            coeffs: dict key -> [Kmax, N, 1]
            mode_mask: [B, Kmax] boolean, True = active
            batch: [N] molecule index per atom

        Returns:
            O: dict key -> [B, Kmax, C_out, 2l+1]
        """
        O = self.forward(h, coeffs, batch)
        if mode_mask is not None:
            # Pad mask to MTO's Kmax if needed (compute_valence_adaptive_k
            # may return a mask narrower than self.num_modes)
            K = self.num_modes
            mask = mode_mask.to(O[next(iter(O.keys()))].device)
            if mask.shape[1] < K:
                pad = torch.zeros(
                    mask.shape[0], K - mask.shape[1],
                    dtype=mask.dtype, device=mask.device,
                )
                mask = torch.cat([mask, pad], dim=1)
            # mode_mask: [B, Kmax] → [B, Kmax, 1, 1]
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)
            O = {key: val * mask_expanded for key, val in O.items()}
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


def compute_valence_adaptive_k(
    z: torch.Tensor,
    batch: torch.Tensor | None = None,
    electrons_per_valence: dict[int, int] | None = None,
    max_modes: int = 32,
    k_min: int = 1,
    k_rounding: str = "ceil",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute K per molecule from neutral-atom valence electron counts.

    K = clamp(ceil(N_val / 2), min=k_min, max=max_modes)

    Args:
        z: [N] atomic numbers
        batch: [N] molecule index per atom
        electrons_per_valence: dict mapping Z → valence electrons
        max_modes: upper bound on K (k_max)
        k_min: lower bound on K
        k_rounding: "ceil" (default) or "floor"

    Returns:
        mode_mask: [B, max_modes] boolean mask
        ks: [B] int K values per molecule
    """
    if electrons_per_valence is None:
        # Standard neutral atom valence counts (main group + transition metals)
        electrons_per_valence = {
            1: 1, 2: 2,
            3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
            11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8,
            19: 1, 20: 2,
            21: 3, 22: 4, 23: 5, 24: 6, 25: 7, 26: 8, 27: 9, 28: 10,
            29: 11, 30: 12,
            31: 3, 32: 4, 33: 5, 34: 6, 35: 7, 36: 8,
            37: 1, 38: 2,
            39: 3, 40: 4, 41: 5, 42: 6, 43: 7, 44: 8, 45: 9, 46: 10,
            47: 11, 48: 12,
            49: 3, 50: 4, 51: 5, 52: 6, 53: 7, 54: 8,
            55: 1, 56: 2,
            57: 3, 58: 4, 59: 5, 60: 6, 61: 7, 62: 8, 63: 9, 64: 10,
            65: 11, 66: 12, 67: 13, 68: 14, 69: 15, 70: 16,
            71: 3, 72: 4, 73: 5, 74: 6, 75: 7, 76: 8, 77: 9, 78: 10,
            79: 11, 80: 12,
            81: 3, 82: 4, 83: 5, 84: 6, 85: 7, 86: 8,
        }

    N = z.shape[0]
    device = z.device
    if batch is None:
        batch = torch.zeros(N, dtype=torch.long, device=device)
    num_molecules = int(batch.max().item()) + 1

    ks = torch.zeros(num_molecules, dtype=torch.long, device=device)
    z_np = z.cpu().tolist()
    batch_np = batch.cpu().tolist()

    for i in range(N):
        b_idx = batch_np[i]
        z_val = int(z_np[i])
        if z_val not in electrons_per_valence:
            raise ValueError(
                f"Unsupported element Z={z_val} for valence-adaptive K. "
                f"Add valence count or use fixed_k mode."
            )
        ks[b_idx] += electrons_per_valence[z_val]

    # K = rounding(N_val / 2), clamped to [k_min, max_modes]
    if k_rounding == "floor":
        ks = (ks // 2)
    else:  # ceil
        ks = (ks + 1) // 2
    ks = torch.clamp(ks, min=k_min, max=max_modes)

    # Always return mask of shape [num_molecules, max_modes] for
    # dimension compatibility with MTO/Gate/Readout modules.
    mode_mask = torch.zeros(num_molecules, max_modes, dtype=torch.bool, device=device)
    for b_idx in range(num_molecules):
        mode_mask[b_idx, :ks[b_idx].item()] = True

    return mode_mask, ks
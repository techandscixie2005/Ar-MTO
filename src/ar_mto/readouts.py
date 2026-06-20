"""Readout heads for MTO modes — batch-aware, mode-mask-respecting.

All readouts produce per-molecule predictions:
  - scalar:      [B, out_dim]
  - vector:      [B, out_dim, 3]
  - rank-2:      [B, out_dim, 3, 3]
  - spectral:    [B, num_spectral_bins]

All weights are invariant. Mode masks are respected.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from e3nn import o3


def _resolve_keys(O: dict, l: int, p: int) -> list:
    """Find keys in O matching order l. Returns list of matching keys."""
    matches = []
    if l in O:
        matches.append(l)
    key = (l, p)
    if key in O:
        matches.append(key)
    return matches


def _get_order(O: dict, l: int, p: int | None = None) -> torch.Tensor | None:
    """Get mode tensor for order l, returning None if absent.

    Tries O[l] first, then O[(l, p)] if p is provided.
    """
    if l in O:
        return O[l]
    if p is not None:
        key = (l, p)
        if key in O:
            return O[key]
    # Fallback: try any tuple key with matching l
    for k in O:
        if isinstance(k, tuple) and k[0] == l:
            return O[k]
    return None


class ScalarReadout(nn.Module):
    """Read scalar targets from MTO l=0 modes and invariant summaries.

    Output: [B, out_dim]
    """

    def __init__(
        self,
        mode_channels: int = 64,
        num_modes: int = 8,
        hidden_dim: int = 128,
        out_dim: int = 1,
    ):
        super().__init__()
        in_dim = num_modes * mode_channels  # flattened K * C
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )
        self.out_dim = out_dim

    def forward(
        self, O: dict[int | tuple, torch.Tensor],
        mode_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict scalar targets per molecule.

        Args:
            O: dict key -> [B, K, C, 2l+1] MTO modes
            mode_mask: [B, K] boolean

        Returns:
            y: [B, out_dim]
        """
        key0 = 0 if 0 in O else (0, 1)
        s = O[key0]  # [B, K, C, 1]
        B, K, C = s.shape[:3]

        if mode_mask is not None:
            s = s * mode_mask.to(s.device).unsqueeze(-1).unsqueeze(-1)

        flat = s.reshape(B, K * C)  # [B, K*C]
        return self.net(flat)  # [B, out_dim]


class VectorReadout(nn.Module):
    """Read vector targets from MTO l=1 modes (odd parity, polar vector).

    Output: [B, out_dim, 3], equivariant under rotation as R·v.
    """

    def __init__(
        self,
        mode_channels: int = 64,
        num_modes: int = 8,
        out_dim: int = 1,
    ):
        super().__init__()
        self.mode_channels = mode_channels
        self.num_modes = num_modes
        self.out_dim = out_dim

        # Per-mode channel reduction weights (invariant)
        self.mode_weights = nn.Parameter(
            torch.randn(num_modes, mode_channels) * 0.02
        )

        # Per-output-dim channel projection
        if out_dim > 1:
            self.out_proj = nn.Parameter(
                torch.randn(out_dim, mode_channels) * 0.02
            )
        else:
            self.out_proj = None

    def forward(
        self, O: dict[int | tuple, torch.Tensor],
        mode_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict vector targets per molecule.

        Args:
            O: dict key -> [B, K, C, 2l+1] MTO modes
            mode_mask: [B, K] boolean

        Returns:
            v: [B, out_dim, 3] vector prediction
        """
        o1 = _get_order(O, 1, -1)  # [B, K, C, 3]
        if o1 is None:
            raise KeyError("VectorReadout requires l=1 (odd parity) modes in O")

        B, K, C = o1.shape[:3]
        device = o1.device

        # Softmax over modes for each channel
        w = self.mode_weights.softmax(dim=0)  # [K, C]

        if mode_mask is not None:
            o1 = o1 * mode_mask.to(device).unsqueeze(-1).unsqueeze(-1)

        # Weighted sum over modes: [B, K, C, 3] → [B, C, 3]
        v = torch.einsum("kc,bkcm->bcm", w, o1)

        if self.out_proj is not None:
            # [B, C, 3] → [B, out_dim, 3]
            v = torch.einsum("oc,bcm->bom", self.out_proj, v)
        else:
            # [B, C, 3] → [B, 1, 3] via mean over channels
            v = v.mean(dim=1, keepdim=True)

        return v


class Rank2TensorReadout(nn.Module):
    """Read rank-2 symmetric tensor targets from MTO modes.

    Combines:
      - Isotropic scalar (l=0e): trace component → (iso/3) * I
      - Traceless tensor (l=2e): anisotropic component → 3×3 symmetric traceless

    Output: [B, out_dim, 3, 3]

    Uses numerically calibrated spherical-to-Cartesian conversion from e3nn's
    integral-normalized real spherical harmonics.
    """

    # Conversion matrix C[6, 5]: 6 Cartesian components (xx,yy,zz,xy,xz,yz)
    # times 5 spherical harmonics (m=-2,-1,0,+1,+2)
    # Computed by numerical quadrature: C[jk,m] = ∫ n_j n_k Y_2^m(n) dΩ
    _C_SPHERICAL_TO_CART = torch.tensor([
        # m=-2       m=-1        m=0         m=+1        m=+2
        [ 0.0,       0.0,       -0.5260,     0.0,        0.9200],  # xx
        [ 0.0,       0.0,       -0.5205,     0.0,       -0.9016],  # yy
        [ 0.0,       0.0,        1.0457,     0.0,        0.0   ],  # zz
        [ 0.9016,    0.0,        0.0,        0.0,        0.0   ],  # xy
        [ 0.0,       0.0,        0.0,        0.9107,     0.0   ],  # xz
        [ 0.0,       0.9016,     0.0,        0.0,        0.0   ],  # yz
    ])

    def __init__(
        self,
        mode_channels: int = 64,
        num_modes: int = 8,
        out_dim: int = 1,
    ):
        super().__init__()
        self.mode_channels = mode_channels
        self.num_modes = num_modes
        self.out_dim = out_dim

        # Isotropic part from l=0 modes
        iso_in_dim = num_modes * mode_channels
        self.iso_net = nn.Sequential(
            nn.Linear(iso_in_dim, mode_channels),
            nn.SiLU(),
            nn.Linear(mode_channels, out_dim),
        )

        # Anisotropic part from l=2 modes
        self.mode_weights_2 = nn.Parameter(
            torch.randn(num_modes, mode_channels) * 0.02
        )

        # Spherical-to-Cartesian conversion for l=2 traceless symmetric tensor
        # C[6, 5]: maps [m=-2,-1,0,+1,+2] → [xx, yy, zz, xy, xz, yz]
        self.register_buffer(
            "_C_6x5",
            Rank2TensorReadout._C_SPHERICAL_TO_CART.clone().float(),
        )

    def _spherical_to_cartesian(self, h2: torch.Tensor) -> torch.Tensor:
        """Convert l=2 spherical components to 3×3 Cartesian traceless symmetric.

        Args:
            h2: [B, C, 5] or [B, 5] spherical components (m=-2,-1,0,+1,+2)

        Returns:
            [B, 3, 3] or [3, 3] Cartesian tensor
        """
        is_batched = h2.dim() == 3
        if not is_batched:
            h2 = h2.unsqueeze(0)

        B = h2.shape[0]
        C = h2.shape[1] if h2.dim() == 3 else 1

        # Average over channels: [B, C, 5] → [B, 5]
        h2_mean = h2.mean(dim=1) if h2.dim() == 3 else h2

        # Apply conversion: [B, 5] @ C^T[5, 6] → [B, 6]
        cart_6 = h2_mean @ self._C_6x5.T  # [B, 6]

        # Build 3×3 tensor from 6 components
        xx, yy, zz, xy, xz, yz = cart_6.unbind(dim=-1)  # each [B]
        cart = torch.zeros(B, 3, 3, device=h2.device, dtype=h2.dtype)
        cart[:, 0, 0] = xx
        cart[:, 1, 1] = yy
        cart[:, 2, 2] = zz
        cart[:, 0, 1] = xy
        cart[:, 1, 0] = xy
        cart[:, 0, 2] = xz
        cart[:, 2, 0] = xz
        cart[:, 1, 2] = yz
        cart[:, 2, 1] = yz

        if not is_batched:
            cart = cart.squeeze(0)
        return cart

    def forward(
        self, O: dict[int | tuple, torch.Tensor],
        mode_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict rank-2 tensor targets per molecule.

        Args:
            O: dict key -> [B, K, C, 2l+1] MTO modes
            mode_mask: [B, K] boolean

        Returns:
            tensor: [B, out_dim, 3, 3] Cartesian tensors
        """
        o0 = _get_order(O, 0, 1)  # [B, K, C, 1]
        o2 = _get_order(O, 2, 1)  # [B, K, C, 5]

        if o0 is None:
            raise KeyError("Rank2TensorReadout requires l=0 modes")
        if o2 is None:
            raise KeyError("Rank2TensorReadout requires l=2e modes")

        B, K, C = o0.shape[:3]
        device = o0.device

        if mode_mask is not None:
            mask = mode_mask.to(device).unsqueeze(-1).unsqueeze(-1)
            o0 = o0 * mask
            o2 = o2 * mask

        # Isotropic part: [B, K*C] → [B, out_dim]
        iso = self.iso_net(o0.reshape(B, K * C))  # [B, out_dim]

        # Anisotropic part from l=2
        w2 = self.mode_weights_2.softmax(dim=0)  # [K, C]
        # Weighted sum over modes: [B, K, C, 5] → [B, C, 5]
        h2 = torch.einsum("kc,bkcm->bcm", w2, o2)

        # Convert each molecule's l=2 spherical to Cartesian
        # [B, C, 5] → [B, 3, 3] traceless symmetric
        cart_traceless = self._spherical_to_cartesian(h2)  # [B, 3, 3]

        # Build full tensor: isotropic + traceless
        eye = torch.eye(3, device=device)  # [3, 3]
        iso_scalar = iso / 3.0  # [B, out_dim]

        if self.out_dim == 1:
            iso_mat = iso_scalar[:, 0:1, None, None] * eye.unsqueeze(0)  # [B, 1, 3, 3]
            tensor = iso_mat + cart_traceless.unsqueeze(1)  # [B, 1, 3, 3]
        else:
            iso_mat = iso_scalar[:, :, None, None] * eye.unsqueeze(0).unsqueeze(1)  # [B, out_dim, 3, 3]
            tensor = iso_mat + cart_traceless.unsqueeze(1)  # [B, out_dim, 3, 3]

        return tensor


class SpectralReadout(nn.Module):
    """Read spectral targets from MTO mode invariant summaries.

    Uses invariant features from all modes: l=0 scalars + tensor norms from l>0.

    Output: [B, num_spectral_bins]
    """

    def __init__(
        self,
        mode_channels: int = 64,
        num_modes: int = 8,
        num_spectral_bins: int = 3501,
        maxl: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_spectral_bins = num_spectral_bins

        # Input: K*C (l=0) + K*maxl (tensor norms per mode)
        in_dim = num_modes * mode_channels + num_modes * maxl

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_spectral_bins),
        )

    def forward(
        self, O: dict[int | tuple, torch.Tensor],
        mode_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict spectrum per molecule.

        Args:
            O: dict key -> [B, K, C, 2l+1] MTO modes
            mode_mask: [B, K] boolean

        Returns:
            spectrum: [B, num_spectral_bins]
        """
        key0 = 0 if 0 in O else (0, 1)
        B, K, C = O[key0].shape[:3]
        device = O[key0].device

        # l=0 invariant features: [B, K*C]
        s = O[key0]  # [B, K, C, 1]
        if mode_mask is not None:
            s = s * mode_mask.to(device).unsqueeze(-1).unsqueeze(-1)
        s_flat = s.reshape(B, K * C)

        # Tensor norms per mode: [B, K*maxl]
        norm_parts = []
        for l in range(1, 4):
            o_l = _get_order(O, l, 1)
            if o_l is None:
                o_l = _get_order(O, l, -1)
            if o_l is not None:
                n = torch.norm(o_l, dim=-1).mean(dim=-1)  # [B, K]
                if mode_mask is not None:
                    n = n * mode_mask.to(device)
                norm_parts.append(n)
            else:
                norm_parts.append(torch.zeros(B, K, device=device))
        norms_flat = torch.cat(norm_parts, dim=1)  # [B, K*maxl]

        # Combine
        features = torch.cat([s_flat, norms_flat], dim=-1)  # [B, in_dim]
        h = self.encoder(features)
        spectrum = self.decoder(h)  # [B, num_spectral_bins]

        return spectrum
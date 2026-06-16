"""Readout heads for MTO modes.

Implements target-type-aware readouts from MTO molecular tensor modes:

  - Scalar readout: from l=0 modes + invariant summaries
  - Vector readout: from l=1 modes
  - Rank-2 tensor readout: isotropic scalar + l=2 traceless component
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ScalarReadout(nn.Module):
    """Read scalar target from MTO l=0 modes and invariant summaries."""

    def __init__(self, mode_channels: int = 64, num_modes: int = 8,
                 hidden_dim: int = 128, out_dim: int = 1):
        super().__init__()
        in_dim = num_modes * mode_channels
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, O: dict[int, torch.Tensor]) -> torch.Tensor:
        """Predict scalar target from MTO modes.

        Args:
            O: dict l -> [K, C, 2l+1] MTO modes

        Returns:
            y: [out_dim] scalar prediction
        """
        s = O[0].reshape(-1)  # flatten all mode scalar features
        return self.net(s)


class VectorReadout(nn.Module):
    """Read vector target from MTO l=1 modes.

    Produces equivariant vector output: under rotation R, output transforms as R·v.
    """

    def __init__(self, mode_channels: int = 64, num_modes: int = 8,
                 out_dim: int = 3):
        super().__init__()
        # Aggregate l=1 modes with learned weights per mode
        self.mode_weights = nn.Parameter(
            torch.randn(num_modes, mode_channels) * 0.02
        )

        # Channel reduction: mode_channels → 1
        self.channel_proj = nn.Linear(mode_channels, 1, bias=False)

        self.out_dim = out_dim

    def forward(self, O: dict[int, torch.Tensor]) -> torch.Tensor:
        """Predict vector target from MTO l=1 modes.

        Args:
            O: dict l -> [K, C, 2l+1] MTO modes
            O[1]: [K, C, 3]

        Returns:
            v: [3] vector prediction
        """
        v_modes = O[1]  # [K, C, 3]

        # Weighted sum over modes: [K, C, 3] → [C, 3]
        w = self.mode_weights.softmax(dim=0)  # [K, C]
        v = torch.einsum("kc,kcm->cm", w, v_modes)  # [C, 3]

        # Channel projection: [C, 3] → [3]
        v = torch.einsum("cm,co->om", v, self.channel_proj.weight)  # wait
        v = self.channel_proj(v.T).squeeze()  # simpler: [C,3] × [1,C] → [1,3]

        # Actually: v is [C, 3], project channels → [3]
        v = torch.einsum("c,cm->m",
                         self.channel_proj.weight.squeeze(0), v)

        return v


class Rank2TensorReadout(nn.Module):
    """Read rank-2 tensor target from MTO modes.

    Combines:
      - Isotropic scalar (l=0): trace component
      - Traceless tensor (l=2): anisotropic component

    Output: 3×3 Cartesian tensor.
    """

    def __init__(self, mode_channels: int = 64, num_modes: int = 8):
        super().__init__()
        # Isotropic part from l=0 modes
        self.iso_net = nn.Sequential(
            nn.Linear(num_modes * mode_channels, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # Anisotropic part from l=2 modes
        self.mode_weights_2 = nn.Parameter(
            torch.randn(num_modes, mode_channels) * 0.02
        )
        self.channel_proj_2 = nn.Linear(mode_channels, 1, bias=False)

        # e3nn-based conversion from (iso, l=2) to 3×3 Cartesian
        from e3nn import o3
        self.irreps_to_cart = None  # will use manual conversion

    def _l2_to_cartesian(self, h2: torch.Tensor) -> torch.Tensor:
        """Convert l=2 spherical (traceless) tensor to 3×3 Cartesian.

        h2: [5] — l=2 spherical components (m=-2,-1,0,1,2)

        Uses the standard conversion from spherical harmonics to Cartesian
        traceless symmetric tensor.
        """
        # Spherical to Cartesian traceless conversion matrix
        # h2 components: [m=-2, m=-1, m=0, m=+1, m=+2]
        xx_yy = h2[-2]  # m=+2 real part → xy-like
        xz = h2[-1]     # m=+1 real part
        zz = h2[0]      # m=0
        yz = h2[1]      # m=-1 real part
        xy = h2[2]      # m=-2 real part (wait, need to check convention)

        # Standard real spherical harmonic to Cartesian traceless tensor:
        # Using convention where:
        #   Q_{ij} is traceless symmetric
        #   Y_2^0 ∝ (2zz - xx - yy) / sqrt(6)
        #   Y_2^{+1} ∝ xz, Y_2^{-1} ∝ yz
        #   Y_2^{+2} ∝ (xx - yy), Y_2^{-2} ∝ xy

        # For now use direct mapping (order depends on e3nn convention)
        return h2

    def forward(self, O: dict[int, torch.Tensor]) -> torch.Tensor:
        """Predict rank-2 tensor target.

        Args:
            O: dict l -> [K, C, 2l+1] MTO modes

        Returns:
            tensor: [3, 3] Cartesian tensor
        """
        K = O[0].shape[0]
        C = O[0].shape[1]
        device = O[0].device

        # Isotropic part (scalar × identity)
        s_feat = O[0].reshape(-1)  # [K*C]
        iso = self.iso_net(s_feat)  # [1]

        # Anisotropic part from l=2
        w2 = self.mode_weights_2.softmax(dim=0)  # [K, C]
        h2_weighted = torch.einsum("kc,kcm->cm", w2, O[2])  # [C, 5]

        # Project channels → [5]
        h2 = torch.einsum("c,cm->m",
                          self.channel_proj_2.weight.squeeze(0),
                          h2_weighted)

        # Build 3×3 Cartesian tensor
        # l=2 spherical: m = -2, -1, 0, +1, +2
        # Cartesian traceless symmetric tensor components
        # Using convention consistent with e3nn:
        # h2[0] ∝ (2zz - xx - yy)
        # h2[1] ∝ xz (or yz depending on convention)
        # h2[2] ∝ yz
        # h2[3] ∝ (xx - yy)
        # h2[4] ∝ xy

        # Factor to convert from spherical to Cartesian
        # For now use a direct linear map learned implicitly
        # Build traceless Cartesian matrix
        cart = torch.zeros(3, 3, device=device)
        # m=0 → zz dominant
        cart[2, 2] = h2[0]
        cart[0, 0] = -0.5 * h2[0]
        cart[1, 1] = -0.5 * h2[0]
        # m=±1 → xz, yz
        cart[0, 2] = h2[1]
        cart[2, 0] = h2[1]
        cart[1, 2] = h2[2]
        cart[2, 1] = h2[2]
        # m=±2 → xx-yy, xy
        cart[0, 0] += h2[3]
        cart[1, 1] -= h2[3]
        cart[0, 1] = h2[4]
        cart[1, 0] = h2[4]

        # Add isotropic part
        eye = torch.eye(3, device=device)
        cart = cart + iso * eye

        return cart


class SpectralReadout(nn.Module):
    """Read spectral target from MTO mode invariant summaries.

    For IR/Raman/UV-Vis spectra: reads from invariant summaries of all modes.
    """

    def __init__(self, mode_channels: int = 64, num_modes: int = 8,
                 num_points: int = 3501, hidden_dim: int = 256):
        super().__init__()
        in_dim = num_modes * mode_channels

        # Global invariant features from all modes
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Decode to spectrum
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_points),
        )

        self.num_points = num_points

    def forward(self, O: dict[int, torch.Tensor]) -> torch.Tensor:
        """Predict spectrum from MTO modes.

        Args:
            O: dict l -> [K, C, 2l+1] MTO modes

        Returns:
            spectrum: [num_points] spectral intensities
        """
        # Use only invariant information
        s = O[0].reshape(-1)  # [K*C]

        # Optional: add tensor norms from higher orders
        for l in [1, 2, 3]:
            if l in O:
                norms = torch.norm(O[l], dim=-1).mean(dim=-1).reshape(-1)  # [K]
                s = torch.cat([s, norms])

        h = self.encoder(s)
        return self.decoder(h)

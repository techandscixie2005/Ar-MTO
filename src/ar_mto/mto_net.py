"""Full MTO-Net model: DetaNet backbone → tensor adapter → MTO assembly.

Pipelines:
  Full tensor MTO (default):
    DetaNet → (S, T) → adapter → h0/h1/h2/h3
    → signed routing → MTO assembly → CG coupling → gates → readouts

  Scalar-only MTO (ablation only):
    DetaNet → (S, T) → adapter → h0 only
    → signed routing → scalar MTO assembly → gates → scalar readout

Config keys (mto_config):
  num_modes: int = 8            # K, number of molecular tensor modes
  mode_channels: int = 64       # output channels per mode
  scalar_only: bool = False     # scalar-only ablation
  use_signed_routing: bool = True
  use_cg_coupling: bool = True
  use_tensor_gate: bool = True
  coupling_type: str = "minimal"  # "minimal" or "full"
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ar_mto.tensor_adapter import TensorAdapter, make_adapter
from ar_mto.signed_routing import SignedRouter
from ar_mto.mto_core import MTOModeAssembly, ScalarOnlyMTO
from ar_mto.cg_coupling import CGCouplingMinimal, CGCoupling
from ar_mto.tensor_gate import TensorGate, NoGate
from ar_mto.readouts import ScalarReadout, VectorReadout, Rank2TensorReadout


class MTOConfig:
    """Configuration for MTO-Net model."""

    def __init__(
        self,
        num_features: int = 128,
        num_modes: int = 8,
        mode_channels: int = 64,
        maxl: int = 3,
        scalar_only: bool = False,
        use_signed_routing: bool = True,
        use_cg_coupling: bool = True,
        use_tensor_gate: bool = True,
        coupling_type: str = "minimal",
        routing_hidden_dim: int = 64,
        gate_hidden_dim: int = 64,
        readout_type: str = "scalar",
    ):
        self.num_features = num_features
        self.num_modes = num_modes
        self.mode_channels = mode_channels
        self.maxl = maxl
        self.scalar_only = scalar_only
        self.use_signed_routing = use_signed_routing
        self.use_cg_coupling = use_cg_coupling
        self.use_tensor_gate = use_tensor_gate
        self.coupling_type = coupling_type
        self.routing_hidden_dim = routing_hidden_dim
        self.gate_hidden_dim = gate_hidden_dim
        self.readout_type = readout_type

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class MTONet(nn.Module):
    """Full MTO-Net model: DetaNet backbone + MTO assembly + readouts.

    Args:
        detanet_model: DetaNet backbone (from make_latent_detanet)
        config: MTO configuration
    """

    def __init__(self, detanet_model: nn.Module, config: MTOConfig):
        super().__init__()
        self.detanet = detanet_model
        self.config = config

        # Tensor adapter
        self.adapter = make_adapter(
            num_features=config.num_features,
            maxl=config.maxl,
        )

        # Routing
        if config.scalar_only:
            self.router = SignedRouter(
                num_features=config.num_features,
                num_modes=config.num_modes,
                hidden_dim=config.routing_hidden_dim,
                use_tensor_norms=False,
                maxl=0,
            )
        else:
            self.router = SignedRouter(
                num_features=config.num_features,
                num_modes=config.num_modes,
                hidden_dim=config.routing_hidden_dim,
                use_tensor_norms=True,
                maxl=config.maxl,
            ) if config.use_signed_routing else None

        # Mode assembly
        if config.scalar_only:
            self.mto = ScalarOnlyMTO(
                num_features=config.num_features,
                mode_channels=config.mode_channels,
                num_modes=config.num_modes,
            )
        else:
            self.mto = MTOModeAssembly(
                num_features=config.num_features,
                mode_channels=config.mode_channels,
                num_modes=config.num_modes,
                maxl=config.maxl,
                scalar_only=False,
            )

        # CG coupling
        if config.use_cg_coupling and not config.scalar_only:
            if config.coupling_type == "minimal":
                self.cg = CGCouplingMinimal(mode_channels=config.mode_channels)
            else:
                self.cg = CGCoupling(
                    mode_channels=config.mode_channels,
                    maxl=config.maxl,
                    coupled_channels=config.mode_channels,
                )
        else:
            self.cg = None

        # Gates
        if config.use_tensor_gate:
            self.gate = TensorGate(
                mode_channels=config.mode_channels,
                num_modes=config.num_modes,
                maxl=config.maxl if not config.scalar_only else 0,
                hidden_dim=config.gate_hidden_dim,
                use_tensor_info=not config.scalar_only,
            )
        else:
            self.gate = NoGate(
                mode_channels=config.mode_channels,
                num_modes=config.num_modes,
                maxl=config.maxl if not config.scalar_only else 0,
            )

        # Readouts
        self.scalar_readout = ScalarReadout(
            mode_channels=config.mode_channels,
            num_modes=config.num_modes,
        )
        self.vector_readout = VectorReadout(
            mode_channels=config.mode_channels,
            num_modes=config.num_modes,
        )
        self.tensor_readout = Rank2TensorReadout(
            mode_channels=config.mode_channels,
            num_modes=config.num_modes,
        )

    def forward(
        self, z: torch.Tensor, pos: torch.Tensor,
        batch: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        return_modes: bool = False,
    ) -> dict:
        """Full MTO forward pass.

        Args:
            z: atomic numbers [N]
            pos: positions [N, 3]
            batch: batch indices [N]
            edge_index: precomputed edges [2, E]
            return_modes: include intermediate MTO modes in output

        Returns:
            dict with keys:
                scalar: scalar prediction
                vector: vector prediction (if applicable)
                tensor: tensor prediction (if applicable)
                modes: dict of MTO modes (if return_modes=True)
                route_stats: routing statistics
                gate_stats: gate statistics
        """
        # DetaNet backbone → (S, T)
        if edge_index is None:
            from ar_mto.detanet_bridge import compute_radius_edges
            edge_index = compute_radius_edges(
                pos=pos, rc=self.detanet.rc, batch=batch
            )
        S, T = self.detanet(z=z, pos=pos, edge_index=edge_index, batch=batch)

        # Tensor adapter → h0/h1/h2/h3
        h = self.adapter(S, T)

        # Signed routing → coefficients
        coeffs = self.router(h) if self.router is not None else {
            l: torch.ones(self.config.num_modes, h[0].shape[0], 1,
                          device=h[0].device) / h[0].shape[0]
            for l in range(self.config.maxl + 1)
        }

        # MTO assembly → O_k^(l)
        O = self.mto(h, coeffs)

        # CG coupling (optional)
        if self.cg is not None:
            O_coupled = self.cg(O)
            # Merge: O_coupled replaces l>0, keep l=0 from original
            O_combined = {0: O[0]}
            for l in range(1, self.config.maxl + 1):
                if l in O_coupled:
                    O_combined[l] = O_coupled[l]
            O = O_combined

        # Gates
        O = self.gate(O)

        # Readouts
        result = {
            "scalar": self.scalar_readout(O),
            "vector": self.vector_readout(O),
            "tensor": self.tensor_readout(O),
        }

        if return_modes:
            result["modes"] = O

        if self.router is not None:
            result["route_stats"] = self.router.route_stats(coeffs)

        result["gate_stats"] = self.gate.gate_stats(O)

        return result


def make_mto_net(detanet_model: nn.Module | None = None, **config_kwargs) -> MTONet:
    """Create a full MTO-Net model.

    Args:
        detanet_model: DetaNet backbone (created if None)
        **config_kwargs: MTOConfig parameters

    Returns:
        MTONet model
    """
    if detanet_model is None:
        from ar_mto.detanet_bridge import make_latent_detanet
        detanet_model = make_latent_detanet(
            num_features=config_kwargs.get("num_features", 128),
            maxl=config_kwargs.get("maxl", 3),
        )
    config = MTOConfig(**config_kwargs)
    return MTONet(detanet_model, config)

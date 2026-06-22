"""Full MTO-Net model: DetaNet backbone → tensor adapter → MTO assembly.

Pipelines:
  Full tensor MTO (default):
    DetaNet → (S, T) → adapter → h dict per (l, p)
    → signed routing (batch-aware) → MTO assembly (per-molecule)
    → CG coupling (parity-correct) → gates (invariant, residual)
    → selected readouts

Architecture:
    h[(l,p)]: [N, C, 2l+1]  atom features
    O[(l,p)]: [B, Kmax, C_out, 2l+1]  MTO modes per molecule
    mode_mask: [B, Kmax]  active mode mask

Config keys (mto_config):
    num_modes: int = 8            Kmax
    mode_channels: int = 64       output channels per mode
    scalar_only: bool = False     scalar-only ablation
    use_signed_routing: bool = True
    use_cg_coupling: bool = True
    use_tensor_gate: bool = True
    k_policy: str = "fixed"       "fixed" or "valence_adaptive"
    normalization: str = "l2"     "l2" or "abs"
    active_heads: list[str]       subset of ["scalar", "vector", "rank2", "spectral"]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ar_mto.detanet_bridge import compute_radius_edges
from ar_mto.tensor_adapter import TensorAdapter, make_adapter
from ar_mto.signed_routing import SignedRouter
from ar_mto.mto_core import MTOModeAssembly, ScalarOnlyMTO, compute_valence_adaptive_k
from ar_mto.cg_coupling import CGCouplingMinimal, CGCoupling
from ar_mto.tensor_gate import TensorGate, NoGate, ScalarOnlyGate
from ar_mto.readouts import (
    ScalarReadout,
    VectorReadout,
    Rank2TensorReadout,
    SpectralReadout,
)


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
        gate_type: str = "tensor_information",
        routing_hidden_dim: int = 64,
        gate_hidden_dim: int = 64,
        k_policy: str = "fixed",
        k_max: int = 32,
        k_min: int = 1,
        k_rounding: str = "ceil",
        k_cap_policy: str = "cap_and_report",
        normalization: str = "l2",
        order_specific_signs: bool = True,
        active_heads: list[str] | None = None,
        readout_hidden_dim: int = 128,
        spectral_bins: int = 3501,
        gate_alpha: float = 0.1,
        baseline_mode: str = "none",  # "none", "direct_sum", "attn_pool"
        **kwargs,
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
        self.gate_type = gate_type
        self.routing_hidden_dim = routing_hidden_dim
        self.gate_hidden_dim = gate_hidden_dim
        self.k_policy = k_policy
        # Normalize deprecated key name
        if self.k_policy == "valence_adaptive":
            self.k_policy = "valence_half"
        self.k_max = k_max
        self.k_min = k_min
        self.k_rounding = k_rounding
        self.k_cap_policy = k_cap_policy
        self.normalization = normalization
        self.order_specific_signs = order_specific_signs
        self.active_heads = active_heads or ["scalar", "vector", "rank2", "spectral"]
        self.readout_hidden_dim = readout_hidden_dim
        self.spectral_bins = spectral_bins
        self.gate_alpha = gate_alpha
        self.baseline_mode = baseline_mode  # "none", "direct_sum", "attn_pool"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class MTONet(nn.Module):
    """Full MTO-Net model: DetaNet backbone + MTO assembly + readouts.

    Args:
        detanet_model: DetaNet backbone (from make_latent_detanet)
        config: MTO configuration

    Kmax: the padded mode bank size used by all modules.
      - k_policy="fixed": Kmax = config.num_modes
      - k_policy="valence_half": Kmax = config.k_max
    """

    def __init__(self, detanet_model: nn.Module, config: MTOConfig):
        super().__init__()
        self.detanet = detanet_model
        self.config = config

        # Resolve Kmax: for valence_half, use k_max; for fixed, use num_modes
        Kmax = config.k_max if config.k_policy == "valence_half" else config.num_modes

        # Tensor adapter
        self.adapter = make_adapter(
            num_features=config.num_features,
            maxl=config.maxl,
        )

        # Routing
        if config.use_signed_routing:
            self.router = SignedRouter(
                num_features=config.num_features,
                num_modes=Kmax,
                hidden_dim=config.routing_hidden_dim,
                use_tensor_norms=not config.scalar_only,
                maxl=0 if config.scalar_only else config.maxl,
                normalization=config.normalization,
                order_specific_signs=config.order_specific_signs,
            )
        else:
            self.router = None

        # Mode assembly
        if config.scalar_only:
            self.mto = ScalarOnlyMTO(
                num_features=config.num_features,
                mode_channels=config.mode_channels,
                num_modes=Kmax,
            )
        else:
            self.mto = MTOModeAssembly(
                num_features=config.num_features,
                mode_channels=config.mode_channels,
                num_modes=Kmax,
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
                    coupled_maxl=2,
                    preserve_uncoupled_l=True,
                )
        else:
            self.cg = None

        # Gates
        if config.use_tensor_gate:
            if config.gate_type == "tensor_information":
                self.gate = TensorGate(
                    mode_channels=config.mode_channels,
                    num_modes=Kmax,
                    maxl=config.maxl if not config.scalar_only else 0,
                    hidden_dim=config.gate_hidden_dim,
                    use_tensor_info=not config.scalar_only,
                    alpha=config.gate_alpha,
                )
            elif config.gate_type == "scalar_only":
                self.gate = ScalarOnlyGate(
                    mode_channels=config.mode_channels,
                    num_modes=Kmax,
                    maxl=config.maxl if not config.scalar_only else 0,
                    hidden_dim=config.gate_hidden_dim,
                    alpha=config.gate_alpha,
                )
            else:
                self.gate = NoGate(
                    mode_channels=config.mode_channels,
                    num_modes=Kmax,
                    maxl=config.maxl if not config.scalar_only else 0,
                )
        else:
            self.gate = NoGate(
                mode_channels=config.mode_channels,
                num_modes=Kmax,
                maxl=config.maxl if not config.scalar_only else 0,
            )

        # Readouts — built on demand by active_heads
        self._build_readouts(Kmax)

    def _build_readouts(self, Kmax: int | None = None):
        """Build readout heads for active target types."""
        if Kmax is None:
            Kmax = self.config.num_modes
        self.scalar_readout = None
        self.vector_readout = None
        self.rank2_readout = None
        self.spectral_readout = None

        heads = set(self.config.active_heads)
        for h in heads:
            if h == "scalar":
                self.scalar_readout = ScalarReadout(
                    mode_channels=self.config.mode_channels,
                    num_modes=Kmax,
                    hidden_dim=self.config.readout_hidden_dim,
                    out_dim=1,
                )
            elif h == "vector":
                self.vector_readout = VectorReadout(
                    mode_channels=self.config.mode_channels,
                    num_modes=Kmax,
                    out_dim=1,
                )
            elif h == "rank2":
                self.rank2_readout = Rank2TensorReadout(
                    mode_channels=self.config.mode_channels,
                    num_modes=Kmax,
                    out_dim=1,
                )
            elif h == "spectral":
                self.spectral_readout = SpectralReadout(
                    mode_channels=self.config.mode_channels,
                    num_modes=Kmax,
                    num_spectral_bins=self.config.spectral_bins,
                    maxl=self.config.maxl,
                    hidden_dim=256,
                )

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        return_modes: bool = False,
        mode_mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> dict:
        """Full MTO forward pass.

        Args:
            z: atomic numbers [N]
            pos: positions [N, 3]
            batch: batch indices [N], None for single molecule
            edge_index: precomputed edges [2, E]
            return_modes: include MTO modes in output
            mode_mask: [B, Kmax] precomputed mode mask (valence-adaptive).
                       If None and k_policy="valence_half", computed automatically.
            return_diagnostics: include routing/gate statistics

        Returns:
            dict with keys:
                scalar: [B, 1] scalar prediction (if scalar head active)
                vector: [B, 1, 3] vector prediction (if vector head active)
                tensor: [B, 1, 3, 3] rank-2 prediction (if rank2 head active)
                spectrum: [B, num_bins] (if spectral head active)
                modes: dict of MTO modes (if return_modes=True)
                mode_mask: [B, Kmax] active mode mask
                ks: [B] per-molecule K values (if valence_half)
                diagnostics: routing/gate stats dict (if return_diagnostics=True)
        """
        N = z.shape[0]
        device = z.device

        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)

        # ── DetaNet backbone ──
        if edge_index is None:
            edge_index = compute_radius_edges(
                pos=pos, rc=self.detanet.rc, batch=batch
            )
        S, T = self.detanet(z=z, pos=pos, edge_index=edge_index, batch=batch)

        # ── Tensor adapter → h0..h3 ──
        h = self.adapter(S, T)  # dict l -> [N, C, 2l+1]

        # ── Baseline dispatch: bypass MTO for direct pooling baselines ──
        if self.config.baseline_mode != "none":
            return self._forward_baseline(
                h, batch=batch, return_modes=return_modes,
                return_diagnostics=return_diagnostics,
            )

        # Auto-compute mode_mask from k_policy
        B = int(batch.max().item()) + 1
        Kmax = self.config.k_max if self.config.k_policy == "valence_half" else self.config.num_modes
        if mode_mask is None:
            if self.config.k_policy == "valence_half":
                mode_mask, ks = compute_valence_adaptive_k(
                    z=z, batch=batch, max_modes=self.config.k_max,
                    k_min=self.config.k_min, k_rounding=self.config.k_rounding,
                )
            else:
                mode_mask = torch.ones(B, Kmax, dtype=torch.bool, device=device)
                ks = None
        else:
            ks = None

        # ── Signed routing ──
        if self.router is not None:
            coeffs = self.router(h, batch=batch)  # dict l -> [K, N, 1]
        else:
            Kmax = self.config.k_max if self.config.k_policy == "valence_half" else self.config.num_modes
            coeffs = {
                l: torch.ones(Kmax, N, 1, device=device) / N
                for l in h.keys()
            }

        # ── MTO assembly ──
        O = self.mto.forward_with_masks(
            h, coeffs, mode_mask=mode_mask, batch=batch
        )  # dict key -> [B, K, C_out, 2l+1]

        # ── CG coupling ──
        if self.cg is not None:
            O_coupled = self.cg(O, mode_mask=mode_mask)  # dict (l,p) -> [B, K, C, 2l+1]

            # Merge: CG output replaces original for orders it covers
            # Preserve original O for orders not coupled
            O_merged = dict(O)  # copy
            for key, coupled_val in O_coupled.items():
                if key in O_merged:
                    O_merged[key] = coupled_val
                else:
                    O_merged[key] = coupled_val
            O = O_merged

        # ── Gates ──
        O = self.gate(O, mode_mask=mode_mask)

        # ── Readouts ──
        result: dict = {}

        if self.scalar_readout is not None:
            result["scalar"] = self.scalar_readout(O, mode_mask=mode_mask)

        if self.vector_readout is not None:
            result["vector"] = self.vector_readout(O, mode_mask=mode_mask)

        if self.rank2_readout is not None:
            result["tensor"] = self.rank2_readout(O, mode_mask=mode_mask)

        if self.spectral_readout is not None:
            result["spectrum"] = self.spectral_readout(O, mode_mask=mode_mask)

        if return_modes:
            result["modes"] = O

        if mode_mask is not None:
            result["mode_mask"] = mode_mask
            if ks is not None:
                result["ks"] = ks

        if return_diagnostics:
            diag: dict = {}
            if self.router is not None:
                diag["route_stats"] = self.router.route_stats(coeffs, mode_mask=mode_mask)
            diag["gate_stats"] = self.gate.gate_stats(O)
            result["diagnostics"] = diag

        return result

    def _forward_baseline(
        self,
        h: dict[int, torch.Tensor],
        batch: torch.Tensor,
        return_modes: bool = False,
        return_diagnostics: bool = False,
    ) -> dict:
        """Pooling baseline forward: DetaNet features → pool → readout.

        baseline_mode="direct_sum": Per-atom sum pooling over tensor features.
        baseline_mode="attn_pool": Learned scalar attention pooling.

        Both are symmetry-legal: sum of equivariant per-atom features
        preserves equivariance; attention weights are invariant.
        """
        B = int(batch.max().item()) + 1
        device = h[0].device
        C = self.config.mode_channels

        # ── Pool atom features to molecule level ──
        pooled: dict[int, torch.Tensor] = {}  # l -> [B, C, 2l+1]

        if self.config.baseline_mode == "direct_sum":
            # Sum pooling over same-batch atoms
            for l_key, h_l in h.items():
                # h_l: [N, C_in, 2l+1]
                C_in = h_l.shape[1]
                multiplicities = 2 * l_key + 1
                # Scatter sum by batch
                out = torch.zeros(B, C_in, multiplicities, device=device, dtype=h_l.dtype)
                out.scatter_add_(0, batch.unsqueeze(-1).unsqueeze(-1).expand_as(h_l), h_l)
                # Channel projection to mode_channels
                if C_in != C:
                    proj = nn.Linear(C_in, C, bias=False, device=device, dtype=h_l.dtype)
                    nn.init.normal_(proj.weight, std=0.02)
                    out = out.transpose(1, 2).reshape(B * multiplicities, C_in)
                    out = proj(out).reshape(B, multiplicities, C).transpose(1, 2)
                pooled[l_key] = out

        elif self.config.baseline_mode == "attn_pool":
            # Scalar attention weights from l=0 features
            h0 = h.get(0, h.get((0, 1)))  # [N, C_in, 1]
            if h0 is None:
                h0 = torch.zeros(h[list(h.keys())[0]].shape[0], C, 1,
                                 device=device, dtype=h[list(h.keys())[0]].dtype)
            h0_flat = h0.reshape(h0.shape[0], -1)  # [N, C_in]

            # Attention: per-atom score from scalar features
            attn_net = nn.Sequential(
                nn.Linear(h0_flat.shape[1], C),
                nn.SiLU(),
                nn.Linear(C, 1),
            ).to(device)
            # Use a non-parameterized soft-attention (temp=1) for now
            scores = attn_net(h0_flat).squeeze(-1)  # [N]
            # Normalize per molecule via softmax
            scores_max = torch.zeros(B, device=device)
            scores_max.scatter_reduce_(0, batch, scores, reduce="amax", include_self=False)
            scores = scores - scores_max[batch]  # stabilize
            exp_scores = torch.exp(scores)
            exp_sum = torch.zeros(B, device=device)
            exp_sum.scatter_add_(0, batch, exp_scores)
            attn = exp_scores / (exp_sum[batch] + 1e-10)  # [N]

            for l_key, h_l in h.items():
                C_in = h_l.shape[1]
                multiplicities = 2 * l_key + 1
                w = attn.view(-1, 1, 1)
                weighted = h_l * w
                out = torch.zeros(B, C_in, multiplicities, device=device, dtype=h_l.dtype)
                out.scatter_add_(0, batch.unsqueeze(-1).unsqueeze(-1).expand_as(h_l), weighted)
                pooled[l_key] = out

        # ── Build pseudo mode dict for readout compatibility ──
        # Pooled tensors may have C_in != mode_channels; project channels first.
        O: dict = {}
        K = 1
        C_out = self.config.mode_channels
        mode_mask = torch.ones(B, K, dtype=torch.bool, device=device)

        for l_key, p_l in list(pooled.items()):
            C_in = p_l.shape[1]  # [B, C_in, 2l+1]
            if C_in != C_out:
                # Project channels: [B, C_in, M] → [B, C_out, M]
                proj = nn.Linear(C_in, C_out, bias=False, device=device, dtype=p_l.dtype)
                nn.init.normal_(proj.weight, std=0.02)
                p_l = p_l.transpose(1, 2).reshape(-1, C_in)
                p_l = proj(p_l).reshape(B, -1, C_out).transpose(1, 2)

            l = l_key if isinstance(l_key, int) else l_key[0]
            O[(l, 1)] = p_l.unsqueeze(1)  # [B, 1, C_out, 2l+1]
            if l == 0:
                O[0] = O[(0, 1)]

        # ── Forward through MTO modules for minimal processing ──
        # Skip CG coupling for baselines
        # Skip gates for baselines
        # Go directly to readouts

        result: dict = {}
        if self.scalar_readout is not None:
            result["scalar"] = self.scalar_readout(O, mode_mask=mode_mask)
        if self.vector_readout is not None:
            result["vector"] = self.vector_readout(O, mode_mask=mode_mask)
        if self.rank2_readout is not None:
            result["tensor"] = self.rank2_readout(O, mode_mask=mode_mask)
        if self.spectral_readout is not None:
            result["spectrum"] = self.spectral_readout(O, mode_mask=mode_mask)

        if return_modes:
            result["modes"] = O
        if mode_mask is not None:
            result["mode_mask"] = mode_mask
        if return_diagnostics:
            result["diagnostics"] = {"route_stats": {}, "gate_stats": {}}

        return result

    def forward_with_adaptive_k(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        return_modes: bool = False,
        return_diagnostics: bool = False,
    ) -> dict:
        """Forward pass with valence-adaptive K.

        Computes mode_mask from atomic numbers and passes it through the pipeline.
        """
        if batch is None:
            batch = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)

        mode_mask, ks = compute_valence_adaptive_k(
            z=z, batch=batch, max_modes=self.config.k_max,
            k_min=self.config.k_min, k_rounding=self.config.k_rounding,
        )

        return self.forward(
            z=z, pos=pos, batch=batch, edge_index=edge_index,
            return_modes=return_modes, mode_mask=mode_mask,
            return_diagnostics=return_diagnostics,
        )


def make_mto_net(
    detanet_model: nn.Module | None = None, **config_kwargs
) -> MTONet:
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
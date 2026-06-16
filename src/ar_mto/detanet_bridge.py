"""Minimal import bridge to DetaNet located in third_party/."""

import sys
from pathlib import Path


def _locate_detanet() -> Path:
    """Locate the DetaNet source directory relative to this project root."""
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "third_party" / "DetaNet",
        Path.cwd() / "third_party" / "DetaNet",
    ]
    for p in candidates:
        if (p / "detanet_model" / "__init__.py").exists():
            return p
    raise FileNotFoundError(
        "Cannot locate DetaNet source. Expected under third_party/DetaNet/ "
        "relative to the Ar-MTO project root."
    )


_DETANET_PATH = _locate_detanet()

if str(_DETANET_PATH) not in sys.path:
    sys.path.insert(0, str(_DETANET_PATH))


def get_detanet_path() -> Path:
    """Return the resolved DetaNet source path."""
    return _DETANET_PATH


def compute_radius_edges(pos, rc, batch=None):
    """Compute edge index for all atom pairs within cutoff radius rc.

    Pure-PyTorch implementation used as a fallback when torch_geometric's
    radius_graph is unavailable (e.g. missing or incompatible pyg-lib).

    Returns:
        edge_index: LongTensor [2, num_edges]
    """
    import torch

    if batch is None:
        # Single molecule: direct pairwise distance computation
        diffs = pos.unsqueeze(0) - pos.unsqueeze(1)  # [n, n, 3]
        dists = torch.norm(diffs, dim=-1)  # [n, n]
        mask = (dists < rc) & (dists > 0.0)  # exclude self-loops
        edge_index = mask.nonzero().t().contiguous()  # [2, num_edges]
    else:
        # Batched: pad to max atoms per molecule, or compute per-molecule
        # Simple approach: compute per-molecule within batch
        edges = []
        for b in batch.unique():
            idx = (batch == b).nonzero(as_tuple=True)[0]
            sub_pos = pos[idx]
            diffs = sub_pos.unsqueeze(0) - sub_pos.unsqueeze(1)
            dists = torch.norm(diffs, dim=-1)
            mask = (dists < rc) & (dists > 0.0)
            local_edges = mask.nonzero().t().contiguous()
            if local_edges.numel() > 0:
                edges.append(idx[local_edges])
        edge_index = torch.cat(edges, dim=1) if edges else torch.zeros(
            2, 0, dtype=torch.long, device=pos.device
        )
    return edge_index


def import_detanet():
    """Import and return the DetaNet class.

    Applies compatibility shims for older e3nn versions (0.4.x) which use
    torch.load() without weights_only=False, incompatible with PyTorch >= 2.6.
    """
    import torch

    # e3nn 0.4.x loads constants.pt with torch.load() — need to allow slice
    # unpickling since PyTorch 2.6+ defaults to weights_only=True.
    torch.serialization.add_safe_globals([slice])

    from detanet_model import DetaNet  # noqa: E402

    return DetaNet


def make_latent_detanet(**kwargs) -> "DetaNet":
    """Build a DetaNet model configured for latent (S, T) feature output.

    This is the primary integration point for MTO: after the interaction
    blocks, DetaNet returns both scalar (S) and tensor irrep (T) features.

    Default kwargs mirror the published DetaNet configuration for QM9S.
    Override as needed.
    """
    DetaNet = import_detanet()

    defaults = dict(
        num_features=128,
        act="swish",
        maxl=3,
        num_block=3,
        radial_type="trainable_bessel",
        num_radial=32,
        attention_head=8,
        rc=5.0,
        dropout=0.0,
        use_cutoff=False,
        max_atomic_number=9,
        scalar_outsize=0,
        irreps_out=None,
        summation=False,
        norm=False,
        out_type="latent",
        grad_type=None,
        device="cpu",
        scale=None,
        atom_ref=None,
    )
    defaults.update(kwargs)

    if defaults["out_type"] != "latent":
        raise ValueError(
            f"Expected out_type='latent' for MTO integration, got {defaults['out_type']!r}"
        )

    return DetaNet(**defaults)


def run_latent_forward(model, z, pos, batch=None, edge_index=None):
    """Run DetaNet in latent mode, returning (S_scalar, T_tensor) features.

    If edge_index is not provided, computes it from positions and model cutoff.

    Returns:
        S: scalar features [num_atoms, num_features]
        T: tensor irrep features [num_atoms, vdim]
    """
    if edge_index is None:
        edge_index = compute_radius_edges(pos=pos, rc=model.rc, batch=batch)
    return model(z=z, pos=pos, edge_index=edge_index, batch=batch)

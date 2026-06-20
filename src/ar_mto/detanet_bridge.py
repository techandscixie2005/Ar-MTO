"""Minimal import bridge to DetaNet located in third_party/.

On systems where torch_geometric / torch_scatter C++ extensions are incompatible
with the installed PyTorch (e.g. torch 2.11.0+cu130 vs pyg-lib compiled for
torch 2.7+cu128), this module installs pure-PyTorch fallback mocks into
sys.modules before any DetaNet import occurs.
"""

import sys
import types
from pathlib import Path


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# PyG compatibility: detect broken C++ extensions; install fallbacks if needed
# ---------------------------------------------------------------------------

_PYG_FALLBACK_ACTIVE = False


def _pyg_compatible() -> bool:
    """Return True if torch_geometric C++ extensions are compatible with torch.

    Checks torch vs pytorch_scatter CUDA ABI versions via package metadata
    (pip show) rather than importing — import can segfault on mismatched libs.
    """
    try:
        import torch
        from importlib.metadata import PackageNotFoundError, version as pkg_version
    except Exception:
        return False

    torch_ver = torch.__version__

    def _cuda_major(ver_str):
        if ver_str is None:
            return None
        for tok in ver_str.replace("+", " ").split():
            if tok.lower().startswith("cu"):
                try:
                    return int(tok[2:])
                except ValueError:
                    continue
        return None

    torch_cu = _cuda_major(torch_ver)

    try:
        ts_ver = pkg_version("torch_scatter")
        ts_cu = _cuda_major(ts_ver)
    except PackageNotFoundError:
        return False
    except Exception:
        return False

    if torch_cu is None or ts_cu is None:
        return False
    if torch_cu == ts_cu:
        return True
    if torch_cu >= 130 and ts_cu <= 128:
        return False
    if abs(torch_cu - ts_cu) >= 2:
        return False
    return True


def _install_pyg_fallbacks():
    """Install pure-PyTorch torch_geometric / torch_scatter stubs in sys.modules.

    Must be called BEFORE any module that does ``import torch_geometric`` or
    ``import torch_scatter`` (e.g. DetaNet's detanet.py top-level imports).
    """
    global _PYG_FALLBACK_ACTIVE
    if "torch_geometric" in sys.modules or "torch_scatter" in sys.modules:
        return

    import torch

    # -- Data stub class (used for pickle deserialization of QM9S dataset) ---

    class _StubGraphData:
        """Stub for torch_geometric.data.Data for pickle deserialization.

        PyG Data stores tensor attributes inside a ``_store`` GlobalStorage,
        which itself stores them in ``_mapping``.  This stub makes ``mol.z``
        work regardless of whether the pickled object is a Data (has _store)
        or a GlobalStorage (has _mapping).
        """

        def __init__(self, **kwargs):
            object.__setattr__(self, "_store", None)
            for k, v in kwargs.items():
                object.__setattr__(self, k, v)

        def __setattr__(self, name, value):
            if name in ("_store", "_mapping", "_parent"):
                object.__setattr__(self, name, value)
                return
            store = self.__dict__.get("_store")
            if store is not None and hasattr(store, "_mapping"):
                setattr(store, name, value)
            elif hasattr(self, "_mapping"):
                self._mapping[name] = value
            else:
                object.__setattr__(self, name, value)

        def __getattr__(self, name):
            if name in ("_store", "_mapping", "_parent"):
                raise AttributeError(name)
            # _mapping path (GlobalStorage / BaseStorage style)
            mapping = self.__dict__.get("_mapping")
            if mapping is not None and name in mapping:
                return mapping[name]
            # _store path (Data style: proxy to inner GlobalStorage)
            store = self.__dict__.get("_store")
            if store is not None and hasattr(store, name):
                return getattr(store, name)
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )

        def __repr__(self):
            mapping = self.__dict__.get("_mapping")
            keys = list(self.__dict__.keys())
            if mapping is not None:
                keys += [k for k in mapping.keys() if not k.startswith("_")]
            return f"StubGraphData(keys={sorted(set(keys))})"

    # -- Stub module factory -------------------------------------------------
    # CPython's module_getattro (PEP 562) looks for __getattr__ in the
    # module's own __dict__, NOT on the type.  So we must set it on each
    # instance rather than defining it on a ModuleType subclass.

    def _make_stub_module(fullname):
        m = types.ModuleType(fullname)
        m.__path__ = []

        def __getattr__(name):
            # Skip private/dunder names
            if name.startswith("_"):
                raise AttributeError(name)
            stub = type(name, (_StubGraphData,), {})
            # Cache on the module so the same class is returned each time
            setattr(m, name, stub)
            return stub

        m.__getattr__ = __getattr__
        return m

    # -- radius_graph (pure PyTorch fallback) --------------------------------

    def _radius_graph(x, r, batch=None, loop=False, max_num_neighbors=32,
                      flow="source_to_target"):
        if batch is None:
            diffs = x.unsqueeze(0) - x.unsqueeze(1)
            dists = torch.norm(diffs, dim=-1)
            mask = (dists < r) if loop else ((dists < r) & (dists > 0.0))
            return mask.nonzero().t().contiguous()

        edges = []
        for b in batch.unique():
            idx = (batch == b).nonzero(as_tuple=True)[0]
            sub_pos = x[idx]
            diffs = sub_pos.unsqueeze(0) - sub_pos.unsqueeze(1)
            dists = torch.norm(diffs, dim=-1)
            mask = (dists < r) if loop else ((dists < r) & (dists > 0.0))
            local_edges = mask.nonzero().t().contiguous()
            if local_edges.numel() > 0:
                edges.append(idx[local_edges])

        if edges:
            return torch.cat(edges, dim=1)
        return torch.zeros(2, 0, dtype=torch.long, device=x.device)

    # -- torch_geometric stub tree --------------------------------------------

    tg = _make_stub_module("torch_geometric")
    tg.__version__ = "2.8.0-fallback"

    tg_nn = _make_stub_module("torch_geometric.nn")
    tg_nn.radius_graph = _radius_graph
    tg.nn = tg_nn

    sys.modules["torch_geometric"] = tg
    sys.modules["torch_geometric.nn"] = tg_nn

    # torch_geometric.data + torch_geometric.data.data
    tg_data = _make_stub_module("torch_geometric.data")
    tg_data_data = _make_stub_module("torch_geometric.data.data")
    tg_data_data.Data = _StubGraphData
    tg_data_data.Batch = _StubGraphData
    tg_data.Data = _StubGraphData
    tg_data.Batch = _StubGraphData

    sys.modules["torch_geometric.data"] = tg_data
    sys.modules["torch_geometric.data.data"] = tg_data_data

    # Other torch_geometric.data.* submodules that pickles may reference
    for _sub in ["torch_geometric.data.storage",
                 "torch_geometric.data.collate",
                 "torch_geometric.data.dataset"]:
        sys.modules[_sub] = _make_stub_module(_sub)

    # -- torch_scatter stub --------------------------------------------------

    ts = types.ModuleType("torch_scatter")
    ts.__version__ = "2.1.2-fallback"

    def _scatter(src, index, dim, dim_size=None, reduce="sum"):
        """Pure-PyTorch scatter matching the torch_scatter interface."""
        if dim_size is None:
            dim_size = int(index.max().item()) + 1

        idx = index.to(src.device)
        while idx.dim() < src.dim():
            idx = idx.unsqueeze(-1) if dim == 0 else idx.unsqueeze(0)
        view = [1] * src.dim()
        view[dim] = src.shape[dim]
        idx = idx.view(view).expand(src.shape)

        shape_out = list(src.shape)
        shape_out[dim] = dim_size
        out = torch.zeros(shape_out, dtype=src.dtype, device=src.device)

        if reduce == "sum":
            return out.scatter_reduce(dim, idx, src, reduce="sum", include_self=False)
        if reduce == "mean":
            s = out.scatter_reduce(dim, idx, src, reduce="sum", include_self=False)
            ones = torch.ones_like(src)
            c = torch.zeros(shape_out, dtype=src.dtype, device=src.device)
            c = c.scatter_reduce(dim, idx, ones, reduce="sum", include_self=False)
            return s / c.clamp(min=1)
        if reduce == "max":
            return out.scatter_reduce(dim, idx, src, reduce="amax", include_self=False)
        raise ValueError(
            f"torch_scatter fallback: reduce='{reduce}' not implemented"
        )

    ts.scatter = _scatter
    sys.modules["torch_scatter"] = ts

    # -- other torch_* stubs (may be imported transitively) ------------------
    for _pkg in ["torch_sparse", "torch_cluster", "torch_spline_conv"]:
        if _pkg not in sys.modules:
            _m = types.ModuleType(_pkg)
            _m.__version__ = "0.0.0-fallback"
            sys.modules[_pkg] = _m

    _PYG_FALLBACK_ACTIVE = True


def _ensure_pyg_available():
    """Check PyG compatibility; install fallbacks if the real lib would crash.

    Called automatically before the first DetaNet import.
    """
    if _pyg_compatible():
        return
    _install_pyg_fallbacks()


def is_pyg_fallback_active() -> bool:
    """Return True if the pure-PyTorch PyG fallback stubs are in use."""
    return _PYG_FALLBACK_ACTIVE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
        diffs = pos.unsqueeze(0) - pos.unsqueeze(1)
        dists = torch.norm(diffs, dim=-1)
        mask = (dists < rc) & (dists > 0.0)
        edge_index = mask.nonzero().t().contiguous()
    else:
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

    If torch_geometric / torch_scatter C++ extensions segfault on the current
    torch version, installs pure-PyTorch fallback stubs before import.
    """
    import torch

    _ensure_pyg_available()

    # e3nn 0.4.x loads constants.pt with torch.load() — need to allow slice
    # unpickling since PyTorch 2.6+ defaults to weights_only=True.
    torch.serialization.add_safe_globals([slice])

    from detanet_model import DetaNet  # noqa: E402

    return DetaNet


def make_latent_detanet(**kwargs):
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

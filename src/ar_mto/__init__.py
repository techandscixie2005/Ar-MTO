"""Ar-MTO: DetaNet-based Molecular Tensor Orbital project.

The central research question:
  Can local equivariant tensor fields be assembled, under symmetry constraints,
  into stable, transferable, and chemically meaningful molecule-level response modes?
"""

__version__ = "0.2.0"

from ar_mto.tensor_adapter import TensorAdapter, make_adapter
from ar_mto.signed_routing import SignedRouter
from ar_mto.mto_core import (
    MTOModeAssembly,
    ScalarOnlyMTO,
    compute_valence_adaptive_k,
)
from ar_mto.cg_coupling import CGCoupling, CGCouplingMinimal
from ar_mto.tensor_gate import TensorGate, NoGate, ScalarOnlyGate
from ar_mto.readouts import (
    ScalarReadout,
    VectorReadout,
    Rank2TensorReadout,
    SpectralReadout,
)
from ar_mto.mto_net import MTOConfig, MTONet, make_mto_net
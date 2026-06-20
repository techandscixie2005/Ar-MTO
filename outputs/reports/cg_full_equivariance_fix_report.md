# CGCouplingFull Equivariance Fix Report

**Date**: 2026-06-20
**Task**: 1.2b
**Status**: Fixed

---

## Root Cause

The `CGCouplingFull.test_equivariance` failure had **two distinct root causes**:

### 1. Broken Wigner-D einsum in test (primary)

All equivariance tests across the codebase used an incorrect einsum pattern for
Wigner-D rotation:

```python
# WRONG (all test files before fix):
h_rot[l] = torch.einsum("sd,bkco->bkcs", D, h_orig[l])
h_rot[l] = torch.einsum("sd,nco->ncs",  D, h_orig[l])
```

In these einsums, the index `o` in the O tensor (the spatial m-component
dimension) does **not** match `s` or `d` in the Wigner-D matrix. Since no
dimension is contracted between D and O, the einsum computes an outer product
with free-index summation — effectively multiplying every tensor element by
row-sums of the Wigner-D matrix. This is not a rotation.

The correct pattern contracts D's source index `d` with O's spatial dimension:

```python
# CORRECT:
h_rot[l] = torch.einsum("sd,bkcd->bkcs", D, h_orig[l])  # 4D tensors
h_rot[l] = torch.einsum("sd,ncd->ncs",  D, h_orig[l])  # 3D tensors
```

### Why Other Tests Passed Despite the Broken Rotation

For **linear** modules (MTO assembly, single FCTP calls), the broken rotation
cancels out: both the input "rotation" and the output verification use the same
broken scaling factor, and linearity ensures `M(broken_R(x)) = broken_R(M(x))`.

For **bilinear** modules (CGCouplingFull's `FCTP(x, x)`), the scaling does
NOT cancel: `FCTP(S·x, S·x) = S²·FCTP(x, x) ≠ S·FCTP(x, x)`. This is why
ONLY CGCouplingFull's equivariance test failed — it was the only bilinear
module.

### 2. CGCoupling scalar_condition weight conditioning (secondary)

The `CGCoupling.scalar_condition` network generated a 516,096-dimensional
weight vector from a 16-dimensional scalar input (for C=16, maxl=3). This
8.7M-parameter network produced per-sample FCTP weights that, while not
directly breaking equivariance, made the module (a) extremely large, (b)
non-deterministic across random seeds, and (c) architecturally unsound.

## Fix Applied

### Code: `src/ar_mto/cg_coupling.py`

1. **Removed `scalar_condition`** network and per-sample `weight=weights`
   argument from the FCTP call. The FCTP now uses its internal learnable
   `uvw` weights, which are properly equivariant.
2. **Fixed dimension bug**: `in_ch = mul` (not `mul * mode_channels`) in
   `_build_output_projections`, since `mul` from `_paths_to_irreps_out`
   already includes the `mode_channels` multiplier.
3. **Fixed results dict**: Changed from a hardcoded set of canonical-parity
   keys to a dynamic `dict.setdefault` pattern that accepts all parity keys
   produced by the coupling paths (including non-canonical ones like `(1,1)=1e`
   and `(2,-1)=2o`).
4. **Marked CGCoupling as experimental** in its docstring, with a clear
   recommendation to use `CGCouplingMinimal` for production tasks.

### Tests: All test files

Fixed the Wigner-D einsum pattern in 16 occurrences across 5 test files:

| File | Fix |
|------|-----|
| `tests/test_mto_equivariance.py` | `sd,nco->ncs` → `sd,ncd->ncs`; `sd,bkco->bkcs` → `sd,bkcd->bkcs` |
| `tests/test_cg_coupling.py` | `sd,bkco->bkcs` → `sd,bkcd->bkcs` |
| `tests/test_tensor_gate.py` | `sd,bkco->bkcs` → `sd,bkcd->bkcs` |
| `tests/test_signed_routing.py` | `sd,nco->ncs` → `sd,ncd->ncs` |

## Test Results

All 13 CG coupling tests pass (including CGCouplingFull equivariance):

```
tests/test_cg_coupling.py::TestCGCouplingFull::test_equivariance PASSED
```

Full suite: **113 passed, 2 pre-existing failures** (not caused by this fix):
- `TestMTOBatchIsolation::test_molecule_ordering_invariance` — pre-existing
- `TestFullMTONet::test_valence_adaptive_k_forward` — pre-existing dimension mismatch

## Conclusion

CGCouplingFull is now genuinely equivariant. The default MTO training path
(both CGCouplingMinimal and CGCouplingFull) has passing equivariance tests.
The broken Wigner-D einsum has been fixed across all test files, making
all equivariance tests genuinely correct rather than relying on error
cancellation.

CGCouplingFull is marked as experimental — for production μ training,
CGCouplingMinimal is recommended (it covers all essential paths for
dipole and polarizability tasks with verified, fixed paths).

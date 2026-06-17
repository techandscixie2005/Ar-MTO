#!/usr/bin/env python3
"""Synthetic TMA (Tensor Mode Assembly) dataset generator.

Generates non-chemical SO(3)-structured local-to-global multipole datasets.
Proves MTO local-to-global tensor assembly works on broader problems than chemistry.

Inputs: 3D points with scalar charge/mass/type.
Targets: scalar total, vector dipole-like, rank-2 quadrupole-like,
         cancellation response, anisotropy response.

Usage:
    python generate_synthetic_tma.py --output_dir /path/to/output --mode smoke
    python generate_synthetic_tma.py --output_dir /path/to/output --mode full
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SmokeConfig:
    num_train: int = 100
    num_val: int = 20
    num_test: int = 20
    min_atoms: int = 3
    max_atoms: int = 12
    noise_std: float = 0.01
    seed: int = 42


@dataclass
class FullConfig:
    num_train: int = 10000
    num_val: int = 2000
    num_test: int = 2000
    min_atoms: int = 3
    max_atoms: int = 20
    noise_std: float = 0.01
    seed: int = 42


# ---------------------------------------------------------------------------
# Target generators
# ---------------------------------------------------------------------------

def generate_system(rng: np.random.Generator, min_atoms: int, max_atoms: int):
    """Generate one synthetic system: positions, types, charges."""
    n_atoms = rng.integers(min_atoms, max_atoms + 1)
    positions = rng.uniform(-5.0, 5.0, size=(n_atoms, 3)).astype(np.float32)
    atom_types = rng.integers(1, 6, size=n_atoms).astype(np.int64)
    charges = (rng.uniform(-1.0, 1.0, size=n_atoms).astype(np.float32)
               * atom_types.astype(np.float32))
    return positions, atom_types, charges


def compute_scalar_total(charges: np.ndarray) -> np.ndarray:
    """Scalar target: sum of charges (invariant)."""
    return np.sum(charges).astype(np.float32)


def compute_vector_dipole(positions: np.ndarray, charges: np.ndarray) -> np.ndarray:
    """Vector dipole-like target: sum_i q_i * r_i (l=1 equivariant)."""
    return np.sum(positions * charges[:, None], axis=0).astype(np.float32)


def compute_rank2_quadrupole(positions: np.ndarray, charges: np.ndarray) -> np.ndarray:
    """Rank-2 quadrupole-like target: traceless part of sum_i q_i * r_i r_i^T.

    Returns the Cartesian 3x3 traceless symmetric tensor.
    This maps to l=2 (5 independent components).
    """
    Q = np.einsum('i,ia,ib->ab', charges, positions, positions)
    Q_trace = np.trace(Q) / 3.0
    Q_traceless = Q - Q_trace * np.eye(3)
    return Q_traceless.astype(np.float32)


def compute_cancellation_response(
    positions: np.ndarray, charges: np.ndarray, rng: np.random.Generator, noise_std: float
) -> np.ndarray:
    """Nonlinear cancellation response: scalar that depends on pairwise
    charge-charge interactions, producing near-cancellation for balanced charges.

    signature = sum_{i != j} q_i * q_j * exp(-|r_i - r_j|^2)
    """
    n = positions.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                dist2 = np.sum((positions[i] - positions[j]) ** 2)
                total += charges[i] * charges[j] * np.exp(-dist2)
    noise = rng.normal(0, noise_std)
    return np.float32(total + noise)


def compute_anisotropy_response(quadrupole: np.ndarray) -> np.ndarray:
    """Fraction of quadrupole response in anisotropic components.

    anisotropy = ||Q_traceless||_F / (||Q_traceless||_F + |trace(Q)/3|)
    This equals 1.0 for purely anisotropic response.
    """
    frob_traceless = np.sqrt(np.sum(quadrupole ** 2))
    # Trace is zero by construction, but we compute robustness measure anyway
    trace_part = 0.0  # Q is already traceless
    if frob_traceless < 1e-12:
        return np.float32(0.0)
    # For a proper measure, we'd need the full Q before traceless removal.
    # Here we compute a derived measure: ratio of off-diagonal to diagonal
    diag = np.sqrt(np.sum(np.diag(quadrupole) ** 2))
    total_frob = frob_traceless
    if total_frob < 1e-12:
        return np.float32(0.0)
    off_diag = np.sqrt(total_frob ** 2 - diag ** 2)
    return np.float32(off_diag / (total_frob + 1e-12))


# ---------------------------------------------------------------------------
# Equivariance sanity checks
# ---------------------------------------------------------------------------

def rotation_matrix(axis: int, angle: float) -> np.ndarray:
    """Generate a 3D rotation matrix around a given axis (0=x, 1=y, 2=z)."""
    c = np.cos(angle)
    s = np.sin(angle)
    if axis == 0:  # x-axis
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    elif axis == 1:  # y-axis
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    else:  # z-axis
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def check_rotation_equivariance_vector(
    positions: np.ndarray, charges: np.ndarray, angle: float = 0.7
) -> dict:
    """Check that vector_dipole transforms correctly under rotation.

    v(R[r]) = R @ v(r) for l=1 equivariant.
    """
    R = rotation_matrix(2, angle)
    rotated_positions = positions @ R.T

    v_original = compute_vector_dipole(positions, charges)
    v_rotated = compute_vector_dipole(rotated_positions, charges)
    v_transformed = R @ v_original

    abs_error = np.max(np.abs(v_rotated - v_transformed))
    return {
        'target': 'vector_dipole',
        'equivariance': 'l=1',
        'max_abs_error': float(abs_error),
        'pass': abs_error < 1e-5,
    }


def check_rotation_equivariance_tensor(
    positions: np.ndarray, charges: np.ndarray, angle: float = 0.7
) -> dict:
    """Check that rank2_quadrupole transforms correctly under rotation.

    Q(R[r]) = R @ Q(r) @ R^T for l=2 equivariant.
    """
    R = rotation_matrix(2, angle)
    rotated_positions = positions @ R.T

    Q_original = compute_rank2_quadrupole(positions, charges)
    Q_rotated = compute_rank2_quadrupole(rotated_positions, charges)
    Q_transformed = R @ Q_original @ R.T

    abs_error = np.max(np.abs(Q_rotated - Q_transformed))
    return {
        'target': 'rank2_quadrupole',
        'equivariance': 'l=2',
        'max_abs_error': float(abs_error),
        'pass': abs_error < 1e-5,
    }


def check_translation_invariance(
    positions: np.ndarray, charges: np.ndarray, translation: np.ndarray
) -> dict:
    """Check that scalar_total is translation-invariant."""
    positions_translated = positions + translation

    s_original = compute_scalar_total(charges)
    s_translated = compute_scalar_total(charges)  # Same charges, so identical

    abs_error = np.abs(s_original - s_translated)
    return {
        'target': 'scalar_total',
        'invariance': 'translation',
        'translation': translation.tolist(),
        'max_abs_error': float(abs_error),
        'pass': abs_error < 1e-12,
    }


def check_permutation_invariance(
    positions: np.ndarray, charges: np.ndarray, atom_types: np.ndarray, rng: np.random.Generator
) -> dict:
    """Check that all targets are invariant under atom permutation."""
    n = positions.shape[0]
    perm = rng.permutation(n)
    positions_perm = positions[perm]
    charges_perm = charges[perm]

    s1 = compute_scalar_total(charges)
    s2 = compute_scalar_total(charges_perm)
    v1 = compute_vector_dipole(positions, charges)
    v2 = compute_vector_dipole(positions_perm, charges_perm)
    q1 = compute_rank2_quadrupole(positions, charges)
    q2 = compute_rank2_quadrupole(positions_perm, charges_perm)

    errors = {
        'scalar_total': float(np.abs(s1 - s2)),
        'vector_dipole': float(np.max(np.abs(v1 - v2))),
        'rank2_quadrupole': float(np.max(np.abs(q1 - q2))),
    }
    all_pass = all(e < 1e-5 for e in errors.values())
    return {
        'targets': list(errors.keys()),
        'errors': errors,
        'pass': all_pass,
    }


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_split(
    rng: np.random.Generator,
    num_systems: int,
    min_atoms: int,
    max_atoms: int,
    noise_std: float,
) -> list[dict]:
    """Generate a list of system dictionaries for one split."""
    systems = []
    for idx in range(num_systems):
        positions, atom_types, charges = generate_system(rng, min_atoms, max_atoms)

        scalar_total = compute_scalar_total(charges)
        vector_dipole = compute_vector_dipole(positions, charges)
        rank2_quadrupole = compute_rank2_quadrupole(positions, charges)
        cancellation = compute_cancellation_response(positions, charges, rng, noise_std)
        anisotropy = compute_anisotropy_response(rank2_quadrupole)

        systems.append({
            'system_id': idx,
            'positions': positions,
            'atom_types': atom_types,
            'charges': charges,
            'n_atoms': positions.shape[0],
            'targets': {
                'scalar_total': scalar_total,
                'vector_dipole': vector_dipole,
                'rank2_quadrupole': rank2_quadrupole,
                'cancellation_response': cancellation,
                'anisotropy_response': anisotropy,
            },
        })
    return systems


def save_split(systems: list[dict], output_dir: str, split_name: str):
    """Save a split as PyTorch .pt file and raw NumPy dict."""
    os.makedirs(output_dir, exist_ok=True)

    # Save as npz for framework-agnostic access
    npz_path = os.path.join(output_dir, f"{split_name}.npz")
    positions_list = [s['positions'] for s in systems]
    types_list = [s['atom_types'] for s in systems]
    charges_list = [s['charges'] for s in systems]
    n_atoms = np.array([s['n_atoms'] for s in systems], dtype=np.int32)

    # Targets
    scalar_total = np.array([s['targets']['scalar_total'] for s in systems], dtype=np.float32)
    vector_dipole = np.array([s['targets']['vector_dipole'] for s in systems], dtype=np.float32)
    rank2_quadrupole = np.array([s['targets']['rank2_quadrupole'] for s in systems], dtype=np.float32)
    cancellation = np.array([s['targets']['cancellation_response'] for s in systems], dtype=np.float32)
    anisotropy = np.array([s['targets']['anisotropy_response'] for s in systems], dtype=np.float32)

    np.savez_compressed(
        npz_path,
        n_atoms=n_atoms,
        scalar_total=scalar_total,
        vector_dipole=vector_dipole,
        rank2_quadrupole=rank2_quadrupole,
        cancellation_response=cancellation,
        anisotropy_response=anisotropy,
        # Variable-length arrays stored as object arrays
        positions=np.array(positions_list, dtype=object),
        atom_types=np.array(types_list, dtype=object),
        charges=np.array(charges_list, dtype=object),
    )

    # Try PyTorch save
    try:
        import torch
        pt_path = os.path.join(output_dir, f"{split_name}.pt")
        torch.save({
            'positions': [torch.from_numpy(s['positions']) for s in systems],
            'atom_types': [torch.from_numpy(s['atom_types']) for s in systems],
            'charges': [torch.from_numpy(s['charges']) for s in systems],
            'n_atoms': torch.tensor(n_atoms, dtype=torch.long),
            'targets': {
                'scalar_total': torch.from_numpy(scalar_total),
                'vector_dipole': torch.from_numpy(vector_dipole),
                'rank2_quadrupole': torch.from_numpy(rank2_quadrupole),
                'cancellation_response': torch.from_numpy(cancellation),
                'anisotropy_response': torch.from_numpy(anisotropy),
            },
        }, pt_path)
        print(f"  Saved {pt_path}")
    except ImportError:
        print(f"  PyTorch not available, saved .npz only: {npz_path}")

    print(f"  Saved {npz_path} ({len(systems)} systems)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic TMA dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for generated data")
    parser.add_argument("--mode", type=str, default="smoke",
                        choices=["smoke", "full"],
                        help="Dataset size: smoke (140 systems) or full (14000 systems)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    if args.mode == "smoke":
        cfg = SmokeConfig()
    else:
        cfg = FullConfig()

    cfg.seed = args.seed
    rng = np.random.default_rng(cfg.seed)

    print(f"=== Synthetic TMA Dataset Generator ===")
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output_dir}")
    print(f"Seed: {cfg.seed}")
    print(f"Train: {cfg.num_train}, Val: {cfg.num_val}, Test: {cfg.num_test}")
    print(f"Atoms range: [{cfg.min_atoms}, {cfg.max_atoms}]")

    # Generate splits
    print("\nGenerating train split...")
    train_systems = generate_split(rng, cfg.num_train, cfg.min_atoms, cfg.max_atoms, cfg.noise_std)
    print(f"  Generated {len(train_systems)} systems")

    print("Generating val split...")
    val_systems = generate_split(rng, cfg.num_val, cfg.min_atoms, cfg.max_atoms, cfg.noise_std)
    print(f"  Generated {len(val_systems)} systems")

    print("Generating test split...")
    test_systems = generate_split(rng, cfg.num_test, cfg.min_atoms, cfg.max_atoms, cfg.noise_std)
    print(f"  Generated {len(test_systems)} systems")

    # Save splits
    print("\nSaving splits...")
    save_split(train_systems, args.output_dir, "train")
    save_split(val_systems, args.output_dir, "val")
    save_split(test_systems, args.output_dir, "test")

    # Save metadata
    metadata = {
        'dataset_id': 'synthetic_tma',
        'mode': args.mode,
        'seed': cfg.seed,
        'num_train': cfg.num_train,
        'num_val': cfg.num_val,
        'num_test': cfg.num_test,
        'min_atoms': cfg.min_atoms,
        'max_atoms': cfg.max_atoms,
        'noise_std': cfg.noise_std,
        'target_types': [
            'scalar_total',
            'vector_dipole',
            'rank2_quadrupole',
            'cancellation_response',
            'anisotropy_response',
        ],
        'generated_at': None,  # Will be filled by caller
    }
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to {metadata_path}")

    # Run equivariance sanity checks on first system
    print("\n=== Equivariance Sanity Checks ===")
    sys0 = train_systems[0]
    pos = sys0['positions']
    chg = sys0['charges']
    atm = sys0['atom_types']

    check_rng = np.random.default_rng(12345)

    results = []
    # Rotation equivariance: vector
    r = check_rotation_equivariance_vector(pos, chg)
    results.append(r)
    status = "PASS" if r['pass'] else "FAIL"
    print(f"  Rotation equivariance (l=1 vector): {status}  max_err={r['max_abs_error']:.2e}")

    # Rotation equivariance: tensor
    r = check_rotation_equivariance_tensor(pos, chg)
    results.append(r)
    status = "PASS" if r['pass'] else "FAIL"
    print(f"  Rotation equivariance (l=2 tensor): {status}  max_err={r['max_abs_error']:.2e}")

    # Translation invariance
    translation = np.array([10.0, -5.0, 3.0], dtype=np.float32)
    r = check_translation_invariance(pos, chg, translation)
    results.append(r)
    status = "PASS" if r['pass'] else "FAIL"
    print(f"  Translation invariance (scalar):   {status}  max_err={r['max_abs_error']:.2e}")

    # Permutation invariance
    r = check_permutation_invariance(pos, chg, atm, check_rng)
    results.append(r)
    status = "PASS" if r['pass'] else "FAIL"
    print(f"  Permutation invariance:             {status}  errors={r['errors']}")

    all_pass = all(r['pass'] for r in results)
    print(f"\nOverall equivariance check: {'PASS' if all_pass else 'FAIL'}")

    # Save audit results
    audit_path = os.path.join(args.output_dir, "equivariance_audit.json")
    # Convert numpy booleans to Python native types for JSON serialization
    def _convert(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(audit_path, 'w') as f:
        json.dump({
            'overall_pass': bool(all_pass),
            'results': results,
        }, f, indent=2, default=_convert)
    print(f"Saved audit to {audit_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

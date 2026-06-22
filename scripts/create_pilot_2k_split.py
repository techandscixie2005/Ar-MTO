#!/usr/bin/env python3
"""create_pilot_2k_split.py — Create locked 2000-molecule pilot split from QM9S.

Samples 1600 train / 200 val / 200 test from the existing QM9S full split
with stratification by dipole norm and heavy atom count.

Usage:
  python scripts/create_pilot_2k_split.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def stratified_sample(indices, features, n_target, rng):
    """Systematic stratified sampling from sorted composite key."""
    n_available = len(indices)
    if n_target >= n_available:
        return list(np.array(indices))

    dipoles = features["dipole_norm"]
    heavies = features["heavy_atom_count"]
    energies = features.get("main_energy", np.zeros_like(dipoles))

    # Composite sort key: heavy count dominant, then dipole, then energy
    sort_key = heavies.astype(float) * 1000 + dipoles * 10 + energies * 0.1
    sorted_order = np.argsort(sort_key)

    step = n_available / n_target
    sample_positions = np.arange(0, n_available, step)[:n_target].astype(int)
    sampled_rel = sorted_order[sample_positions]
    sampled = sorted([indices[int(i)] for i in sampled_rel])

    print(f"  Sampled {len(sampled)} from {n_available} "
          f"(coverage: {len(sampled) / n_available:.3f})")
    return sampled


def build_feature_arrays(data, indices):
    """Build feature arrays from molecule data, indexed by position in `indices`."""
    n = len(indices)
    dipoles = np.zeros(n)
    main_energies = np.zeros(n)
    heavy_atom_counts = np.zeros(n)
    for pos, idx in enumerate(indices):
        mol = data[idx]
        d = mol.dipole.reshape(-1)
        dipoles[pos] = float(torch.norm(d).item())
        e = getattr(mol, "tran_energy", None)
        if e is not None:
            main_energies[pos] = float(e.reshape(-1)[0].item())
        z = mol.z
        heavy_atom_counts[pos] = int((z > 1).sum().item())
    return {
        "dipole_norm": dipoles,
        "main_energy": main_energies,
        "heavy_atom_count": heavy_atom_counts,
    }


def summarize_split(name, indices, idx_to_pos, feat_arrays):
    """Build summary stats for a split using position mapping."""
    d_vals = []
    e_vals = []
    h_vals = []
    for idx in indices:
        pos = idx_to_pos.get(idx)
        if pos is None:
            continue
        d_vals.append(feat_arrays["dipole_norm"][pos])
        if feat_arrays["main_energy"][pos] > 0:
            e_vals.append(feat_arrays["main_energy"][pos])
        h_vals.append(feat_arrays["heavy_atom_count"][pos])

    d_arr = np.array(d_vals)
    e_arr = np.array(e_vals) if e_vals else np.zeros(1)
    h_arr = np.array(h_vals)

    return {
        "name": name,
        "n": len(indices),
        "dipole_mean": float(d_arr.mean()),
        "dipole_std": float(d_arr.std()),
        "dipole_q25": float(np.percentile(d_arr, 25)),
        "dipole_q50": float(np.percentile(d_arr, 50)),
        "dipole_q75": float(np.percentile(d_arr, 75)),
        "main_energy_mean": float(e_arr.mean()),
        "main_energy_std": float(e_arr.std()),
        "heavy_atom_mean": float(h_arr.mean()),
        "heavy_atom_std": float(h_arr.std()),
    }


def main():
    src_dir = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(src_dir))
    from ar_mto.detanet_bridge import _ensure_pyg_available
    _ensure_pyg_available()
    global torch
    import torch
    torch.serialization.add_safe_globals([slice])

    SEED = 42
    N_TRAIN = 1600
    N_VAL = 200
    N_TEST = 200

    project_root = Path(__file__).resolve().parent.parent

    # Load existing full split
    splits_path = project_root / "data/qm9s/splits/full/splits.json"
    with open(splits_path) as f:
        full_split = json.load(f)

    train_idx_full = full_split["train"]
    val_idx_full = full_split["val"]
    test_idx_full = full_split["test"]

    print(f"Full split: train={len(train_idx_full)}, val={len(val_idx_full)}, "
          f"test={len(test_idx_full)}")

    # Load dataset for stratification
    dataset_path = project_root / "data/qm9s/qm9s.pt"
    print(f"Loading dataset for stratification features...")
    data = torch.load(str(dataset_path), map_location="cpu", weights_only=False)

    # Build feature arrays per source split (position-indexed)
    print("Building feature arrays from full splits...")
    train_feat = build_feature_arrays(data, train_idx_full)
    val_feat = build_feature_arrays(data, val_idx_full)
    test_feat = build_feature_arrays(data, test_idx_full)

    rng = np.random.RandomState(SEED)

    # Sample
    print("\nSampling train split:")
    train_pilot = stratified_sample(train_idx_full, train_feat, N_TRAIN, rng)

    print("Sampling val split:")
    val_pilot = stratified_sample(val_idx_full, val_feat, N_VAL, rng)

    print("Sampling test split:")
    test_pilot = stratified_sample(test_idx_full, test_feat, N_TEST, rng)

    # Leakage check
    train_set = set(train_pilot)
    val_set = set(val_pilot)
    test_set = set(test_pilot)
    assert len(train_set & val_set) == 0, "OVERLAP: train-val"
    assert len(train_set & test_set) == 0, "OVERLAP: train-test"
    assert len(val_set & test_set) == 0, "OVERLAP: val-test"
    print("\nLeakage check: PASSED")

    # Subset check
    assert train_set <= set(train_idx_full), "train not subset of full train"
    assert val_set <= set(val_idx_full), "val not subset of full val"
    assert test_set <= set(test_idx_full), "test not subset of full test"
    print("Subset check: PASSED")

    # Save split files
    outdir = project_root / "outputs/splits"
    outdir.mkdir(parents=True, exist_ok=True)

    split_data = {
        "train": sorted(train_pilot),
        "val": sorted(val_pilot),
        "test": sorted(test_pilot),
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "n_total": N_TRAIN + N_VAL + N_TEST,
        "seed": SEED,
        "source_split": "full",
        "description": "2k pilot split for mu+UV closed-loop experiment",
    }

    for name, indices in [("train", train_pilot), ("val", val_pilot),
                           ("test", test_pilot)]:
        path = outdir / f"qm9s_2k_pilot_{name}.json"
        with open(path, "w") as f:
            json.dump(sorted(indices), f)
        print(f"Saved: {path}")

    # Save .pt index files
    import torch as _torch
    for name, indices in [("train", train_pilot), ("val", val_pilot),
                           ("test", test_pilot)]:
        _torch.save(sorted(indices),
                    str(outdir / f"qm9s_2k_pilot_{name}_indices.pt"))

    # Compute and save hash
    split_json = json.dumps(split_data, sort_keys=True, indent=2)
    split_hash = hashlib.sha256(split_json.encode()).hexdigest()
    hash_path = outdir / "qm9s_2k_pilot_hash.txt"
    with open(hash_path, "w") as f:
        f.write(f"sha256: {split_hash}\nsha256_short: {split_hash[:16]}\n")
    print(f"\nSplit hash: {split_hash[:16]}")
    print(f"Saved: {hash_path}")

    # Build position maps for summary generation
    train_pos = {v: i for i, v in enumerate(train_idx_full)}
    val_pos = {v: i for i, v in enumerate(val_idx_full)}
    test_pos = {v: i for i, v in enumerate(test_idx_full)}

    # Generate summary
    summary_rows = [
        summarize_split("train", train_pilot, train_pos, train_feat),
        summarize_split("val", val_pilot, val_pos, val_feat),
        summarize_split("test", test_pilot, test_pos, test_feat),
    ]

    summary_path = outdir / "qm9s_2k_pilot_summary.csv"
    keys = ["name", "n", "dipole_mean", "dipole_std",
            "dipole_q25", "dipole_q50", "dipole_q75",
            "main_energy_mean", "main_energy_std",
            "heavy_atom_mean", "heavy_atom_std"]
    with open(summary_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in summary_rows:
            f.write(",".join(str(row[k]) for k in keys) + "\n")
    print(f"Saved: {summary_path}")

    print("\n=== Split Summary ===")
    for row in summary_rows:
        print(f"  {row['name']:6s}: n={row['n']:4d}  "
              f"mu={row['dipole_mean']:.3f}({row['dipole_std']:.3f})  "
              f"E1={row['main_energy_mean']:.2f}({row['main_energy_std']:.2f}) eV  "
              f"heavy={row['heavy_atom_mean']:.1f}({row['heavy_atom_std']:.1f})")

    print("\nDone. All split files saved to outputs/splits/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

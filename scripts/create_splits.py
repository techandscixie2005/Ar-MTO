#!/usr/bin/env python3
"""create_splits.py — Generate train/val/test split indices for QM9S datasets.

Uses zipfile to count molecules (no PyTorch import needed), generates
random split indices, and saves them as PyTorch .pt files.

Usage:
  python scripts/create_splits.py --dataset data/qm9s/subset_smoke/qm9s.pt --outdir data/qm9s/splits --train 0.8 --val 0.1 --seed 42
  python scripts/create_splits.py --dataset data/qm9s/qm9s.pt --outdir data/qm9s/splits --train 0.8 --val 0.1 --seed 42
"""

import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path


def count_molecules(dataset_path: str) -> int:
    """Count molecules in a QM9S .pt file using zipfile metadata.

    Each torch_geometric Data object stores each tensor attribute as a
    separate zip entry under 'data/<index>'. QM9S Data objects have
    approximately 16 tensor fields each.
    """
    with zipfile.ZipFile(dataset_path) as zf:
        data_indices = set()
        for info in zf.infolist():
            m = re.search(r'/data/(\d+)$', info.filename)
            if m:
                data_indices.add(int(m.group(1)))
    # Count unique tensor entries. 16 tensors per molecule.
    TENSORS_PER_MOL = 16
    n_entries = len(data_indices)
    n_molecules = n_entries // TENSORS_PER_MOL
    return n_molecules


def generate_splits(
    n_total: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict:
    """Generate train/val/test split indices."""
    import random
    rng = random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)

    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)

    return {
        "train": sorted(indices[:n_train]),
        "val": sorted(indices[n_train:n_train + n_val]),
        "test": sorted(indices[n_train + n_val:]),
        "n_total": n_total,
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
    }


def main():
    parser = argparse.ArgumentParser(description="Create QM9S split files")
    parser.add_argument("--dataset", required=True, help="Path to QM9S .pt file")
    parser.add_argument("--outdir", required=True, help="Output directory for split files")
    parser.add_argument("--train", type=float, default=0.8, help="Train fraction")
    parser.add_argument("--val", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-molecules", type=int, default=None,
                        help="Override molecule count (skip zipfile counting)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Count or use override
    if args.n_molecules is not None:
        n_total = args.n_molecules
        print(f"Using override: {n_total} molecules")
    else:
        print(f"Counting molecules in {dataset_path}...")
        n_total = count_molecules(str(dataset_path))
        print(f"Found {n_total} molecules")

    # Generate splits
    splits = generate_splits(
        n_total=n_total,
        train_frac=args.train,
        val_frac=args.val,
        seed=args.seed,
    )

    # Save as JSON (human-readable)
    json_path = outdir / "splits.json"
    with open(json_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Saved: {json_path}")

    # Save as PyTorch .pt for easy loading
    import torch
    pt_path = outdir / "splits.pt"
    torch.save(splits, str(pt_path))
    print(f"Saved: {pt_path}")

    # Print summary
    print(f"\nSplit summary:")
    print(f"  Total:  {n_total}")
    print(f"  Train:  {len(splits['train'])} ({len(splits['train'])/n_total:.1%})")
    print(f"  Val:    {len(splits['val'])} ({len(splits['val'])/n_total:.1%})")
    print(f"  Test:   {len(splits['test'])} ({len(splits['test'])/n_total:.1%})")
    print(f"  Seed:   {args.seed}")


if __name__ == "__main__":
    main()

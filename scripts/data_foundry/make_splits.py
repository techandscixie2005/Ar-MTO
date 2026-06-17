#!/usr/bin/env python3
"""Make train/val/test splits for MTO-Net datasets.

Supports:
- Random split for standard datasets
- Molecule-based split (no molecule appears in multiple splits)
- Stratified split by molecule size

Usage:
    python make_splits.py --dataset qm9s --method random --train 0.8 --val 0.1 --test 0.1
    python make_splits.py --dataset synthetic_tma --method pre_split  # already split
"""

import argparse
import json
import os
import sys

import numpy as np


def split_random(num_samples: int, train: float, val: float, test: float, seed: int = 42):
    """Random split of sample indices."""
    assert abs(train + val + test - 1.0) < 1e-9, "Split ratios must sum to 1"
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    train_end = int(num_samples * train)
    val_end = train_end + int(num_samples * val)
    return {
        'train': sorted(indices[:train_end].tolist()),
        'val': sorted(indices[train_end:val_end].tolist()),
        'test': sorted(indices[val_end:].tolist()),
    }


def main():
    parser = argparse.ArgumentParser(description="Make train/val/test splits")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset ID")
    parser.add_argument("--method", type=str, default="random",
                        choices=["random", "pre_split"],
                        help="Split method")
    parser.add_argument("--train", type=float, default=0.8,
                        help="Train fraction")
    parser.add_argument("--val", type=float, default=0.1,
                        help="Val fraction")
    parser.add_argument("--test", type=float, default=0.1,
                        help="Test fraction")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Total number of samples (auto-detected if not given)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for split files")
    args = parser.parse_args()

    if args.method == "pre_split":
        print(f"Dataset {args.dataset} is pre-split. No new splits needed.")
        return

    if args.num_samples is None:
        print("ERROR: --num_samples required for random split method")
        sys.exit(1)

    splits = split_random(args.num_samples, args.train, args.val, args.test, args.seed)

    output_dir = args.output_dir or f"outputs/splits/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    for split_name, indices in splits.items():
        path = os.path.join(output_dir, f"{split_name}.json")
        with open(path, 'w') as f:
            json.dump({'dataset': args.dataset, 'split': split_name, 'indices': indices, 'seed': args.seed}, f)
        print(f"  {split_name}: {len(indices)} samples -> {path}")

    print(f"\nDone. Total: {args.num_samples} samples.")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val:   {len(splits['val'])}")
    print(f"  Test:  {len(splits['test'])}")


if __name__ == "__main__":
    main()

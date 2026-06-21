#!/usr/bin/env python3
"""Audit QM9S valence electron counts and recommend K_max.

Computes:
  N_val = sum_i valence[Z_i]
  K_half = ceil(N_val / 2)

Outputs: csv table, audit report
"""
import json
import os
import sys
import torch

# Valence counts (same as in ar_mto.mto_core.compute_valence_adaptive_k)
VALENCE = {
    1: 1, 2: 2,
    3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
    11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8,
    19: 1, 20: 2,
    21: 3, 22: 4, 23: 5, 24: 6, 25: 7, 26: 8, 27: 9, 28: 10,
    29: 11, 30: 12,
    31: 3, 32: 4, 33: 5, 34: 6, 35: 7, 36: 8,
    37: 1, 38: 2,
    39: 3, 40: 4, 41: 5, 42: 6, 43: 7, 44: 8, 45: 9, 46: 10,
    47: 11, 48: 12,
    49: 3, 50: 4, 51: 5, 52: 6, 53: 7, 54: 8,
    55: 1, 56: 2,
    57: 3, 58: 4, 59: 5, 60: 6, 61: 7, 62: 8, 63: 9, 64: 10,
    65: 11, 66: 12, 67: 13, 68: 14, 69: 15, 70: 16,
    71: 3, 72: 4, 73: 5, 74: 6, 75: 7, 76: 8, 77: 9, 78: 10,
    79: 11, 80: 12,
    81: 3, 82: 4, 83: 5, 84: 6, 85: 7, 86: 8,
}

torch.serialization.add_safe_globals([slice])


def compute_n_val(z_list):
    """Compute N_val per molecule."""
    n_vals = []
    for z in z_list:
        total = sum(VALENCE[int(zi)] for zi in z)
        n_vals.append(total)
    return torch.tensor(n_vals, dtype=torch.long)


def main():
    datasets = {}

    # Load subset_medium
    medium_path = "data/qm9s/subset_medium/qm9s.pt"
    if os.path.exists(medium_path):
        print("Loading subset_medium...")
        data = torch.load(medium_path, map_location="cpu", weights_only=False)
        z_list = [mol.z for mol in data]
        datasets["subset_medium"] = z_list
        print(f"  {len(z_list)} molecules")
    else:
        print("subset_medium not found locally, will try server")

    # Check for full QM9S
    full_path = "data/qm9s/qm9s.pt"
    if os.path.exists(full_path):
        print("Loading full QM9S...")
        data = torch.load(full_path, map_location="cpu", weights_only=False)
        z_list = [mol.z for mol in data]
        datasets["full"] = z_list
        print(f"  {len(z_list)} molecules")

    if not datasets:
        print("No datasets found locally.")
        sys.exit(1)

    for name, z_list in datasets.items():
        n_val = compute_n_val(z_list)
        k_half = (n_val + 1) // 2  # ceil division

        # Statistics
        stats = {
            "count": len(n_val),
            "N_val_min": int(n_val.min()),
            "N_val_max": int(n_val.max()),
            "N_val_mean": float(n_val.float().mean()),
            "N_val_std": float(n_val.float().std()),
            "K_half_min": int(k_half.min()),
            "K_half_max": int(k_half.max()),
            "K_half_mean": float(k_half.float().mean()),
            "K_half_std": float(k_half.float().std()),
        }

        # Percentiles
        for p in [50, 90, 95, 99]:
            stats[f"N_val_p{p}"] = int(torch.quantile(n_val.float(), p / 100))
            stats[f"K_half_p{p}"] = int(torch.quantile(k_half.float(), p / 100))

        print(f"\n{'='*60}")
        print(f"Dataset: {name} ({stats['count']} molecules)")
        print(f"{'='*60}")
        print(f"N_val: min={stats['N_val_min']}, max={stats['N_val_max']}, "
              f"mean={stats['N_val_mean']:.1f}, std={stats['N_val_std']:.1f}")
        print(f"K_half: min={stats['K_half_min']}, max={stats['K_half_max']}, "
              f"mean={stats['K_half_mean']:.1f}, std={stats['K_half_std']:.1f}")
        print(f"Percentiles (N_val): p50={stats['N_val_p50']}, p90={stats['N_val_p90']}, "
              f"p95={stats['N_val_p95']}, p99={stats['N_val_p99']}")
        print(f"Percentiles (K_half): p50={stats['K_half_p50']}, p90={stats['K_half_p90']}, "
              f"p95={stats['K_half_p95']}, p99={stats['K_half_p99']}")

        # K_half distribution
        k_counts = torch.bincount(k_half, minlength=stats['K_half_max'] + 1)
        print(f"\nK_half distribution:")
        for k in range(stats['K_half_min'], min(stats['K_half_max'] + 1, 50)):
            if k_counts[k] > 0:
                print(f"  K={k}: {k_counts[k]} molecules ({100*k_counts[k]/stats['count']:.1f}%)")

        # Cap analysis
        for cap in [16, 20, 24, 28, 32, 36, 40, 48, 64]:
            capped = (k_half > cap).sum().item()
            print(f"  K_max={cap}: {capped} molecules capped ({100*capped/stats['count']:.2f}%)")

        # Element analysis
        all_z = torch.cat([z.long() for z in z_list])
        unique_z = all_z.unique().sort()[0]
        z_counts = [(int(z), (all_z == z).sum().item()) for z in unique_z]
        print(f"\nElement occurrences ({len(unique_z)} elements):")
        for z, count in sorted(z_counts, key=lambda x: -x[1])[:15]:
            print(f"  Z={z:3d}: {count:8d} atoms, {count/stats['count']:.1f} per mol avg")

        # Save CSV
        os.makedirs("outputs/tables", exist_ok=True)
        csv_path = f"outputs/tables/qm9s_valence_k_distribution_{name}.csv"
        with open(csv_path, "w") as f:
            f.write("n_val,k_half\n")
            for nv, kh in zip(n_val.tolist(), k_half.tolist()):
                f.write(f"{nv},{kh}\n")
        print(f"\nWrote {csv_path}")

        # Save stats JSON
        stats_path = f"outputs/tables/qm9s_valence_k_stats_{name}.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Wrote {stats_path}")

    # Recommendation
    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")
    for name in datasets:
        if name == "subset_medium":
            # Use medium stats for K_max recommendation
            k_half_med = compute_n_val(datasets[name])
            k_half_med = (k_half_med + 1) // 2
            k_max_rec = int(k_half_med.max())
            capped_frac = 0.0
        else:
            k_half_full = compute_n_val(datasets[name])
            k_half_full = (k_half_full + 1) // 2
            k_max_rec = int(k_half_full.max())
            capped_frac = 0.0

    print(f"K_max covering all molecules: {k_max_rec}")
    print(f"Capped fraction at K_max={k_max_rec}: {capped_frac*100:.2f}%")
    print(f"\nFor smoke training on subset_medium (5000 molecules), recommend K_max=32.")
    print(f"This covers most organic molecules: p99 K_half = {stats.get('K_half_p99', 'N/A')}")
    print(f"For full QM9S, use K_max=32-48 depending on memory budget.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compute SHA-256 checksums for all files in external dataset directories.

Generates manifests in CSV and JSON formats.
Used to verify data integrity after download and upload.

Usage:
    python compute_checksums.py --data_root /data/home/scwc008/run/xxy/MTO/data/external
    python compute_checksums.py --data_root /path/to/data --dataset qm7x
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone


def sha256_file(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def process_directory(root_dir: str, checksum_dir: str, manifest_dir: str) -> dict:
    """Compute checksums for all files under root_dir. Skips existing checksum/manifest dirs."""
    if not os.path.exists(root_dir):
        return {'error': f'Directory not found: {root_dir}', 'files': []}

    os.makedirs(checksum_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)

    files = []
    total_bytes = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip checksum and manifest directories
        dirnames[:] = [d for d in dirnames if d not in ('checksums', 'manifests')]
        for fn in filenames:
            filepath = os.path.join(dirpath, fn)
            try:
                size = os.path.getsize(filepath)
                sha = sha256_file(filepath)
                rel_path = os.path.relpath(filepath, root_dir)
                files.append({
                    'path': rel_path,
                    'size_bytes': size,
                    'sha256': sha,
                })
                total_bytes += size
                print(f"  {sha[:16]}  {rel_path}  ({size/1024/1024:.1f} MiB)")
            except Exception as e:
                files.append({
                    'path': os.path.relpath(filepath, root_dir),
                    'size_bytes': -1,
                    'sha256': None,
                    'error': str(e),
                })
                print(f"  ERROR: {os.path.relpath(filepath, root_dir)} — {e}")

    return {
        'root': root_dir,
        'file_count': len(files),
        'total_size_bytes': total_bytes,
        'total_size_human': f"{total_bytes/1024/1024/1024:.3f} GiB",
        'files': files,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute checksums for external datasets")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root external data directory")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Specific dataset to checksum (default: all)")
    args = parser.parse_args()

    data_root = args.data_root
    if not os.path.exists(data_root):
        print(f"ERROR: Data root not found: {data_root}")
        sys.exit(1)

    datasets_to_process = []
    if args.dataset:
        datasets_to_process.append(args.dataset)
    else:
        # Discover all subdirectories
        for name in sorted(os.listdir(data_root)):
            path = os.path.join(data_root, name)
            if os.path.isdir(path) and name not in ('.', '..'):
                datasets_to_process.append(name)

    print(f"Computing checksums for datasets: {datasets_to_process}")
    print(f"Data root: {data_root}")
    print()

    all_results = {}
    timestamp = datetime.now(timezone.utc).isoformat()

    for ds_id in datasets_to_process:
        ds_path = os.path.join(data_root, ds_id)
        raw_path = os.path.join(ds_path, 'raw')
        checksum_dir = os.path.join(ds_path, 'checksums')
        manifest_dir = os.path.join(ds_path, 'manifests')

        print(f"[{ds_id}]")

        # If there's a raw/ subdirectory, checksum that; otherwise checksum the dataset dir
        if os.path.isdir(raw_path):
            target = raw_path
        else:
            target = ds_path

        result = process_directory(target, checksum_dir, manifest_dir)
        all_results[ds_id] = {
            'dataset_id': ds_id,
            'target_directory': target,
            'timestamp': timestamp,
            **result,
        }

        if result.get('error'):
            print(f"  {result['error']}")
            continue

        # Write CSV manifest
        csv_path = os.path.join(checksum_dir, 'checksums.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['path', 'size_bytes', 'sha256'])
            writer.writeheader()
            for entry in result['files']:
                writer.writerow({
                    'path': entry['path'],
                    'size_bytes': entry['size_bytes'],
                    'sha256': entry.get('sha256', 'ERROR'),
                })
        print(f"  CSV manifest: {csv_path}")

        # Write JSON manifest
        json_path = os.path.join(checksum_dir, 'checksums.json')
        with open(json_path, 'w') as f:
            json.dump({
                'dataset_id': ds_id,
                'timestamp': timestamp,
                'file_count': result['file_count'],
                'total_size_bytes': result['total_size_bytes'],
                'files': result['files'],
            }, f, indent=2)
        print(f"  JSON manifest: {json_path}")

        # Write file manifest (without checksums, for quick listing)
        file_manifest_path = os.path.join(manifest_dir, 'file_manifest.csv')
        with open(file_manifest_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['path', 'size_bytes'])
            writer.writeheader()
            for entry in result['files']:
                writer.writerow({'path': entry['path'], 'size_bytes': entry['size_bytes']})
        print(f"  File manifest: {file_manifest_path}")

    # Combined manifest
    combined_path = os.path.join(data_root, '..', '..', 'outputs', 'audit', 'data_foundry')
    # Actually write to a known location
    if os.path.exists('/home/xiangyu_xie/Ar-MTO-worktrees/wt-06-data-foundry'):
        combined_dir = '/home/xiangyu_xie/Ar-MTO-worktrees/wt-06-data-foundry/outputs/audit/data_foundry'
    else:
        combined_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'audit', 'data_foundry')

    os.makedirs(combined_dir, exist_ok=True)

    all_checksums_path = os.path.join(combined_dir, 'external_checksum_manifest.json')
    with open(all_checksums_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'data_root': data_root,
            'datasets': all_results,
        }, f, indent=2, default=str)
    print(f"\nCombined checksum manifest: {all_checksums_path}")


if __name__ == "__main__":
    main()

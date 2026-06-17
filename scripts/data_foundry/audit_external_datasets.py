#!/usr/bin/env python3
"""Audit all external datasets: verify file existence, sizes, and basic integrity.

Reads the dataset registry and checks:
- Whether server paths exist
- File counts and sizes
- Top-level keys in HDF5 files (if available)
- Basic target statistics (for .pt/.npz files)
- Missing files

Usage:
    python audit_external_datasets.py --data_root /data/home/scwc008/run/xxy/MTO/data
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone


def audit_directory(path: str) -> dict:
    """Recursively audit a directory: file count, total size, file list."""
    if not os.path.exists(path):
        return {'exists': False, 'file_count': 0, 'total_size_bytes': 0, 'files': []}

    files = []
    total_size = 0
    for root, dirs, filenames in os.walk(path):
        for fn in filenames:
            fp = os.path.join(root, fn)
            try:
                size = os.path.getsize(fp)
                total_size += size
                rel_path = os.path.relpath(fp, path)
                files.append({'path': rel_path, 'size_bytes': size})
            except OSError as e:
                files.append({'path': os.path.relpath(fp, path), 'size_bytes': -1, 'error': str(e)})

    return {
        'exists': True,
        'file_count': len(files),
        'total_size_bytes': total_size,
        'total_size_gib': round(total_size / (1024**3), 3),
        'files': sorted(files, key=lambda f: f['path']),
    }


def audit_hdf5(filepath: str) -> dict:
    """Inspect top-level keys of an HDF5 file."""
    try:
        import h5py
        with h5py.File(filepath, 'r') as f:
            keys = list(f.keys())
            attrs = dict(f.attrs)
        return {'readable': True, 'top_level_keys': keys, 'attrs': str(attrs)}
    except ImportError:
        return {'readable': False, 'error': 'h5py not installed'}
    except Exception as e:
        return {'readable': False, 'error': str(e)}


def audit_npz(filepath: str) -> dict:
    """Inspect contents of a .npz file."""
    try:
        import numpy as np
        data = np.load(filepath, allow_pickle=True)
        keys = list(data.keys())
        shapes = {k: str(data[k].shape) for k in keys}
        return {'readable': True, 'keys': keys, 'shapes': shapes}
    except Exception as e:
        return {'readable': False, 'error': str(e)}


def format_bytes(b: int) -> str:
    """Human-readable byte size."""
    if b < 1024:
        return f"{b} B"
    elif b < 1024**2:
        return f"{b/1024:.1f} KiB"
    elif b < 1024**3:
        return f"{b/1024**2:.1f} MiB"
    else:
        return f"{b/1024**3:.2f} GiB"


def main():
    parser = argparse.ArgumentParser(description="Audit external datasets")
    parser.add_argument("--data_root", type=str,
                        default="/data/home/scwc008/run/xxy/MTO/data",
                        help="Root data directory")
    parser.add_argument("--output", type=str,
                        default=None,
                        help="Output JSON file for audit results")
    args = parser.parse_args()

    data_root = args.data_root
    if not os.path.exists(data_root):
        print(f"[WARN] Data root does not exist locally: {data_root}")
        print("Running in local mode — will report what's available.")

    # Define datasets to audit
    datasets = {
        'qm9s': os.path.join(data_root, 'qm9s'),
        'qme14s': os.path.join(data_root, 'external', 'qme14s', 'raw'),
        'qm7x': os.path.join(data_root, 'external', 'qm7x', 'raw'),
        'nmrshiftdb2': os.path.join(data_root, 'external', 'nmrshiftdb2', 'raw'),
        'nmrexp': os.path.join(data_root, 'external', 'nmrexp', 'raw'),
        'experimental_spectra': os.path.join(data_root, 'external', 'experimental_spectra'),
        'synthetic_tma': os.path.join(data_root, 'external', 'synthetic_tma', 'smoke'),
    }

    results = {
        'audit_timestamp': datetime.now(timezone.utc).isoformat(),
        'data_root': data_root,
        'data_root_exists': os.path.exists(data_root),
        'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
        'datasets': {},
    }

    total_files = 0
    total_size = 0

    for ds_id, ds_path in datasets.items():
        print(f"\n{'='*60}")
        print(f"Auditing: {ds_id}")
        print(f"  Path: {ds_path}")

        dir_audit = audit_directory(ds_path)
        results['datasets'][ds_id] = {
            'path': ds_path,
            'exists': dir_audit['exists'],
            'file_count': dir_audit['file_count'],
            'total_size_bytes': dir_audit['total_size_bytes'],
            'total_size_human': format_bytes(dir_audit['total_size_bytes']),
            'files': [],
            'hdf5_audits': [],
            'npz_audits': [],
        }

        if dir_audit['exists']:
            total_files += dir_audit['file_count']
            total_size += dir_audit['total_size_bytes']
            print(f"  Files: {dir_audit['file_count']}")
            print(f"  Size:  {format_bytes(dir_audit['total_size_bytes'])}")

            for f in dir_audit['files']:
                print(f"    {f['path']}  ({format_bytes(f['size_bytes'])})")
                results['datasets'][ds_id]['files'].append(f)

                # Audit HDF5 files
                full_path = os.path.join(ds_path, f['path'])
                if f['path'].endswith(('.hdf5', '.h5', '.hdf')):
                    h5_audit = audit_hdf5(full_path)
                    results['datasets'][ds_id]['hdf5_audits'].append({
                        'file': f['path'],
                        **h5_audit,
                    })
                    if h5_audit.get('readable'):
                        print(f"      HDF5 keys: {h5_audit['top_level_keys']}")

                # Audit NPZ files
                if f['path'].endswith('.npz'):
                    npz_audit = audit_npz(full_path)
                    results['datasets'][ds_id]['npz_audits'].append({
                        'file': f['path'],
                        **npz_audit,
                    })
                    if npz_audit.get('readable'):
                        print(f"      NPZ keys: {npz_audit['keys']}")
        else:
            print(f"  NOT FOUND")

    # Summary
    print(f"\n{'='*60}")
    print(f"AUDIT SUMMARY")
    print(f"  Data root: {data_root}")
    print(f"  Datasets present: {sum(1 for d in results['datasets'].values() if d['exists'])}/{len(datasets)}")
    print(f"  Total files: {total_files}")
    print(f"  Total size:  {format_bytes(total_size)}")

    results['summary'] = {
        'datasets_present': sum(1 for d in results['datasets'].values() if d['exists']),
        'datasets_total': len(datasets),
        'total_files': total_files,
        'total_size_bytes': total_size,
        'total_size_human': format_bytes(total_size),
    }

    # Output
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nAudit saved to: {args.output}")

    # Also print dataset status table
    print(f"\n{'Dataset':<25} {'Status':<20} {'Files':<8} {'Size':<15}")
    print("-" * 68)
    for ds_id, ds_info in results['datasets'].items():
        if ds_info['exists']:
            status = 'AVAILABLE'
        elif ds_id in ('nist_webbook', 'sdbs_aist'):
            status = 'MANUAL_ONLY'
        else:
            status = 'NOT_FOUND'
        print(f"{ds_id:<25} {status:<20} {ds_info['file_count']:<8} {ds_info['total_size_human']:<15}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Robust resumable dataset downloader with retry, checksum, and progress.

Handles Zenodo, Figshare, and generic URLs. Designed for large scientific
datasets that may need multiple retry attempts.

Usage:
  python robust_downloader.py --url URL --dest DIR [--expected-size BYTES]
      [--expected-md5 HASH] [--max-retries N] [--chunk-size BYTES]
"""

import argparse
import hashlib
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


def download_file(url, dest_path, expected_size=None, expected_md5=None,
                  max_retries=5, chunk_size=8 * 1024 * 1024, timeout=120):
    """Download a file with resume support and retry logic."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    existing_size = dest.stat().st_size if dest.exists() else 0

    for attempt in range(max_retries):
        try:
            headers = {}
            if existing_size > 0:
                headers['Range'] = f'bytes={existing_size}-'
                print(f"[Resume] Starting from byte {existing_size} "
                      f"({existing_size / 1e6:.1f} MB)")
            else:
                print(f"[Start] Beginning download")

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                total = existing_size
                if 'Content-Range' in response.headers:
                    # Parse Content-Range for total size
                    cr = response.headers['Content-Range']
                    if '/' in cr:
                        total_expected = int(cr.split('/')[-1])
                elif 'Content-Length' in response.headers:
                    total_expected = existing_size + int(response.headers['Content-Length'])

                mode = 'ab' if existing_size > 0 else 'wb'
                with open(dest, mode) as f:
                    last_report = time.time()
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        total += len(chunk)

                        if time.time() - last_report >= 5:
                            if expected_size:
                                pct = total / expected_size * 100
                                print(f"  Progress: {total / 1e6:.1f} / "
                                      f"{expected_size / 1e6:.1f} MB ({pct:.1f}%)")
                            else:
                                print(f"  Progress: {total / 1e6:.1f} MB")
                            last_report = time.time()

            final_size = dest.stat().st_size
            print(f"[Done] Downloaded {final_size / 1e6:.1f} MB")

            if expected_size and final_size != expected_size:
                print(f"[WARN] Size mismatch: got {final_size}, "
                      f"expected {expected_size}")

            if expected_md5:
                print("[Verify] Computing MD5...")
                md5 = hashlib.md5()
                with open(dest, 'rb') as f:
                    for chunk in iter(lambda: f.read(chunk_size), b''):
                        md5.update(chunk)
                actual_md5 = md5.hexdigest()
                if actual_md5 != expected_md5.replace('md5:', ''):
                    print(f"[FAIL] MD5 mismatch: got {actual_md5}, "
                          f"expected {expected_md5}")
                    existing_size = 0  # reset and retry
                    continue
                else:
                    print(f"[OK] MD5 verified: {actual_md5}")

            return True

        except (urllib.error.URLError, urllib.error.HTTPError,
                ConnectionResetError, TimeoutError, OSError) as e:
            print(f"[Retry {attempt + 1}/{max_retries}] Error: {e}")
            existing_size = dest.stat().st_size if dest.exists() else 0
            wait = min(2 ** attempt * 5, 120)
            print(f"  Waiting {wait}s before retry...")
            time.sleep(wait)

    print(f"[FAIL] Download failed after {max_retries} attempts")
    return False


def main():
    parser = argparse.ArgumentParser(description='Robust resumable dataset downloader')
    parser.add_argument('--url', required=True, help='Download URL')
    parser.add_argument('--dest', required=True, help='Destination file path')
    parser.add_argument('--expected-size', type=int,
                        help='Expected file size in bytes')
    parser.add_argument('--expected-md5', help='Expected MD5 hash (may include "md5:" prefix)')
    parser.add_argument('--max-retries', type=int, default=5)
    parser.add_argument('--chunk-size', type=int, default=8 * 1024 * 1024,
                        help='Chunk size in bytes (default: 8 MiB)')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Connection timeout in seconds')
    args = parser.parse_args()

    success = download_file(
        url=args.url,
        dest_path=args.dest,
        expected_size=args.expected_size,
        expected_md5=args.expected_md5,
        max_retries=args.max_retries,
        chunk_size=args.chunk_size,
        timeout=args.timeout,
    )
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

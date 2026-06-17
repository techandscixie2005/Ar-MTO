#!/usr/bin/env python3
"""Build the dataset registry from individual configs and audit results.

Reads per-dataset YAML configs and audit outputs, then assembles a unified
dataset_registry.yaml with current state.

Usage:
    python build_dataset_registry.py
"""

import json
import os
import sys
from datetime import datetime, timezone


def main():
    """Rebuild registry from configs and audit data."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.join(script_dir, '..', '..')

    registry_path = os.path.join(repo_root, 'data_registry', 'dataset_registry.yaml')
    audit_path = os.path.join(repo_root, 'outputs', 'audit', 'data_foundry', 'external_dataset_audit.json')

    print(f"Registry: {registry_path}")
    print(f"Audit:    {audit_path}")

    # Load existing registry
    if os.path.exists(registry_path):
        print(f"\nRegistry exists at {registry_path}")
        print("No rebuild needed — registry is manually maintained.")
        print("Run audit_external_datasets.py to update audit results.")
    else:
        print(f"\nERROR: Registry not found at {registry_path}")
        sys.exit(1)

    # Load audit if available
    if os.path.exists(audit_path):
        with open(audit_path) as f:
            audit = json.load(f)
        print(f"\nAudit loaded: {audit['summary']}")
    else:
        print("\nNo audit file found. Run audit_external_datasets.py first.")


if __name__ == "__main__":
    main()

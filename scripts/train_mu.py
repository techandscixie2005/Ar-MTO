#!/usr/bin/env python3
"""train_mu.py — Config-driven MTO-Net training for dipole moment prediction.

Usage:
  # Dry-run: parse config, init model, single forward/backward pass
  python scripts/train_mu.py --config configs/model/mto_full.yaml --dry-run

  # Train with smoke subset
  python scripts/train_mu.py --config configs/model/mto_full.yaml \\
      --dataset data/qm9s/subset_smoke/qm9s.pt \\
      --splits data/qm9s/splits/smoke/splits.json \\
      --run-dir runs/mu_smoke_test

  # Train with full QM9S
  python scripts/train_mu.py --config configs/model/mto_full.yaml \\
      --dataset data/qm9s/qm9s.pt \\
      --splits data/qm9s/splits/full/splits.json \\
      --run-dir runs/mu_full/seed0 \\
      --epochs 200 --batch-size 64 --lr 5e-4 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml

# Add project src to path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


# ============================================================================
# Metrics
# ============================================================================

def compute_vector_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict:
    """Compute metrics for vector-valued predictions.

    Args:
        pred: [B, D, 3] predicted vectors
        target: [B, D, 3] target vectors

    Returns:
        dict with keys: vec_mae, norm_mae, rmse, r2, ang_err_deg
    """
    B, D = pred.shape[:2]
    pred = pred.reshape(B * D, 3)
    target = target.reshape(B * D, 3)

    diff = pred - target  # [N, 3]

    # Vector MAE (component-wise)
    vec_mae = diff.abs().mean().item()

    # Norm MAE
    pred_norm = torch.norm(pred, dim=-1)
    target_norm = torch.norm(target, dim=-1)
    norm_mae = (pred_norm - target_norm).abs().mean().item()

    # RMSE
    rmse = torch.sqrt((diff ** 2).mean()).item()

    # R²
    ss_res = (diff ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = (1.0 - ss_res / ss_tot).item() if ss_tot > 0 else float("nan")

    # Angular error (degrees) — only meaningful for vector targets
    cos_sim = torch.nn.functional.cosine_similarity(pred, target, dim=-1)
    cos_sim = cos_sim.clamp(-1.0, 1.0)
    ang_err_rad = torch.acos(cos_sim)
    ang_err_deg = ang_err_rad.mean().item() * 180.0 / 3.141592653589793

    return {
        "vec_mae": vec_mae,
        "norm_mae": norm_mae,
        "rmse": rmse,
        "r2": r2,
        "ang_err_deg": ang_err_deg,
    }


def format_metrics(metrics: dict) -> str:
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append("%s=%.6f" % (k, v))
        else:
            parts.append("%s=%s" % (k, v))
    return " ".join(parts)


# ============================================================================
# Dataset
# ============================================================================

def load_qm9s_dataset(dataset_path: str, split_indices: list[int] | None = None):
    """Load QM9S dataset and optionally filter by split indices.

    Supports both torch_geometric Data objects and plain dict entries.
    """
    torch.serialization.add_safe_globals([slice])
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)

    if split_indices is not None:
        idx_set = set(split_indices)
        data = [data[i] for i in split_indices if i < len(data)]

    return data


def collate_molecules(batch: list) -> tuple:
    """Collate a list of Data objects into a batched tensor dictionary.

    Each Data object has: z, pos, edge_index (optional), dipole, etc.
    Returns dict with batched tensors.
    """
    z_list = []
    pos_list = []
    dipole_list = []
    batch_idx_list = []
    edge_index_list = []
    offset = 0

    for i, mol in enumerate(batch):
        n_atoms = mol.z.shape[0]
        z_list.append(mol.z)
        pos_list.append(mol.pos)
        batch_idx_list.append(torch.full((n_atoms,), i, dtype=torch.long))
        # dipole: [1, 3] → squeeze to [3]
        d = mol.dipole
        if d.dim() == 2:
            d = d.reshape(-1)
        dipole_list.append(d)
        if hasattr(mol, "edge_index") and mol.edge_index is not None:
            ei = mol.edge_index + offset
            edge_index_list.append(ei)
        offset += n_atoms

    z = torch.cat(z_list, dim=0)
    pos = torch.cat(pos_list, dim=0)
    batch_idx = torch.cat(batch_idx_list, dim=0)
    dipole = torch.stack(dipole_list, dim=0)  # [B, 3]

    if edge_index_list:
        edge_index = torch.cat(edge_index_list, dim=1)  # [2, E_total]
    else:
        edge_index = None

    return {
        "z": z,
        "pos": pos,
        "batch": batch_idx,
        "edge_index": edge_index,
        "dipole": dipole,
    }


# ============================================================================
# Training state and checkpointing
# ============================================================================

class TrainingState:
    """Holds training state for checkpointing and resumption."""

    def __init__(self, config: dict, run_dir: Path, device: torch.device):
        self.config = config
        self.run_dir = run_dir
        self.device = device
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.global_step = 0
        self.metrics_history: list[dict] = []
        self.mto_cache: list[dict] = []

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        filename: str = "last.ckpt",
    ):
        path = self.run_dir / filename
        ckpt = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": self.config,
            "metrics_history": self.metrics_history,
        }
        torch.save(ckpt, str(path))
        return path

    def load_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                        filename: str = "last.ckpt") -> bool:
        path = self.run_dir / filename
        if not path.exists():
            return False
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.metrics_history = ckpt.get("metrics_history", [])
        print("Loaded checkpoint: %s (epoch %d)" % (filename, self.epoch))
        return True

    def save_predictions(self, pred: torch.Tensor, target: torch.Tensor, split: str):
        path = self.run_dir / ("pred_%s_epoch%04d.pt" % (split, self.epoch))
        torch.save({"pred": pred.cpu(), "target": target.cpu(), "epoch": self.epoch}, str(path))

    def save_mto_cache(self, modes: dict, epoch: int):
        path = self.run_dir / ("mto_cache_epoch%04d.pt" % epoch)
        cpu_modes = {str(k): v.cpu() for k, v in modes.items()}
        torch.save(cpu_modes, str(path))
        self.mto_cache.append({"epoch": epoch, "path": str(path)})

    def save_routing_stats(self, diagnostics: dict, epoch: int):
        if diagnostics is None:
            return
        path = self.run_dir / ("routing_stats_epoch%04d.pt" % epoch)
        route = diagnostics.get("route_stats", {})
        gate = diagnostics.get("gate_stats", {})
        out = {"epoch": epoch, "route_stats": route, "gate_stats": gate}
        # Convert tensors to float
        out_serializable = _to_serializable(out)
        torch.save(out_serializable, str(path))

    def save_metrics_csv(self):
        if not self.metrics_history:
            return
        path = self.run_dir / "metrics.csv"
        keys = list(self.metrics_history[0].keys())
        with open(path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in self.metrics_history:
                f.write(",".join(str(row[k]) for k in keys) + "\n")


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    elif isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.tolist()
    else:
        return obj


# ============================================================================
# Training loop
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    state: TrainingState,
    grad_clip: float = 1.0,
) -> dict:
    """Run one training epoch. Returns average loss and metrics."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_data in loader:
        z = batch_data["z"].to(device)
        pos = batch_data["pos"].to(device)
        b = batch_data["batch"].to(device)
        edge_index = batch_data.get("edge_index")
        if edge_index is not None:
            edge_index = edge_index.to(device)
        target = batch_data["dipole"].to(device)  # [B, 3]

        output = model(z=z, pos=pos, batch=b, edge_index=edge_index)
        pred = output["vector"]  # [B, 1, 3]

        loss = nn.functional.mse_loss(pred.reshape(-1, 3), target.reshape(-1, 3))

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        state.global_step += 1

    avg_loss = total_loss / max(n_batches, 1)
    return {"train_loss": avg_loss}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    state: TrainingState | None = None,
    save_predictions: bool = False,
    save_mto: bool = False,
    split: str = "val",
) -> dict:
    """Evaluate model and return metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    n_batches = 0

    for batch_data in loader:
        z = batch_data["z"].to(device)
        pos = batch_data["pos"].to(device)
        b = batch_data["batch"].to(device)
        edge_index = batch_data.get("edge_index")
        if edge_index is not None:
            edge_index = edge_index.to(device)
        target = batch_data["dipole"].to(device)

        return_diag = save_mto and state is not None
        output = model(
            z=z, pos=pos, batch=b, edge_index=edge_index,
            return_modes=save_mto,
            return_diagnostics=return_diag,
        )
        pred = output["vector"]

        loss = nn.functional.mse_loss(pred.reshape(-1, 3), target.reshape(-1, 3))
        total_loss += loss.item()
        n_batches += 1

        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())

        # Save first batch MTO modes for analysis
        if save_mto and state is not None and n_batches == 1 and "modes" in output:
            state.save_mto_cache(output["modes"], state.epoch)
        if return_diag and state is not None and n_batches == 1 and "diagnostics" in output:
            state.save_routing_stats(output["diagnostics"], state.epoch)

    preds = torch.cat(all_preds, dim=0)  # [N, 1, 3]
    targets = torch.cat(all_targets, dim=0)

    avg_loss = total_loss / max(n_batches, 1)
    metrics = compute_vector_metrics(preds, targets)
    metrics["%s_loss" % split] = avg_loss

    if save_predictions and state is not None:
        state.save_predictions(preds, targets, split)

    return metrics


# ============================================================================
# Dry-run
# ============================================================================

def dry_run(config_path: str, model_config: dict, device: torch.device):
    """Initialize model, run single forward/backward pass, verify everything works."""
    print("=" * 60)
    print("DRY RUN: %s" % config_path)
    print("=" * 60)

    # Build model
    print("\n[1] Building MTONet...")
    from ar_mto.mto_net import make_mto_net
    from ar_mto.detanet_bridge import make_latent_detanet

    detanet = make_latent_detanet(
        num_features=model_config.get("num_features", 128),
        maxl=model_config.get("maxl", 3),
        num_block=2,
        device=str(device),
    )
    model = make_mto_net(detanet_model=detanet, **model_config)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("  Parameters: %s" % f"{n_params:,}")
    print("  Config: %s" % model_config.get("num_modes", "?") + " modes, "
          + str(model_config.get("mode_channels", "?")) + " channels")

    # Synthetic batch
    print("\n[2] Creating synthetic batch (4 molecules, 3-7 atoms each)...")
    n_mols = 4
    atoms_per_mol = [3, 5, 7, 4]
    z_parts = []
    pos_parts = []
    batch_parts = []
    for i, n in enumerate(atoms_per_mol):
        z_parts.append(torch.randint(1, 9, (n,)))
        pos_parts.append(torch.randn(n, 3))
        batch_parts.append(torch.full((n,), i, dtype=torch.long))

    z = torch.cat(z_parts).to(device)
    pos = torch.cat(pos_parts).to(device)
    batch = torch.cat(batch_parts).to(device)

    print("  Total atoms: %d" % z.shape[0])

    # Forward pass
    print("\n[3] Forward pass...")
    model.train()
    output = model(z=z, pos=pos, batch=batch, return_modes=True, return_diagnostics=True)
    pred = output["vector"]
    print("  vector pred shape: %s" % str(list(pred.shape)))
    print("  vector pred mean:  %.6f" % pred.mean().item())
    print("  vector pred std:   %.6f" % pred.std().item())

    # Check for NaN/inf
    pred_ok = not torch.isnan(pred).any() and not torch.isinf(pred).any()
    print("  vector pred clean: %s" % pred_ok)

    # Mode shapes
    if "modes" in output:
        for k, v in output["modes"].items():
            print("  modes[%s]: shape=%s" % (k, list(v.shape)))

    # Diagnostics
    if "diagnostics" in output:
        diag = output["diagnostics"]
        route = diag.get("route_stats", {})
        gate = diag.get("gate_stats", {})
        print("  route entropy: %s" % route.get("route_entropy", "N/A"))
        for l_key in gate:
            if l_key not in ("mean_gate",):
                print("  gate stats[%s]: %s" % (l_key, gate[l_key]))

    # Backward pass
    print("\n[4] Backward pass...")
    target = torch.randn(n_mols, 1, 3, device=device)
    loss = nn.functional.mse_loss(pred, target)
    loss.backward()
    print("  loss: %.6f" % loss.item())

    # Gradient sanity
    grad_norm = sum(
        p.grad.norm().item() ** 2
        for p in model.parameters()
        if p.grad is not None
    ) ** 0.5
    print("  grad norm: %.6f" % grad_norm)
    grad_ok = not (torch.isnan(torch.tensor(grad_norm)) or torch.isinf(torch.tensor(grad_norm)))
    print("  grad ok: %s" % grad_ok)

    # Check all modules produce finite values
    model.eval()
    with torch.no_grad():
        output2 = model(z=z, pos=pos, batch=batch)
        pred2 = output2["vector"]
    print("  eval forward ok: %s" % (not torch.isnan(pred2).any() and not torch.isinf(pred2).any()))

    success = pred_ok and grad_ok
    print("\n" + ("=" * 60))
    print("DRY RUN %s" % ("PASSED" if success else "FAILED"))
    print("=" * 60)
    return success


# ============================================================================
# Main training
# ============================================================================

def main_train(args):
    """Full training run."""
    # Load config
    with open(args.config) as f:
        model_config = yaml.safe_load(f)

    # Override with CLI args
    if args.batch_size is not None:
        model_config["batch_size"] = args.batch_size
    if args.lr is not None:
        model_config["lr"] = args.lr
    if args.epochs is not None:
        model_config["epochs"] = args.epochs
    if args.seed is not None:
        model_config["seed"] = args.seed

    batch_size = model_config.get("batch_size", 64)
    lr = model_config.get("lr", 5e-4)
    epochs = model_config.get("epochs", 200)
    seed = model_config.get("seed", 0)
    grad_clip = model_config.get("grad_clip", 1.0)
    val_every = model_config.get("val_every", 1)
    save_every = model_config.get("save_every", 10)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    # Set seed
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Set up run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else Path(
        "runs/mu_%s_s%d" % (timestamp, seed)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print("Run directory: %s" % run_dir)

    # Save run config
    run_config = {
        "model_config": model_config,
        "dataset": args.dataset,
        "splits": args.splits,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "seed": seed,
        "command": " ".join(sys.argv),
        "timestamp": timestamp,
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2, default=str)

    # Load splits
    if args.splits:
        with open(args.splits) as f:
            splits = json.load(f)
        train_idx = splits["train"]
        val_idx = splits["val"]
        test_idx = splits["test"]
        print("Splits: train=%d, val=%d, test=%d" % (len(train_idx), len(val_idx), len(test_idx)))
    else:
        train_idx = val_idx = test_idx = None

    # Load dataset
    print("Loading dataset: %s" % args.dataset)
    t0 = time.time()
    full_data = load_qm9s_dataset(args.dataset)
    print("  Loaded %d molecules in %.1fs" % (len(full_data), time.time() - t0))

    # Split
    if train_idx is not None:
        train_data = [full_data[i] for i in train_idx if i < len(full_data)]
        val_data = [full_data[i] for i in val_idx if i < len(full_data)]
        test_data = [full_data[i] for i in test_idx if i < len(full_data)]
    else:
        # Default: 80/10/10 split
        n = len(full_data)
        rng = __import__("random").Random(seed)
        indices = list(range(n))
        rng.shuffle(indices)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        train_data = [full_data[i] for i in indices[:n_train]]
        val_data = [full_data[i] for i in indices[n_train:n_train + n_val]]
        test_data = [full_data[i] for i in indices[n_train + n_val:]]
        print("Default split: train=%d, val=%d, test=%d" % (
            len(train_data), len(val_data), len(test_data)))

    # Build model
    print("Building MTONet...")
    from ar_mto.mto_net import make_mto_net
    from ar_mto.detanet_bridge import make_latent_detanet

    detanet = make_latent_detanet(
        num_features=model_config.get("num_features", 128),
        maxl=model_config.get("maxl", 3),
        num_block=model_config.get("num_block", 3),
        device=str(device),
    )
    model = make_mto_net(detanet_model=detanet, **model_config)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("  Parameters: %s" % f"{n_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )

    # Create data loaders
    print("Creating data loaders (batch_size=%d)..." % batch_size)
    train_loader = DataLoaderWrapper(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoaderWrapper(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoaderWrapper(test_data, batch_size=batch_size, shuffle=False)

    # Training state
    state = TrainingState(config=model_config, run_dir=run_dir, device=device)

    # Resume if checkpoint exists
    if args.resume:
        if state.load_checkpoint(model, optimizer, "last.ckpt"):
            print("Resuming from epoch %d" % state.epoch)

    # Training loop
    print("\nStarting training for %d epochs..." % epochs)
    print("=" * 60)

    for epoch in range(state.epoch + 1, epochs + 1):
        state.epoch = epoch
        t_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, state, grad_clip=grad_clip
        )

        # Evaluate
        epoch_metrics = {"epoch": epoch}
        epoch_metrics.update(train_metrics)

        if epoch % val_every == 0:
            save_mto = (epoch % save_every == 0)
            val_metrics = evaluate(
                model, val_loader, device, state,
                save_predictions=save_mto,
                save_mto=save_mto,
                split="val",
            )
            epoch_metrics.update(val_metrics)

            # Scheduler step
            val_loss = val_metrics.get("val_loss", float("inf"))
            scheduler.step(val_loss)

            # Best checkpoint
            if val_loss < state.best_val_loss:
                state.best_val_loss = val_loss
                state.save_checkpoint(model, optimizer, "best.ckpt")
                print("  -> new best: val_loss=%.6f" % val_loss)

        # Save periodic
        if epoch % save_every == 0:
            state.save_checkpoint(model, optimizer, "last.ckpt")
            state.save_metrics_csv()

        # Log
        elapsed = time.time() - t_start
        lr_now = optimizer.param_groups[0]["lr"]
        print("epoch %4d/%d | %s | lr=%.2e | %.1fs" % (
            epoch, epochs, format_metrics(epoch_metrics), lr_now, elapsed))

        state.metrics_history.append(epoch_metrics)

    # Final save
    state.save_checkpoint(model, optimizer, "last.ckpt")
    state.save_metrics_csv()

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Final test evaluation")
    print("=" * 60)
    test_metrics = evaluate(
        model, test_loader, device, state,
        save_predictions=True, save_mto=True, split="test",
    )
    test_metrics["epoch"] = epochs
    print("Test: %s" % format_metrics(test_metrics))
    state.metrics_history.append(test_metrics)
    state.save_metrics_csv()

    # Save test results
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\nDone. Run directory: %s" % run_dir)
    return 0


# ============================================================================
# Simple DataLoader wrapper (no torch_geometric dependency for basic use)
# ============================================================================

class DataLoaderWrapper:
    """Simple batching DataLoader for QM9S Data objects.

    Uses random sampling for shuffle mode to produce batches of
    collated molecules suitable for DetaNet.
    """

    def __init__(self, dataset: list, batch_size: int = 64, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            indices = torch.randperm(n).tolist()
        else:
            indices = list(range(n))

        for start in range(0, n, self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch = [self.dataset[i] for i in batch_indices]
            yield collate_molecules(batch)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="MTO-Net dipole moment training")
    p.add_argument("--config", default="configs/model/mto_full.yaml",
                   help="Path to model config YAML")
    p.add_argument("--dataset", default=None,
                   help="Path to QM9S .pt dataset")
    p.add_argument("--splits", default=None,
                   help="Path to splits JSON file")
    p.add_argument("--run-dir", default=None,
                   help="Output directory for checkpoints and artifacts")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                   help="Resume from last checkpoint")
    p.add_argument("--dry-run", action="store_true",
                   help="Initialize model and run single fwd/bwd pass")
    p.add_argument("--device", default=None,
                   help="Device override (cpu, cuda:0, etc.)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print("ERROR: config not found: %s" % config_path)
        sys.exit(1)
    with open(config_path) as f:
        model_config = yaml.safe_load(f)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if args.dry_run:
        success = dry_run(str(config_path), model_config, device)
        sys.exit(0 if success else 1)
    else:
        if args.dataset is None:
            print("ERROR: --dataset required for training (or use --dry-run)")
            sys.exit(1)
        sys.exit(main_train(args))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""train_pilot.py — Task-agnostic MTO-Net training for the 2k mu+UV Pilot.

Supports target types: mu (dipole vector), uv (transition-broadened spectrum),
and multitask (mu+uv jointly).

Replaces: scripts/train_mu.py for the pilot.
Backward-compatible: --target-type mu produces identical output to train_mu.py.

Usage:
  # μ only
  python scripts/train_pilot.py --config configs/model/mto_full.yaml \\
      --dataset data/qm9s/qm9s.pt \\
      --splits outputs/splits/qm9s_2k_pilot_train.json \\
           outputs/splits/qm9s_2k_pilot_val.json \\
           outputs/splits/qm9s_2k_pilot_test.json \\
      --run-dir outputs/pilot_2k_mu_uv/checkpoints/mu_mto_k8_s0 \\
      --target-type mu --seed 0

  # UV only
  python scripts/train_pilot.py --config configs/model/mto_full.yaml \\
      --dataset data/qm9s/qm9s.pt \\
      --splits ... --run-dir ... --target-type uv --seed 0

  # multitask
  python scripts/train_pilot.py --config configs/model/mto_full.yaml \\
      --dataset ... --splits ... --run-dir ... --target-type multitask --seed 0

  # Dry-run
  python scripts/train_pilot.py --config configs/model/mto_full.yaml --dry-run
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

import numpy as np
import torch
import torch.nn as nn
import yaml

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from ar_mto.detanet_bridge import _ensure_pyg_available  # noqa: E402
_ensure_pyg_available()
from ar_mto.detanet_bridge import is_pyg_fallback_active, compute_radius_edges  # noqa: E402


# ============================================================================
# Spectrum synthesis from transition data
# ============================================================================

def synthesize_uv_spectrum(
    tran_energy: torch.Tensor,    # [..., N_trans]
    tran_dipole: torch.Tensor,    # [..., N_trans, 3]
    n_bins: int = 3501,
    e_min: float = 0.0,
    e_max: float = 35.0,
    sigma: float = 0.2,          # Gaussian broadening (eV)
) -> torch.Tensor:
    """Synthesize UV-Vis absorption spectrum from transition data.

    Uses Gaussian broadening of oscillator strengths:
        spectrum(E) = sum_i f_i * exp(-(E - E_i)^2 / (2 * sigma^2))

    where f_i = (2/3) * E_i * |mu_i|^2 is the oscillator strength.

    Args:
        tran_energy: transition energies in eV, shape [..., N_trans]
        tran_dipole: transition dipole moments, shape [..., N_trans, 3]
        n_bins: number of energy bins
        e_min, e_max: energy range in eV
        sigma: Gaussian broadening width in eV

    Returns:
        spectrum: broadened absorption spectrum, shape [..., n_bins]
    """
    device = tran_energy.device
    dtype = tran_energy.dtype

    # Oscillator strength: f_i = (2/3) * E_i * |mu_i|^2
    osc_strength = (2.0 / 3.0) * tran_energy * (tran_dipole ** 2).sum(dim=-1)  # [..., N_trans]

    # Energy grid
    grid = torch.linspace(e_min, e_max, n_bins, device=device, dtype=dtype)  # [n_bins]

    # Broadcast for Gaussian: spectrum[bins, transitions]
    # E_i: [..., N_trans] -> [..., 1, N_trans]
    # grid: [n_bins] -> [1, n_bins, 1]
    e_i = tran_energy.unsqueeze(-2)   # [..., 1, N_trans]
    g = grid.view(*([1] * (e_i.dim() - 2)), n_bins, 1)  # [1, ..., n_bins, 1]

    # Gaussian: exp(-(E - E_i)^2 / (2*sigma^2))
    diff = (g - e_i) / sigma
    gauss = torch.exp(-0.5 * diff ** 2)  # [..., n_bins, N_trans]

    # Weight by oscillator strength
    f = osc_strength.unsqueeze(-2)  # [..., 1, N_trans]
    spectrum = (f * gauss).sum(dim=-1)  # [..., n_bins]

    # Normalize per molecule to unit integral
    integral = spectrum.sum(dim=-1, keepdim=True)
    spectrum = spectrum / (integral + 1e-10)

    return spectrum


# ============================================================================
# Metrics
# ============================================================================

def compute_vector_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute μ (dipole vector) metrics.

    Args:
        pred: [B, D, 3]
        target: [B, D, 3]
    """
    B, D = pred.shape[:2]
    pred = pred.reshape(B * D, 3)
    target = target.reshape(B * D, 3)

    diff = pred - target
    vec_mae = diff.abs().mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()

    pred_norm = torch.norm(pred, dim=-1)
    target_norm = torch.norm(target, dim=-1)
    norm_mae = (pred_norm - target_norm).abs().mean().item()

    ss_res = (diff ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = (1.0 - ss_res / ss_tot).item() if ss_tot > 0 else float("nan")

    cos_sim_val = nn.functional.cosine_similarity(pred, target, dim=-1).clamp(-1.0, 1.0)
    ang_err_rad = torch.acos(cos_sim_val)
    ang_err_deg = ang_err_rad.mean().item() * 180.0 / 3.141592653589793
    cosine_sim = cos_sim_val.mean().item()

    # R² per component
    r2_components = {}
    for i, name in enumerate(["x", "y", "z"]):
        ss_res_c = (diff[:, i] ** 2).sum()
        ss_tot_c = ((target[:, i] - target[:, i].mean()) ** 2).sum()
        r2_components[f"r2_{name}"] = (1.0 - ss_res_c / ss_tot_c).item() if ss_tot_c > 0 else float("nan")

    return {
        "vec_mae": vec_mae, "norm_mae": norm_mae, "rmse": rmse, "r2": r2,
        "ang_err_deg": ang_err_deg, "cosine_sim": cosine_sim,
        **r2_components,
    }


def compute_spectral_metrics(pred: torch.Tensor, target: torch.Tensor,
                              grid_start: float = 0.0, grid_end: float = 35.0) -> dict:
    """Compute UV spectrum metrics.

    Args:
        pred: [B, n_bins]
        target: [B, n_bins]
    """
    B = pred.shape[0]
    diff = pred - target

    mae = diff.abs().mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()

    # R²
    ss_res = (diff ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = (1.0 - ss_res / ss_tot).item() if ss_tot > 0 else float("nan")

    # Cosine similarity (per molecule, then mean)
    cos_sims = nn.functional.cosine_similarity(pred, target, dim=-1)
    cosine_sim = cos_sims.mean().item()

    # Spearman rank correlation
    spearman_vals = []
    for i in range(B):
        if B > 1:
            from scipy.stats import spearmanr as _sr
            try:
                rho, _ = _sr(pred[i].detach().cpu().numpy(),
                             target[i].detach().cpu().numpy())
                spearman_vals.append(rho)
            except Exception:
                spearman_vals.append(0.0)
        else:
            spearman_vals.append(0.0)
    spearman_r = float(np.mean(spearman_vals))

    # Peak metrics
    n_bins = pred.shape[1]
    grid_resolution = (grid_end - grid_start) / n_bins

    peak_energy_errors = []
    peak_intensity_errors = []
    top3_recall = []

    for i in range(B):
        p = pred[i].detach().cpu().numpy()
        t = target[i].detach().cpu().numpy()

        # Main peak (argmax)
        p_peak_idx = np.argmax(p)
        t_peak_idx = np.argmax(t)
        peak_energy_errors.append(abs(p_peak_idx - t_peak_idx) * grid_resolution)
        peak_intensity_errors.append(abs(p[p_peak_idx] - t[t_peak_idx]))

        # Top-3 recall: how many of the true top-3 peaks are within ±5 bins of predicted top-3
        t_top3 = set(np.argsort(t)[-3:])
        p_top3 = set(np.argsort(p)[-3:])
        recall = sum(1 for ti in t_top3 for pi in p_top3 if abs(ti - pi) <= 5) / 3.0
        top3_recall.append(recall)

    # Integrated intensity error
    integ_pred = pred.sum(dim=-1)
    integ_target = target.sum(dim=-1)
    integ_error = (integ_pred - integ_target).abs().mean().item()

    return {
        "spec_mae": mae,
        "spec_rmse": rmse,
        "spec_r2": r2,
        "spec_cosine_sim": cosine_sim,
        "spec_spearman_r": spearman_r,
        "peak_energy_mae_ev": float(np.mean(peak_energy_errors)),
        "peak_intensity_mae": float(np.mean(peak_intensity_errors)),
        "top3_peak_recall": float(np.mean(top3_recall)),
        "integrated_intensity_error": integ_error,
    }


def compute_aux_transition_metrics(pred: dict, target: dict) -> dict:
    """Compute auxiliary transition-level metrics if tran_energy/dipole labels exist."""
    metrics = {}
    if "tran_energy" in pred and "tran_energy" in target:
        diff_e = pred["tran_energy"] - target["tran_energy"]
        metrics["tran_energy_mae"] = diff_e.abs().mean().item()
    if "tran_dipole" in pred and "tran_dipole" in target:
        diff_d = pred["tran_dipole"] - target["tran_dipole"]
        metrics["tran_dipole_mae"] = diff_d.abs().mean().item()
    return metrics


# ============================================================================
# Data loading
# ============================================================================

def load_qm9s_dataset(dataset_path: str, split_indices: list[int] | None = None):
    """Load QM9S dataset and optionally filter by split indices."""
    torch.serialization.add_safe_globals([slice])
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if split_indices is not None:
        idx_set = set(split_indices)
        data = [data[i] for i in split_indices if i < len(data)]
    return data


def collate_molecules(batch: list, target_type: str = "mu") -> tuple:
    """Collate a list of Data objects into a batched tensor dictionary.

    Args:
        batch: list of _StubGraphData objects
        target_type: "mu", "uv", or "multitask"

    Returns:
        dict with batched tensors and targets
    """
    z_list, pos_list, batch_idx_list = [], [], []
    edge_index_list = []
    offset = 0

    # Targets
    dipole_list = []
    tran_e_list, tran_d_list = [], []

    for i, mol in enumerate(batch):
        n_atoms = mol.z.shape[0]
        z_list.append(mol.z)
        pos_list.append(mol.pos)
        batch_idx_list.append(torch.full((n_atoms,), i, dtype=torch.long))

        if hasattr(mol, "edge_index") and mol.edge_index is not None:
            ei = mol.edge_index + offset
            edge_index_list.append(ei)
        offset += n_atoms

        # Collect targets
        if target_type in ("mu", "multitask"):
            d = mol.dipole
            if d.dim() == 2:
                d = d.reshape(-1)
            dipole_list.append(d)

        if target_type in ("uv", "multitask"):
            e = mol.tran_energy.reshape(-1)    # [10]
            td = mol.tran_dipole.reshape(-1, 3)  # [10, 3]
            tran_e_list.append(e)
            tran_d_list.append(td)

    result = {
        "z": torch.cat(z_list, dim=0),
        "pos": torch.cat(pos_list, dim=0),
        "batch": torch.cat(batch_idx_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1) if edge_index_list else None,
    }

    # Targets
    if target_type in ("mu", "multitask"):
        result["dipole"] = torch.stack(dipole_list, dim=0).reshape(-1, 1, 3)  # [B, 1, 3]

    if target_type in ("uv", "multitask"):
        result["tran_energy"] = torch.stack(tran_e_list, dim=0)      # [B, 10]
        result["tran_dipole"] = torch.stack(tran_d_list, dim=0)      # [B, 10, 3]

    return result


class DataLoaderWrapper:
    """Simple DataLoader wrapper over a list of molecules."""
    def __init__(self, dataset: list, batch_size: int = 64, shuffle: bool = False,
                 target_type: str = "mu"):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_type = target_type
        self._order = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            self._order = torch.randperm(len(self.dataset)).tolist()
        for start in range(0, len(self.dataset), self.batch_size):
            batch_indices = self._order[start:start + self.batch_size]
            batch = [self.dataset[i] for i in batch_indices]
            yield collate_molecules(batch, target_type=self.target_type)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# ============================================================================
# Training state
# ============================================================================

class TrainingState:
    def __init__(self, config: dict, run_dir: Path, device: torch.device):
        self.config = config
        self.run_dir = run_dir
        self.device = device
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.global_step = 0
        self.metrics_history: list[dict] = []
        self.mto_cache: list[dict] = []

    def save_checkpoint(self, model, optimizer, filename="last.ckpt"):
        path = self.run_dir / filename
        ckpt = {
            "epoch": self.epoch, "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": self.config, "metrics_history": self.metrics_history,
        }
        torch.save(ckpt, str(path))
        return path

    def load_checkpoint(self, model, optimizer, filename="last.ckpt") -> bool:
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

    def save_predictions(self, pred, target, split):
        path = self.run_dir / ("pred_%s_epoch%04d.pt" % (split, self.epoch))
        pred_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in (pred.items() if isinstance(pred, dict) else {"pred": pred, "target": target}.items())}
        target_cpu = target if not isinstance(target, torch.Tensor) else target.cpu()
        data = {"pred": pred_cpu, "target": target_cpu, "epoch": self.epoch}
        if isinstance(target, dict):
            data["target"] = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                              for k, v in target.items()}
        torch.save(data, str(path))

    def save_mto_cache(self, modes, epoch):
        path = self.run_dir / ("mto_cache_epoch%04d.pt" % epoch)
        cpu_modes = {str(k): v.cpu() for k, v in modes.items()}
        torch.save(cpu_modes, str(path))
        self.mto_cache.append({"epoch": epoch, "path": str(path)})

    def save_routing_stats(self, diagnostics, epoch):
        if diagnostics is None:
            return
        path = self.run_dir / ("routing_stats_epoch%04d.pt" % epoch)
        route = diagnostics.get("route_stats", {})
        gate = diagnostics.get("gate_stats", {})
        out = {"epoch": epoch, "route_stats": route, "gate_stats": gate}
        torch.save(_to_serializable(out), str(path))

    def save_metrics_csv(self):
        if not self.metrics_history:
            return
        path = self.run_dir / "metrics.csv"
        keys = list(self.metrics_history[0].keys())
        for row in self.metrics_history[1:]:
            for k in row:
                if k not in keys:
                    keys.append(k)
        with open(path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in self.metrics_history:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    elif isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.tolist()
    return obj


# ============================================================================
# Training loop
# ============================================================================

def train_epoch(model, loader, optimizer, device, state, target_type="mu",
                grad_clip=1.0):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_data in loader:
        z = batch_data["z"].to(device)
        pos = batch_data["pos"].to(device)
        b = batch_data["batch"].to(device)
        ei = batch_data.get("edge_index")
        if ei is not None:
            ei = ei.to(device)

        output = model(z=z, pos=pos, batch=b, edge_index=ei)

        if target_type == "mu":
            pred = output["vector"]  # [B, 1, 3]
            target = batch_data["dipole"].to(device)
            loss = nn.functional.mse_loss(pred.reshape(-1, 3), target.reshape(-1, 3))

        elif target_type == "uv":
            pred_spectrum = output.get("spectrum")  # [B, n_bins]
            if pred_spectrum is None:
                pred_spectrum = output.get("scalar")
            # Synthesize target spectrum from transition data
            tran_e = batch_data["tran_energy"].to(device)
            tran_d = batch_data["tran_dipole"].to(device)
            target_spectrum = synthesize_uv_spectrum(tran_e, tran_d,
                                                      n_bins=pred_spectrum.shape[-1])
            loss = nn.functional.mse_loss(pred_spectrum, target_spectrum)

        elif target_type == "multitask":
            # μ loss
            pred_mu = output["vector"]
            target_mu = batch_data["dipole"].to(device)
            loss_mu = nn.functional.mse_loss(pred_mu.reshape(-1, 3), target_mu.reshape(-1, 3))

            # UV loss
            pred_spectrum = output.get("spectrum") or output.get("scalar")
            tran_e = batch_data["tran_energy"].to(device)
            tran_d = batch_data["tran_dipole"].to(device)
            target_spectrum = synthesize_uv_spectrum(tran_e, tran_d,
                                                      n_bins=pred_spectrum.shape[-1])
            loss_uv = nn.functional.mse_loss(pred_spectrum, target_spectrum)

            loss = loss_mu + loss_uv

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        state.global_step += 1

    return {"train_loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def evaluate(model, loader, device, state, target_type="mu",
             save_predictions=False, save_mto=False, split="val"):
    model.eval()
    total_loss = 0.0
    all_preds_mu, all_targets_mu = [], []
    all_preds_uv, all_targets_uv = [], []
    n_batches = 0

    for batch_data in loader:
        z = batch_data["z"].to(device)
        pos = batch_data["pos"].to(device)
        b = batch_data["batch"].to(device)
        ei = batch_data.get("edge_index")
        if ei is not None:
            ei = ei.to(device)

        return_diag = save_mto and state is not None
        output = model(
            z=z, pos=pos, batch=b, edge_index=ei,
            return_modes=save_mto,
            return_diagnostics=return_diag,
        )

        if target_type in ("mu", "multitask"):
            pred_mu = output["vector"].cpu()
            target_mu = batch_data["dipole"]
            all_preds_mu.append(pred_mu)
            all_targets_mu.append(target_mu)

        if target_type in ("uv", "multitask"):
            pred_uv = (output.get("spectrum") or output.get("scalar")).cpu()
            tran_e = batch_data["tran_energy"]
            tran_d = batch_data["tran_dipole"]
            target_uv = synthesize_uv_spectrum(tran_e, tran_d,
                                                n_bins=pred_uv.shape[-1])
            all_preds_uv.append(pred_uv)
            all_targets_uv.append(target_uv.cpu())

        # Compute loss
        if target_type == "mu":
            p = output["vector"]
            t = batch_data["dipole"].to(device)
            loss = nn.functional.mse_loss(p.reshape(-1, 3), t.reshape(-1, 3))
        elif target_type == "uv":
            p = output.get("spectrum") or output.get("scalar")
            t = synthesize_uv_spectrum(tran_e.to(device), tran_d.to(device),
                                        n_bins=p.shape[-1]).to(device)
            loss = nn.functional.mse_loss(p, t)
        elif target_type == "multitask":
            p_mu = output["vector"]
            t_mu = batch_data["dipole"].to(device)
            loss_mu = nn.functional.mse_loss(p_mu.reshape(-1, 3), t_mu.reshape(-1, 3))
            p_uv = output.get("spectrum") or output.get("scalar")
            t_uv = synthesize_uv_spectrum(tran_e.to(device), tran_d.to(device),
                                           n_bins=p_uv.shape[-1]).to(device)
            loss_uv = nn.functional.mse_loss(p_uv, t_uv)
            loss = loss_mu + loss_uv

        total_loss += loss.item()
        n_batches += 1

        if save_mto and state is not None and n_batches == 1 and "modes" in output:
            state.save_mto_cache(output["modes"], state.epoch)
        if return_diag and state is not None and n_batches == 1 and "diagnostics" in output:
            state.save_routing_stats(output["diagnostics"], state.epoch)

    avg_loss = total_loss / max(n_batches, 1)
    metrics = {"%s_loss" % split: avg_loss}

    if target_type in ("mu", "multitask") and all_preds_mu:
        preds_mu = torch.cat(all_preds_mu, dim=0)
        tgts_mu = torch.cat(all_targets_mu, dim=0).reshape(-1, 1, 3)
        mu_metrics = compute_vector_metrics(preds_mu, tgts_mu)
        metrics.update({("%s_%s" % (split, k) if k != "%s_loss" % split else k): v
                        for k, v in mu_metrics.items()})

    if target_type in ("uv", "multitask") and all_preds_uv:
        preds_uv = torch.cat(all_preds_uv, dim=0)
        tgts_uv = torch.cat(all_targets_uv, dim=0)
        uv_metrics = compute_spectral_metrics(preds_uv, tgts_uv)
        metrics.update({("%s_%s" % (split, k)): v for k, v in uv_metrics.items()})

    if save_predictions and state is not None:
        pred_dict = {}
        if target_type in ("mu", "multitask") and all_preds_mu:
            pred_dict["vector"] = torch.cat(all_preds_mu, dim=0)
        if target_type in ("uv", "multitask") and all_preds_uv:
            pred_dict["spectral"] = torch.cat(all_preds_uv, dim=0)
        target_dict = {}
        if target_type in ("mu", "multitask") and all_targets_mu:
            target_dict["dipole"] = torch.cat(all_targets_mu, dim=0).reshape(-1, 1, 3)
        if target_type in ("uv", "multitask") and all_targets_uv:
            target_dict["uv_spectrum"] = torch.cat(all_targets_uv, dim=0)
        state.save_predictions(pred_dict, target_dict, split)

    return metrics


# ============================================================================
# Dry-run
# ============================================================================

def dry_run(config_path, model_config, device, target_type="mu"):
    print("=" * 60)
    print("DRY RUN: %s | target=%s" % (config_path, target_type))
    print("=" * 60)

    from ar_mto.detanet_bridge import is_pyg_fallback_active as _pyg
    print("\nPyG fallback active: %s" % _pyg())

    from ar_mto.mto_net import make_mto_net
    from ar_mto.detanet_bridge import make_latent_detanet

    # Build model
    print("\n[1] Building MTONet...")
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

    # Synthetic batch
    print("\n[2] Creating synthetic batch (4 molecules, 3-7 atoms each)...")
    n_mols = 4
    atoms_per_mol = [3, 5, 7, 4]
    z_parts, pos_parts, batch_parts = [], [], []
    for i, n in enumerate(atoms_per_mol):
        z_parts.append(torch.randint(1, 9, (n,)))
        pos_parts.append(torch.randn(n, 3))
        batch_parts.append(torch.full((n,), i, dtype=torch.long))
    z = torch.cat(z_parts).to(device)
    pos = torch.cat(pos_parts).to(device)
    batch_t = torch.cat(batch_parts).to(device)

    # Forward pass
    print("\n[3] Forward pass...")
    model.train()
    output = model(z=z, pos=pos, batch=batch_t, return_modes=True, return_diagnostics=True)

    success = True
    for key in ["scalar", "vector", "rank2", "spectral"]:
        if key in output:
            p = output[key]
            ok = not torch.isnan(p).any() and not torch.isinf(p).any()
            print("  %s pred shape: %s, ok=%s" % (key, list(p.shape), ok))
            if not ok:
                success = False

    # Mode shapes
    if "modes" in output:
        for k, v in output["modes"].items():
            print("  modes[%s]: shape=%s" % (k, list(v.shape)))

    # Diagnostics
    if "diagnostics" in output:
        diag = output["diagnostics"]
        route = diag.get("route_stats", {})
        print("  route entropy: %s" % route.get("route_entropy", "N/A"))
        gate = diag.get("gate_stats", {})
        for l_key in gate:
            if l_key != "mean_gate":
                print("  gate[%s]: %s" % (l_key, gate[l_key]))

    # Backward pass
    print("\n[4] Backward pass...")
    if target_type == "mu":
        target = torch.randn(n_mols, 1, 3, device=device)
        pred = output["vector"]
        loss = nn.functional.mse_loss(pred, target)
    elif target_type == "uv":
        # Synthetic UV spectrum target
        target = torch.randn(n_mols, model_config.get("spectral_bins", 3501), device=device)
        pred = output.get("spectrum") or output.get("scalar")
        loss = nn.functional.mse_loss(pred, target)
    elif target_type == "multitask":
        t_mu = torch.randn(n_mols, 1, 3, device=device)
        t_uv = torch.randn(n_mols, model_config.get("spectral_bins", 3501), device=device)
        pred_uv = output.get("spectrum")
        if pred_uv is None:
            pred_uv = output.get("scalar")
        loss = nn.functional.mse_loss(output["vector"], t_mu) + \
               nn.functional.mse_loss(pred_uv, t_uv)

    loss.backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters()
                    if p.grad is not None) ** 0.5
    grad_ok = not (torch.isnan(torch.tensor(grad_norm)) or torch.isinf(torch.tensor(grad_norm)))
    print("  loss: %.6f, grad_norm: %.6f, grad_ok: %s" % (loss.item(), grad_norm, grad_ok))

    if not grad_ok:
        success = False

    # Eval forward
    model.eval()
    with torch.no_grad():
        output2 = model(z=z, pos=pos, batch=batch_t)
    for key in list(output2.keys()):
        if isinstance(output2[key], torch.Tensor):
            ok = not torch.isnan(output2[key]).any()
            if not ok:
                print("  eval %s has NaN!" % key)
                success = False

    print("\n" + "=" * 60)
    print("DRY RUN %s" % ("PASSED" if success else "FAILED"))
    print("=" * 60)
    return success


# ============================================================================
# Main
# ============================================================================

def format_metrics(metrics):
    parts = []
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            parts.append("%s=%.4f" % (k, v))
    return " | ".join(parts)


def main_train(args):
    # Load config
    with open(args.config) as f:
        model_config = yaml.safe_load(f)

    # CLI overrides
    for key in ["batch_size", "lr", "epochs", "seed"]:
        val = getattr(args, key, None)
        if val is not None:
            model_config[key] = val

    target_type = args.target_type
    batch_size = model_config.get("batch_size", 64)
    lr = model_config.get("lr", 5e-4)
    epochs = model_config.get("epochs", 200)
    seed = model_config.get("seed", 0)
    grad_clip = model_config.get("grad_clip", 1.0)
    val_every = model_config.get("val_every", 1)
    save_every = model_config.get("save_every", 10)
    spectral_bins = model_config.get("spectral_bins", 3501)

    # For UV and multitask, ensure spectral head is active
    if target_type in ("uv", "multitask"):
        active_heads = model_config.get("active_heads", ["scalar", "vector", "rank2", "spectral"])
        if "spectral" not in active_heads:
            active_heads.append("spectral")
            model_config["active_heads"] = active_heads
        model_config["spectral_bins"] = spectral_bins

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s | Target: %s | Seed: %d" % (device, target_type, seed))

    # Seeds
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else Path(
        "runs/pilot_%s_s%d_%s" % (target_type, seed, timestamp)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print("Run directory: %s" % run_dir)

    # Save run config
    run_config = {
        "model_config": model_config,
        "dataset": args.dataset,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "target_type": target_type,
        "batch_size": batch_size, "lr": lr, "epochs": epochs, "seed": seed,
        "command": " ".join(sys.argv), "timestamp": timestamp,
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2, default=str)

    # Load splits
    def load_split_indices(path):
        with open(path) as f:
            data = json.load(f)
        # Could be a plain list or a dict with "train"/"val"/"test"
        if isinstance(data, list):
            return data
        return data  # Return as dict

    train_idx = load_split_indices(args.train_split)
    val_idx = load_split_indices(args.val_split)
    test_idx = load_split_indices(args.test_split)

    # Handle both list format and dict format
    if isinstance(train_idx, dict):
        train_idx = train_idx["train"]
        val_idx = val_idx["val"]
        test_idx = test_idx["test"]

    print("Splits: train=%d, val=%d, test=%d" % (len(train_idx), len(val_idx), len(test_idx)))

    # Load dataset
    print("Loading dataset: %s" % args.dataset)
    t0 = time.time()
    full_data = load_qm9s_dataset(args.dataset)
    print("  Loaded %d molecules in %.1fs" % (len(full_data), time.time() - t0))

    train_data = [full_data[i] for i in train_idx if i < len(full_data)]
    val_data = [full_data[i] for i in val_idx if i < len(full_data)]
    test_data = [full_data[i] for i in test_idx if i < len(full_data)]

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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )

    # Data loaders
    train_loader = DataLoaderWrapper(train_data, batch_size=batch_size, shuffle=True,
                                      target_type=target_type)
    val_loader = DataLoaderWrapper(val_data, batch_size=batch_size, shuffle=False,
                                    target_type=target_type)
    test_loader = DataLoaderWrapper(test_data, batch_size=batch_size, shuffle=False,
                                     target_type=target_type)

    state = TrainingState(config=model_config, run_dir=run_dir, device=device)

    if args.resume:
        if state.load_checkpoint(model, optimizer, "last.ckpt"):
            print("Resuming from epoch %d" % state.epoch)

    # Training loop
    print("\nStarting training for %d epochs..." % epochs)
    print("=" * 60)

    for epoch in range(state.epoch + 1, epochs + 1):
        state.epoch = epoch
        t_start = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, device, state,
                                     target_type=target_type, grad_clip=grad_clip)
        epoch_metrics = {"epoch": epoch}
        epoch_metrics.update(train_metrics)

        if epoch % val_every == 0:
            do_save = (epoch % save_every == 0)
            val_metrics = evaluate(model, val_loader, device, state,
                                    target_type=target_type,
                                    save_predictions=do_save,
                                    save_mto=do_save, split="val")
            epoch_metrics.update(val_metrics)

            val_loss = val_metrics.get("val_loss", float("inf"))
            scheduler.step(val_loss)

            if val_loss < state.best_val_loss:
                state.best_val_loss = val_loss
                state.save_checkpoint(model, optimizer, "best.ckpt")
                print("  -> new best: val_loss=%.6f" % val_loss)

        if epoch % save_every == 0:
            state.save_checkpoint(model, optimizer, "last.ckpt")
            state.save_metrics_csv()

        elapsed = time.time() - t_start
        lr_now = optimizer.param_groups[0]["lr"]
        print("epoch %4d/%d | %s | lr=%.2e | %.1fs" % (
            epoch, epochs, format_metrics(epoch_metrics), lr_now, elapsed))
        state.metrics_history.append(epoch_metrics)

    # Final save
    state.save_checkpoint(model, optimizer, "last.ckpt")
    state.save_metrics_csv()

    # Test evaluation
    print("\n" + "=" * 60)
    print("Final test evaluation")
    print("=" * 60)
    test_metrics = evaluate(model, test_loader, device, state,
                             target_type=target_type,
                             save_predictions=True, save_mto=True, split="test")
    test_metrics["epoch"] = epochs
    print("Test: %s" % format_metrics(test_metrics))
    state.metrics_history.append(test_metrics)
    state.save_metrics_csv()

    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(_to_serializable(test_metrics), f, indent=2)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(_to_serializable(state.metrics_history), f, indent=2)

    # Environment snapshot
    try:
        import subprocess
        env_info = {
            "hostname": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip(),
            "python_version": subprocess.run([sys.executable, "--version"],
                                             capture_output=True, text=True).stdout.strip(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        with open(run_dir / "environment.txt", "w") as f:
            for k, v in env_info.items():
                f.write("%s: %s\n" % (k, v))
    except Exception:
        pass

    # Git state
    try:
        import subprocess
        git_info = {
            "commit": subprocess.run(["git", "rev-parse", "HEAD"],
                                     capture_output=True, text=True, cwd=SRC_DIR.parent).stdout.strip(),
            "branch": subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                     capture_output=True, text=True, cwd=SRC_DIR.parent).stdout.strip(),
        }
        with open(run_dir / "git_state.json", "w") as f:
            json.dump(git_info, f, indent=2)
    except Exception:
        pass

    # Save config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(model_config, f)

    # MTO cache
    import glob as _glob
    mto_files = sorted(_glob.glob(str(run_dir / "mto_cache_epoch*.pt")))
    if mto_files:
        import shutil as _shutil
        _shutil.copy(mto_files[-1], str(run_dir / "mto_cache_test.pt"))

    # Route/gate stats
    routing_files = sorted(_glob.glob(str(run_dir / "routing_stats_epoch*.pt")))
    if routing_files:
        try:
            final_stats = torch.load(routing_files[-1], map_location="cpu", weights_only=False)
            for key_suffix, key in [("routing_stats.json", "route_stats"),
                                     ("mode_stats.json", "route_stats"),
                                     ("gate_stats.json", "gate_stats")]:
                try:
                    data = final_stats.get(key, {})
                    with open(run_dir / key_suffix, "w") as f:
                        json.dump(_to_serializable(data), f, indent=2)
                except Exception:
                    pass
        except Exception:
            pass

    # Predictions CSV
    import numpy as _np
    pred_files = sorted(_glob.glob(str(run_dir / "pred_test_epoch*.pt")))
    if pred_files:
        try:
            pred_data = torch.load(pred_files[-1], map_location="cpu", weights_only=False)
            with open(run_dir / "predictions_test.csv", "w") as f:
                if "pred" in pred_data and isinstance(pred_data["pred"], dict):
                    # Multitask format
                    pd = pred_data["pred"]
                    td = pred_data["target"]
                    # Write μ predictions
                    if "vector" in pd:
                        p_arr = pd["vector"].cpu().numpy() if isinstance(pd["vector"], torch.Tensor) else _np.array(pd["vector"])
                        t_arr = td["dipole"].cpu().numpy() if isinstance(td["dipole"], torch.Tensor) else _np.array(td["dipole"])
                        p_arr = p_arr.reshape(-1, 3)
                        t_arr = t_arr.reshape(-1, 3)
                        f.write("idx,pred_x,pred_y,pred_z,target_x,target_y,target_z\n")
                        for i in range(p_arr.shape[0]):
                            f.write("%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (
                                i, p_arr[i, 0], p_arr[i, 1], p_arr[i, 2],
                                t_arr[i, 0], t_arr[i, 1], t_arr[i, 2]))
                else:
                    p_arr = pred_data["pred"].cpu().numpy()
                    t_arr = pred_data["target"].cpu().numpy()
                    if p_arr.ndim == 3 and p_arr.shape[2] == 3:
                        f.write("idx,pred_x,pred_y,pred_z,target_x,target_y,target_z\n")
                        p_flat = p_arr.reshape(-1, 3)
                        t_flat = t_arr.reshape(-1, 3)
                        for i in range(p_flat.shape[0]):
                            f.write("%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (
                                i, p_flat[i, 0], p_flat[i, 1], p_flat[i, 2],
                                t_flat[i, 0], t_flat[i, 1], t_flat[i, 2]))
        except Exception:
            pass

    print("\nTraining complete. Run directory: %s" % run_dir)
    return 0


def main():
    parser = argparse.ArgumentParser(description="MTO-Net Pilot Training")
    parser.add_argument("--config", required=True, help="Model config YAML")
    parser.add_argument("--target-type", default="mu", choices=["mu", "uv", "multitask"],
                        help="Target type (default: mu)")
    parser.add_argument("--dataset", default=None, help="Path to QM9S .pt dataset")
    parser.add_argument("--train-split", default=None, help="JSON file with train split indices")
    parser.add_argument("--val-split", default=None, help="JSON file with val split indices")
    parser.add_argument("--test-split", default=None, help="JSON file with test split indices")
    parser.add_argument("--run-dir", default=None, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Dry run
    if args.dry_run:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ok = dry_run(args.config, cfg, device, target_type=args.target_type)
        sys.exit(0 if ok else 1)

    # Check required args
    if args.dataset is None or args.train_split is None:
        print("ERROR: --dataset, --train-split, --val-split, --test-split required")
        sys.exit(1)

    return main_train(args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Comprehensive MTO effective mode analysis for valence_half vs fixed-K=8 smoke training.

Produces all Phase 3.3b output tables: summary, mode importance, top-r masking, order masking.
Also computes correlations: K_eff vs atom count, N_val, dipole magnitude, error/loss.
"""
import argparse, json, os, sys, csv
from pathlib import Path
import torch, numpy as np
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))
sys.path.insert(0, str(SCRIPT_DIR / "third_party" / "DetaNet"))
torch.serialization.add_safe_globals([slice])

from ar_mto.detanet_bridge import _ensure_pyg_available
_ensure_pyg_available()

from ar_mto.mto_net import make_mto_net
from ar_mto.detanet_bridge import make_latent_detanet
from ar_mto.mto_core import compute_valence_adaptive_k

VALENCE = {1:1,2:2,3:1,4:2,5:3,6:4,7:5,8:6,9:7,10:8,11:1,12:2,13:3,14:4,15:5,16:6,17:7,18:8,19:1,20:2,21:3,22:4,23:5,24:6,25:7,26:8,27:9,28:10,29:11,30:12,31:3,32:4,33:5,34:6,35:7,36:8,37:1,38:2,39:3,40:4,41:5,42:6,43:7,44:8,45:9,46:10,47:11,48:12,49:3,50:4,51:5,52:6,53:7,54:8,55:1,56:2,57:3,58:4,59:5,60:6,61:7,62:8,63:9,64:10,65:11,66:12,67:13,68:14,69:15,70:16,71:3,72:4,73:5,74:6,75:7,76:8,77:9,78:10,79:11,80:12,81:3,82:4,83:5,84:6,85:7,86:8}


def load_run(run_dir):
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        with open(run_dir / "run_config.json") as f:
            rc = json.load(f)
            config = rc.get("model_config", {})
    else:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
    metrics_path = run_dir / "metrics.json"
    metrics = json.load(open(metrics_path)) if metrics_path.exists() else []
    ckpt = run_dir / "best.ckpt"
    if not ckpt.exists():
        ckpt = run_dir / "last.ckpt"
    return config, metrics, str(ckpt)


def load_dataset(run_dir):
    run_dir = Path(run_dir)
    try:
        with open(run_dir / "run_config.json") as f:
            rc = json.load(f)
        ds_path = rc.get("dataset", "data/qm9s/subset_medium/qm9s.pt")
        splits_path = rc.get("splits", "data/qm9s/splits/medium/splits.json")
    except Exception:
        ds_path = "data/qm9s/subset_medium/qm9s.pt"
        splits_path = "data/qm9s/splits/medium/splits.json"
    if not Path(ds_path).exists():
        ds_path = str(SCRIPT_DIR / ds_path)
    if not Path(splits_path).exists():
        splits_path = str(SCRIPT_DIR / splits_path)
    data = torch.load(ds_path, map_location="cpu", weights_only=False)
    with open(splits_path) as f:
        splits = json.load(f)
    test_idx = set(splits["test"])
    return [data[i] for i in test_idx if i < len(data)]


def collate_batch(mols):
    z_list, pos_list, batch_list, dip_list = [], [], [], []
    for i, mol in enumerate(mols):
        n = mol.z.shape[0]
        z_list.append(mol.z); pos_list.append(mol.pos)
        batch_list.append(torch.full((n,), i, dtype=torch.long))
        d = mol.dipole
        dip_list.append(d.reshape(-1) if d.dim() == 2 else d)
    return {"z": torch.cat(z_list), "pos": torch.cat(pos_list),
            "batch": torch.cat(batch_list), "dipole": torch.stack(dip_list, dim=0)}


def get_atom_metadata(test_data):
    """Get per-molecule N_val, atom_count, and dipole magnitude."""
    results = []
    for mol in test_data:
        total_v = sum(VALENCE[int(zi)] for zi in mol.z)
        k_half = (total_v + 1) // 2
        results.append({
            "n_atoms": mol.z.shape[0],
            "n_val": total_v,
            "k_half": k_half,
            "dipole_mag": float(torch.norm(mol.dipole.reshape(-1)).item()),
        })
    return results


def compute_all_metrics(model, test_data, device, batch_size=16):
    """Full forward pass on test set, returning predictions, modes, masks, ks, gate stats."""
    model.eval()
    all_preds, all_targets, all_masks, all_ks = [], [], [], []
    mode_norms = {l: [] for l in range(4)}
    gate_stats_all = {}

    for start in range(0, len(test_data), batch_size):
        mols = test_data[start:start + batch_size]
        bdata = collate_batch(mols)
        z = bdata["z"].to(device)
        pos = bdata["pos"].to(device)
        b = bdata["batch"].to(device)
        target = bdata["dipole"].to(device)

        with torch.no_grad():
            out = model(z=z, pos=pos, batch=b, return_modes=True, return_diagnostics=(start == 0))

        all_preds.append(out["vector"].cpu())
        all_targets.append(target.cpu())

        if start == 0 and "diagnostics" in out and "gate_stats" in out["diagnostics"]:
            gate_stats_all = out["diagnostics"]["gate_stats"]

        modes = out["modes"]
        for l in range(4):
            if l in modes:
                n = torch.norm(modes[l], dim=-1).mean(dim=-1)
                mode_norms[l].append(n.cpu())

        if "mode_mask" in out:
            all_masks.append(out["mode_mask"].cpu())
        if "ks" in out:
            all_ks.append(out["ks"].cpu())

    preds = torch.cat(all_preds, dim=0)  # [N, 1, 3]
    targets = torch.cat(all_targets, dim=0)

    results = {"predictions": preds, "targets": targets}
    for l in range(4):
        if mode_norms[l]:
            results[f"l{l}_norms"] = torch.cat(mode_norms[l], dim=0)
    if all_masks:
        results["mode_mask"] = torch.cat(all_masks, dim=0)
    if all_ks:
        results["ks"] = torch.cat(all_ks, dim=0)
    results["gate_stats"] = gate_stats_all

    # Per-molecule errors
    diff = preds.reshape(-1, 3) - targets.reshape(-1, 3)
    results["per_mol_mse"] = (diff ** 2).mean(dim=1)

    return results


def compute_k_eff(activity, mode_mask=None):
    B, K = activity.shape
    eps = 1e-10
    if mode_mask is not None:
        activity = activity.clone()
        activity[~mode_mask] = -float("inf")
    k_bank = mode_mask.sum(dim=1).float() if mode_mask is not None else torch.full((B,), K, dtype=torch.float32)

    probs = F.softmax(activity, dim=1)
    entropy = -(probs * (probs + eps).log()).sum(dim=1)
    k_entropy = entropy.exp()
    pr = 1.0 / ((probs ** 2).sum(dim=1) + eps)
    sorted_probs, _ = probs.sort(dim=1, descending=True)
    cumsum = sorted_probs.cumsum(dim=1)
    k_80 = (cumsum < 0.80).sum(dim=1).float() + 1
    k_90 = (cumsum < 0.90).sum(dim=1).float() + 1
    k_95 = (cumsum < 0.95).sum(dim=1).float() + 1

    # Gini
    sorted_vals, _ = activity.clamp(min=0).sort(dim=1)
    n = sorted_vals.shape[1]
    rank = torch.arange(1, n + 1, dtype=torch.float32, device=activity.device)
    gini = (2 * (rank * sorted_vals).sum(dim=1) - (n + 1) * sorted_vals.sum(dim=1)) / (n * sorted_vals.sum(dim=1) + eps)

    top1 = sorted_probs[:, 0]
    top3 = sorted_probs[:, :3].sum(dim=1)
    top5 = sorted_probs[:, :5].sum(dim=1)
    n_1pct = (probs > 0.01).sum(dim=1).float()
    n_2pct = (probs > 0.02).sum(dim=1).float()
    n_5pct = (probs > 0.05).sum(dim=1).float()
    dead_frac = (probs < 0.001).sum(dim=1).float() / K

    stats = {
        "k_entropy_mean": float(k_entropy.mean()), "k_entropy_std": float(k_entropy.std()),
        "k_pr_mean": float(pr.mean()), "k_pr_std": float(pr.std()),
        "k_80_mean": float(k_80.mean()), "k_80_std": float(k_80.std()),
        "k_90_mean": float(k_90.mean()), "k_90_std": float(k_90.std()),
        "k_95_mean": float(k_95.mean()), "k_95_std": float(k_95.std()),
        "gini_mean": float(gini.mean()), "gini_std": float(gini.std()),
        "top1_share_mean": float(top1.mean()), "top3_share_mean": float(top3.mean()),
        "top5_share_mean": float(top5.mean()), "active_1pct_mean": float(n_1pct.mean()),
        "active_2pct_mean": float(n_2pct.mean()), "active_5pct_mean": float(n_5pct.mean()),
        "dead_frac_mean": float(dead_frac.mean()), "k_bank_mean": float(k_bank.mean()),
        "k_bank_std": float(k_bank.std()),
        "k_entropy_over_kbank_mean": float((k_entropy / k_bank.clamp(min=1)).mean()),
        "k_entropy_over_kbank_std": float((k_entropy / k_bank.clamp(min=1)).std()),
        "k_pr_over_kbank_mean": float((pr / k_bank.clamp(min=1)).mean()),
        "k_pr_over_kbank_std": float((pr / k_bank.clamp(min=1)).std()),
    }
    return stats, k_entropy, pr, k_bank, probs


def compute_top_r_masking(model, test_data, device, r_values=None, max_mols=64):
    if r_values is None:
        r_values = [1, 2, 4, 8, 16, 32]
    mols = test_data[:min(max_mols, len(test_data))]
    agg = {r: {"retention": [], "mse": [], "vec_mae": [], "delta_norm": []} for r in r_values}

    for mol in mols:
        bdata = collate_batch([mol])
        z = bdata["z"].to(device); pos = bdata["pos"].to(device)
        b = bdata["batch"].to(device); target = bdata["dipole"].to(device)
        model.eval()
        with torch.no_grad():
            full_out = model(z=z, pos=pos, batch=b, return_modes=True)
            full_pred = full_out["vector"]
            modes = full_out["modes"]
            mode_mask = full_out.get("mode_mask", None)
            if 1 in modes:
                activity = torch.norm(modes[1], dim=-1).mean(dim=-1)
            else:
                activity = torch.norm(modes[0], dim=-1).mean(dim=-1)

        full_err = F.mse_loss(full_pred.reshape(-1, 3), target.reshape(-1, 3)).item()
        total_act = activity.sum().item()
        active_activity = activity.clone()
        if mode_mask is not None:
            active_activity[~mode_mask] = -float("inf")
        _, top_indices = active_activity.topk(activity.shape[1], dim=1)

        for r in r_values:
            keep = set(top_indices[0, :r].tolist())
            # For top-r, create a new mode_mask that only has those modes active
            # But also respect the original mask
            keep_mask = mode_mask.clone() if mode_mask is not None else torch.ones(1, activity.shape[1], dtype=torch.bool)
            keep_mask[0, :] = False
            for idx in keep:
                keep_mask[0, idx] = True

            with torch.no_grad():
                mr_out = model(z=z, pos=pos, batch=b, mode_mask=keep_mask)
                mr_pred = mr_out["vector"]

            mr_err = F.mse_loss(mr_pred.reshape(-1, 3), target.reshape(-1, 3)).item()
            delta = torch.norm(mr_pred - full_pred).item()
            vmae = (mr_pred - target).abs().mean().item()
            retention = sum(activity[0, list(keep)].item() for _ in [1]) if r > 0 else 0
            retention = activity[0, list(keep)].sum().item() / (total_act + 1e-10)

            agg[r]["retention"].append(retention)
            agg[r]["mse"].append(mr_err)
            agg[r]["vec_mae"].append(vmae)
            agg[r]["delta_norm"].append(delta)

    return [{**{"r": r}, **{k: float(np.mean(v)) for k, v in vals.items()},
             **{f"{k}_std": float(np.std(v)) for k, v in vals.items()}}
            for r, vals in agg.items()]


def compute_lomo(model, test_data, device, max_mols=128):
    mols = test_data[:min(max_mols, len(test_data))]
    all_deltas = {}  # k -> list of deltas
    for mol in mols:
        bdata = collate_batch([mol])
        z = bdata["z"].to(device); pos = bdata["pos"].to(device)
        b = bdata["batch"].to(device); target = bdata["dipole"].to(device)
        model.eval()
        with torch.no_grad():
            full_out = model(z=z, pos=pos, batch=b)
            full_pred = full_out["vector"]
            mode_mask = full_out.get("mode_mask", None)
        Kmax = mode_mask.shape[1] if mode_mask is not None else model.config.num_modes
        K_active = int(mode_mask[0].sum()) if mode_mask is not None else Kmax
        K_limit = min(K_active, 32)  # cap LOMO at 32 modes

        for k in range(K_limit):
            lomo_mask = mode_mask.clone() if mode_mask is not None else torch.ones(1, Kmax, dtype=torch.bool)
            lomo_mask[0, k] = False
            with torch.no_grad():
                lo_out = model(z=z, pos=pos, batch=b, mode_mask=lomo_mask)
                lo_pred = lo_out["vector"]
            delta = torch.norm(lo_pred - full_pred).item()
            all_deltas.setdefault(k, []).append(delta)

    return [{"k": k, "mean_delta": float(np.mean(v)), "std_delta": float(np.std(v))}
            for k, v in sorted(all_deltas.items())]


def compute_metrics_batch(preds, targets):
    """Compute vector metrics for precomputed predictions."""
    diff = preds - targets
    vec_mae = diff.abs().mean().item()
    pn = torch.norm(preds, dim=-1); tn = torch.norm(targets, dim=-1)
    norm_mae = (pn - tn).abs().mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()
    ss_res = (diff ** 2).sum(); ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else float("nan")
    cos = F.cosine_similarity(preds, targets, dim=-1).clamp(-1, 1)
    ang = torch.acos(cos).mean().item() * 180.0 / np.pi
    return {"vec_mae": vec_mae, "norm_mae": norm_mae, "rmse": rmse, "r2": r2,
            "ang_err_deg": ang, "mse": (diff ** 2).mean().item()}


def pearson_r(x, y):
    x, y = x.detach().float(), y.detach().float()
    mx, my = x.mean(), y.mean()
    num = ((x - mx) * (y - my)).sum()
    den = torch.sqrt(((x - mx)**2).sum() * ((y - my)**2).sum())
    return float((num / (den + 1e-10)).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--compare", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output-prefix", default="phase3_3_valence_half")
    p.add_argument("--skip-topr", action="store_true")
    p.add_argument("--skip-lomo", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures/phase3_3_valence_half_mu", exist_ok=True)

    all_runs = [(args.run_dir, "valence_half")]
    if args.compare:
        all_runs.append((args.compare, "fixed_k8"))

    run_results = {}

    for run_dir, label in all_runs:
        print(f"\n{'='*60}")
        print(f"Analyzing: {label} ({run_dir})")
        print(f"{'='*60}")

        config, metrics, ckpt_path = load_run(run_dir)
        print(f"Checkpoint: {ckpt_path}")
        print(f"k_policy={config.get('k_policy','fixed')}, k_max={config.get('k_max','N/A')}, num_modes={config.get('num_modes','N/A')}")

        # Load model
        detanet = make_latent_detanet(num_features=config.get("num_features", 128),
                                       maxl=config.get("maxl", 3),
                                       num_block=config.get("num_block", 3),
                                       device=str(device))
        model = make_mto_net(detanet_model=detanet, **config)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_params:,} params")

        # Load test data (always from subset_medium)
        test_data = load_dataset(run_dir)
        print(f"Test molecules: {len(test_data)}")
        mol_meta = get_atom_metadata(test_data)

        # Full forward pass
        print("Computing full test pass...")
        results = compute_all_metrics(model, test_data, device, batch_size=args.batch_size)

        # Prediction metrics
        preds = results["predictions"].reshape(-1, 3)
        targets = results["targets"].reshape(-1, 3)
        pred_metrics = compute_metrics_batch(preds, targets)
        print(f"Test metrics: vec_mae={pred_metrics['vec_mae']:.4f}, rmse={pred_metrics['rmse']:.4f}, r2={pred_metrics['r2']:.4f}")

        # K_eff from l=1 norms
        l1_norms = results.get("l1_norms")
        mode_mask = results.get("mode_mask")
        ks = results.get("ks")
        gate_stats = results.get("gate_stats", {})

        k_eff_data = {}
        if l1_norms is not None:
            k_eff_data, k_entropy, k_pr, k_bank, probs = compute_k_eff(l1_norms, mode_mask)
            print(f"K_bank: {k_eff_data['k_bank_mean']:.1f} ± {k_eff_data['k_bank_std']:.1f}")
            print(f"K_entropy: {k_eff_data['k_entropy_mean']:.2f} ± {k_eff_data['k_entropy_std']:.2f}")
            print(f"K_PR: {k_eff_data['k_pr_mean']:.2f} ± {k_eff_data['k_pr_std']:.2f}")
            print(f"K_entropy/K_bank: {k_eff_data['k_entropy_over_kbank_mean']:.4f}")
            print(f"K_PR/K_bank: {k_eff_data['k_pr_over_kbank_mean']:.4f}")
            print(f"Gini: {k_eff_data['gini_mean']:.4f}")
            print(f"Top-1 share: {k_eff_data['top1_share_mean']:.4f}")
            print(f"Top-3 share: {k_eff_data['top3_share_mean']:.4f}")
            print(f"Dead frac: {k_eff_data['dead_frac_mean']:.4f}")

        # Gate stats
        print(f"Gate stats:")
        for k, v in sorted(gate_stats.items()):
            print(f"  {k}: {v}")

        # Save summary CSV
        summary = {"label": label, "n_params": n_params}
        summary.update(pred_metrics)
        if k_eff_data:
            summary.update(k_eff_data)
        for k, v in gate_stats.items():
            summary[k] = v
        # Training final metrics
        if metrics:
            last = metrics[-1]
            summary["train_loss"] = last.get("train_loss", "")
            summary["val_loss"] = last.get("val_loss", "")

        csv_path = f"outputs/tables/{args.output_prefix}_mu_summary_{label}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in summary.items():
                w.writerow([k, v])
        print(f"Wrote {csv_path}")

        # Top-r masking
        if not args.skip_topr:
            print("Computing top-r masking...")
            top_r = compute_top_r_masking(model, test_data, device)
            top_r_path = f"outputs/tables/{args.output_prefix}_top_r_masking_{label}.csv"
            with open(top_r_path, "w", newline="") as f:
                keys = ["r", "retention", "mse", "vec_mae", "delta_norm",
                        "retention_std", "mse_std", "vec_mae_std", "delta_norm_std"]
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for row in top_r:
                    w.writerow(row)
            print(f"Wrote {top_r_path}")

        # LOMO
        if not args.skip_lomo:
            print("Computing leave-one-mode-out...")
            lomo = compute_lomo(model, test_data, device)
            lomo_path = f"outputs/tables/{args.output_prefix}_mode_importance_{label}.csv"
            with open(lomo_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["k", "mean_delta_norm", "std_delta_norm"])
                for r in lomo:
                    w.writerow([r["k"], r["mean_delta"], r["std_delta"]])
            print(f"Wrote {lomo_path}")

            # K_lomo stats
            if lomo:
                deltas = torch.tensor([x["mean_delta"] for x in lomo])
                probs_lomo = F.softmax(deltas, dim=0)
                entropy_lomo = float((-probs_lomo * (probs_lomo + 1e-10).log()).sum().exp())
                pr_lomo = float(1.0 / ((probs_lomo ** 2).sum() + 1e-10))
                print(f"K_lomo_entropy: {entropy_lomo:.2f}, K_lomo_PR: {pr_lomo:.2f}")
                summary["k_lomo_entropy"] = entropy_lomo
                summary["k_lomo_pr"] = pr_lomo

        # Correlations
        if l1_norms is not None:
            n_atoms = torch.tensor([m["n_atoms"] for m in mol_meta], dtype=torch.float32)
            n_vals = torch.tensor([m["n_val"] for m in mol_meta], dtype=torch.float32)
            dipole_mags = torch.tensor([m["dipole_mag"] for m in mol_meta], dtype=torch.float32)
            per_mol_mse = results["per_mol_mse"]
            corrs = {
                "k_entropy_vs_n_atoms": pearson_r(k_entropy, n_atoms),
                "k_entropy_vs_n_val": pearson_r(k_entropy, n_vals),
                "k_entropy_vs_dipole_mag": pearson_r(k_entropy, dipole_mags),
                "k_entropy_vs_mse": pearson_r(k_entropy, per_mol_mse),
                "k_pr_vs_n_atoms": pearson_r(k_pr, n_atoms),
                "k_pr_vs_n_val": pearson_r(k_pr, n_vals),
                "k_pr_vs_dipole_mag": pearson_r(k_pr, dipole_mags),
                "k_pr_vs_mse": pearson_r(k_pr, per_mol_mse),
                "k_eff_over_kbank_vs_n_val": pearson_r(k_entropy / k_bank.clamp(min=1), n_vals),
                "k_eff_over_kbank_vs_dipole_mag": pearson_r(k_entropy / k_bank.clamp(min=1), dipole_mags),
                "k_eff_over_kbank_vs_mse": pearson_r(k_entropy / k_bank.clamp(min=1), per_mol_mse),
                "top1_share_vs_n_atoms": pearson_r(probs[:, 0], n_atoms),
                "top1_share_vs_dipole_mag": pearson_r(probs[:, 0], dipole_mags),
                "gini_vs_n_atoms": pearson_r(torch.tensor([k_eff_data["gini_mean"]]).expand(len(n_atoms)), n_atoms),  # FIXME: per-mol gini
            }
            print("Correlations:")
            for k, v in corrs.items():
                print(f"  {k}: {v:.4f}")
            for k, v in corrs.items():
                summary[k] = v

        # Re-save summary with all data
        csv_path = f"outputs/tables/{args.output_prefix}_mu_summary_{label}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in summary.items():
                w.writerow([k, v])
        print(f"Final summary written to {csv_path}")

        # K_half distribution (only for valence_half)
        if ks is not None:
            ks_int = ks.long()
            k_dist = torch.bincount(ks_int, minlength=int(ks_int.max()) + 1)
            kdist_path = f"outputs/tables/{args.output_prefix}_k_bank_distribution.csv"
            with open(kdist_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["k", "count"])
                for k in range(int(ks_int.min()), min(int(ks_int.max()) + 1, 50)):
                    if k_dist[k] > 0:
                        w.writerow([k, int(k_dist[k])])
            print(f"Wrote {kdist_path}")

        # Order analysis (norms per l)
        order_path = f"outputs/tables/{args.output_prefix}_order_masking_{label}.csv"
        with open(order_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["order", "mean_norm"])
            for l in range(4):
                key = f"l{l}_norms"
                if key in results:
                    w.writerow([f"l{l}", float(results[key].mean())])
        print(f"Wrote {order_path}")

        run_results[label] = {"summary": summary, "metrics": metrics, "config": config}

    # Cross-comparison
    if len(all_runs) == 2:
        print(f"\n{'='*60}")
        print("CROSS-COMPARISON: valence_half vs fixed_K=8")
        print(f"{'='*60}")
        s_vh = run_results["valence_half"]["summary"]
        s_fk = run_results["fixed_k8"]["summary"]
        compare_keys = [
            "vec_mae", "norm_mae", "rmse", "r2", "ang_err_deg", "mse",
            "k_bank_mean", "k_entropy_mean", "k_pr_mean",
            "k_entropy_over_kbank_mean", "k_pr_over_kbank_mean",
            "gini_mean", "top1_share_mean", "top3_share_mean", "dead_frac_mean",
            "k_80_mean", "k_90_mean", "val_loss",
        ]
        comp_path = f"outputs/tables/{args.output_prefix}_comparison.csv"
        with open(comp_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "fixed_k8", "valence_half", "ratio_vh_fk"])
            for key in compare_keys:
                fk_v = s_fk.get(key, float("nan"))
                vh_v = s_vh.get(key, float("nan"))
                ratio = ""
                if isinstance(fk_v, (int, float)) and isinstance(vh_v, (int, float)) and abs(fk_v) > 1e-10:
                    ratio = f"{vh_v / fk_v:.4f}"
                w.writerow([key, fk_v, vh_v, ratio])
        print(f"Wrote {comp_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired SBF loss ablation on synthetic or saved ADNI coefficients.

This diagnostic deliberately bypasses dictionary learning.  It imports the
frozen SBF model/loss implementation from the adaptive-ridge run and varies
only two factors: learned uncertainty weights and the prediction-weight
penalty.  No held-out test target or test coefficient is accessed.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


CONDITIONS = {
    "current": {"auto_weight": [1, 1, 1, 1], "lambdas": [0.5, 0.5, 0.5, 0.5]},
    "freeze_auto": {"auto_weight": [0, 0, 0, 0], "lambdas": [0.5, 0.5, 0.5, 0.5]},
    "no_pred": {"auto_weight": [1, 1, 1, 0], "lambdas": [0.5, 0.5, 0.5, 0.0]},
    "freeze_auto_no_pred": {"auto_weight": [0, 0, 0, 0], "lambdas": [0.5, 0.5, 0.5, 0.0]},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, obj) -> None:
    with path.open("w") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_frozen_module(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_sbf_utils", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def safe_corr(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def prediction_metrics(y, pred) -> dict:
    y = np.asarray(y, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    keep = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[keep], pred[keep]
    residual = y - pred
    sse = float(np.dot(residual, residual))
    centered = y - np.mean(y)
    sst = float(np.dot(centered, centered))
    pred_sd = float(np.std(pred, ddof=0))
    target_sd = float(np.std(y, ddof=0))
    if pred_sd > 0:
        slope, intercept = np.polyfit(pred, y, 1)
    else:
        slope, intercept = float("nan"), float("nan")
    return {
        "n": int(y.size),
        "r": safe_corr(y, pred),
        "r2_conventional": float(1.0 - sse / sst) if sst > 0 else float("nan"),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "target_mean": float(np.mean(y)),
        "target_sd": target_sd,
        "prediction_mean": float(np.mean(pred)),
        "prediction_sd": pred_sd,
        "prediction_sd_ratio": float(pred_sd / target_sd) if target_sd > 0 else float("nan"),
        "prediction_min": float(np.min(pred)),
        "prediction_max": float(np.max(pred)),
        "calibration_slope_y_on_prediction": float(slope),
        "calibration_intercept_y_on_prediction": float(intercept),
    }


def metrics_gate(mode: str, train: dict, validation: dict) -> bool:
    finite = all(math.isfinite(float(validation[k])) for k in (
        "r", "r2_conventional", "rmse", "prediction_sd_ratio",
        "calibration_slope_y_on_prediction"))
    if not finite:
        return False
    if mode == "synthetic":
        return bool(
            validation["r"] >= 0.70
            and validation["r2_conventional"] >= 0.40
            and 0.50 <= validation["prediction_sd_ratio"] <= 1.50
            and 0.50 <= validation["calibration_slope_y_on_prediction"] <= 1.50
            and train["r2_conventional"] >= 0.40
        )
    return bool(
        validation["r"] >= 0.30
        and validation["r2_conventional"] > 0.0
        and 0.25 <= validation["prediction_sd_ratio"] <= 2.0
        and 0.25 <= validation["calibration_slope_y_on_prediction"] <= 2.0
        and train["r2_conventional"] > 0.0
    )


def synthetic_data(seed: int, n_train=192, n_validation=41,
                   nmod=5, nfea=250, shared_dim=50):
    """Generate a strongly identifiable distributed multimodal target."""
    rng = np.random.default_rng(seed)
    n = n_train + n_validation
    z = rng.normal(size=(n, shared_dim))
    x = []
    for modality in range(nmod):
        mixing = rng.normal(size=(shared_dim, nfea)) / np.sqrt(shared_dim)
        private = rng.normal(scale=0.30 + 0.05 * modality, size=(n, nfea))
        matrix = z @ mixing + private
        mean = matrix[:n_train].mean(axis=0)
        sd = matrix[:n_train].std(axis=0)
        sd[sd == 0] = 1.0
        x.append(((matrix - mean) / sd).astype(np.float32))
    beta = np.zeros(shared_dim)
    beta[:10] = np.array([1.2, -1.0, 0.9, -0.8, 0.7,
                          -0.6, 0.5, -0.4, 0.3, -0.2])
    signal = z @ beta
    y = signal + rng.normal(scale=0.35 * np.std(signal), size=n)
    y = 1.5 + 1.5 * (y - y[:n_train].mean()) / y[:n_train].std()
    return ([m[:n_train] for m in x], [m[n_train:] for m in x],
            y[:n_train, None].astype(np.float32),
            y[n_train:, None].astype(np.float32),
            {"generator": "shared_latent_linear", "shared_dim": shared_dim,
             "target_nonzero_latent_dimensions": 10,
             "noise_sd_fraction_of_signal_sd": 0.35})


def load_adni_data(root: Path, target_train: Path, target_validation: Path):
    x_train, x_validation, artifacts = [], [], []
    for modality in range(5):
        coeff_path = root / "coefficient_diagnostics" / f"modality_{modality}_raw_coefficients.npz"
        norm_path = root / "dictionary_artifacts" / f"modality_{modality}_coefficient_normalization.npz"
        with np.load(coeff_path) as coeff:
            # Deliberately request only training and validation keys.  The
            # held-out test member in the archive is never read.
            raw_train = np.asarray(coeff["train"], dtype=np.float64)
            raw_validation = np.asarray(coeff["validation"], dtype=np.float64)
        with np.load(norm_path) as norm:
            mean = np.asarray(norm["mean"], dtype=np.float64)
            sd = np.asarray(norm["sd"], dtype=np.float64)
            active = np.asarray(norm["active"], dtype=bool)
        if raw_train.shape != (192, 250) or raw_validation.shape != (41, 250):
            raise ValueError(f"Unexpected modality {modality} shapes: {raw_train.shape}, {raw_validation.shape}")
        if not np.allclose(raw_train.mean(axis=0), mean, atol=1e-7, rtol=1e-5):
            raise ValueError(f"Stored normalization mean mismatch for modality {modality}")
        divisor = np.where(active, sd, 1.0)
        train_z = (raw_train - mean) / divisor
        validation_z = (raw_validation - mean) / divisor
        train_z[:, ~active] = 0.0
        validation_z[:, ~active] = 0.0
        if not np.isfinite(train_z).all() or not np.isfinite(validation_z).all():
            raise ValueError(f"Nonfinite standardized coefficients for modality {modality}")
        x_train.append(train_z.astype(np.float32))
        x_validation.append(validation_z.astype(np.float32))
        artifacts.append({
            "modality_index": modality,
            "coefficient_path": str(coeff_path),
            "coefficient_sha256": sha256(coeff_path),
            "normalization_path": str(norm_path),
            "normalization_sha256": sha256(norm_path),
            "active_directions": int(active.sum()),
            "validation_standardized_abs_max": float(np.max(np.abs(validation_z))),
        })
    train_targets_full = np.genfromtxt(target_train, delimiter=",", dtype=np.float64)
    validation_targets_full = np.genfromtxt(target_validation, delimiter=",", dtype=np.float64)
    y_train = np.atleast_2d(train_targets_full)[:, [0]].astype(np.float32)
    y_validation = np.atleast_2d(validation_targets_full)[:, [0]].astype(np.float32)
    if y_train.shape != (192, 1) or y_validation.shape != (41, 1):
        raise ValueError(f"Unexpected target shapes {y_train.shape}, {y_validation.shape}")
    if not np.isfinite(y_train).all() or not np.isfinite(y_validation).all():
        raise ValueError("Nonfinite CDR-SB targets")
    provenance = {
        "source_root": str(root),
        "target_train": str(target_train),
        "target_train_sha256": sha256(target_train),
        "target_validation": str(target_validation),
        "target_validation_sha256": sha256(target_validation),
        "held_out_test_accessed": False,
        "artifacts": artifacts,
    }
    return x_train, x_validation, y_train, y_validation, provenance


def grad_norm(parameter) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().norm().cpu().item())


def evaluate(model, x, y_z, y_mean, y_sd, device):
    model.eval()
    tensors = [torch.as_tensor(m, dtype=torch.float32, device=device) for m in x]
    with torch.no_grad():
        _, _, latent, pred_z, _ = model(tensors, device)
    pred = pred_z.cpu().numpy() * y_sd + y_mean
    y = y_z * y_sd + y_mean
    return prediction_metrics(y, pred), pred, latent.cpu().numpy()


def spatial_collinearity(model) -> dict:
    summaries = []
    mod_weight = torch.softmax(model.mod_weight.detach().cpu(), dim=1).numpy()
    for modality, loading in enumerate(model.spatial_loading):
        effective = loading.detach().cpu().numpy() * mod_weight[:, modality][None, :]
        corr = np.corrcoef(effective.T)
        values = np.abs(corr[np.triu_indices_from(corr, 1)])
        summaries.append({
            "modality_index": modality,
            "median_abs_pairwise_correlation": float(np.nanmedian(values)),
            "p95_abs_pairwise_correlation": float(np.nanpercentile(values, 95)),
            "fraction_abs_correlation_gt_0_95": float(np.nanmean(values > 0.95)),
        })
    return {"per_modality": summaries}


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    condition = CONDITIONS[args.condition]
    frozen_path = Path(args.frozen_utils).resolve()
    sbf = load_frozen_module(frozen_path)

    if args.mode == "synthetic":
        x_train, x_validation, y_train, y_validation, data_provenance = synthetic_data(args.seed)
    else:
        x_train, x_validation, y_train, y_validation, data_provenance = load_adni_data(
            Path(args.adni_root), Path(args.target_train), Path(args.target_validation))

    nlat = 50
    y_mean = y_train.mean(axis=0)
    y_sd = y_train.std(axis=0)
    y_sd[y_sd == 0] = 1.0
    y_train_z = ((y_train - y_mean) / y_sd).astype(np.float32)
    y_validation_z = ((y_validation - y_mean) / y_sd).astype(np.float32)

    configuration = {
        "status": "running",
        "mode": args.mode,
        "condition": args.condition,
        "factor_learned_uncertainty_weights": args.condition in ("current", "no_pred"),
        "factor_prediction_weight_penalty": args.condition in ("current", "freeze_auto"),
        "auto_weight": condition["auto_weight"],
        "lambdas": condition["lambdas"],
        "seed": args.seed,
        "nlat": nlat,
        "n_features_per_modality": [int(m.shape[1]) for m in x_train],
        "n_train": int(len(y_train)),
        "n_validation": int(len(y_validation)),
        "max_epoch": args.max_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "dropout": args.dropout,
        "uncertainty_optimizer_starts_epoch": 10,
        "checkpoint_rule": "minimum validation RMSE among epochs satisfying prespecified gate",
        "gate": ({
            "validation_r_min": 0.70, "validation_r2_min": 0.40,
            "validation_prediction_sd_ratio": [0.50, 1.50],
            "validation_calibration_slope": [0.50, 1.50],
            "training_r2_min": 0.40,
        } if args.mode == "synthetic" else {
            "validation_r_min": 0.30, "validation_r2_min_exclusive": 0.0,
            "validation_prediction_sd_ratio": [0.25, 2.0],
            "validation_calibration_slope": [0.25, 2.0],
            "training_r2_min_exclusive": 0.0,
        }),
        "frozen_utils": str(frozen_path),
        "frozen_utils_sha256": sha256(frozen_path),
        "data_provenance": data_provenance,
        "held_out_test_accessed": False,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    json_write(output / "configuration.json", configuration)

    set_seed(args.seed)
    device = torch.device(args.device)
    model = sbf.SupervisedFLICAmodel(
        [m.shape[1] for m in x_train], nlat, 1,
        dropout=args.dropout, device=device).to(device)
    loss_fun = sbf.loss_SuperBigFLICA_regression(
        1, len(x_train), len(y_train), device=device,
        auto_weight=condition["auto_weight"], lambdas=condition["lambdas"]).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer_loss = torch.optim.Adam(loss_fun.parameters(), lr=args.lr)
    train_loader = DataLoader(
        sbf.load_Multimodal_data(x_train, y_train_z),
        batch_size=args.batch_size, shuffle=True)
    model_steps = max(1, (args.max_epoch + 1) * len(train_loader))
    loss_steps = max(1, max(0, args.max_epoch - 9) * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=model_steps)
    scheduler_loss = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_loss, T_max=loss_steps)

    history = []
    selected = None
    for epoch in range(args.max_epoch + 1):
        torch.manual_seed(epoch)
        torch.cuda.manual_seed_all(epoch)
        model.train()
        accum = {k: 0.0 for k in ("total", "reconstruction", "spatial", "mse", "prediction_penalty", "prediction_weight_grad_norm", "sigma_grad_norm")}
        batches = 0
        for data_batch, labels in train_loader:
            data_batch = [m.to(device) for m in data_batch]
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            if epoch >= 10:
                optimizer_loss.zero_grad(set_to_none=True)
            recon, spatial, latent, pred, pred_weight = model(data_batch, device)
            losses = loss_fun(recon, data_batch, spatial, pred, labels, pred_weight, latent)
            losses[0].backward()
            accum["prediction_weight_grad_norm"] += grad_norm(model.weight_pred)
            accum["sigma_grad_norm"] += sum(grad_norm(p) for p in loss_fun.parameters())
            optimizer.step()
            scheduler.step()
            if epoch >= 10:
                optimizer_loss.step()
                scheduler_loss.step()
            for name, value in zip(("total", "reconstruction", "spatial", "mse", "prediction_penalty"), losses):
                accum[name] += float(value.detach().cpu().item())
            batches += 1

        train_metrics, pred_train, latent_train = evaluate(
            model, x_train, y_train_z, y_mean, y_sd, device)
        validation_metrics, pred_validation, latent_validation = evaluate(
            model, x_validation, y_validation_z, y_mean, y_sd, device)
        gate = metrics_gate(args.mode, train_metrics, validation_metrics)
        sigmas = {
            name: float(getattr(loss_fun, name).detach().cpu().ravel()[0])
            for name in ("sigma1", "sigma2", "sigma3", "sigma4", "sigma5")
        }
        row = {
            "epoch": epoch,
            "gate_pass": gate,
            **{f"loss_{k}": v / batches for k, v in accum.items()},
            **{f"raw_{k}": v for k, v in sigmas.items()},
            "prediction_weight_l1": float(model.weight_pred.detach().abs().sum().cpu().item()),
            "prediction_weight_l2": float(model.weight_pred.detach().norm().cpu().item()),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"validation_{k}": v for k, v in validation_metrics.items()},
        }
        history.append(row)
        if gate and (selected is None or validation_metrics["rmse"] < selected["validation_metrics"]["rmse"]):
            selected = {
                "epoch": epoch,
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "state_dict": copy.deepcopy(model.state_dict()),
            }
        print(json.dumps({
            "epoch": epoch, "condition": args.condition, "gate": gate,
            "train_r2": train_metrics["r2_conventional"],
            "validation_r": validation_metrics["r"],
            "validation_r2": validation_metrics["r2_conventional"],
            "validation_sd_ratio": validation_metrics["prediction_sd_ratio"],
            "validation_calibration_slope": validation_metrics["calibration_slope_y_on_prediction"],
            "raw_sigma3": sigmas["sigma3"], "raw_sigma4": sigmas["sigma4"],
            "raw_sigma5": sigmas["sigma5"],
        }), flush=True)

    with (output / "epoch_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    final = history[-1]
    summary = {
        "status": "success",
        "mode": args.mode,
        "condition": args.condition,
        "passes_prespecified_gate": selected is not None,
        "selected_epoch": None if selected is None else selected["epoch"],
        "selected_train_metrics": None if selected is None else selected["train_metrics"],
        "selected_validation_metrics": None if selected is None else selected["validation_metrics"],
        "final_train_metrics": {k.removeprefix("train_"): v for k, v in final.items() if k.startswith("train_")},
        "final_validation_metrics": {k.removeprefix("validation_"): v for k, v in final.items() if k.startswith("validation_")},
        "final_spatial_collinearity": spatial_collinearity(model),
        "held_out_test_accessed": False,
    }
    if selected is not None:
        torch.save(selected["state_dict"], output / "selected_model_state.pth")
    json_write(output / "summary.json", summary)
    configuration["status"] = "success"
    json_write(output / "configuration.json", configuration)
    (output / "_SUCCESS").write_text("success\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("synthetic", "adni"), required=True)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frozen-utils", required=True)
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--max-epoch", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--adni-root")
    parser.add_argument("--target-train")
    parser.add_argument("--target-validation")
    args = parser.parse_args()
    if args.mode == "adni" and not all((args.adni_root, args.target_train, args.target_validation)):
        parser.error("ADNI mode requires --adni-root, --target-train, and --target-validation")
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

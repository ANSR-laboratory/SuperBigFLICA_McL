#!/usr/bin/env python3
"""Training-only nested schedule selection for ten saved ADNI SBF rounds.

This diagnostic is conditional on the saved adaptive-ridge dictionaries and
coefficient normalization. It never requests the held-out-test NPZ member or
test targets. Three rank-balanced folds within the 192 training subjects select
the neural epoch; the 41-subject validation set is evaluated once afterward.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import run_loss_ablation as base
import sbf_utils as sbf


def load_training_only(root, target_train):
    x_train, artifacts = [], []
    for modality in range(5):
        coeff_path = root / "coefficient_diagnostics" / f"modality_{modality}_raw_coefficients.npz"
        norm_path = root / "dictionary_artifacts" / f"modality_{modality}_coefficient_normalization.npz"
        with np.load(coeff_path) as coefficient_archive:
            raw = np.asarray(coefficient_archive["train"], dtype=np.float64)
        with np.load(norm_path) as normalization:
            mean = np.asarray(normalization["mean"], dtype=np.float64)
            sd = np.asarray(normalization["sd"], dtype=np.float64)
            active = np.asarray(normalization["active"], dtype=bool)
        if raw.shape != (192, 250):
            raise ValueError(f"Unexpected training shape for modality {modality}: {raw.shape}")
        divisor = np.where(active, sd, 1.0)
        standardized = (raw - mean) / divisor
        standardized[:, ~active] = 0.0
        if not np.isfinite(standardized).all():
            raise ValueError(f"Nonfinite training coefficients for modality {modality}")
        x_train.append(standardized.astype(np.float32))
        artifacts.append({
            "modality_index": modality,
            "coefficient_path": str(coeff_path),
            "coefficient_sha256": base.sha256(coeff_path),
            "normalization_path": str(norm_path),
            "normalization_sha256": base.sha256(norm_path),
            "active_directions": int(active.sum()),
        })
    target_values = np.genfromtxt(target_train, delimiter=",", dtype=np.float64)
    target = np.atleast_2d(target_values)[:, [0]].astype(np.float32)
    if target.shape != (192, 1) or not np.isfinite(target).all():
        raise ValueError(f"Invalid training target shape/content: {target.shape}")
    provenance = {
        "source_root": str(root),
        "target_train": str(target_train),
        "target_train_sha256": base.sha256(target_train),
        "external_validation_accessed_during_selection": False,
        "held_out_test_accessed": False,
        "artifacts": artifacts,
    }
    return x_train, target, provenance


def load_external_validation_after_selection(root, target_validation):
    x_validation = []
    for modality in range(5):
        coeff_path = root / "coefficient_diagnostics" / f"modality_{modality}_raw_coefficients.npz"
        norm_path = root / "dictionary_artifacts" / f"modality_{modality}_coefficient_normalization.npz"
        with np.load(coeff_path) as coefficient_archive:
            raw = np.asarray(coefficient_archive["validation"], dtype=np.float64)
        with np.load(norm_path) as normalization:
            mean = np.asarray(normalization["mean"], dtype=np.float64)
            sd = np.asarray(normalization["sd"], dtype=np.float64)
            active = np.asarray(normalization["active"], dtype=bool)
        if raw.shape != (41, 250):
            raise ValueError(f"Unexpected validation shape for modality {modality}: {raw.shape}")
        divisor = np.where(active, sd, 1.0)
        standardized = (raw - mean) / divisor
        standardized[:, ~active] = 0.0
        if not np.isfinite(standardized).all():
            raise ValueError(f"Nonfinite validation coefficients for modality {modality}")
        x_validation.append(standardized.astype(np.float32))
    target_values = np.genfromtxt(target_validation, delimiter=",", dtype=np.float64)
    target = np.atleast_2d(target_values)[:, [0]].astype(np.float32)
    if target.shape != (41, 1) or not np.isfinite(target).all():
        raise ValueError(f"Invalid validation target shape/content: {target.shape}")
    return x_validation, target, {
        "target_validation": str(target_validation),
        "target_validation_sha256": base.sha256(target_validation),
        "external_validation_loaded_after_epoch_selection": True,
        "held_out_test_accessed": False,
    }


def reconstruction_r2(original, reconstructed):
    result = []
    for x, xhat in zip(original, reconstructed):
        x = np.asarray(x, dtype=np.float64)
        xhat = np.asarray(xhat, dtype=np.float64)
        sse = float(np.sum((x - xhat) ** 2))
        sst = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
        result.append(float(1.0 - sse / sst))
    return result


def evaluate(model, x, y_z, y_mean, y_sd, device):
    model.eval()
    tensors = [torch.as_tensor(a, dtype=torch.float32, device=device) for a in x]
    with torch.no_grad():
        recon, _, latent, prediction_z, _ = model(tensors, device)
    prediction = prediction_z.cpu().numpy() * y_sd + y_mean
    target = y_z * y_sd + y_mean
    return {
        "prediction": base.prediction_metrics(target, prediction),
        "reconstruction_r2": reconstruction_r2(
            x, [item.cpu().numpy() for item in recon]),
        "latent": latent.cpu().numpy(),
        "predicted_values": prediction,
    }


def spatial_collinearity(model):
    q = torch.softmax(model.mod_weight.detach().cpu(), dim=1).numpy()
    result = []
    for modality, parameter in enumerate(model.spatial_loading):
        maps = parameter.detach().cpu().numpy() * q[:, modality][None, :]
        matrix = np.abs(np.corrcoef(maps.T))
        values = matrix[np.triu_indices_from(matrix, 1)]
        values = values[np.isfinite(values)]
        result.append({
            "modality_index": modality,
            "median_abs_r": float(np.median(values)),
            "p95_abs_r": float(np.quantile(values, 0.95)),
            "fraction_abs_r_gt_0p95": float(np.mean(values > 0.95)),
        })
    return result


def make_rank_balanced_folds(target, nfold, seed):
    """Distribute neighboring target ranks across folds reproducibly."""
    target = np.asarray(target).ravel()
    order = np.argsort(target, kind="stable")
    rng = np.random.default_rng(seed + 41000)
    assignment = np.empty(target.size, dtype=int)
    for start in range(0, target.size, nfold):
        block = order[start:start + nfold]
        labels = rng.permutation(nfold)[:len(block)]
        assignment[block] = labels
    folds = []
    for fold in range(nfold):
        valid = np.flatnonzero(assignment == fold)
        train = np.flatnonzero(assignment != fold)
        folds.append((train, valid))
    return assignment, folds


def build_model(x, y_z, seed, dropout, device):
    base.set_seed(seed)
    init_spatial, init_weight, init_bias, _, diagnostics = (
        sbf.Initialize_SuperBigFLICA(x, y_z, 50))
    model = sbf.SupervisedFLICAmodel(
        [a.shape[1] for a in x], 50, 1, dropout=dropout, device=device,
        init_spatial_loading=init_spatial,
        init_weight_pred=init_weight,
        init_bais_pred=init_bias).to(device)
    return model, diagnostics


def fit_history(x_train, y_train, x_eval, y_eval, seed, args, keep_final):
    y_mean = y_train.mean(axis=0)
    y_sd = y_train.std(axis=0)
    y_sd[y_sd == 0] = 1.0
    y_train_z = ((y_train - y_mean) / y_sd).astype(np.float32)
    y_eval_z = ((y_eval - y_mean) / y_sd).astype(np.float32)
    device = torch.device(args.device)
    model, initialization = build_model(
        x_train, y_train_z, seed, args.dropout, device)
    loss_fun = sbf.loss_SuperBigFLICA_regression(
        1, len(x_train), len(y_train), device=device,
        auto_weight=[0, 0, 0, 0], lambdas=[1, 1, 1, 1]).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9)
    generator = torch.Generator().manual_seed(seed + 900)
    loader = DataLoader(
        sbf.load_Multimodal_data(x_train, y_train_z),
        batch_size=args.batch_size, shuffle=True, generator=generator)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(args.epochs + 1) * len(loader))

    history = []
    for epoch in range(args.epochs + 1):
        torch.manual_seed(seed + epoch)
        model.train()
        losses_sum = np.zeros(5, dtype=float)
        for batch, labels in loader:
            batch = [item.to(device) for item in batch]
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch, device)
            losses = loss_fun(outputs[0], batch, outputs[1], outputs[3],
                              labels, outputs[4], outputs[2])
            losses[0].backward()
            optimizer.step()
            scheduler.step()
            losses_sum += np.asarray([float(x.detach().cpu()) for x in losses])
        training = evaluate(model, x_train, y_train_z, y_mean, y_sd, device)
        evaluation = evaluate(model, x_eval, y_eval_z, y_mean, y_sd, device)
        spatial = spatial_collinearity(model)
        pred = evaluation["prediction"]
        recon_mean = float(np.mean(evaluation["reconstruction_r2"]))
        p95_max = max(item["p95_abs_r"] for item in spatial)
        finite = all(math.isfinite(float(pred[k])) for k in (
            "r", "r2_conventional", "rmse", "prediction_sd_ratio",
            "calibration_slope_y_on_prediction"))
        gate = bool(
            finite and training["prediction"]["r2_conventional"] > 0
            and pred["r2_conventional"] > 0
            and 0.5 <= pred["prediction_sd_ratio"] <= 1.5
            and 0.5 <= pred["calibration_slope_y_on_prediction"] <= 1.5
            and recon_mean > 0 and p95_max < 0.8)
        row = {
            "epoch": epoch,
            "gate": gate,
            "score": float(pred["r2_conventional"] + 0.25 * recon_mean),
            "eval_r": pred["r"],
            "eval_r2": pred["r2_conventional"],
            "eval_rmse": pred["rmse"],
            "eval_prediction_sd_ratio": pred["prediction_sd_ratio"],
            "eval_calibration_slope": pred["calibration_slope_y_on_prediction"],
            "eval_reconstruction_r2_mean": recon_mean,
            "map_p95_abs_r_max": p95_max,
            "train_r2": training["prediction"]["r2_conventional"],
            **{f"loss_{i}": float(v / len(loader))
               for i, v in enumerate(losses_sum)},
        }
        history.append(row)
    return history, model if keep_final else None, loss_fun, initialization


def select_epoch(fold_histories):
    aggregate = []
    for epoch in range(len(fold_histories[0])):
        rows = [history[epoch] for history in fold_histories]
        scores = np.asarray([row["score"] for row in rows], dtype=float)
        aggregate.append({
            "epoch": epoch,
            "mean_score": float(np.mean(scores)),
            "score_se": float(np.std(scores, ddof=1) / np.sqrt(len(scores))),
            "gate_fraction": float(np.mean([row["gate"] for row in rows])),
            "mean_inner_r": float(np.mean([row["eval_r"] for row in rows])),
            "mean_inner_r2": float(np.mean([row["eval_r2"] for row in rows])),
            "mean_inner_reconstruction_r2": float(np.mean(
                [row["eval_reconstruction_r2_mean"] for row in rows])),
            "mean_map_p95_abs_r_max": float(np.mean(
                [row["map_p95_abs_r_max"] for row in rows])),
        })
    eligible = [row for row in aggregate if row["gate_fraction"] >= 2 / 3]
    if not eligible:
        raise RuntimeError("No epoch passed the pre-specified inner-CV gate")
    best = max(eligible, key=lambda row: row["mean_score"])
    cutoff = best["mean_score"] - best["score_se"]
    selected = min((row for row in eligible if row["mean_score"] >= cutoff),
                   key=lambda row: row["epoch"])
    return aggregate, best, selected


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to reuse {output}")
    output.mkdir(parents=True)
    source_root = Path(args.adni_root)
    x_all, y_all, provenance = load_training_only(
        source_root, Path(args.target_train))
    assignment, folds = make_rank_balanced_folds(y_all, args.folds, args.seed)
    config = {
        "status": "running",
        "round": args.round,
        "seed": args.seed,
        "method": "fixed uncertainty + equal loss weights + exact released mMIGP",
        "inner_cv": "three rank-balanced folds; one-standard-error epoch rule",
        "conditional_on_saved_full_training_dictionary": True,
        "dictionary_learning_rerun": False,
        "external_validation_used_for_epoch_selection": False,
        "external_validation_evaluations": 1,
        "held_out_test_coefficients_accessed": False,
        "held_out_test_targets_accessed": False,
        "held_out_test_evaluated": False,
        "epochs_considered": list(range(args.epochs + 1)),
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "fold_assignment": assignment.tolist(),
        "fold_target_summaries": [
            {"fold": fold, "n": int(np.sum(assignment == fold)),
             "mean": float(np.mean(y_all[assignment == fold])),
             "sd": float(np.std(y_all[assignment == fold]))}
            for fold in range(args.folds)],
        "data_provenance": provenance,
        "slurm": {key: os.environ.get(key) for key in (
            "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID")},
    }
    base.json_write(output / "configuration.json", config)

    fold_histories, init_diagnostics = [], []
    for fold, (train_idx, valid_idx) in enumerate(folds):
        x_train = [x[train_idx] for x in x_all]
        x_valid = [x[valid_idx] for x in x_all]
        history, _, _, init = fit_history(
            x_train, y_all[train_idx], x_valid, y_all[valid_idx],
            args.seed + 10000 * (fold + 1), args, keep_final=False)
        fold_histories.append(history)
        init_diagnostics.append(init)
        write_csv(output / f"inner_fold{fold + 1}_epoch_metrics.csv", history)
        print(json.dumps({"round": args.round, "fold": fold + 1,
                          "status": "complete"}), flush=True)

    aggregate, best, selected = select_epoch(fold_histories)
    write_csv(output / "inner_cv_aggregate.csv", aggregate)
    selected_epoch = int(selected["epoch"])
    # This is the first point at which the external validation coefficient key
    # and target file are accessed.
    x_external, y_external, external_provenance = (
        load_external_validation_after_selection(
            source_root, Path(args.target_validation)))
    config["external_validation_provenance"] = external_provenance
    config["external_validation_loaded_after_epoch_selection"] = True
    original_epochs = args.epochs
    args.epochs = selected_epoch
    # Retrain for the frozen epoch count without repeatedly consulting the
    # external validation set. Training data serve as the monitoring set here;
    # the external validation set is evaluated once below.
    final_history, final_model, final_loss, final_init = fit_history(
        x_all, y_all, x_all, y_all, args.seed, args, keep_final=True)
    args.epochs = original_epochs
    assert final_model is not None

    y_mean = y_all.mean(axis=0)
    y_sd = y_all.std(axis=0)
    y_sd[y_sd == 0] = 1.0
    y_all_z = ((y_all - y_mean) / y_sd).astype(np.float32)
    y_external_z = ((y_external - y_mean) / y_sd).astype(np.float32)
    train_eval = evaluate(final_model, x_all, y_all_z, y_mean, y_sd,
                          torch.device(args.device))
    external_eval = evaluate(final_model, x_external, y_external_z, y_mean,
                             y_sd, torch.device(args.device))
    spatial = spatial_collinearity(final_model)
    torch.save(final_model.state_dict(), output / "nested_selected_model_state.pth")
    np.savez_compressed(
        output / "nested_selected_results.npz",
        latent_train=train_eval["latent"],
        latent_validation=external_eval["latent"],
        prediction_train=train_eval["predicted_values"],
        prediction_validation=external_eval["predicted_values"],
        modality_weights=torch.softmax(
            final_model.mod_weight.detach().cpu(), dim=1).numpy(),
        prediction_weights=final_model.weight_pred.detach().cpu().numpy())
    write_csv(output / "full_training_selected_epoch_metrics.csv", final_history)
    result = {
        "status": "success",
        "round": args.round,
        "seed": args.seed,
        "inner_cv_best_epoch": best,
        "inner_cv_selected_epoch_one_se": selected,
        "selected_epoch": selected_epoch,
        "training_prediction": train_eval["prediction"],
        "external_validation_prediction_single_evaluation": external_eval["prediction"],
        "training_reconstruction_r2_by_modality": train_eval["reconstruction_r2"],
        "external_validation_reconstruction_r2_by_modality": external_eval["reconstruction_r2"],
        "spatial_collinearity_by_modality": spatial,
        "inner_initialization_diagnostics": init_diagnostics,
        "full_training_initialization_diagnostics": final_init,
        "conditional_on_saved_full_training_dictionary": True,
        "external_validation_used_for_epoch_selection": False,
        "held_out_test_evaluated": False,
    }
    base.json_write(output / "results.json", result)
    config["status"] = "success"
    config["selected_epoch"] = selected_epoch
    base.json_write(output / "configuration.json", config)
    (output / "_SUCCESS").write_text(
        "computational_success; external validation evaluated once; test untouched\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adni-root", required=True)
    parser.add_argument("--target-train", required=True)
    parser.add_argument("--target-validation", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

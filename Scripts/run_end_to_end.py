#!/usr/bin/env python3
"""Final ADNI SBF: fresh dictionaries and validation-selected neural state.

Each invocation is one seed/round. It reads raw training and validation
imaging from a view containing no held-out-test directories, learns five new
dictionaries, projects/normalizes coefficients, and trains the repaired model
for epochs 0..120 with fixed equal loss weights. Validation selects one epoch
using conventional R2 only after positive-correlation, finite-range,
prediction-scale, and calibration gates. Validation never supplies gradients
and held-out test data remain inaccessible to this training process.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

import run_loss_ablation as base
import run_nested_cv as common
import sbf_utils as sbf


MODALITIES = ["CT", "TAU", "PSA", "GM", "AMY"]
FILETYPES = [".mgh", ".gz", ".mgh", ".gz", ".gz"]
EXPECTED_N = {"train": 192, "validation": 41}
UPSTREAM_COMMIT = "cb0b2e7d07526d203204df2b8822107eab78d971"


def prediction_selection_gate(train_metrics: dict, validation_metrics: dict,
                              prediction: np.ndarray,
                              allowed_min: np.ndarray,
                              allowed_max: np.ndarray) -> tuple[bool, dict]:
    """Prespecified gate applied before validation R2 can select an epoch."""
    required = (
        validation_metrics["r"], validation_metrics["r2_conventional"],
        validation_metrics["rmse"], validation_metrics["prediction_sd_ratio"],
        validation_metrics["calibration_slope_y_on_prediction"],
        train_metrics["r2_conventional"],
    )
    finite = bool(np.isfinite(np.asarray(required, dtype=float)).all() and
                  np.isfinite(prediction).all())
    in_range = bool(finite and
                    np.all(prediction >= allowed_min[None, :]) and
                    np.all(prediction <= allowed_max[None, :]))
    positive_r = bool(finite and validation_metrics["r"] > 0.0)
    positive_r2 = bool(finite and validation_metrics["r2_conventional"] > 0.0)
    positive_train_r2 = bool(finite and train_metrics["r2_conventional"] > 0.0)
    sd_valid = bool(finite and
                    0.25 <= validation_metrics["prediction_sd_ratio"] <= 2.0)
    calibration_valid = bool(
        finite and 0.25 <=
        validation_metrics["calibration_slope_y_on_prediction"] <= 2.0)
    audit = {
        "finite": finite,
        "prediction_in_training_derived_range": in_range,
        "positive_validation_r": positive_r,
        "positive_validation_conventional_r2": positive_r2,
        "positive_training_conventional_r2": positive_train_r2,
        "validation_prediction_sd_ratio_in_0p25_to_2": sd_valid,
        "validation_calibration_slope_in_0p25_to_2": calibration_valid,
    }
    return bool(all(audit.values())), audit


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, obj: object) -> None:
    with path.open("w") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False, default=str)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_target(path: Path, n: int) -> np.ndarray:
    values = sbf.load_nidp_csv(path)
    target = np.asarray(values[:, [0]], dtype=np.float32)
    if target.shape != (n, 1) or not np.isfinite(target).all():
        raise ValueError(f"Invalid target {path}: {target.shape}")
    return target


def split_name(name: str) -> str:
    lower = name.lower()
    if lower.endswith("_train"):
        return "train"
    if lower.endswith("_validation"):
        return "validation"
    if lower.endswith("_test"):
        raise RuntimeError(f"Held-out-test directory unexpectedly visible: {name}")
    raise ValueError(f"Cannot identify split: {name}")


def finite_correlation_summary(maps: np.ndarray) -> dict:
    maps = np.asarray(maps, dtype=np.float64)
    if maps.ndim != 2 or maps.shape[1] != 50:
        raise ValueError(f"Expected feature x 50 maps, got {maps.shape}")
    if not np.isfinite(maps).all() or np.any(np.std(maps, axis=0) == 0):
        raise RuntimeError("Spatial maps are nonfinite or contain a constant component")
    corr = np.abs(np.corrcoef(maps.T))
    values = corr[np.triu_indices(50, 1)]
    if values.size != 1225 or not np.isfinite(values).all():
        raise RuntimeError("Invalid spatial-map correlation matrix")
    return {
        "median_abs_r": float(np.median(values)),
        "p90_abs_r": float(np.quantile(values, 0.90)),
        "p95_abs_r": float(np.quantile(values, 0.95)),
        "p99_abs_r": float(np.quantile(values, 0.99)),
        "max_abs_r": float(np.max(values)),
        "fraction_abs_r_gt_0p95": float(np.mean(values > 0.95)),
    }


def bh_significant_loading_correlations(latent: np.ndarray,
                                        target: np.ndarray) -> dict:
    from scipy.stats import pearsonr

    target = np.asarray(target).ravel()
    rows = []
    for component in range(latent.shape[1]):
        result = pearsonr(np.asarray(latent[:, component]), target)
        rows.append({"component": component + 1,
                     "r": float(result.statistic), "p": float(result.pvalue)})
    order = np.argsort([row["p"] for row in rows])
    m = len(rows)
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for rank_index in range(m - 1, -1, -1):
        original_index = int(order[rank_index])
        rank = rank_index + 1
        running = min(running, rows[original_index]["p"] * m / rank)
        adjusted[original_index] = min(1.0, running)
    for index, row in enumerate(rows):
        row["p_bh"] = float(adjusted[index])
        row["significant_bh_0p05"] = bool(adjusted[index] < 0.05)
    return {
        "n_significant_bh_0p05": int(sum(row["significant_bh_0p05"] for row in rows)),
        "median_abs_r": float(np.median(np.abs([row["r"] for row in rows]))),
        "max_abs_r": float(np.max(np.abs([row["r"] for row in rows]))),
        "components": rows,
    }


def prepare_dictionary_data(args, output: Path, train_target: np.ndarray):
    fs_home = Path(os.environ["FREESURFER_HOME"])
    fsl_home = Path(os.environ["FSLDIR"])
    opts = {
        "brain_data_main_folder": str(Path(args.input_view)),
        "output_dir": str(output),
        "modalities_order": MODALITIES,
        "modalities_order_filetypes": FILETYPES,
        "fs_path": str(fs_home / "bin"),
        "path_to_fsavg": str(fs_home / "subjects" / "fsaverage"),
        "fsl_path": str(fsl_home),
    }
    visible = sorted(path.name for path in Path(args.input_view).iterdir()
                     if path.is_dir())
    expected = sorted([f"{m}_train" for m in MODALITIES] +
                      [f"{m}_validation" for m in MODALITIES])
    if visible != expected:
        raise RuntimeError(f"Input view must contain exactly {expected}; found {visible}")

    # Dummy masks allow raw validation imaging to be projected without opening
    # validation targets.  No test directories are visible to this process.
    img_data, img_info, loaded_names = sbf.SBF_load(
        opts, train_target, np.zeros((1, 1), dtype=np.float32),
        np.zeros((EXPECTED_N["validation"], 1), dtype=np.float32))
    grouped = {split: {} for split in EXPECTED_N}
    for directory_name, matrix, loader_scale in zip(
            loaded_names, img_data, img_info["scaling_data_transform"]):
        split = split_name(directory_name)
        modality = sbf.clean_modality_name(directory_name)
        if modality in grouped[split]:
            raise RuntimeError(f"Duplicate {split} {modality}")
        # Match the validated adaptive-dictionary preprocessing: retain the
        # legacy within-split voxel centering but undo its split-specific RMS.
        grouped[split][modality] = np.asarray(matrix * loader_scale,
                                               dtype=np.float32)
    for split, n in EXPECTED_N.items():
        if set(grouped[split]) != set(MODALITIES):
            raise RuntimeError(f"Missing {split} modality: {sorted(grouped[split])}")
        for modality in MODALITIES:
            if grouped[split][modality].shape[0] != n:
                raise RuntimeError(f"Bad {split} {modality} shape: "
                                   f"{grouped[split][modality].shape}")
    train = [grouped["train"][m] for m in MODALITIES]
    validation = [grouped["validation"][m] for m in MODALITIES]
    return opts, img_info, train, validation


def fresh_dictionary_stage(args, output: Path, train, validation):
    dictionary_dir = output / "dictionary_artifacts"
    coefficient_dir = output / "coefficient_diagnostics"
    dictionary_dir.mkdir()
    coefficient_dir.mkdir()

    # The third API slot is a duplicate of validation, solely because the
    # upstream function has a fixed three-split signature.  It is not test data.
    placeholder = [matrix.copy() for matrix in validation]
    (dictionaries, raw_train, raw_validation, raw_placeholder,
     solver_diagnostics, input_transforms) = sbf.parallel_dictionary_learning(
        [matrix.copy() for matrix in train],
        [matrix.copy() for matrix in validation], placeholder,
        args.dictionary_dim, n_jobs=1, lambda_l1=args.alpha,
        projection_max_iter=args.projection_max_iter,
        projection_tol=args.projection_tol, random_seed=args.seed,
        dictionary_update_ridge=args.ridge_factor)

    write_json(coefficient_dir / "fista_solver_diagnostics.json",
               solver_diagnostics)
    unconverged = [(entry["modality_index"], split)
                   for entry in solver_diagnostics
                   for split in ("train", "validation", "test")
                   if not entry[split]["converged"]]
    if unconverged:
        raise RuntimeError(f"Fixed-dictionary FISTA did not converge: {unconverged}")
    for index, (dictionary, transform) in enumerate(zip(
            dictionaries, input_transforms)):
        np.save(dictionary_dir / f"modality_{index}_dictionary.npy", dictionary)
        np.savez_compressed(
            dictionary_dir / f"modality_{index}_input_transform.npz",
            feature_mean=transform["feature_mean"],
            global_rms=np.asarray([transform["global_rms"]], dtype=np.float64))
        np.savez_compressed(
            coefficient_dir / f"modality_{index}_raw_coefficients.npz",
            train=raw_train[index], validation=raw_validation[index])

    standardized_train, standardized_validation = [], []
    scale_diagnostics = []
    for index in range(len(MODALITIES)):
        z_train, z_validation, z_placeholder, active, diagnostics = (
            sbf._coefficient_scale_guard(
                raw_train[index], raw_validation[index], raw_placeholder[index],
                args.inactive_sd_ratio))
        if not np.array_equal(z_validation, z_placeholder):
            raise RuntimeError("Validation placeholder projection was not identical")
        mean = np.mean(raw_train[index], axis=0, dtype=np.float64)
        sd = np.std(raw_train[index], axis=0, ddof=0, dtype=np.float64)
        np.savez_compressed(
            dictionary_dir / f"modality_{index}_coefficient_normalization.npz",
            mean=mean, sd=sd, active=active)
        centered = raw_train[index][:, active].astype(np.float64) - mean[active]
        effective_rank = int(np.linalg.matrix_rank(centered))
        diagnostics.update({
            "modality_index": index,
            "active_direction_indices": np.flatnonzero(active).tolist(),
            "active_coefficient_effective_rank": effective_rank,
            "third_projection_role": "validation duplicate; no test data",
        })
        if int(np.sum(active)) < 50 or effective_rank < 50:
            raise RuntimeError(f"Modality {index} has fewer than 50 usable directions")
        if diagnostics["split_stats"]["validation"]["standardized_abs_max"] > 100:
            raise RuntimeError(f"Modality {index} violates coefficient scale gate")
        standardized_train.append(z_train)
        standardized_validation.append(z_validation)
        scale_diagnostics.append(diagnostics)
    write_json(coefficient_dir / "scale_guard_diagnostics.json", scale_diagnostics)
    return dictionaries, standardized_train, standardized_validation


def map_diagnostics(model, latent_train, coefficient_train, dictionaries):
    q = torch.softmax(model.mod_weight.detach().cpu(), dim=1).numpy()
    parameter_summaries, output_summaries, output_dictionary_maps = [], [], []
    covariate = np.ones((latent_train.shape[0], 1), dtype=np.float32)
    for index in range(len(MODALITIES)):
        parameter_dictionary = (model.spatial_loading[index].detach().cpu().numpy()
                                * q[:, index][None, :])
        parameter_maps = dictionaries[index] @ parameter_dictionary
        output_dictionary = sbf.sKPCR_regression(
            latent_train, coefficient_train[index], covariate)
        output_maps = dictionaries[index] @ output_dictionary
        parameter_summaries.append({"modality_index": index,
                                    "modality": MODALITIES[index],
                                    **finite_correlation_summary(parameter_maps)})
        output_summaries.append({"modality_index": index,
                                 "modality": MODALITIES[index],
                                 **finite_correlation_summary(output_maps)})
        output_dictionary_maps.append(output_dictionary)
    return parameter_summaries, output_summaries, output_dictionary_maps


def run(args) -> None:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to reuse {output}")
    output.mkdir(parents=True)
    try:
        train_target_path = Path(args.target_train)
        validation_target_path = Path(args.target_validation)
        train_target = load_target(train_target_path, EXPECTED_N["train"])
        configuration = {
            "status": "running",
            "started_utc": utc_now(),
            "round": args.round,
            "seed": args.seed,
            "workflow": "fresh raw imaging -> fresh dictionary -> repaired neural SBF",
            "dictionary_learning_rerun": True,
            "prior_dictionary_coefficients_used_as_inputs": False,
            "input_view": str(Path(args.input_view)),
            "held_out_test_directories_visible": False,
            "held_out_test_imaging_accessed": False,
            "held_out_test_targets_accessed": False,
            "validation_target_used_for_model_selection": True,
            "validation_target_used_for_gradients": False,
            "dictionary": {
                "directions": args.dictionary_dim,
                "canonical_alpha": args.alpha,
                "ridge_policy": "factor * mean(diag(A.T @ A)) each outer iteration",
                "ridge_factor": args.ridge_factor,
                "fista_restart": "corrected O'Donoghue-Candes gradient restart",
                "projection_max_iter": args.projection_max_iter,
                "projection_tol": args.projection_tol,
                "n_jobs": 1,
            },
            "neural_repair": {
                "components": 50,
                "initialization": "exact released mMIGP",
                "uncertainty_weights": "fixed for all four loss groups",
                "uncertainty_optimizer": None,
                "uncertainty_learning_rate": None,
                "loss_weights": [1, 1, 1, 1],
                "optimizer": "RMSprop",
                "learning_rate": args.lr,
                "momentum": 0.9,
                "schedule": "cosine over 121 passes",
                "epochs_evaluated_for_selection": [10, args.epochs],
                "validation_checkpoint_selection": True,
                "selection_rule": (
                    "among gate-passing epochs, maximize validation conventional "
                    "R2; break ties by lower RMSE then earlier epoch"),
                "selection_gate": {
                    "validation_r": ">0",
                    "validation_conventional_r2": ">0",
                    "training_conventional_r2": ">0",
                    "finite_predictions": True,
                    "validation_prediction_range": (
                        "within train min-range to train max+range"),
                    "validation_prediction_sd_ratio": [0.25, 2.0],
                    "validation_calibration_slope_y_on_prediction": [0.25, 2.0],
                },
                "all_epoch_model_states_saved": True,
                "external_projection_bundle_saved": True,
            },
            "target_train": {"path": str(train_target_path),
                             "sha256": sha256(train_target_path)},
            "target_validation": {"path": str(validation_target_path),
                                  "role": "checkpoint selection only; no gradients"},
            "software": {"python": platform.python_version(),
                         "numpy": np.__version__, "torch": torch.__version__,
                         "nibabel": nib.__version__,
                         "upstream_commit": UPSTREAM_COMMIT},
            "scripts": {name: sha256(Path(__file__).parent / name) for name in
                        ("run_end_to_end.py", "sbf_utils.py", "run_nested_cv.py",
                         "run_loss_ablation.py")},
            "slurm": {key: os.environ.get(key) for key in
                      ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID",
                       "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK")},
        }
        write_json(output / "configuration.json", configuration)

        opts, img_info, raw_train, raw_validation = prepare_dictionary_data(
            args, output, train_target)
        dictionaries, x_train, x_validation = fresh_dictionary_stage(
            args, output, raw_train, raw_validation)

        y_mean = train_target.mean(axis=0)
        y_sd = train_target.std(axis=0)
        y_sd[y_sd == 0] = 1.0
        y_train_z = ((train_target - y_mean) / y_sd).astype(np.float32)
        validation_target = load_target(validation_target_path,
                                        EXPECTED_N["validation"])
        validation_target_sha = sha256(validation_target_path)
        y_validation_z = ((validation_target - y_mean) / y_sd).astype(np.float32)
        target_min = np.min(train_target, axis=0)
        target_max = np.max(train_target, axis=0)
        target_range = np.maximum(target_max - target_min, y_sd)
        allowed_min = target_min - target_range
        allowed_max = target_max + target_range
        device = torch.device(args.device)
        model, initialization = common.build_model(
            x_train, y_train_z, args.seed, args.dropout, device)
        loss_fun = sbf.loss_SuperBigFLICA_regression(
            1, len(x_train), len(train_target), device=device,
            auto_weight=[0, 0, 0, 0], lambdas=[1, 1, 1, 1]).to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr,
                                        momentum=0.9)
        generator = torch.Generator().manual_seed(args.seed + 900)
        loader = DataLoader(sbf.load_Multimodal_data(x_train, y_train_z),
                            batch_size=args.batch_size, shuffle=True,
                            generator=generator)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(args.epochs + 1) * len(loader))
        configuration["initialization_diagnostics"] = initialization
        write_json(output / "configuration.json", configuration)

        history, states = [], {}
        selected = None
        for epoch in range(args.epochs + 1):
            torch.manual_seed(args.seed + epoch)
            model.train()
            sums = np.zeros(5, dtype=float)
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
                sums += np.asarray([float(value.detach().cpu()) for value in losses])
            train_eval = common.evaluate(model, x_train, y_train_z,
                                         y_mean, y_sd, device)
            validation_eval = common.evaluate(
                model, x_validation, y_validation_z, y_mean, y_sd, device)
            gate, gate_audit = prediction_selection_gate(
                train_eval["prediction"], validation_eval["prediction"],
                validation_eval["predicted_values"], allowed_min, allowed_max)
            sigma_values = {
                name: getattr(loss_fun, name).detach().cpu().numpy().tolist()
                for name in ("sigma1", "sigma2", "sigma3", "sigma4", "sigma5")
            }
            row = {"epoch": epoch, "selection_gate_pass": gate}
            row.update({f"loss_{name}": float(sums[index] / len(loader))
                        for index, name in enumerate(("total", "reconstruction",
                                                     "spatial", "target",
                                                     "prediction_penalty"))})
            row.update({f"train_{key}": value
                        for key, value in train_eval["prediction"].items()})
            row.update({f"validation_{key}": value
                        for key, value in validation_eval["prediction"].items()})
            row.update({f"gate_{key}": value for key, value in gate_audit.items()})
            history.append(row)
            states[epoch] = copy.deepcopy(model.state_dict())
            if gate and epoch >= 10:
                candidate_key = (
                    float(validation_eval["prediction"]["r2_conventional"]),
                    -float(validation_eval["prediction"]["rmse"]),
                    -int(epoch),
                )
                if selected is None or candidate_key > selected["key"]:
                    selected = {
                        "key": candidate_key,
                        "epoch": epoch,
                        "model_state": copy.deepcopy(model.state_dict()),
                        "uncertainty_state": copy.deepcopy(loss_fun.state_dict()),
                        "train_prediction": train_eval["prediction"],
                        "validation_prediction": validation_eval["prediction"],
                        "gate_audit": gate_audit,
                        "sigma_values": sigma_values,
                    }
            print(json.dumps({"round": args.round, "epoch": epoch,
                              "loss_total": row["loss_total"],
                              "gate": gate,
                              "validation_r": row["validation_r"],
                              "validation_r2": row["validation_r2_conventional"]}),
                  flush=True)
        if len(states) != args.epochs + 1:
            raise RuntimeError("Not all epoch states were retained")
        if selected is None:
            raise RuntimeError("No epoch passed the prespecified selection gate")
        write_csv(output / "training_loss_history.csv", history)
        torch.save(states, output / "all_epoch_model_states.pth")

        model.load_state_dict(selected["model_state"])
        loss_fun.load_state_dict(selected["uncertainty_state"])
        train_eval = common.evaluate(model, x_train, y_train_z,
                                     y_mean, y_sd, device)
        validation_eval = common.evaluate(model, x_validation,
                                          y_validation_z, y_mean, y_sd, device)
        parameter_spatial, output_spatial, output_dictionary_maps = (
            map_diagnostics(model, train_eval["latent"], x_train, dictionaries))
        loading_target = bh_significant_loading_correlations(
            train_eval["latent"], train_target)
        torch.save(selected["model_state"], output / "selected_model_state.pth")
        torch.save(selected["uncertainty_state"],
                   output / "selected_uncertainty_state.pth")
        np.savez_compressed(
            output / "selected_results.npz",
            selected_epoch=np.asarray([selected["epoch"]], dtype=np.int64),
            latent_train=train_eval["latent"],
            latent_validation=validation_eval["latent"],
            prediction_train=train_eval["predicted_values"],
            prediction_validation=validation_eval["predicted_values"],
            modality_weights=torch.softmax(
                model.mod_weight.detach().cpu(), dim=1).numpy(),
            prediction_weights=model.weight_pred.detach().cpu().numpy())
        np.savez_compressed(
            output / "target_normalization.npz",
            mean=y_mean.astype(np.float64), sd=y_sd.astype(np.float64),
            training_min=target_min.astype(np.float64),
            training_max=target_max.astype(np.float64),
            allowed_min=allowed_min.astype(np.float64),
            allowed_max=allowed_max.astype(np.float64))
        architecture = {
            "model_class": "SupervisedFLICAmodel",
            "modalities": MODALITIES,
            "standardized_coefficient_dimensions": [
                int(matrix.shape[1]) for matrix in x_train],
            "components": 50,
            "targets": 1,
            "dropout": args.dropout,
            "target_standardization": "training mean and population SD",
        }
        write_json(output / "model_architecture.json", architecture)

        projection_artifacts = [
            output / "selected_model_state.pth",
            output / "target_normalization.npz",
            output / "model_architecture.json",
        ]
        for index in range(len(MODALITIES)):
            projection_artifacts.extend([
                output / "dictionary_artifacts" /
                f"modality_{index}_dictionary.npy",
                output / "dictionary_artifacts" /
                f"modality_{index}_input_transform.npz",
                output / "dictionary_artifacts" /
                f"modality_{index}_coefficient_normalization.npz",
            ])
        projection_manifest = {
            "schema_version": 1,
            "purpose": (
                "Locked external-cohort projection without dictionary learning, "
                "neural retraining, or checkpoint reselection"),
            "round": args.round,
            "seed": args.seed,
            "selected_epoch": selected["epoch"],
            "modality_order": MODALITIES,
            "raw_feature_counts": [int(matrix.shape[1]) for matrix in raw_train],
            "standardized_coefficient_dimensions": [
                int(matrix.shape[1]) for matrix in x_train],
            "input_contract": (
                "Each external matrix must be participant x feature, use the "
                "same modality-specific vertex/voxel order and support as ADNI, "
                "and receive no cohort-specific refitting."),
            "dictionary_projection": {
                "alpha": args.alpha,
                "max_iter": args.projection_max_iter,
                "tol": args.projection_tol,
                "fista_restart": "corrected O'Donoghue-Candes gradient restart",
            },
            "artifacts": [{
                "relative_path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            } for path in projection_artifacts],
            "held_out_test_used_to_fit_or_select": False,
        }
        write_json(output / "external_projection_bundle_manifest.json",
                   projection_manifest)

        map_dir = output / f"spatial_maps_selected_epoch{selected['epoch']}_raw_pre_gmm"
        map_dir.mkdir()
        sbf.SBF_save_everything(str(map_dir), output_dictionary_maps,
                                img_info, {**opts,
                                "modalities_order_filetypes": FILETYPES,
                                "modalities_order": MODALITIES}, dictionaries)
        map_files = sorted(path for path in map_dir.iterdir() if path.is_file())
        map_manifest = [{"path": str(path), "bytes": path.stat().st_size,
                         "sha256": sha256(path)} for path in map_files]
        # CT and PSA each produce LH/RH MGH files; TAU, GM, and AMY each
        # produce one NIfTI, for seven files covering five modalities.
        if len(map_manifest) != 7:
            raise RuntimeError(f"Expected seven selected output images, got {len(map_manifest)}")

        result = {
            "status": "success",
            "round": args.round,
            "seed": args.seed,
            "dictionary_learning_rerun": True,
            "prior_dictionary_coefficients_used_as_inputs": False,
            "selected_epoch": selected["epoch"],
            "selection_rule": configuration["neural_repair"]["selection_rule"],
            "selection_gate_audit": selected["gate_audit"],
            "selected_uncertainty_values": selected["sigma_values"],
            "selected_result": {
                "training_prediction": train_eval["prediction"],
                "validation_prediction": validation_eval["prediction"],
                "training_coefficient_reconstruction_r2_by_modality":
                    train_eval["reconstruction_r2"],
                "validation_coefficient_reconstruction_r2_by_modality":
                    validation_eval["reconstruction_r2"],
                "training_loading_target_correlations": loading_target,
                "raw_parameter_map_collinearity": parameter_spatial,
                "raw_skpcr_output_map_collinearity": output_spatial,
            },
            "selected_raw_pre_gmm_map_manifest": map_manifest,
            "external_projection_bundle_manifest": {
                "path": str(output / "external_projection_bundle_manifest.json"),
                "sha256": sha256(
                    output / "external_projection_bundle_manifest.json"),
            },
            "validation_checkpoint_selection": True,
            "held_out_test_evaluated": False,
        }
        write_json(output / "results.json", result)
        configuration.update({
            "status": "success",
            "completed_utc": utc_now(),
            "validation_target_used_for_model_selection": True,
            "validation_target_used_for_gradients": False,
            "target_validation": {"path": str(validation_target_path),
                                  "sha256": validation_target_sha},
            "held_out_test_imaging_accessed": False,
            "held_out_test_targets_accessed": False,
            "held_out_test_evaluated": False,
        })
        write_json(output / "configuration.json", configuration)
        (output / "_SUCCESS").write_text(
            "fresh dictionaries + fixed uncertainty + validation-R2-selected SBF; test absent\n")
    except Exception as error:
        if output.exists():
            write_json(output / "_FAILURE.json", {
                "status": "failed", "failed_utc": utc_now(),
                "error_type": type(error).__name__, "error": str(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-view", required=True)
    parser.add_argument("--target-train", required=True)
    parser.add_argument("--target-validation", required=True)
    parser.add_argument("--dictionary-dim", type=int, default=250)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--ridge-factor", type=float, default=1e-4)
    parser.add_argument("--inactive-sd-ratio", type=float, default=1e-3)
    parser.add_argument("--projection-max-iter", type=int, default=5000)
    parser.add_argument("--projection-tol", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs != 120 or args.dictionary_dim != 250:
        raise ValueError("This confirmatory workflow requires epochs=120 and dictionary_dim=250")
    run(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FATAL: {type(error).__name__}: {error}", file=sys.stderr,
              flush=True)
        raise

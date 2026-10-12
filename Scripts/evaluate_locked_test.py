#!/usr/bin/env python3
"""Evaluate one final ADNI round on the true test set without refitting.

The training validator must have completed before this script is launched.
It loads the validation-selected state, saved training dictionaries, input
transforms, coefficient normalization, and target normalization. Test outcomes
never affect checkpoint or round selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import traceback
from datetime import datetime, timezone

import numpy as np
import torch

import sbf_utils as sbf


MODALITIES = ["CT", "TAU", "PSA", "GM", "AMY"]
FILETYPES = [".mgh", ".gz", ".mgh", ".gz", ".gz"]
SEEDS = [555, 666, 777, 888, 999, 1111, 2222, 3333, 4444, 5555]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_target(path: Path) -> np.ndarray:
    values = sbf.load_nidp_csv(path)
    result = np.asarray(values[:, [0]], dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 1 or not np.isfinite(result).all():
        raise ValueError(f"Invalid target file {path}: {result.shape}")
    return result


def prediction_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid locked test prediction")
    error = prediction - target
    target_centered = target - np.mean(target)
    sst = float(np.sum(target_centered ** 2))
    prediction_sd = float(np.std(prediction, ddof=0))
    target_sd = float(np.std(target, ddof=0))
    if sst <= 0 or prediction_sd <= 0 or target_sd <= 0:
        raise RuntimeError("Degenerate target or prediction distribution")
    slope = float(np.cov(prediction, target, ddof=0)[0, 1] /
                  np.var(prediction, ddof=0))
    return {
        "n": int(target.size),
        "r": float(np.corrcoef(prediction, target)[0, 1]),
        "r2_conventional": float(1.0 - np.sum(error ** 2) / sst),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "prediction_min": float(np.min(prediction)),
        "prediction_max": float(np.max(prediction)),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_sd": prediction_sd,
        "target_min": float(np.min(target)),
        "target_max": float(np.max(target)),
        "target_mean": float(np.mean(target)),
        "target_sd": target_sd,
        "prediction_sd_ratio": float(prediction_sd / target_sd),
        "calibration_slope_y_on_prediction": slope,
    }


def split_name(name: str) -> str:
    lower = name.lower()
    for split in ("validation", "train", "test"):
        if lower.endswith(f"_{split}"):
            return split
    raise ValueError(f"Cannot identify split: {name}")


def evaluate(args) -> None:
    root = Path(args.root)
    seed = SEEDS[args.round - 1]
    source = root / "results" / f"round{args.round}_seed{seed}"
    output = root / "test_evaluation" / f"round{args.round}_seed{seed}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    try:
        if not (root / "_TEN_ROUNDS_VALIDATED").is_file():
            raise RuntimeError("Training validator marker is absent")
        if not (source / "_SUCCESS").is_file() or list(source.glob("_FAILURE*")):
            raise RuntimeError("Source training round is not cleanly validated")
        config = json.loads((source / "configuration.json").read_text())
        result = json.loads((source / "results.json").read_text())
        if (config["round"], config["seed"], result["selected_epoch"]) != (
                args.round, seed, result["selected_epoch"]):
            raise RuntimeError("Round/seed identity mismatch")
        if config["held_out_test_evaluated"] is not False:
            raise RuntimeError("Training configuration reports prior test evaluation")

        input_root = Path(args.input_view)
        visible = sorted(path.name for path in input_root.iterdir() if path.is_dir())
        expected = sorted(f"{modality}_{split}" for modality in MODALITIES
                          for split in ("train", "validation", "test"))
        if visible != expected:
            raise RuntimeError(f"Test view mismatch: {visible}")
        target_root = Path(args.target_root)
        train_target_path = target_root / "nIDP_CDRSB_matrix_training.csv"
        validation_target_path = target_root / "nIDP_CDRSB_matrix_validation.csv"
        test_target_path = target_root / "nIDP_CDRSB_matrix_test.csv"
        train_target = load_target(train_target_path)
        validation_target = load_target(validation_target_path)
        test_target = load_target(test_target_path)

        fs_home = Path(os.environ["FREESURFER_HOME"])
        fsl_home = Path(os.environ["FSLDIR"])
        options = {
            "brain_data_main_folder": str(input_root),
            "output_dir": str(output),
            "modalities_order": MODALITIES,
            "modalities_order_filetypes": FILETYPES,
            "fs_path": str(fs_home / "bin"),
            "path_to_fsavg": str(fs_home / "subjects" / "fsaverage"),
            "fsl_path": str(fsl_home),
        }
        imaging, image_info, names = sbf.SBF_load(
            options, train_target, test_target, validation_target)
        test_by_modality = {}
        for name, matrix, loader_scale in zip(
                names, imaging, image_info["scaling_data_transform"]):
            if split_name(name) != "test":
                continue
            modality = sbf.clean_modality_name(name)
            if modality in test_by_modality:
                raise RuntimeError(f"Duplicate test modality {modality}")
            test_by_modality[modality] = np.asarray(
                matrix * loader_scale, dtype=np.float32)
        if set(test_by_modality) != set(MODALITIES):
            raise RuntimeError(f"Missing test modality: {sorted(test_by_modality)}")

        manifest = json.loads(
            (source / "external_projection_bundle_manifest.json").read_text())
        normalized, projection_diagnostics = [], []
        for index, modality in enumerate(MODALITIES):
            dictionary = np.load(
                source / "dictionary_artifacts" /
                f"modality_{index}_dictionary.npy", mmap_mode="r")
            transform = np.load(
                source / "dictionary_artifacts" /
                f"modality_{index}_input_transform.npz", allow_pickle=False)
            matrix = test_by_modality[modality]
            if matrix.shape[1] != manifest["raw_feature_counts"][index]:
                raise RuntimeError(
                    f"{modality} feature mismatch: {matrix.shape[1]}")
            transformed = ((matrix - transform["feature_mean"].astype(np.float32)) /
                           float(transform["global_rms"].reshape(-1)[0])).astype(
                               np.float32)
            coefficients, solver = sbf.apply_dictionary(
                dictionary, transformed,
                float(manifest["dictionary_projection"]["alpha"]),
                int(manifest["dictionary_projection"]["max_iter"]),
                float(manifest["dictionary_projection"]["tol"]),
                "cpu", True)
            if not solver["converged"]:
                raise RuntimeError(f"Test FISTA failed for {modality}: {solver}")
            normalization = np.load(
                source / "dictionary_artifacts" /
                f"modality_{index}_coefficient_normalization.npz",
                allow_pickle=False)
            mean = normalization["mean"].astype(np.float64)
            sd = normalization["sd"].astype(np.float64)
            active = normalization["active"].astype(bool)
            z = ((coefficients.astype(np.float64) - mean) /
                 np.where(active, sd, 1.0))
            z[:, ~active] = 0.0
            z = z.astype(np.float32)
            abs_max = float(np.max(np.abs(z)))
            if not np.isfinite(z).all() or abs_max > 100.0:
                raise RuntimeError(
                    f"{modality} standardized test coefficient failure: {abs_max}")
            normalized.append(torch.as_tensor(z, dtype=torch.float32))
            projection_diagnostics.append({
                "modality": modality,
                "raw_shape": list(matrix.shape),
                "coefficient_shape": list(coefficients.shape),
                "active_directions": int(np.sum(active)),
                "standardized_abs_max": abs_max,
                "solver": solver,
            })

        architecture = json.loads((source / "model_architecture.json").read_text())
        model = sbf.SupervisedFLICAmodel(
            architecture["standardized_coefficient_dimensions"],
            architecture["components"], architecture["targets"],
            dropout=architecture["dropout"], device=torch.device("cpu")).to("cpu")
        state_path = source / "selected_model_state.pth"
        model.load_state_dict(torch.load(
            state_path, map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            _, _, latent, prediction_z, _ = model(normalized, "cpu")
        target_norm = np.load(source / "target_normalization.npz",
                              allow_pickle=False)
        prediction = (prediction_z.cpu().numpy().astype(np.float64) *
                      target_norm["sd"] + target_norm["mean"])
        latent = latent.cpu().numpy().astype(np.float64)
        metrics = prediction_metrics(test_target, prediction)
        audit = {
            "status": "success",
            "round": args.round,
            "seed": seed,
            "selected_epoch": int(result["selected_epoch"]),
            "evaluated_utc": datetime.now(timezone.utc).isoformat(),
            "source_directory": str(source),
            "checkpoint_sha256": sha256(state_path),
            "bundle_manifest_sha256": sha256(
                source / "external_projection_bundle_manifest.json"),
            "test_target_sha256": sha256(test_target_path),
            "selection_lock": (
                "Checkpoint was selected by validation conventional R2 before "
                "test data were made visible; no retraining or reselection."),
            "test_metrics": metrics,
            "projection_diagnostics": projection_diagnostics,
            "held_out_test_evaluated": True,
            "test_used_for_checkpoint_or_round_selection": False,
            "scientific_caveat": (
                "All ten locked rounds are evaluated for robustness; therefore "
                "this test set is no longer an untouched single-use test set."),
            "slurm": {key: os.environ.get(key) for key in
                      ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID",
                       "SLURM_ARRAY_TASK_ID")},
        }
        np.savetxt(output / "pred_test.csv", prediction, delimiter=",")
        np.savetxt(output / "lat_test.csv", latent, delimiter=",")
        (output / "test_evaluation.json").write_text(
            json.dumps(audit, indent=2, allow_nan=False) + "\n")
        (output / "_SUCCESS").write_text("locked true-test inference PASS\n")
        print(json.dumps(audit, indent=2, allow_nan=False), flush=True)
    except Exception as error:
        (output / "_FAILURE.json").write_text(json.dumps({
            "status": "failed", "round": args.round, "seed": seed,
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
            "failed_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2) + "\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--round", type=int, choices=range(1, 11), required=True)
    parser.add_argument("--input-view", required=True)
    parser.add_argument("--target-root", required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()

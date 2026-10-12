#!/usr/bin/env python3
"""Independent audit of the ten final fixed-weight ADNI runs."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import nibabel as nib
import numpy as np


ROUNDS = list(range(1, 11))
SEEDS = [555, 666, 777, 888, 999, 1111, 2222, 3333, 4444, 5555]


def read_json(path: Path):
    with path.open() as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def as_bool(text: str) -> bool:
    if text == "True":
        return True
    if text == "False":
        return False
    raise ValueError(f"Not a serialized Boolean: {text!r}")


def validate_history(path: Path, selected_epoch: int) -> dict:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(121)):
        raise RuntimeError(f"Epoch history is incomplete: {epochs[:3]}...{epochs[-3:]}")
    eligible = []
    for row in rows:
        for key, value in row.items():
            if key == "epoch" or key.startswith("gate_") or key == "selection_gate_pass":
                continue
            if not math.isfinite(float(value)):
                raise RuntimeError(f"Nonfinite epoch metric {key}={value}")
        if as_bool(row["selection_gate_pass"]) and int(row["epoch"]) >= 10:
            eligible.append(row)
    if not eligible:
        raise RuntimeError("No epoch passed the prespecified selection gate")
    expected = max(
        eligible,
        key=lambda row: (float(row["validation_r2_conventional"]),
                         -float(row["validation_rmse"]), -int(row["epoch"])))
    if int(expected["epoch"]) != selected_epoch:
        raise RuntimeError(
            f"Selected epoch {selected_epoch} != audited optimum {expected['epoch']}")
    return {
        "n_epochs": len(rows),
        "n_gate_passing_epochs_10_to_120": len(eligible),
        "audited_selected_epoch": int(expected["epoch"]),
        "audited_selected_validation_r": float(expected["validation_r"]),
        "audited_selected_validation_r2": float(
            expected["validation_r2_conventional"]),
        "audited_selected_validation_rmse": float(expected["validation_rmse"]),
    }


def main(root: Path) -> None:
    marker = root / "_TEN_ROUNDS_VALIDATED"
    if marker.exists():
        raise FileExistsError("Refusing to overwrite completed validation")
    validation = root / "validation"
    validation.mkdir(exist_ok=False)
    summary_rows, audits = [], []
    for round_number, seed in zip(ROUNDS, SEEDS):
        run = root / "results" / f"round{round_number}_seed{seed}"
        if list(run.glob("_FAILURE*")) or not (run / "_SUCCESS").is_file():
            raise RuntimeError(f"Round {round_number} missing success or has failure")
        config = read_json(run / "configuration.json")
        result = read_json(run / "results.json")
        if config.get("status") != "success" or result.get("status") != "success":
            raise RuntimeError(f"Round {round_number} is not successful")
        required = {
            "round": round_number,
            "seed": seed,
            "dictionary_learning_rerun": True,
            "prior_dictionary_coefficients_used_as_inputs": False,
            "held_out_test_imaging_accessed": False,
            "held_out_test_targets_accessed": False,
            "held_out_test_evaluated": False,
            "validation_target_used_for_model_selection": True,
            "validation_target_used_for_gradients": False,
        }
        for key, expected in required.items():
            if config.get(key) != expected:
                raise RuntimeError(
                    f"Round {round_number}: {key}={config.get(key)!r}, expected {expected!r}")
        dictionary = config["dictionary"]
        if (dictionary["directions"], dictionary["canonical_alpha"],
                dictionary["ridge_factor"]) != (250, 0.1, 1e-4):
            raise RuntimeError(f"Round {round_number} dictionary configuration changed")
        neural = config["neural_repair"]
        if neural["initialization"] != "exact released mMIGP":
            raise RuntimeError("mMIGP initialization changed")
        if neural["uncertainty_weights"] != "fixed for all four loss groups":
            raise RuntimeError("Uncertainty weights are not fixed")
        if neural["uncertainty_optimizer"] is not None:
            raise RuntimeError("Unexpected uncertainty optimizer")
        if not neural["validation_checkpoint_selection"]:
            raise RuntimeError("Validation selection is disabled")

        solver = read_json(run / "coefficient_diagnostics" /
                           "fista_solver_diagnostics.json")
        scale = read_json(run / "coefficient_diagnostics" /
                          "scale_guard_diagnostics.json")
        if len(solver) != 5 or len(scale) != 5:
            raise RuntimeError(f"Round {round_number} modality diagnostics missing")
        for entry in solver:
            if any(not entry[split]["converged"]
                   for split in ("train", "validation", "test")):
                raise RuntimeError(f"Round {round_number} FISTA failure")
            if entry["dictionary_update_ridge_factor"] != 1e-4:
                raise RuntimeError("Adaptive ridge changed")
        for entry in scale:
            if entry["n_active_atoms"] < 50 or entry["active_coefficient_effective_rank"] < 50:
                raise RuntimeError(f"Round {round_number} coefficient rank failure")
            if entry["split_stats"]["validation"]["standardized_abs_max"] > 100:
                raise RuntimeError(f"Round {round_number} coefficient scale failure")

        selected_epoch = int(result["selected_epoch"])
        if not 10 <= selected_epoch <= 120:
            raise RuntimeError(f"Round {round_number} invalid selected epoch {selected_epoch}")
        history_audit = validate_history(run / "training_loss_history.csv",
                                         selected_epoch)
        for state_name in ("all_epoch_model_states.pth",
                           "selected_model_state.pth",
                           "selected_uncertainty_state.pth",
                           "selected_results.npz",
                           "target_normalization.npz",
                           "model_architecture.json",
                           "external_projection_bundle_manifest.json"):
            state = run / state_name
            if not state.is_file() or state.stat().st_size == 0:
                raise RuntimeError(f"Round {round_number} missing {state_name}")
        projection_manifest = read_json(
            run / "external_projection_bundle_manifest.json")
        if (projection_manifest["round"], projection_manifest["seed"],
                projection_manifest["selected_epoch"]) != (
                round_number, seed, selected_epoch):
            raise RuntimeError(f"Round {round_number} projection identity mismatch")
        if projection_manifest["modality_order"] != ["CT", "TAU", "PSA", "GM", "AMY"]:
            raise RuntimeError(f"Round {round_number} projection modality order changed")
        if len(projection_manifest["artifacts"]) != 18:
            raise RuntimeError(f"Round {round_number} projection bundle incomplete")
        for artifact in projection_manifest["artifacts"]:
            path = run / artifact["relative_path"]
            if (not path.is_file() or path.stat().st_size != artifact["bytes"] or
                    sha256(path) != artifact["sha256"]):
                raise RuntimeError(
                    f"Round {round_number} projection artifact integrity failure: {path}")
        selected = result["selected_result"]
        if not finite_tree(selected) or not finite_tree(
                result["selected_uncertainty_values"]):
            raise RuntimeError(f"Round {round_number} has nonfinite selected results")
        train = selected["training_prediction"]
        valid = selected["validation_prediction"]
        if not (valid["r"] > 0 and valid["r2_conventional"] > 0 and
                train["r2_conventional"] > 0 and
                0.25 <= valid["prediction_sd_ratio"] <= 2.0 and
                0.25 <= valid["calibration_slope_y_on_prediction"] <= 2.0):
            raise RuntimeError(f"Round {round_number} selected prediction gate failure")
        if not all(result["selection_gate_audit"].values()):
            raise RuntimeError(f"Round {round_number} recorded gate failure")
        output_spatial = selected["raw_skpcr_output_map_collinearity"]
        if len(output_spatial) != 5 or any(
                item["fraction_abs_r_gt_0p95"] >= 0.10 for item in output_spatial):
            raise RuntimeError(f"Round {round_number} map collinearity gate failure")

        image_audit = []
        manifest = result["selected_raw_pre_gmm_map_manifest"]
        if len(manifest) != 7:
            raise RuntimeError(f"Round {round_number} map count {len(manifest)}")
        for item in manifest:
            path = Path(item["path"])
            if not path.is_file() or sha256(path) != item["sha256"]:
                raise RuntimeError(f"Round {round_number} map integrity failure")
            shape = tuple(int(value) for value in nib.load(path).shape)
            if shape[-1] != 50:
                raise RuntimeError(f"Round {round_number} map shape {shape}")
            image_audit.append({"path": str(path), "shape": list(shape),
                                "sha256": item["sha256"]})

        loading = selected["training_loading_target_correlations"]
        summary_rows.append({
            "round": round_number,
            "seed": seed,
            "selected_epoch": selected_epoch,
            "train_r": train["r"],
            "train_r2": train["r2_conventional"],
            "validation_r": valid["r"],
            "validation_r2": valid["r2_conventional"],
            "validation_mae": valid["mae"],
            "validation_rmse": valid["rmse"],
            "validation_prediction_sd_ratio": valid["prediction_sd_ratio"],
            "validation_calibration_slope": valid[
                "calibration_slope_y_on_prediction"],
            "mean_train_coefficient_reconstruction_r2": float(np.mean(
                selected["training_coefficient_reconstruction_r2_by_modality"])),
            "mean_validation_coefficient_reconstruction_r2": float(np.mean(
                selected["validation_coefficient_reconstruction_r2_by_modality"])),
            "n_loading_cdr_significant_bh_0p05": loading[
                "n_significant_bh_0p05"],
            "median_abs_loading_cdr_r": loading["median_abs_r"],
            "max_abs_loading_cdr_r": loading["max_abs_r"],
            "worst_output_map_p95_abs_r": max(
                item["p95_abs_r"] for item in output_spatial),
            "worst_output_map_fraction_abs_r_gt_0p95": max(
                item["fraction_abs_r_gt_0p95"] for item in output_spatial),
            "n_gate_passing_epochs": history_audit[
                "n_gate_passing_epochs_10_to_120"],
        })
        audits.append({"round": round_number, "seed": seed,
                       "history_audit": history_audit,
                       "image_audit": image_audit})

    csv_path = validation / "selected_epoch_all_rounds.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    numeric = [key for key in summary_rows[0] if key not in {"round", "seed"}]
    aggregate = {
        key: {
            "mean": float(np.mean([row[key] for row in summary_rows])),
            "min": float(np.min([row[key] for row in summary_rows])),
            "max": float(np.max([row[key] for row in summary_rows])),
        }
        for key in numeric
    }
    summary = {
        "status": "PASS",
        "rounds": 10,
        "fresh_dictionary_learning_per_round": True,
        "fixed_uncertainty_all_four_groups": True,
        "validation_checkpoint_selection": True,
        "selection_metric": "conventional validation R2",
        "external_projection_bundle_validated": True,
        "held_out_test_accessed": False,
        "aggregate": aggregate,
        "rows": summary_rows,
        "audits": audits,
    }
    with (validation / "validation_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    marker.write_text(
        "PASS: ten fresh dictionaries + fixed uncertainty + validation-R2 selection; test absent\n")
    print(json.dumps({"status": "PASS", "csv": str(csv_path),
                      "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_and_summarize.py ROOT")
    main(Path(sys.argv[1]))

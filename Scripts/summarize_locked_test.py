#!/usr/bin/env python3
"""Independently summarize the ten locked true-test evaluations."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


SEEDS = [555, 666, 777, 888, 999, 1111, 2222, 3333, 4444, 5555]
METRICS = [
    "r", "r2_conventional", "mae", "rmse", "prediction_min",
    "prediction_max", "prediction_mean", "prediction_sd", "target_min",
    "target_max", "target_mean", "target_sd", "prediction_sd_ratio",
    "calibration_slope_y_on_prediction",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(root: Path) -> None:
    marker = root / "_TRUE_TEST_EVALUATION_VALIDATED"
    if marker.exists():
        raise FileExistsError("Refusing to overwrite test summary")
    out = root / "test_evaluation" / "summary"
    out.mkdir(parents=True, exist_ok=False)
    rows, audits = [], []
    validation = json.loads(
        (root / "validation" / "validation_summary.json").read_text())
    if validation["status"] != "PASS" or not validation["fixed_uncertainty_all_four_groups"]:
        raise RuntimeError("Training validation did not PASS as fixed-weight final model")
    validation_by_round = {int(row["round"]): row for row in validation["rows"]}
    for round_number, seed in enumerate(SEEDS, 1):
        directory = root / "test_evaluation" / f"round{round_number}_seed{seed}"
        if list(directory.glob("_FAILURE*")) or not (directory / "_SUCCESS").is_file():
            raise RuntimeError(f"Test round {round_number} is not successful")
        audit_path = directory / "test_evaluation.json"
        audit = json.loads(audit_path.read_text())
        if (audit["status"], audit["round"], audit["seed"]) != (
                "success", round_number, seed):
            raise RuntimeError(f"Test round {round_number} identity failure")
        if (not audit["held_out_test_evaluated"] or
                audit["test_used_for_checkpoint_or_round_selection"]):
            raise RuntimeError(f"Test round {round_number} selection-lock failure")
        selected_epoch = int(validation_by_round[round_number]["selected_epoch"])
        if int(audit["selected_epoch"]) != selected_epoch:
            raise RuntimeError(f"Test round {round_number} checkpoint changed")
        metrics = audit["test_metrics"]
        if not all(math.isfinite(float(metrics[key])) for key in METRICS):
            raise RuntimeError(f"Test round {round_number} has nonfinite metrics")
        for item in audit["projection_diagnostics"]:
            if (not item["solver"]["converged"] or
                    item["standardized_abs_max"] > 100.0):
                raise RuntimeError(f"Test round {round_number} projection failure")
        pred = np.loadtxt(directory / "pred_test.csv", delimiter=",", ndmin=2)
        latent = np.loadtxt(directory / "lat_test.csv", delimiter=",", ndmin=2)
        if (pred.shape[1] != 1 or latent.shape != (pred.shape[0], 50) or
                not np.isfinite(pred).all() or not np.isfinite(latent).all()):
            raise RuntimeError(f"Test round {round_number} saved array failure")
        row = {
            "round": round_number,
            "seed": seed,
            "validation_selected_epoch": selected_epoch,
            "validation_r": validation_by_round[round_number]["validation_r"],
            "validation_r2_conventional": validation_by_round[round_number]["validation_r2"],
            "validation_mae": validation_by_round[round_number]["validation_mae"],
            "validation_rmse": validation_by_round[round_number]["validation_rmse"],
        }
        row.update({f"test_{key}": metrics[key] for key in METRICS})
        rows.append(row)
        audits.append({
            "round": round_number,
            "test_evaluation_sha256": sha256(audit_path),
            "prediction_sha256": sha256(directory / "pred_test.csv"),
            "latent_sha256": sha256(directory / "lat_test.csv"),
        })
    csv_path = out / "validation_and_locked_test_all_rounds.csv"
    with csv_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    numeric = [key for key in rows[0] if key not in {"round", "seed"}]
    aggregate = {
        key: {
            "mean": float(np.mean([float(row[key]) for row in rows])),
            "sd_across_rounds": float(np.std(
                [float(row[key]) for row in rows], ddof=1)),
            "min": float(np.min([float(row[key]) for row in rows])),
            "max": float(np.max([float(row[key]) for row in rows])),
        }
        for key in numeric
    }
    summary = {
        "status": "PASS",
        "rounds": 10,
        "locked_validation_selected_checkpoints": True,
        "test_used_for_checkpoint_or_round_selection": False,
        "held_out_test_evaluated": True,
        "test_set_status": (
            "No longer an untouched single-use test set because all ten locked "
            "rounds were evaluated at the user's explicit request."),
        "aggregate": aggregate,
        "rows": rows,
        "file_audits": audits,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")
    marker.write_text(
        "PASS: ten validation-selected checkpoints evaluated once without refitting\n")
    print(json.dumps({"status": "PASS", "csv": str(csv_path),
                      "aggregate": aggregate}, indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_locked_test.py ROOT")
    main(Path(sys.argv[1]))

#!/usr/bin/env python3
"""Project aligned external participant-by-feature matrices through one round.

This utility performs inference only. It never learns dictionaries, refits
normalization, updates neural parameters, or selects a checkpoint. Each input
must use the exact ADNI feature support/order recorded in the bundle manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import sbf_utils as sbf


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_matrix(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        matrix = np.load(path, mmap_mode="r")
    elif path.suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        if archive.files != ["data"]:
            raise ValueError(f"NPZ must contain only 'data': {archive.files}")
        matrix = archive["data"]
    else:
        matrix = np.loadtxt(path, delimiter=",", ndmin=2)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid participant x feature matrix {path}: {matrix.shape}")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--tau", type=Path, required=True)
    parser.add_argument("--psa", type=Path, required=True)
    parser.add_argument("--gm", type=Path, required=True)
    parser.add_argument("--amy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    source = args.bundle_dir
    manifest_path = source / "external_projection_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for artifact in manifest["artifacts"]:
        path = source / artifact["relative_path"]
        if (not path.is_file() or path.stat().st_size != artifact["bytes"] or
                sha256(path) != artifact["sha256"]):
            raise RuntimeError(f"Bundle artifact integrity failure: {path}")

    paths = [args.ct, args.tau, args.psa, args.gm, args.amy]
    raw = [load_matrix(path) for path in paths]
    n = raw[0].shape[0]
    if any(matrix.shape[0] != n for matrix in raw):
        raise RuntimeError("External modalities have different participant counts")
    normalized, diagnostics = [], []
    for index, (modality, matrix, path) in enumerate(zip(
            manifest["modality_order"], raw, paths)):
        if matrix.shape[1] != manifest["raw_feature_counts"][index]:
            raise RuntimeError(
                f"{modality} feature count {matrix.shape[1]} != "
                f"{manifest['raw_feature_counts'][index]}")
        dictionary = np.load(source / "dictionary_artifacts" /
                             f"modality_{index}_dictionary.npy", mmap_mode="r")
        transform = np.load(source / "dictionary_artifacts" /
                            f"modality_{index}_input_transform.npz",
                            allow_pickle=False)
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
            raise RuntimeError(f"External FISTA failed for {modality}: {solver}")
        scaling = np.load(source / "dictionary_artifacts" /
                          f"modality_{index}_coefficient_normalization.npz",
                          allow_pickle=False)
        mean = scaling["mean"].astype(np.float64)
        sd = scaling["sd"].astype(np.float64)
        active = scaling["active"].astype(bool)
        z = ((coefficients.astype(np.float64) - mean) /
             np.where(active, sd, 1.0))
        z[:, ~active] = 0.0
        z = z.astype(np.float32)
        abs_max = float(np.max(np.abs(z)))
        if not np.isfinite(z).all() or abs_max > 100.0:
            raise RuntimeError(
                f"External coefficient scale failure for {modality}: {abs_max}")
        normalized.append(torch.as_tensor(z, dtype=torch.float32))
        diagnostics.append({
            "modality": modality,
            "input_path": str(path),
            "input_sha256": sha256(path),
            "raw_shape": list(matrix.shape),
            "active_directions": int(np.sum(active)),
            "standardized_abs_max": abs_max,
            "solver": solver,
        })

    architecture = json.loads((source / "model_architecture.json").read_text())
    model = sbf.SupervisedFLICAmodel(
        architecture["standardized_coefficient_dimensions"],
        architecture["components"], architecture["targets"],
        dropout=architecture["dropout"], device=torch.device("cpu")).to("cpu")
    checkpoint = source / "selected_model_state.pth"
    model.load_state_dict(torch.load(
        checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        _, _, latent, prediction_z, _ = model(normalized, "cpu")
    target_norm = np.load(source / "target_normalization.npz", allow_pickle=False)
    prediction = (prediction_z.cpu().numpy().astype(np.float64) *
                  target_norm["sd"] + target_norm["mean"])
    latent = latent.cpu().numpy().astype(np.float64)
    np.savetxt(args.output / "predicted_cdrsb.csv", prediction, delimiter=",")
    np.savetxt(args.output / "component_loadings.csv", latent, delimiter=",")
    provenance = {
        "status": "PASS",
        "inference_only": True,
        "dictionary_refit": False,
        "normalization_refit": False,
        "neural_retraining": False,
        "checkpoint_reselection": False,
        "round": manifest["round"],
        "seed": manifest["seed"],
        "selected_epoch": manifest["selected_epoch"],
        "participants": n,
        "bundle_manifest_sha256": sha256(manifest_path),
        "checkpoint_sha256": sha256(checkpoint),
        "projection_diagnostics": diagnostics,
        "outputs": {
            "prediction_shape": list(prediction.shape),
            "loading_shape": list(latent.shape),
        },
    }
    (args.output / "projection_provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    (args.output / "_SUCCESS").write_text("locked external projection PASS\n")
    print(json.dumps(provenance, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

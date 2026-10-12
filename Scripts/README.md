# Scripts

See the main README for installation, input preparation, and full run commands.

## Typical workflow

| Step | Script | What it does |
|---|---|---|
| 1. Train | `run_end_to_end.py` | Trains SuperBigFLICA on the training set and selects the best model using the validation set. |
| 2. Check | `validate_and_summarize.py` | After all training rounds finish, checks their outputs and summarizes performance. |
| 3. Test | `evaluate_locked_test.py` | Evaluates the selected model once on the held-out test set. |
|  | `summarize_locked_test.py` | Summarizes the test-set results. |
| 4. Apply to new data | `project_external_cohort.py` | Applies a trained model to an external dataset. |

## Running on a cluster

- `RUN_SBF_sbatch.sh`: SLURM script that runs all ten training rounds. Set the paths described in the main README before submitting.

## Supporting files

- `sbf_utils.py`: core model, data loading, and projection functions used by the scripts above.
- `run_loss_ablation.py`, `run_nested_cv.py`: helper code used by `run_end_to_end.py`. They also contain experimental options that aren't needed for standard use.
- `SBF.py`: older entry point kept for compatibility; accepts the same options as `run_end_to_end.py`.
- `smoke_test.py`: quick check on synthetic data that the installation works.
- `SOURCE_HASHES.json`: file checksums for verifying the code version.

This folder contains code only; no data or results are included.

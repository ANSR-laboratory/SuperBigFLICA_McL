# SBF Data Fusion

This repository contains scripts for running **SuperBigFLICA (SBF)** data fusion analyses.

## Files
- **SBF.py** – main script for running data fusion.
- **sbf_utils.py** – helper functions used by `SBF.py`.
- **RUN_SBF_sbatch.sh** – SLURM batch script for running jobs on an HPC cluster.

## Usage
### Local run
```bash
python SBF.py
```

### Slurm cluster run
```bash
sbatch RUN_SBF_sbatch.sh
```

## Notes
- `SBF.py` uses `sbf_utils.load_nidp_csv` (`np.genfromtxt`) to load CSVs and allows empty cells (filled as `NaN`), which are filtered during processing.
- Device selection uses `sbf_utils.select_device()` and can be overridden with `SBF_DEVICE` (e.g., `export SBF_DEVICE=cuda`).
  - Auto-selection order: `cuda` → `mps` → `cpu` (falls back to CPU if GPU is unavailable).
  - Valid device options: `cpu`, `cuda`, `mps`.
- Set `save_all_epochs` in `SBF.py` to `True` to save per-epoch correlation metrics, spatial maps, and latent loadings under `output_dir/epoch_details`.
- On clusters, load the appropriate CUDA toolkit module (for example: `module load shared` and `module load cuda11.8/toolkit/11.8.0`).
- If deterministic algorithms are enabled, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` (or `:16:8`) before running with CUDA.

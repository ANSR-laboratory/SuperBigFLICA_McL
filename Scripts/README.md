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
- `SBF.py` uses `np.genfromtxt` to load CSVs and allows empty cells (filled as `NaN`), which are filtered during processing.
- Set `save_all_epochs` in `SBF.py` to `True` to save per-epoch correlation metrics and spatial maps under `output_dir/epoch_details`.

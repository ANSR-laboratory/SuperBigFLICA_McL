# SuperBigFLICA

SuperBigFLICA is a semi-supervised framework for multimodal neuroimaging data fusion. It learns shared latent components and modality-specific spatial maps while predicting a clinical or behavioral target.

The current pipeline learns fresh dictionaries, uses fixed equal loss weights, and selects a model checkpoint using validation R² with prediction-quality checks. Training and held-out test evaluation are separate steps. Saved dictionaries and preprocessing state support projection of aligned external cohorts.

**Versions.** This is the current version of the pipeline, v2.0 (October 2026); see [CHANGELOG.md](CHANGELOG.md) for changes.
For the repository snapshot from the period of the 2026 DADM publication, see tag [v1.0-DADM-2026](https://github.com/ANSR-laboratory/SuperBigFLICA_McL/tree/v1.0-DADM-2026).
A correction to the published results is in preparation; this page will link to it once available.

## Installation

Use Python 3.11 in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt         # Linux
# On macOS, use: pip install -r requirements.macos.txt
```

Install FSL and FreeSurfer, then configure their paths:

```bash
export FSLDIR=/path/to/fsl
export FREESURFER_HOME=/path/to/freesurfer
```

The loader uses the FreeSurfer `fsaverage` cortex support. Run `python Scripts/run_end_to_end.py --help` to see the training options.

## Prepare inputs

The supplied workflow is configured for five ADNI modalities: cortical thickness (CT), tau PET (TAU), pial surface area (PSA), gray matter (GM), and amyloid PET (AMY). It expects 192 training, 41 validation, and 41 test participants.

Create a training view containing ten folders, one per modality and split:

```text
train_validation_view/
├── CT_train/           CT_validation/
├── TAU_train/          TAU_validation/
├── PSA_train/          PSA_validation/
├── GM_train/           GM_validation/
└── AMY_train/          AMY_validation/
```

CT and PSA use paired left/right hemisphere MGH files; TAU, GM, and AMY use NIfTI files. Participant columns must follow the same order across all modalities and their corresponding headerless numeric target CSVs. Keep test folders outside the training view.

## Run training

```bash
python Scripts/run_end_to_end.py \
  --round 1 --seed 555 --output /path/to/run/results/round1_seed555 \
  --input-view /path/to/train_validation_view \
  --target-train /path/to/nIDP_CDRSB_matrix_training.csv \
  --target-validation /path/to/nIDP_CDRSB_matrix_validation.csv \
  --dictionary-dim 250 --epochs 120 --device cpu
```

The workflow uses 50 latent components and trains through epochs 0–120. The output directory must be new. `Scripts/SBF.py` accepts the same command-line arguments as a compatibility entry point.

For all ten rounds on a SLURM cluster, activate your environment, configure FSL/FreeSurfer, and set these absolute paths:

```bash
export SBF_RUN_ROOT=/path/to/run
export SBF_INPUT_VIEW=/path/to/train_validation_view
export SBF_TARGET_ROOT=/path/to/target_csvs
export SBF_SCRIPTS_DIR=/path/to/SuperBigFLICA_McL/Scripts
mkdir -p logs
sbatch Scripts/RUN_SBF_sbatch.sh
```

The array uses seeds 555, 666, 777, 888, 999, 1111, 2222, 3333, 4444, and 5555. Adjust cluster resource directives for your environment.

## Validate and evaluate

After all ten rounds finish, validate their outputs:

```bash
python Scripts/validate_and_summarize.py /path/to/run
```

After validation passes, evaluate a frozen model on the test set:

```bash
python Scripts/evaluate_locked_test.py \
  --root /path/to/run --round 1 \
  --input-view /path/to/all_split_view \
  --target-root /path/to/target_csvs
```

The evaluation view must contain all fifteen modality/split folders (`_train`, `_validation`, and `_test`). The target folder must contain `nIDP_CDRSB_matrix_training.csv`, `nIDP_CDRSB_matrix_validation.csv`, and `nIDP_CDRSB_matrix_test.csv`. Use validation data for tuning and checkpoint selection; reserve test data for evaluation.

For external-cohort inference, run `python Scripts/project_external_cohort.py --help`. Supply aligned participant-by-feature matrices and the saved projection bundle.

A quick synthetic check is available without participant data:

```bash
python Scripts/smoke_test.py
```

## Outputs and privacy

Each training round saves configuration records, the selected checkpoint, dictionaries and projection state, latent components, predictions, training history, and modality-specific component maps. Separate test evaluation saves test loadings, predictions, and aggregate performance metrics.

Spatial maps are raw backprojected regression maps, not standardized z maps or significance maps.

This repository distributes code, not participant data or trained checkpoints. Keep inputs, participant identifiers, loading/prediction tables, arrays, and model outputs in private storage. `.gitignore` excludes common data and output formats, checkpoints, runtime logs, and `.claude/`. Review files before committing: ignore rules do not remove files already tracked by Git.

## Citation

SuperBigFLICA was originally proposed in:

> **Supervised Phenotype Discovery From Multimodal Brain Imaging**
> Weikang Gong, Shuang Bai, Yong-Qiang Zheng, Stephen M. Smith, Christian F. Beckmann
> *IEEE Transactions on Medical Imaging*, 42(3):834–849, 2023. https://doi.org/10.1109/TMI.2022.3218720

```bibtex
@article{gong2023supervised,
  title={Supervised Phenotype Discovery From Multimodal Brain Imaging},
  author={Gong, Weikang and Bai, Shuang and Zheng, Yong-Qiang and Smith, Stephen M. and Beckmann, Christian F.},
  journal={IEEE Transactions on Medical Imaging},
  volume={42},
  number={3},
  pages={834--849},
  year={2023},
  doi={10.1109/TMI.2022.3218720}
}
```

If you use this codebase in your research, please also cite:

> **Investigating the Amyloid-Tau-Neurodegeneration Framework in Alzheimer's Disease Using Semi-Supervised Multimodal Imaging Data Fusion**
> You Cheng, Adrián Medina, Cole Korponay, Christian F. Beckmann, David Harper, Lisa Nickerson, and the Alzheimer's Disease Neuroimaging Initiative
> *Alzheimer's & Dementia: Diagnosis, Assessment & Disease Monitoring*, 2026. https://doi.org/10.1002/dad2.70360

```bibtex
@article{cheng2025sbf,
  title={Investigating the Amyloid-Tau-Neurodegeneration Framework in Alzheimer's Disease Using Semi-Supervised Multimodal Imaging Data Fusion},
  author={Cheng, You and Medina, Adri{\'a}n and Korponay, Cole and Beckmann, Christian F. and Harper, David and Nickerson, Lisa and {Alzheimer's Disease Neuroimaging Initiative}},
  journal={Alzheimer's \& Dementia: Diagnosis, Assessment \& Disease Monitoring},
  year={2026},
  doi={10.1002/dad2.70360},
  url={http://dx.doi.org/10.1002/dad2.70360}
}
```


## License

MIT — see [LICENSE](LICENSE).

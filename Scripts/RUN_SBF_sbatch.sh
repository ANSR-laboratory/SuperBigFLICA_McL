#!/bin/bash
#SBATCH --job-name=SBF_v2
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-9%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
: "${SBF_RUN_ROOT:?Set output run root}"
: "${SBF_INPUT_VIEW:?Set train/validation-only imaging view}"
: "${SBF_TARGET_ROOT:?Set target CSV folder}"
: "${SBF_SCRIPTS_DIR:?Set absolute path to repository Scripts}"
SEEDS=(555 666 777 888 999 1111 2222 3333 4444 5555)
INDEX=${SLURM_ARRAY_TASK_ID}
SEED=${SEEDS[$INDEX]}
ROUND=$((INDEX + 1))
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 PYTHONHASHSEED=${SEED}
python3 -u "${SBF_SCRIPTS_DIR}/run_end_to_end.py"   --round "${ROUND}" --seed "${SEED}"   --output "${SBF_RUN_ROOT}/results/round${ROUND}_seed${SEED}"   --input-view "${SBF_INPUT_VIEW}"   --target-train "${SBF_TARGET_ROOT}/nIDP_CDRSB_matrix_training.csv"   --target-validation "${SBF_TARGET_ROOT}/nIDP_CDRSB_matrix_validation.csv"   --dictionary-dim 250 --alpha 0.1 --ridge-factor 1e-4   --inactive-sd-ratio 1e-3 --projection-max-iter 5000   --projection-tol 1e-5 --epochs 120 --lr 0.01   --batch-size 32 --dropout 0.2 --device cpu

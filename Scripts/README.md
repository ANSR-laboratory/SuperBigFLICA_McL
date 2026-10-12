# August pipeline entry points

See the root README for input alignment, version boundaries, installation and run commands.

- run_end_to_end.py: fresh dictionaries and validation-selected training.
- sbf_utils.py: August model, projection and loader utilities.
- run_loss_ablation.py / run_nested_cv.py: imported training support; their additional experimental entry points are not the main run recipe.
- validate_and_summarize.py: ten-round audit before locked test evaluation.
- evaluate_locked_test.py / summarize_locked_test.py: frozen test evaluation and aggregate summary.
- project_external_cohort.py: saved-bundle inference on aligned external matrices.
- smoke_test.py: existing synthetic selection-gate checks.
- SBF.py: compatibility alias accepting run_end_to_end CLI flags.
- RUN_SBF_sbatch.sh: portable ten-round CPU submission recipe.
- AUGUST_SOURCE_HASHES.json: exact archived source identities.

Scientific Python sources were copied unchanged from the August execution archive; wrapper paths are configurable. No data or results are included.

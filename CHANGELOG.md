# Changelog

## v2.0 — October 2026 (current)

**Bug fixes**
- Training, validation, and test data are now all encoded against the same final dictionary (previously, training data used an earlier dictionary state).
- Held-out data are encoded with the same sparse-coding solver as training data (previously a short 20-step approximation).

**Method changes**
- Initialization: the model is initialized with multimodal MIGP (mMIGP), following Gong et al.'s released implementation, instead of random initialization.
- Loss weighting: fixed equal weights for the four loss terms, instead of weights learned adaptively during training.
- Model selection: the final model is chosen by validation R², with checks on fit, calibration, and prediction range (previously validation correlation).
- Training: 120 epochs at learning rate 0.01 with a cosine schedule (previously 30 epochs at 0.05).
- Cortical surface data: vertices are selected using the FreeSurfer fsaverage cortex labels.

**New features**
- Separate steps for training, validating all runs, one-time test evaluation, and applying a trained model to new cohorts.
- Fitted dictionaries and preprocessing are saved, so trained models can be applied to new data without refitting.
- Expanded performance metrics (R², MAE, RMSE, calibration).

## v1.0-DADM-2026 — May 2026
Repository snapshot from the period of the DADM 2026 publication (Cheng et al., 2026, doi:10.1002/dad2.70360). Exact match to the paper's original run is not verified.

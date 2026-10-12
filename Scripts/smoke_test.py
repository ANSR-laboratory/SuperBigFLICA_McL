#!/usr/bin/env python3
"""Fast pre-submission checks that do not access imaging or targets."""
import numpy as np

import run_end_to_end as workflow


train = {"r2_conventional": 0.8}
validation = {
    "r": 0.7,
    "r2_conventional": 0.5,
    "rmse": 1.0,
    "prediction_sd_ratio": 0.8,
    "calibration_slope_y_on_prediction": 1.1,
}
prediction = np.asarray([[1.0], [2.0]])
passed, audit = workflow.prediction_selection_gate(
    train, validation, prediction, np.asarray([-10.0]), np.asarray([20.0]))
assert passed and all(audit.values())
bad = dict(validation, prediction_sd_ratio=0.1)
passed, audit = workflow.prediction_selection_gate(
    train, bad, prediction, np.asarray([-10.0]), np.asarray([20.0]))
assert not passed and not audit["validation_prediction_sd_ratio_in_0p25_to_2"]
print("REMOTE_IMPORT_AND_GATE_PASS")

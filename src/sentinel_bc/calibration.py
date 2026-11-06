"""Sigmoid calibration fits on its own held-out development partition."""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


def calibrate(estimator, x, y):
    if not np.array_equal(estimator.classes_, [0, 1]):
        raise ValueError("Expected classes [BENIGN=0, MALIGNANT=1]")
    calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid").fit(x, y)
    if not np.array_equal(calibrated.classes_, [0, 1]):
        raise ValueError("Calibration changed class mapping")
    return calibrated


"""MAPIE v1 split conformal sets on an independent conformalization set."""
import numpy as np
from mapie.classification import SplitConformalClassifier


def conformalize(calibrated, x, y, confidence=.90):
    cp = SplitConformalClassifier(estimator=calibrated, confidence_level=confidence,
                                  conformity_score="lac", prefit=True, random_state=42)
    cp.conformalize(x, y)
    return cp


def prediction_sets(cp, x):
    _, sets = cp.predict_set(x)
    return np.asarray(sets[:, :, 0], dtype=bool)


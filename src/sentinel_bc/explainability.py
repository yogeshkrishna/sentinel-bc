"""Bounded SHAP attribution for the calibrated malignant probability."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


DESCRIPTIONS = {
    "radius": "Average distance from the nucleus center to its perimeter",
    "texture": "Variation in image gray levels",
    "perimeter": "Length of the nucleus boundary",
    "area": "Area enclosed by the nucleus boundary",
    "smoothness": "Local variation in radius lengths",
    "compactness": "Boundary compactness derived from perimeter and area",
    "concavity": "Severity of inward boundary curves",
    "concave points": "Number of inward boundary portions",
    "symmetry": "Nucleus symmetry measurement",
    "fractal dimension": "Boundary complexity measurement",
}


def explain(engine, frame, max_evals=183):
    import shap
    features = engine.features
    background = engine.bundle["background"]
    def probability(values):
        return engine.bundle["calibrated"].predict_proba(pd.DataFrame(values, columns=features))[:, 1]
    explainer = shap.Explainer(probability, background, algorithm="permutation", seed=42)
    values = explainer(frame, max_evals=max_evals, silent=True)
    return values


def local_table(engine, frame):
    values = explain(engine, frame.iloc[:1])
    out = pd.DataFrame({"measurement": engine.features, "value": frame.iloc[0].to_numpy(),
                        "contribution_to_p_malignant": values.values[0]})
    out["direction"] = np.where(out.contribution_to_p_malignant >= 0, "toward malignant", "toward benign")
    out["meaning"] = [next((v for k, v in DESCRIPTIONS.items() if k in f), "Morphology measurement") for f in engine.features]
    return out.iloc[np.argsort(-np.abs(out.contribution_to_p_malignant))].reset_index(drop=True)


def global_permutation(engine, x, y):
    importance = permutation_importance(engine.bundle["calibrated"], x, y, n_repeats=8,
                                         random_state=42, scoring="balanced_accuracy", n_jobs=1)
    return pd.DataFrame({"measurement": engine.features, "mean_accuracy_drop": importance.importances_mean,
                         "std": importance.importances_std}).sort_values("mean_accuracy_drop", ascending=False)


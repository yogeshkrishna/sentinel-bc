"""The bundled UCI WDBC data, explicitly remapped to malignant = 1."""
import hashlib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    data = load_breast_cancer(as_frame=True)
    assert list(data.target_names) == ["malignant", "benign"]
    x = data.data.copy()
    y = (data.target.to_numpy() == 0).astype(np.int64)
    if x.duplicated().any():
        raise ValueError("Duplicate measurements require grouped partitioning")
    return x, y


def data_digest(x, y) -> str:
    return hashlib.sha256(np.asarray(x, dtype="<f8").tobytes() + np.asarray(y, dtype="<i8").tobytes()).hexdigest()


def partitions(y: np.ndarray, config: dict) -> dict[str, np.ndarray]:
    seed = config["seed"]
    dev, test = train_test_split(np.arange(len(y)), test_size=config["test_fraction"], stratify=y, random_state=seed)
    fit, policy = train_test_split(dev, test_size=config["policy_size"], stratify=y[dev], random_state=seed + 1)
    fit, conformal = train_test_split(fit, test_size=config["conformal_size"], stratify=y[fit], random_state=seed + 2)
    fit, calibration = train_test_split(fit, test_size=config["calibration_size"], stratify=y[fit], random_state=seed + 3)
    return dict(fit=fit, calibration=calibration, conformal=conformal, policy=policy, test=test)


def describe_data(x, y) -> dict:
    corr = x.corr().abs()
    pairs = [(x.columns[i], x.columns[j], float(corr.iloc[i, j]))
             for i in range(30) for j in range(i) if corr.iloc[i, j] > .95]
    return {"source": "UCI WDBC ID 17 via sklearn.datasets.load_breast_cancer",
            "source_url": "https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic",
            "sha256": data_digest(x, y), "n": len(y), "n_features": 30,
            "class_mapping": {"0": "BENIGN", "1": "MALIGNANT"},
            "benign": int(sum(y == 0)), "malignant": int(sum(y == 1)),
            "missing_values": int(x.isna().sum().sum()), "duplicate_rows": int(x.duplicated().sum()),
            "correlation_above_095": pairs,
            "related_sample_limit": "Patient linkage is unavailable in the bundled data; independence beyond exact duplicates cannot be established.",
            "feature_summary": x.describe().to_dict()}


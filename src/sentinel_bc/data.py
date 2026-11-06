"""The bundled UCI WDBC data, explicitly remapped to malignant = 1."""
import hashlib
from dataclasses import dataclass
from pathlib import Path
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


def partitions(y: np.ndarray, config: dict, locked_test=None) -> dict[str, np.ndarray]:
    seed = config["seed"]
    if locked_test is None:
        dev, test = train_test_split(np.arange(len(y)), test_size=config["test_fraction"], stratify=y, random_state=seed)
    else:
        test = np.asarray(locked_test)
        dev = np.setdiff1d(np.arange(len(y)), test)
    fit, policy = train_test_split(dev, test_size=config["policy_size"], stratify=y[dev], random_state=seed + 1)
    fit, conformal = train_test_split(fit, test_size=config["conformal_size"], stratify=y[fit], random_state=seed + 2)
    fit, calibration = train_test_split(fit, test_size=config["calibration_size"], stratify=y[fit], random_state=seed + 3)
    return dict(fit=fit, calibration=calibration, conformal=conformal, policy=policy, test=test)


@dataclass
class TrainingData:
    x: pd.DataFrame
    y: np.ndarray
    provenance: dict
    config: dict
    split: dict


def row_digests(x):
    # Eight decimals preserve WDBC's recorded precision while recognizing CSV
    # round-trip differences. IDs and row order do not affect corpus identity.
    return [hashlib.sha256(row.tobytes()).hexdigest()
            for row in np.asarray(np.round(x, 8), dtype="<f8")]


def corpus_digest(x, y):
    records = sorted(f"{h}:{int(label)}" for h, label in zip(row_digests(x), y))
    return hashlib.sha256("\n".join(records).encode()).hexdigest()


def load_training_data(source=None, config=None, mode="quick") -> TrainingData:
    """Validate one supervised WDBC-schema corpus before any model is fitted."""
    from copy import deepcopy
    from .config import load_config
    config = deepcopy(config or load_config())
    if mode not in ("quick", "full"):
        raise ValueError("Choose Quick or Full training")
    builtin_x, builtin_y = load_data()
    features = list(builtin_x.columns)
    excluded = []
    if source is None:
        x, y = builtin_x, builtin_y
    else:
        path = Path(source)
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("Training CSV exceeds 10 MiB")
        try:
            import csv
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream, strict=True)
                headers = next(reader)
                if not headers or len(set(headers)) != len(headers):
                    raise ValueError("The CSV header is empty or contains duplicate column names")
                for row in reader:
                    if row and len(row) != len(headers):
                        raise ValueError(f"CSV line {reader.line_num} does not match the header column count")
            frame = pd.read_csv(path, float_precision="round_trip")
        except (UnicodeError, csv.Error, pd.errors.ParserError, StopIteration) as exc:
            raise ValueError("Upload a UTF-8 CSV with a header row") from exc
        targets = [c for c in ("label", "diagnosis", "target") if c in frame]
        if not targets:
            raise ValueError("Supervised training needs label, diagnosis or target. Use unlabeled CSVs in Predict.")
        missing = sorted(set(features) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(features) - set(targets) - {"sample_id", "id"})
        if missing or extra:
            raise ValueError(f"Expected exactly 30 WDBC measurements. Missing: {', '.join(missing) or 'none'}. Unrecognized: {', '.join(extra) or 'none'}.")
        mapping = {"0": 0, "0.0": 0, "B": 0, "BENIGN": 0,
                   "1": 1, "1.0": 1, "M": 1, "MALIGNANT": 1}
        labels = []
        for name in targets:
            normalized = frame[name].astype(str).str.strip().str.upper().map(mapping)
            if normalized.isna().any():
                raise ValueError(f"Invalid {name} labels. Use 0/B/BENIGN or 1/M/MALIGNANT; malignant is always 1.")
            labels.append(normalized.to_numpy(dtype=np.int64))
        if any(not np.array_equal(labels[0], other) for other in labels[1:]):
            raise ValueError("Target columns disagree. diagnosis, label and target must describe identical classes.")
        y = labels[0]
        try:
            x = frame.loc[:, features].astype(float)
        except (ValueError, TypeError) as exc:
            raise ValueError("All 30 measurements must be numeric") from exc
        excluded = targets + [c for c in ("sample_id", "id") if c in frame]
    if not 350 <= len(x) <= 5000:
        raise ValueError(f"Found {len(x)} samples. Training requires 350–5000 rows to support five disjoint data roles and bounded local computation.")
    if x.isna().any().any():
        raise ValueError(f"Found {int(x.isna().sum().sum())} missing measurements. Supply complete rows; labels and missing inputs are never silently filled.")
    if not np.isfinite(x.to_numpy()).all() or (x.to_numpy() < 0).any() or (x.to_numpy() > 1e12).any():
        raise ValueError("Measurements must be finite, nonnegative source values no larger than 1e12")
    hashes = row_digests(x)
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate measurement records found (including conflicting labels). Remove duplicates or provide a grouped-study protocol.")
    counts = np.bincount(y, minlength=2)
    if min(counts) < 70:
        raise ValueError(f"Insufficient class representation: benign={counts[0]}, malignant={counts[1]}. At least 70 of each class is required.")
    builtin_hashes = row_digests(builtin_x)
    builtin_map = dict(zip(builtin_hashes, builtin_y))
    same = len(x) == len(builtin_x) and set(hashes) == set(builtin_hashes)
    if same and any(builtin_map[h] != label for h, label in zip(hashes, y)):
        raise ValueError("This WDBC copy conflicts with the official labels; check for inverted target mapping")
    benchmark_config = load_config() | {"seed": 42, "test_fraction": .20}
    benchmark_split = partitions(builtin_y, benchmark_config)
    locked_hashes = {builtin_hashes[i] for i in benchmark_split["test"]}
    if not same and locked_hashes.intersection(hashes):
        raise ValueError("This mixed/subset CSV contains known benchmark test observations. They cannot enter a new fitting pool. Use the complete WDBC corpus or an independent compatible dataset.")
    if same:
        # Canonical order and the historical test observations survive shuffled
        # uploads and changed development seeds. This is repeated WDBC research.
        x, y = builtin_x, builtin_y
    else:
        config["calibration_size"] = max(40, round(len(x) * .10))
        config["conformal_size"] = max(40, round(len(x) * .10))
        config["policy_size"] = max(40, round(len(x) * .12))
    try:
        preserve = same and (config["seed"] != 42 or config["test_fraction"] != .20)
        split = partitions(y, config, benchmark_split["test"] if preserve else None)
    except ValueError as exc:
        raise ValueError("Not enough observations for stratified fitting, calibration, conformalization, policy and test partitions") from exc
    for role, idx in split.items():
        minimum = 30 if role == "fit" else 8
        if min(np.bincount(y[idx], minlength=2)) < minimum:
            raise ValueError(f"The {role} partition needs at least {minimum} observations of each class. Add more minority-class examples.")
    if len(split["fit"]) < 120 or config[mode]["folds"] != 3:
        raise ValueError("The experiment needs at least 120 fitting rows and the existing three-fold protocol")
    provenance = {"source": "built-in WDBC" if source is None else "uploaded compatible CSV",
                  "fingerprint": corpus_digest(x, y), "n": len(y), "n_features": 30,
                  "benign": int(sum(y == 0)), "malignant": int(sum(y == 1)),
                  "missing_values": 0, "duplicate_rows": 0, "excluded_columns": excluded,
                  "same_underlying_wdbc": same, "benchmark_feature_overlap": len(set(hashes) & set(builtin_hashes)),
                  "interpretation": "Repeated WDBC experiment; not independent external validation" if same else "User-supplied labels and measurement provenance are unverified; this is an internal held-out experiment, not established external clinical validation.",
                  "class_mapping": {"0": "BENIGN", "1": "MALIGNANT"}}
    return TrainingData(x.reset_index(drop=True), y, provenance, config, split)


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


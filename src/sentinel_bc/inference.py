"""Version-checked offline inference. No implicit fitting or calibration."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import threading
import joblib
import numpy as np
import pandas as pd
from .config import ROOT
from .preprocessing import validate_frame
from .conformal import prediction_sets
from .uncertainty import signals
from .safety_gate import apply_policy, SafetyPolicy


class Engine:
    def __init__(self, artifact_dir: str | Path = ROOT / "artifacts/full"):
        self.directory = Path(artifact_dir)
        manifest = self.directory / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError("Training artifacts are missing. Run: python train.py --mode full")
        self.metadata = json.loads(manifest.read_text(encoding="utf-8"))
        for name, digest in self.metadata["files"].items():
            path = self.directory / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Missing or modified model component: {name}")
        for package, version in self.metadata["runtime_versions"].items():
            if importlib.metadata.version(package) != version:
                raise ValueError(f"Artifact requires {package}=={version}; install requirements-lock.txt")
        # joblib is executable serialization: load only your own trusted local run.
        self.bundle = joblib.load(self.directory / "bundle.joblib")
        required = {"base", "calibrated", "conformal", "neural", "screen", "policy", "features", "version"}
        if not required.issubset(self.bundle):
            raise ValueError("Incomplete artifact bundle")
        if self.bundle["version"] != self.metadata["version"]:
            raise ValueError("Mixed model versions")
        if self.metadata["class_mapping"] != {"0": "BENIGN", "1": "MALIGNANT"}:
            raise ValueError("Invalid class mapping")
        for key in ["base", "calibrated", "neural"]:
            if not np.array_equal(self.bundle[key].classes_, [0, 1]):
                raise ValueError(f"Wrong class order in {key}")
        self.policy = SafetyPolicy(**self.bundle["policy"])
        self.features = self.bundle["features"]
        if self.features != self.metadata["features"] or len(set(self.features)) != 30:
            raise ValueError("Invalid feature schema")
        self._lock = threading.RLock()

    def predict(self, frame: pd.DataFrame) -> list[dict]:
        x = validate_frame(frame, self.features, self.metadata["config"]["max_batch_rows"])
        valid, ood, distance = self.bundle["screen"].inspect(x)
        # Invalid rows are represented without feeding nonfinite values to a model.
        output = [None] * len(x)
        for i in np.flatnonzero(~valid):
            output[i] = {"model_version": self.metadata["version"], "decision": "OUT_OF_SCOPE_INPUT",
                         "reasons": ["INVALID_INPUT"], "p_malignant": None, "p_benign": None,
                         "prediction_set": [], "uncertainty": {}, "ood": True,
                         "notice": "Research only. Never clinical clearance."}
        idx = np.flatnonzero(valid)
        if len(idx):
            clean = x.iloc[idx]
            with self._lock:
                p = self.bundle["calibrated"].predict_proba(clean)[:, 1]
                sets = prediction_sets(self.bundle["conformal"], clean)
                u = signals(p, self.bundle["neural"], clean, self.metadata["config"]["mc_passes"])
            decisions = apply_policy(p, u, sets, valid[idx], ood[idx], self.policy)
            for j, i in enumerate(idx):
                output[i] = {"model_version": self.metadata["version"], "decision": decisions[j][0], "reasons": decisions[j][1],
                             "predicted_class": "MALIGNANT" if p[j] >= .5 else "BENIGN",
                             "p_malignant": float(p[j]), "p_benign": float(1-p[j]),
                             "prediction_set": [name for k, name in enumerate(["BENIGN", "MALIGNANT"]) if sets[j, k]],
                             "uncertainty": {k: float(v[j]) for k, v in u.items()},
                             "ood": bool(ood[i]), "ood_distance": float(distance[i]),
                             "notice": "Research only. Never clinical clearance."}
        return output

    def predict_csv(self, path):
        if Path(path).stat().st_size > self.metadata["config"]["max_upload_bytes"]:
            raise ValueError("CSV exceeds the 1 MiB upload limit")
        # pandas renames duplicate headers, so also reject them via schema validation.
        return self.predict(pd.read_csv(path))


"""Quick genuine training in CI; full artifacts are verified locally when present."""
from pathlib import Path
import json
import shutil
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sentinel_bc.config import ROOT
from sentinel_bc.data import load_data, partitions
from sentinel_bc.config import load_config
from sentinel_bc.preprocessing import transformer
from sentinel_bc.calibration import calibrate
from sentinel_bc.conformal import conformalize, prediction_sets
from sentinel_bc.models import train_network, NeuralPredictor
from sentinel_bc.inference import Engine


def test_real_training_calibration_conformal_serialization(tmp_path):
    x, y = load_data()
    roles = partitions(y, load_config())
    a, b, c = [roles[k] for k in ["fit", "calibration", "conformal"]]
    model = make_pipeline(transformer(), LogisticRegression(max_iter=2000)).fit(x.iloc[a], y[a])
    cal = calibrate(model, x.iloc[b], y[b])
    cp = conformalize(cal, x.iloc[c], y[c])
    p = cal.predict_proba(x.iloc[c])
    np.testing.assert_allclose(p.sum(axis=1), 1)
    assert np.array_equal(cal.classes_, [0, 1])
    sets = prediction_sets(cp, x.iloc[c[:3]])
    assert sets.shape == (3, 2)
    joblib.dump((cal, cp), tmp_path / "components.joblib")
    cal2, cp2 = joblib.load(tmp_path / "components.joblib")
    np.testing.assert_allclose(cal2.predict_proba(x.iloc[c]), p)
    assert np.array_equal(prediction_sets(cp2, x.iloc[c[:3]]), sets)


def test_keras_training_and_stable_mc(tmp_path):
    x, y = load_data()
    a = partitions(y, load_config())["fit"]
    prep, network, history = train_network(x.iloc[a], y[a], 42, epochs=2, patience=2)
    assert len(history["loss"]) == 2
    adapter = NeuralPredictor([(prep, network)])
    rows = x.iloc[[0, 19]]
    p = adapter.predict_proba(rows)
    assert p.shape == (2, 2) and np.array_equal(adapter.classes_, [0, 1])
    batch = adapter.mc_probabilities(rows, 8)
    single = adapter.mc_probabilities(rows.iloc[:1], 8)
    np.testing.assert_allclose(batch[:, :1], single, atol=1e-7)
    assert np.std(batch[:, 0]) > 0
    joblib.dump(adapter, tmp_path / "neural.joblib")
    restored = joblib.load(tmp_path / "neural.joblib")
    np.testing.assert_allclose(p, restored.predict_proba(rows), atol=1e-7)


def test_missing_artifacts_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py"):
        Engine(tmp_path)


@pytest.fixture
def full_engine():
    if not (ROOT / "artifacts/full/manifest.json").exists():
        pytest.skip("Full experiment artifacts are absent; CI uses small real-training tests")
    return Engine()


def test_full_single_batch_csv_consistency(full_engine, tmp_path):
    x, _ = load_data()
    rows = x.iloc[[0, 19, 30]]
    a = full_engine.predict(rows)
    b = sum([full_engine.predict(rows.iloc[i:i+1]) for i in range(3)], [])
    for one, two in zip(a, b):
        assert one["decision"] == two["decision"] and one["prediction_set"] == two["prediction_set"]
        assert one["p_malignant"] == pytest.approx(two["p_malignant"], abs=1e-6)
        assert one["uncertainty"]["mc_std"] == pytest.approx(two["uncertainty"]["mc_std"], abs=1e-6)
    rows.to_csv(tmp_path / "batch.csv", index=False)
    csv = full_engine.predict_csv(tmp_path / "batch.csv")
    np.testing.assert_allclose([r["p_malignant"] for r in a], [r["p_malignant"] for r in csv], atol=1e-6)


def test_tampering_rejected(full_engine, tmp_path):
    source = ROOT / "artifacts/full"
    shutil.copytree(source, tmp_path / "model")
    with (tmp_path / "model/bundle.joblib").open("ab") as f: f.write(b"corrupt")
    with pytest.raises(ValueError, match="modified"):
        Engine(tmp_path / "model")


def test_full_invalid_ood_and_malignant(full_engine):
    x, _ = load_data()
    row = x.iloc[:1].copy()
    row.iloc[0, 0] = np.inf
    assert full_engine.predict(row)[0]["decision"] == "OUT_OF_SCOPE_INPUT"
    assert full_engine.predict(x.iloc[:1] * 100)[0]["decision"] == "OUT_OF_SCOPE_INPUT"
    for r in full_engine.predict(x.iloc[:20]):
        if r["p_malignant"] >= .5:
            assert r["decision"] != "LOW_RISK_ESTIMATE"


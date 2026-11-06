import numpy as np
import pandas as pd
import pytest
from dataclasses import replace
from sentinel_bc.config import load_config
from sentinel_bc.data import load_data, partitions
from sentinel_bc.evaluation import classification_metrics, selective_metrics, wilson, ece
from sentinel_bc.preprocessing import validate_frame, InputScreen
from sentinel_bc.safety_gate import SafetyPolicy, decide, apply_policy, select_policy, LOW, REVIEW, OUT
from sentinel_bc.uncertainty import entropy
from sentinel_bc.storage import ReviewStore


@pytest.fixture
def policy():
    return SafetyPolicy(.15, .65, .15, .15, True)


U = {"entropy": .2, "disagreement": .02, "mc_std": .02}


def test_dataset_and_canonical_mapping():
    x, y = load_data()
    assert x.shape == (569, 30)
    assert sum(y == 1) == 212 and sum(y == 0) == 357
    assert y[0] == 1 and y[19] == 0
    assert x.columns[0] == "mean radius" and x.columns[-1] == "worst fractal dimension"
    assert x.isna().sum().sum() == 0


def test_disjoint_reproducible_partitions():
    x, y = load_data()
    split = partitions(y, load_config())
    assert {k: len(v) for k, v in split.items()} == dict(fit=273, calibration=57, conformal=57, policy=68, test=114)
    all_idx = np.concatenate(list(split.values()))
    assert len(np.unique(all_idx)) == len(x)
    assert all(np.array_equal(v, partitions(y, load_config())[k]) for k, v in split.items())


@pytest.mark.parametrize("p", [.5, .95, 1.0])
def test_malignant_always_reviewed(policy, p):
    assert decide(p, U, [True, False], True, False, policy)[0] == REVIEW


def test_high_uncertainty_and_sets(policy):
    assert decide(.02, U, [True, False], True, False, policy)[0] == LOW
    for key, value in [("entropy", .9), ("disagreement", .3), ("mc_std", .4)]:
        assert decide(.02, U | {key: value}, [True, False], True, False, policy)[0] == REVIEW
    for s in [[False, True], [True, True], [False, False]]:
        assert decide(.02, U, s, True, False, policy)[0] == REVIEW


@pytest.mark.parametrize("valid,ood", [(False, False), (False, True), (True, True)])
def test_invalid_or_ood_cannot_be_low(policy, valid, ood):
    assert decide(.001, U, [True, False], valid, ood, policy)[0] == OUT


def test_missing_or_nonfinite_signal_fails_closed(policy):
    for p in [np.nan, np.inf, -.1, 1.1]:
        assert decide(p, U, [True, False], True, False, policy)[0] == OUT
    assert decide(.01, {}, [True, False], True, False, policy)[0] == OUT
    assert decide(.01, U, [True, False], True, False, SafetyPolicy())[0] == REVIEW


def test_probability_monotonicity(policy):
    flags = [decide(p, U, [True, False], True, False, policy)[0] != LOW for p in np.linspace(0, 1, 201)]
    assert np.all(np.diff(np.array(flags, dtype=int)) >= 0)


def test_policy_validation():
    for t in [-.01, .51, np.nan]:
        with pytest.raises(ValueError):
            SafetyPolicy(t)


def test_threshold_selection_and_conservative_fallback():
    y, p = np.array([0, 0, 1, 1]), np.array([.01, .02, .20, .90])
    u = {k: np.repeat(v, 4) for k, v in U.items()}
    settings = load_config()["safety"]
    policy, rows = select_policy(y, p, u, np.array([[True, False]]*4), np.ones(4, bool), np.zeros(4, bool), settings)
    assert policy.feasible and policy.max_p_malignant <= .2
    assert max(r["accepted"] for r in rows if r["feasible"]) == 2
    bad, _ = select_policy(np.ones(4), p, u, np.array([[True, False]]*4), np.ones(4, bool), np.zeros(4, bool), settings)
    assert not bad.feasible


def test_csv_schema_and_screen():
    x, _ = load_data()
    assert validate_frame(x.iloc[:2, ::-1], list(x.columns)).equals(x.iloc[:2])
    with pytest.raises(ValueError): validate_frame(x.iloc[:1, :29], list(x.columns))
    with pytest.raises(ValueError): validate_frame(x.iloc[:101], list(x.columns))
    screen = InputScreen().fit(x.iloc[:200])
    valid, ood, _ = screen.inspect(x.iloc[:1] * 100)
    assert valid[0] and ood[0]
    invalid = x.iloc[:1].copy(); invalid.iloc[0, 0] = np.nan
    assert not screen.inspect(invalid)[0][0]


def test_metrics_positive_class_and_empty_acceptance():
    result = classification_metrics(np.array([0, 0, 1, 1]), np.array([.1, .8, .3, .9]))
    assert result["fn"] == 1 and result["fp"] == 1 and result["malignant_recall"] == .5
    assert result["accuracy"] == .5
    assert selective_metrics(np.array([0, 1]), [(REVIEW, []), (REVIEW, [])])["accepted_accuracy"] is None
    assert wilson(0, 0) == [None, None]
    assert ece(np.array([0, 1]), np.array([0., 1.])) == 0
    np.testing.assert_allclose(entropy([0, .5, 1]), [0, 1, 0], atol=1e-9)


def test_persistent_queue_and_audit(tmp_path):
    store = ReviewStore(tmp_path / "queue.db")
    original = {"model_version": "unit-test", "decision": REVIEW, "p_malignant": .9}
    case = store.add(original, {"mean radius": 12})
    store.review(case, "BENIGN", "Teaching override")
    reopened = ReviewStore(tmp_path / "queue.db")
    row = reopened.inspect(case)
    assert row["original_output"] == original
    assert row["reviewer_decision"] == "BENIGN" and len(row["audit"]) == 2
    with reopened.connection() as db:
        with pytest.raises(Exception, match="immutable"):
            db.execute("UPDATE cases SET original_output='{}' WHERE case_id=?", (case,))
    export = pd.read_csv(reopened.export(tmp_path / "export.csv"))
    assert "notes" not in export.columns and "measurements" not in export.columns
    with pytest.raises(ValueError): store.review("missing", "MALIGNANT")


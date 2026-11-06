from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from sentinel_bc.config import load_config
from sentinel_bc.data import load_data, load_training_data, partitions


@pytest.fixture
def csv_frame():
    x, y = load_data()
    frame = x.copy()
    frame["diagnosis"] = np.where(y == 1, "M", "B")
    frame["sample_id"] = [f"public_{i}" for i in range(len(y))]
    return frame


def save(tmp_path, frame):
    path = tmp_path / "input.csv"
    frame.to_csv(path, index=False)
    return path


def test_valid_csv_mapping_and_shuffled_copy(csv_frame, tmp_path):
    dataset = load_training_data(save(tmp_path, csv_frame.sample(frac=1, random_state=12)))
    x, y = load_data()
    assert dataset.x.equals(x) and np.array_equal(dataset.y, y)
    assert dataset.provenance["same_underlying_wdbc"]
    assert dataset.provenance["excluded_columns"] == ["diagnosis", "sample_id"]
    assert dataset.provenance["fingerprint"] == load_training_data().provenance["fingerprint"]
    assert len(dataset.x.columns) == 30
    expected = partitions(y, load_config())
    assert all(np.array_equal(dataset.split[role], idx) for role, idx in expected.items())


def test_csv_rejects_mismatched_row_width(csv_frame, tmp_path):
    path = save(tmp_path, csv_frame)
    lines = path.read_text().splitlines()
    lines[1] = "unexpected," + lines[1]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="column count"):
        load_training_data(path)


@pytest.mark.parametrize("labels", ["numeric", "words", "both"])
def test_equivalent_labels(csv_frame, tmp_path, labels):
    _, y = load_data()
    if labels == "numeric": csv_frame["diagnosis"] = y
    elif labels == "words": csv_frame["diagnosis"] = np.where(y, "malignant", "benign")
    else: csv_frame["label"] = y
    assert np.array_equal(load_training_data(save(tmp_path, csv_frame)).y, y)


@pytest.mark.parametrize("problem,match", [("missing","Missing"),("extra","Unrecognized"),("unlabeled","Supervised"),
    ("badlabel","Invalid"),("conflict","disagree"),("duplicate","Duplicate"),("numeric","numeric"),
    ("missingvalue","missing"),("fewrows","350"),("fewclass","class representation"),("negative","nonnegative")])
def test_invalid_training_data(csv_frame, tmp_path, problem, match):
    frame = csv_frame.copy()
    if problem == "missing": frame = frame.drop(columns=["mean radius"])
    elif problem == "extra": frame["risk_score"] = 1
    elif problem == "unlabeled": frame = frame.drop(columns="diagnosis")
    elif problem == "badlabel": frame.loc[0,"diagnosis"] = "cancer?"
    elif problem == "conflict": frame["label"] = 0
    elif problem == "duplicate": frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "numeric": frame["mean radius"] = frame["mean radius"].astype(object); frame.loc[0,"mean radius"] = "not a number"
    elif problem == "missingvalue": frame.loc[0,"mean radius"] = np.nan
    elif problem == "fewrows": frame = frame.iloc[:100]
    elif problem == "fewclass": frame["diagnosis"] = ["M"]*20 + ["B"]*(len(frame)-20)
    elif problem == "negative": frame.loc[0,"mean radius"] = -1
    with pytest.raises(ValueError, match=match): load_training_data(save(tmp_path, frame))


def test_original_test_rows_never_move_into_fitting(csv_frame, tmp_path):
    config = load_config(); config["seed"] = 88
    changed = load_training_data(save(tmp_path, csv_frame), config)
    original = partitions(load_data()[1], load_config())
    assert set(changed.split["test"]) == set(original["test"])
    assert not set(changed.split["fit"]) & set(original["test"])
    altered = csv_frame.copy()
    altered.loc[0,"mean radius"] += .5
    with pytest.raises(ValueError, match="benchmark test"): load_training_data(save(tmp_path, altered))


def test_independent_compatible_corpus_has_isolated_roles(csv_frame, tmp_path):
    features = list(load_data()[0].columns)
    csv_frame.loc[:, features] *= 1.003  # Controlled fixture; never called external clinical evidence.
    dataset = load_training_data(save(tmp_path, csv_frame))
    assert not dataset.provenance["same_underlying_wdbc"]
    assert dataset.provenance["benchmark_feature_overlap"] == 0
    idx = np.concatenate(list(dataset.split.values()))
    assert len(set(idx)) == len(dataset.x)
    assert dataset.provenance["fingerprint"] != load_training_data().provenance["fingerprint"]

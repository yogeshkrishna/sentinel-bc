import pytest
from sentinel_bc.config import ROOT
from sentinel_bc.ui import build_ui


def test_missing_artifacts_ui_is_explicit(tmp_path):
    app = build_ui(tmp_path / "absent", tmp_path / "review.sqlite3")
    assert "Training artifacts unavailable" in str(app.config)


def test_full_ui_configuration(tmp_path):
    if not (ROOT / "artifacts/full/manifest.json").exists():
        pytest.skip("Full artifacts are not part of lightweight CI")
    app = build_ui(database=tmp_path / "review.sqlite3")
    config = str(app.config)
    for tab in ["Overview", "Prediction Console", "Model Comparison", "Reliability and Safety", "Human Review Queue", "Research and Methodology"]:
        assert tab in config
    for name in ["predict_case", "predict_csv", "record_review", "what_if"]:
        assert name in config

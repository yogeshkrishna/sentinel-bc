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


@pytest.mark.parametrize("bad_value", [-1, None])
def test_missing_and_negative_ui_input_fail_closed(tmp_path, bad_value):
    if not (ROOT / "artifacts/full/manifest.json").exists():
        pytest.skip("Full artifacts are not part of lightweight CI")
    from sentinel_bc.data import load_data
    import gradio as gr
    app = build_ui(database=tmp_path / "review.sqlite3")
    callbacks = {f.api_name: f.fn for f in app.fns.values()}
    values = load_data()[0].iloc[0].tolist()
    values[0] = bad_value
    result = callbacks["predict_case"](*values)
    assert result[2]["decision"] == "OUT_OF_SCOPE_INPUT"
    with pytest.raises(gr.Error, match="Invalid measurements"):
        callbacks["queue_case"](result[4])

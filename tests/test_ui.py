import pandas as pd
import pytest
from sentinel_bc.config import ROOT
from sentinel_bc.ui import build_ui, percent


def test_no_model_still_opens_training_studio(tmp_path):
    app = build_ui(tmp_path / "absent", tmp_path / "review.sqlite3", tmp_path / "runs")
    tabs = [c["props"].get("label") for c in app.config["components"] if c["type"] == "tabitem"]
    assert tabs == ["Train", "Predict", "Review", "Results"]
    assert "No active model" in str(app.config)
    assert "Train Models" in str(app.config)
    with pytest.raises(ValueError, match="No active model"):
        app.studio.predict("Manual", None, None, [1]*30, "")


def test_id_dropdowns_keep_selected_values_when_they_lose_focus(tmp_path):
    app = build_ui(tmp_path / "absent", tmp_path / "review.sqlite3", tmp_path / "runs")
    dropdowns = {c["props"]["label"]: c["props"] for c in app.config["components"] if c["type"] == "dropdown"}
    for label in ("Completed runs · benchmark and your experiments", "Select a case"):
        # Gradio commits the displayed text on blur when custom values are enabled.
        assert dropdowns[label]["allow_custom_value"] is False


@pytest.fixture
def app(tmp_path):
    if not (ROOT / "artifacts/full/manifest.json").exists():
        pytest.skip("Historical artifact integration is verified locally")
    return build_ui(database=tmp_path / "review.sqlite3", runs_root=tmp_path / "runs")


def test_four_tabs_and_collapsed_manual_inputs(app):
    components = app.config["components"]
    assert [c["props"].get("label") for c in components if c["type"]=="tabitem"] == ["Train","Predict","Review","Results"]
    accordion = next(c for c in components if c["props"].get("label")=="Advanced: Enter 30 Measurements Manually")
    assert accordion["props"]["open"] is False
    names = {f.api_name for f in app.fns.values()}
    assert {"start_training","validate_dataset","activate_run","predict_case","record_review","what_if"} <= names
    assert percent(1.0) == "100.00%" and percent(None) == "Not defined"


@pytest.mark.parametrize("bad_value", [-1, None])
def test_invalid_manual_input_fails_closed(app, bad_value):
    from sentinel_bc.data import load_data
    values = load_data()[0].iloc[0].tolist(); values[0] = bad_value
    version = app.studio.manager.active.engine.metadata["version"]
    records, state, report = app.studio.predict("Manual", None, None, values, version)
    assert records[0]["decision"] == "OUT_OF_SCOPE_INPUT"
    assert not records[0]["review_case_id"]
    assert len(app.studio.store.list_cases()) == 0


def test_auto_referral_is_idempotent_and_batch_is_opt_in(app, tmp_path):
    examples = app.studio.examples()
    malignant = examples.loc[examples.example_name.str.startswith("Malignant referral")].iloc[0]
    version = app.studio.manager.active.engine.metadata["version"]
    a = app.studio.predict("Example",None,malignant.example_name,[],version)[0][0]
    b = app.studio.predict("Example",None,malignant.example_name,[],version)[0][0]
    assert a["review_case_id"] == b["review_case_id"]
    assert len(app.studio.store.list_cases()) == 1
    engine = app.studio.manager.active.engine
    path = tmp_path / "batch.csv"
    examples[engine.features].to_csv(path,index=False)
    records, _, _ = app.studio.predict("CSV",str(path),None,[],version)
    assert len(records) == len(examples) and len(app.studio.store.list_cases()) == 1


@pytest.mark.parametrize("submit_label", [False, True])
def test_activation_clears_prediction_state_and_updates_results(app, submit_label):
    callbacks = {f.api_name:f.fn for f in app.fns.values()}
    dropdown = next(c for c in app.config["components"] if c.get("props", {}).get("label", "").startswith("Completed runs"))
    label, run_id = dropdown["props"]["choices"][0]
    selection = label if submit_label else run_id
    if submit_label:
        activation = next(f for f in app.fns.values() if f.api_name == "activate_run")
        assert activation.preprocess is False
        assert "Historical WDBC benchmark" in callbacks["selected_run"](selection)[0]
    result = callbacks["activate_run"](selection)
    assert result[5] is None and result[6] is None and result[8] is None
    assert app.studio.manager.active.engine.metadata["version"] in result[41]
    assert "100.00%" in result[42] and "33.33%" in result[42]
    assert "Model ready" in result[-1] and "Predict" in result[-1]


def test_upload_callback_reports_blocking_error(app, tmp_path):
    path = tmp_path / "bad.csv"; pd.DataFrame({"label":[0,1]}).to_csv(path,index=False)
    callback = next(f.fn for f in app.fns.values() if f.api_name=="validate_dataset")
    summary, preview, button = callback("Upload labeled CSV",str(path),"quick")
    assert "Cannot train" in summary and preview.empty and button["interactive"] is False


def test_educational_threshold_preserves_policy(app):
    manager = app.studio.manager
    before = manager.active.engine.policy
    callback = next(f.fn for f in app.fns.values() if f.api_name=="what_if")
    text = callback(.2,manager.active.engine.metadata["version"])
    assert "0.20" in text and "remains 0.02" in text
    assert manager.active.engine.policy == before

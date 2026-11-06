"""Local Gradio research console. Launching never trains a model."""
import json
from pathlib import Path
import uuid
import gradio as gr
import numpy as np
import pandas as pd
from .config import ROOT, write_json
from .inference import Engine
from .storage import ReviewStore
from .explainability import local_table

CSS = """
.gradio-container {max-width: 1320px !important; font-family: 'Segoe UI', Arial, sans-serif;}
#hero {background: linear-gradient(115deg,#102d36,#185760); color: #f1faf9; padding: 34px 38px; border-radius: 16px; margin-bottom: 16px;}
#hero h1 {color: white; font-size: 42px; letter-spacing: -1.8px; margin: 5px 0 12px;}
#hero p {color: #d6e8e7; max-width: 850px; font-size: 16px; line-height: 1.65;}
#hero .eyebrow {font-size: 12px; font-weight: 700; letter-spacing: 2px; color: #8fd6ca;}
.notice {border-left: 4px solid #bd8732; padding: 12px 18px; background: #fff8e9; color: #514027; border-radius: 4px;}
.notice b {color: #514027 !important;}
#decision {padding: 20px; border: 1px solid #b7d6cf; border-radius: 12px; background: #f0faf7;}
#decision h3, #decision p {color: #174f47 !important;}
.tab-nav button {font-weight: 600 !important;}
footer {display: none !important;}
@media(max-width:700px) {#hero {padding: 22px;} #hero h1 {font-size: 32px;}}
"""


def build_ui(artifact_dir=None, database=None):
    artifact_dir = Path(artifact_dir or ROOT / "artifacts/full")
    database = Path(database or ROOT / "runtime/review.sqlite3")
    theme = gr.themes.Soft(primary_hue="teal", secondary_hue="slate", neutral_hue="slate", font=["Segoe UI", "Arial", "sans-serif"])
    try:
        engine = Engine(artifact_dir)
    except (FileNotFoundError, ValueError) as exc:
        with gr.Blocks(theme=theme, css=CSS, title="SENTINEL-BC · Setup required") as empty:
            gr.Markdown("# SENTINEL-BC\n### Training artifacts unavailable")
            gr.Markdown(str(exc))
            gr.Code("python train.py --mode full\npython app.py", language="shell")
            gr.Markdown("No model is loaded and no performance claims are available. Launching this page does not train a model.")
        return empty
    store = ReviewStore(database)
    mode = engine.metadata["mode"]
    directory = artifact_dir.parent.parent / "results" / mode
    metrics = json.loads((directory / "metrics.json").read_text())
    if metrics["version"] != engine.metadata["version"]:
        raise ValueError("Evaluation results and model artifacts belong to different versions")
    cv = json.loads((directory / "cross_validation.json").read_text())
    example_path = directory / "demo_cases.csv"
    examples = pd.read_csv(example_path) if example_path.exists() else None
    labels = [str(s) for s in examples.example_name] if examples is not None else []
    feature_inputs = []
    def load_example(label):
        row = examples.loc[examples.example_name == label].iloc[0]
        return [float(row[f]) for f in engine.features]
    defaults = load_example(labels[0]) if labels else [0.0]*30

    def predict(*values):
        frame = pd.DataFrame([values], columns=engine.features)
        try:
            result = engine.predict(frame)[0]
            explanation = local_table(engine, frame).head(10) if result["p_malignant"] is not None and not result["ood"] else pd.DataFrame()
        except (ValueError, TypeError) as exc:
            raise gr.Error(str(exc)) from exc
        title = result["decision"].replace("_", " ").title()
        text = f"### {title}\n" + " · ".join(result["reasons"]) + "\n\nResearch estimate. Human assessment remains essential."
        probability = {"MALIGNANT": result["p_malignant"], "BENIGN": result["p_benign"]} if result["p_malignant"] is not None else {}
        state = {"prediction": result, "measurements": dict(zip(engine.features, [float(v) for v in values]))}
        path = ROOT / "runtime/reports" / (uuid.uuid4().hex + ".json")
        write_json(path, result)
        return text, probability, result, explanation, state, str(path)

    def save_case(state):
        if not state:
            raise gr.Error("Run a prediction first")
        if not all(np.isfinite(list(state["measurements"].values()))):
            raise gr.Error("Invalid measurements cannot be stored as a case")
        case_id = store.add(state["prediction"], state["measurements"])
        return f"Saved **{case_id}**. Open Human Review Queue to inspect the original output."

    def batch(path):
        if not path:
            raise gr.Error("Choose a CSV first")
        try:
            records = engine.predict_csv(path)
        except (ValueError, TypeError, pd.errors.ParserError) as exc:
            raise gr.Error(str(exc)) from exc
        flat = [{"row": i+1, "p_malignant": r["p_malignant"], "decision": r["decision"], "reasons": ", ".join(r["reasons"])} for i, r in enumerate(records)]
        target = ROOT / "runtime/reports" / (uuid.uuid4().hex + ".json")
        write_json(target, records)
        return pd.DataFrame(flat), str(target)

    def inspect(case):
        try: return store.inspect(case)
        except ValueError as exc: raise gr.Error(str(exc)) from exc

    def review(case, decision, notes):
        try:
            store.review(case, decision, notes)
            return "Review saved. Original prediction preserved.", store.list_cases(), store.inspect(case)
        except ValueError as exc: raise gr.Error(str(exc)) from exc

    def what_if(t):
        rows = pd.read_json(directory / "threshold_sensitivity.json")
        row = rows.iloc[(rows.threshold - t).abs().argmin()]
        return {"educational_threshold": float(row.threshold), "coverage": float(row.coverage),
                "review_rate": float(row.review_rate), "accepted_malignant": int(row.accepted_malignant),
                "accepted_accuracy": None if pd.isna(row.accepted_accuracy) else float(row.accepted_accuracy),
                "validated_deployed_threshold_unchanged": engine.policy.max_p_malignant,
                "warning": "Descriptive test-set exploration only. These values do not replace the frozen policy."}

    def figure(name, label):
        path = directory / "figures" / (name + ".png")
        if path.exists(): gr.Image(str(path), label=label, interactive=False, show_download_button=True)

    with gr.Blocks(theme=theme, css=CSS, title="SENTINEL-BC · Research Console", analytics_enabled=False) as demo:
        gr.HTML('''<div id="hero"><div class="eyebrow">RELIABILITY RESEARCH / WDBC</div><h1>SENTINEL-BC</h1><p>Breast mass classification with calibrated probabilities, uncertainty estimates and an explicit human referral policy. Explore the evidence behind each decision.</p></div>''')
        gr.HTML('<div class="notice"><b>Academic research only.</b> Uses extracted numerical measurements. No image diagnosis, patient records or clinical clearance. Keep this application local.</div>')
        with gr.Tab("Overview"):
            m, s = metrics["test"], metrics["selective"]
            gr.Markdown(f"## A measured prediction, with a route to review\nLoaded **{metrics['selected_model']}** · `{metrics['version']}` · **{mode.upper()} experiment**")
            with gr.Row():
                gr.Number(value=round(m["accuracy"]*100, 2), label=f"Classifier accuracy (%) · {m['correct']}/{m['n']}", interactive=False)
                gr.Number(value=round(m["malignant_recall"]*100, 2), label=f"Malignant recall (%) · {m['fn']} false negatives", interactive=False)
                gr.Number(value=round(s["coverage"]*100, 2), label=f"Automated coverage (%) · {s['accepted']} cases", interactive=False)
                gr.Number(value=round(s["review_rate"]*100, 2), label="Review / reject rate (%)", interactive=False)
            gr.Markdown(f"**Accepted-case accuracy:** {s['accepted_accuracy'] if s['accepted_accuracy'] is not None else 'undefined (no accepted cases)'}. Always interpret it with the coverage above.\n\n569 source observations · 357 benign · 212 malignant · 30 morphology measurements. Positive class: **MALIGNANT = 1**.")
            gr.Markdown("### Workflow\n30 measurements → validity and feature-space checks → calibrated malignant probability → neural uncertainty and MAPIE set → frozen safety gate → low-risk research estimate or human review")
            gr.Markdown(f"### What the gate checks\nThe maximum malignant probability for acceptance is **{engine.policy.max_p_malignant:.2f}**. Acceptance also requires a benign-only conformal set, low entropy, low neural dispersion, valid measurements and no OOD flag. Strong malignant predictions always require review.\n\nOnly **68 development cases** support policy selection. Zero observed accepted malignancies does not establish zero risk.")
            gr.JSON(value=engine.bundle["policy"], label="Frozen deployed policy")
        with gr.Tab("Prediction Console"):
            gr.Markdown("## Inspect one case\nExamples below are labeled historical WDBC development rows. Enter the source dataset's numerical values. UCI does not specify physical units consistently; no conversion to mm or pixels is assumed.")
            selector = gr.Dropdown(labels, value=labels[0] if labels else None, label="Labeled source example")
            with gr.Row():
                for group, start in [("Mean measurements", 0), ("Standard errors", 10), ("Worst measurements", 20)]:
                    with gr.Column():
                        gr.Markdown(f"### {group}")
                        for i in range(start, start+10):
                            feature_inputs.append(gr.Number(value=defaults[i], label=engine.features[i], precision=6))
            selector.change(load_example, selector, feature_inputs, api_name="load_example")
            run = gr.Button("Evaluate case", variant="primary")
            state = gr.State()
            decision_md = gr.Markdown("Choose an example and evaluate it to see a decision.", elem_id="decision")
            with gr.Row():
                probability = gr.Label(label="Calibrated probability", num_top_classes=2)
                output = gr.JSON(label="Prediction set, uncertainty and reason codes")
            attribution = gr.Dataframe(label="Local SHAP contributions to malignant probability", interactive=False)
            gr.Markdown("Positive attribution increases p(malignant) relative to the background. Correlated features can share or redistribute attribution. Feature importance is not causation.")
            with gr.Row():
                save = gr.Button("Add to review queue")
                download = gr.File(label="Download prediction JSON")
            saved = gr.Markdown()
            run.click(predict, feature_inputs, [decision_md, probability, output, attribution, state, download], api_name="predict_case")
            save.click(save_case, state, saved, api_name="queue_case")
            with gr.Accordion("Batch CSV · up to 100 rows / 1 MiB", open=False):
                gr.Markdown("Exactly the 30 named measurement columns, in any order. Exclude names, IDs and diagnosis labels. Batch results do not enter the queue automatically.")
                template = ROOT / "runtime/measurement_template.csv"
                template.parent.mkdir(exist_ok=True)
                pd.DataFrame(columns=engine.features).to_csv(template, index=False)
                gr.File(str(template), label="Download column template")
                upload = gr.File(file_types=[".csv"], type="filepath", label="Measurements CSV")
                batch_button = gr.Button("Evaluate CSV")
                batch_table = gr.Dataframe(label="Batch decisions", interactive=False)
                batch_file = gr.File(label="Download batch JSON")
                batch_button.click(batch, upload, [batch_table, batch_file], api_name="predict_csv")
        with gr.Tab("Model Comparison"):
            gr.Markdown("## Development selected the model\nSelection maximizes the CV mean of accuracy and malignant recall, with smaller NLL breaking ties. Held-out results below were recorded after model and policy freezing.")
            leaderboard = pd.DataFrame([{k: r[k] for k in ["model", "accuracy_mean", "accuracy_std", "malignant_recall_mean", "malignant_recall_std", "brier_mean", "seconds"]} for r in cv])
            gr.Dataframe(leaderboard, label="Fitting-partition cross-validation", interactive=False)
            gr.Dataframe(pd.DataFrame(metrics["models"]).T.reset_index().rename(columns={"index": "model"})[["model", "accuracy", "malignant_recall", "f1", "roc_auc", "pr_auc", "brier", "fn"]], label="Locked test comparison · n=114", interactive=False)
            figure("discrimination", "ROC and precision–recall")
            figure("confusion_matrices", "Confusion matrices · B=benign, M=malignant")
            figure("learning_curves", "Actual neural training histories")
            gr.JSON(engine.metadata, label="Training metadata and integrity hashes")
        with gr.Tab("Reliability and Safety"):
            gr.Markdown("## How much evidence supports acceptance?\nCalibration improves interpretation of probabilities but can worsen some finite-sample scores. Conformal coverage is marginal under exchangeability, not a guarantee for a patient or subgroup.")
            with gr.Row():
                gr.JSON(metrics["uncalibrated"], label="Before sigmoid calibration")
                gr.JSON(metrics["test"], label="After sigmoid calibration")
            figure("calibration", "Calibration and entropy")
            figure("uncertainty", "Confidence, error and neural dispersion")
            figure("safety", "Risk, coverage and referral workload")
            with gr.Row():
                gr.JSON(metrics["selective"], label="Frozen safety policy")
                gr.JSON(metrics["conformal"], label="90% MAPIE prediction sets")
            slider = gr.Slider(.01, .49, value=engine.policy.max_p_malignant or .01, step=.01, label="Educational what-if probability threshold")
            educational = gr.JSON(value=what_if(engine.policy.max_p_malignant), label="Descriptive exploration · deployed policy stays fixed")
            slider.change(what_if, slider, educational, api_name="what_if")
            gr.Dataframe(pd.read_csv(directory / "false_negatives.csv"), label="Ordinary classifier false negatives · source row indices", interactive=False)
            gr.JSON(json.loads((directory / "robustness.json").read_text()), label="Controlled development-data stress tests")
            gr.JSON(json.loads((directory / "ablation.json").read_text()), label="Ablations · each acceptance score includes coverage")
        with gr.Tab("Human Review Queue"):
            gr.Markdown("## A durable record of the original prediction\nTeaching cases only. Use anonymous IDs. Do not enter patient information or identifying notes. Review decisions never trigger retraining. Review/reject rate in the evaluation includes OOD rejection.")
            with gr.Row():
                status_filter = gr.Dropdown(["ALL", "PENDING", "REVIEWED"], value="ALL", label="Status filter")
                refresh = gr.Button("Refresh queue")
            queue = gr.Dataframe(store.list_cases(), interactive=False, label="Saved cases")
            refresh.click(store.list_cases, status_filter, queue, api_name="refresh_queue")
            status_filter.change(store.list_cases, status_filter, queue)
            case = gr.Textbox(label="Anonymous case ID", placeholder="case_…")
            inspect_button = gr.Button("Inspect case")
            inspection = gr.JSON(label="Original output, measurements and append-only audit")
            inspect_button.click(inspect, case, inspection, api_name="inspect_case")
            with gr.Row():
                reviewer = gr.Dropdown(["BENIGN", "MALIGNANT", "FURTHER_REVIEW"], value="FURTHER_REVIEW", label="Reviewer decision")
                notes = gr.Textbox(label="Teaching notes · no personal data", lines=3, max_length=2000)
            record = gr.Button("Record human review", variant="primary")
            review_status = gr.Markdown()
            record.click(review, [case, reviewer, notes], [review_status, queue, inspection], api_name="record_review")
            export_button = gr.Button("Export anonymous decisions")
            export_file = gr.File(label="Export excludes free text and measurements")
            export_button.click(lambda: store.export(ROOT / "runtime/review_export.csv"), outputs=export_file, api_name="export_reviews")
        with gr.Tab("Research and Methodology"):
            gr.Markdown((ROOT / "docs/methodology.md").read_text(encoding="utf-8") if (ROOT / "docs/methodology.md").exists() else "Methodology report will accompany the completed experiment.")
            figure("feature_importance", "Attribution and permutation importance on development data")
            gr.Markdown("### Reproducibility\n`python train.py --mode quick` verifies installation. `python train.py --mode full` runs all families and three independent Keras seeds. Existing completed runs are protected. Use `--output-root experiments/run2` for a distinct repeat.\n\n### Boundaries\nHistorical, small, correlated data; no external clinical validation; uncertain physical units; unknown deployment distribution. This software does not read biopsy images. Local joblib artifacts must be trusted. SQLite is a local academic audit trail, not a regulated clinical record system.\n\n### References\n[UCI WDBC](https://doi.org/10.24432/C5DW2B) · [MAPIE](https://mapie.readthedocs.io/) · [SHAP](https://github.com/shap/shap) · [Research and reuse audit](https://github.com/yogeshkrishna/sentinel-bc/blob/main/docs/third_party_reuse.md)")
    return demo


"""Four-tab local Training Studio over the existing research pipeline."""
import json
from pathlib import Path
import shutil
import uuid
import gradio as gr
import numpy as np
import pandas as pd
from .config import ROOT, load_config, write_json
from .data import load_data, load_training_data
from .explainability import local_table
from .runs import RunManager
from .storage import ReviewStore

ASSETS = ROOT / "runtime/studio-assets"
EXPORTS = ROOT / "runtime/reports"
CSS = """
.gradio-container {max-width:1200px !important;font-family:'Segoe UI',Arial,sans-serif;}
#hero {background:linear-gradient(110deg,#102e37,#155d60);padding:25px 30px;border-radius:14px;margin-bottom:10px;}
#hero h1 {color:white;font-size:34px;letter-spacing:-1px;margin:0 0 7px;}
#hero p {color:#d5e9e5;margin:0;font-size:15px;}
#research {font-size:13px;color:var(--body-text-color-subdued);padding:4px 2px;}
.tab-nav button {font-weight:650 !important;}
.metrics {display:grid;grid-template-columns:repeat(4,1fr);gap:22px;padding:20px 0;border-top:1px solid #56706c55;border-bottom:1px solid #56706c55;}
.metric strong {font-size:28px;color:var(--body-text-color);display:block;}
.metric span {font-size:13px;color:var(--body-text-color-subdued);}
#decision {padding:18px 22px;border-left:4px solid #289a88;background:#1b777512;border-radius:6px;}
#active {font-size:13px;} footer {display:none !important;}
@media(max-width:700px){.metrics{grid-template-columns:repeat(2,1fr);gap:12px;}#hero{padding:20px;}#hero h1{font-size:29px;}}
"""


def percent(value):
    return "Not defined" if value is None else f"{100*value:.2f}%"


def reason_text(record):
    descriptions = {
        "MALIGNANT_PROBABILITY": "Malignant probability exceeds the frozen acceptance threshold.",
        "HIGH_ENTROPY": "The calibrated estimate is ambiguous.",
        "HIGH_DISAGREEMENT": "The neural models disagree.",
        "HIGH_MC_STD": "Dropout estimates vary substantially.",
        "CONFORMAL_SET_NOT_BENIGN_SINGLETON": "The prediction set does not support benign-only acceptance.",
        "FEATURE_SPACE_OOD": "These measurements are unfamiliar to this trained model.",
        "INVALID_INPUT": "Measurements are missing, negative, nonfinite or too extreme.",
        "NO_FEASIBLE_DEVELOPMENT_POLICY": "No development threshold met the acceptance constraints.",
        "SUPPORTED_LOW_RISK_RESEARCH_ESTIMATE": "All frozen acceptance checks passed.",
    }
    return " ".join(descriptions.get(code, code.replace("_", " ").capitalize()) for code in record["reasons"])


class Studio:
    """Capture one active engine under the activation lock for every operation."""
    def __init__(self, manager, store):
        self.manager, self.store = manager, store

    def examples(self):
        with self.manager.active.lock:
            self.manager.active.require()
            path = self.manager.active.results / "demo_cases.csv"
            return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def predict(self, source, upload, example, values, version, queue_batch=False):
        with self.manager.active.lock:
            engine = self.manager.active.require(version)
            if source == "CSV":
                if not upload:
                    raise ValueError("Choose an unlabeled prediction CSV first")
                records = engine.predict_csv(upload)
                frame = pd.read_csv(upload).loc[:, engine.features].astype(float)
            else:
                if source == "Example":
                    examples = self.examples()
                    match = examples.loc[examples.example_name == example]
                    if match.empty:
                        raise ValueError("Choose a labeled example for the active model")
                    frame = match.iloc[:1].loc[:, engine.features].astype(float)
                else:
                    frame = pd.DataFrame([values], columns=engine.features)
                records = engine.predict(frame)
            for i, record in enumerate(records):
                record["run_id"] = self.manager.active.run_id
                if source != "CSV" or queue_batch:
                    measurements = {name: float(frame.iloc[i][name]) if frame.iloc[i][name] is not None else None for name in engine.features}
                    record["review_case_id"] = self.store.refer_once(record, measurements)
            target = EXPORTS / (uuid.uuid4().hex + ".json")
            write_json(target, records if source == "CSV" else records[0])
            state = None if source == "CSV" else {"record": records[0], "frame": frame.to_dict("list"), "version": engine.metadata["version"]}
            return records, state, str(target)

    def explain(self, state):
        if not state:
            raise ValueError("Predict a single case first")
        with self.manager.active.lock:
            engine = self.manager.active.require(state["version"])
            if state["record"]["ood"]:
                raise ValueError("Feature explanations are unavailable for invalid or out-of-scope cases")
            return local_table(engine, pd.DataFrame(state["frame"])).head(10)


def build_ui(artifact_dir=None, database=None, runs_root=None):
    ASSETS.mkdir(parents=True, exist_ok=True)
    EXPORTS.mkdir(parents=True, exist_ok=True)
    manager = RunManager(runs_root, artifact_dir)
    store = ReviewStore(database or ROOT / "runtime/review.sqlite3")
    studio = Studio(manager, store)
    config = load_config()
    features = list(load_data()[0].columns)
    for name, columns in [("training-template.csv", features+["label"]), ("prediction-template.csv", features)]:
        pd.DataFrame(columns=columns).to_csv(ASSETS / name, index=False)
    x, y = load_data()
    sample = x.copy(); sample["diagnosis"] = np.where(y, "M", "B")
    sample.to_csv(ASSETS / "wdbc-labeled-example.csv", index=False)

    def choices():
        return [(f"{'Historical benchmark' if r['run_id']=='benchmark' else r['run_id'][4:19]} · {r['mode'].title()} · {r['model']} · {percent(r['accuracy'])}", r["run_id"]) for r in manager.completed()]

    def active_text():
        if manager.active.engine is None:
            return "**No active model.** Train your data below; no old model download is required."
        engine = manager.active.engine
        return f"**Active:** {engine.metadata['selected_model']} · `{engine.metadata['version']}` · {'Historical benchmark' if manager.active.run_id=='benchmark' else 'Independent training run'}"

    def validate(choice, upload, mode):
        try:
            if choice == "Upload labeled CSV" and not upload:
                return "Choose a labeled CSV to validate it before training.", pd.DataFrame(), gr.update(interactive=False)
            data = load_training_data(upload if choice == "Upload labeled CSV" else None, mode=mode)
            p = data.provenance
            roles = ", ".join(f"{name}: {len(idx)}" for name, idx in data.split.items())
            message = f"**Ready to train · {p['n']} samples · 30 valid features**\n\nBenign **{p['benign']}** · Malignant **{p['malignant']}** · Missing **0** · Duplicate records **0**\n\n{p['interpretation']}\n\nPartitions — {roles}."
            if p["excluded_columns"]:
                message += "\n\nExcluded from features: " + ", ".join(p["excluded_columns"])
            preview = data.x.head().copy(); preview["label"] = data.y[:5]
            return message, preview, gr.update(interactive=True)
        except (ValueError, OSError) as exc:
            return f"**Cannot train this dataset**\n\n{exc}", pd.DataFrame(), gr.update(interactive=False)

    def start_training(choice, upload, mode, seed, epochs):
        try:
            if choice == "Upload labeled CSV" and not upload:
                raise ValueError("Choose a labeled CSV first")
            run_id = manager.start(mode, upload if choice == "Upload labeled CSV" else None, seed, epochs)
            return run_id, f"**Training started** · `{run_id}`", "Preparing the local worker…", gr.update(interactive=False), gr.update(interactive=True)
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    def poll(run_id):
        if not run_id:
            return (gr.skip(),)*7
        state = manager.status(run_id)
        path = manager.root / run_id / "events.jsonl"
        events = []
        for line in path.read_text().splitlines()[-6:] if path.exists() else []:
            try: events.append(json.loads(line))
            except json.JSONDecodeError: pass  # A worker may still be appending the last line.
        recent = "\n".join(f"{r['phase']} — {r['detail']}" for r in events)
        message = f"**{state['phase']}** · {state['elapsed_seconds']:.0f}s elapsed\n\n{state.get('detail','')}"
        terminal = state["status"] not in ("QUEUED", "RUNNING", "CANCELLING")
        if terminal:
            if state["status"] == "COMPLETED":
                m, s = state["summary"]["test"], state["summary"]["selective"]
                message = f"**Training complete** · {state['duration_seconds']:.1f}s · **{state['selected_model']}**\n\nAccuracy **{percent(m['accuracy'])}** · Malignant recall **{percent(m['malignant_recall'])}** · False negatives **{m['fn']}**\n\nGate coverage **{percent(s['coverage'])}** · Review/reject rate **{percent(s['review_rate'])}**. Choose **Use This Trained Model** to activate it."
            else:
                message = f"**{state['status'].title()}**\n\n{state.get('error','Worker interrupted')}. The active model has not changed."
        selected = run_id if state["status"] == "COMPLETED" else manager.active.run_id
        return ("" if terminal else run_id, message, recent,
                gr.update(interactive=True) if terminal else gr.skip(),
                gr.update(choices=choices(), value=selected) if terminal else gr.skip(),
                gr.update(interactive=terminal and state["status"] == "COMPLETED"), gr.update(interactive=not terminal))

    def selected_info(run_id):
        if not run_id:
            return "No completed experiment selected.", pd.DataFrame()
        try:
            metadata, metrics, _ = manager.validate_run(run_id, load_engine=False)
            dataset = metadata.get("dataset", {"source": "Historical WDBC benchmark", "n": 569, "interpretation": "Preserved v1.0.0 evidence"})
            message = f"**{dataset['source']}** · {dataset['n']} rows · {metadata['mode'].title()} · `{metrics['version']}`\n\n{dataset['interpretation']}"
            if "fingerprint" in dataset:
                message += f"\n\nDataset fingerprint `{dataset['fingerprint'][:12]}` · Seed {metadata['config']['seed']}."
            rows = [{"Model": name, "Accuracy (%)": round(100*m["accuracy"], 2), "Malignant recall (%)": round(100*m["malignant_recall"], 2), "False negatives": m["fn"]} for name, m in metrics["models"].items()]
            return message, pd.DataFrame(rows)
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    def queue_values(status="ALL", selected=None):
        frame = store.queue_view(status)
        options = [(f"{r['Case'][-6:]} · {r['Prediction'].title()} · {r['Status']}", r["Case"]) for r in frame.to_dict("records")]
        ids = [value for _, value in options]
        return frame, gr.update(choices=options, value=selected if selected in ids else (ids[0] if ids else None))

    def inspect_case(case):
        if not case:
            return "Select a referred case to review it.", None, "FURTHER_REVIEW", ""
        row = store.inspect(case); r = row["original_output"]
        summary = f"**{r.get('predicted_class','Unavailable').title()}** · Malignant probability **{percent(r.get('p_malignant'))}**\n\n{reason_text(r)}\n\nStatus: **{row['status']}** · Original model `{row['model_version']}`"
        return summary, row, row["reviewer_decision"] or "FURTHER_REVIEW", row["notes"]

    def record_review(case, decision, notes, status):
        try:
            store.review(case, decision, notes)
            frame, dropdown = queue_values(status, case)
            summary, inspection, _, _ = inspect_case(case)
            return "Review recorded. Original prediction preserved.", frame, dropdown, summary, inspection
        except (ValueError, TypeError) as exc:
            raise gr.Error(str(exc)) from exc

    def result_values():
        if manager.active.engine is None:
            return ["Train and activate a completed run to see measured results.", "", pd.DataFrame()] + [None]*7 + [pd.DataFrame()]*5 + ["", None]
        engine, metrics, directory = manager.active.engine, manager.active.metrics, manager.active.results
        dataset = engine.metadata.get("dataset", {"source": "Historical WDBC benchmark", "n": 569, "interpretation": "Preserved v1.0.0 benchmark; not a newly trained run."})
        title = f"### {dataset['source']}\n{dataset['interpretation']}\n\n{metrics['test']['n']} held-out cases · `{metrics['version']}` · Selected by development CV."
        m, s = metrics["test"], metrics["selective"]
        cells = [(percent(m["accuracy"]), "Accuracy", f"{m['correct']}/{m['n']} correct"),
                 (percent(m["malignant_recall"]), "Malignant recall", f"{m['fn']} false negatives"),
                 (percent(s["accepted_accuracy"]), "Accepted-case accuracy", f"{percent(s['coverage'])} coverage · {s['accepted']} cases"),
                 (percent(s["review_rate"]), "Human review / reject rate", "Frozen safety policy")]
        stats = '<div class="metrics">'+"".join(f'<div class="metric"><span>{label}</span><strong>{value}</strong><span>{detail}</span></div>' for value,label,detail in cells)+'</div>'
        asset_dir = ASSETS / engine.metadata["version"]; asset_dir.mkdir(exist_ok=True)
        images = []
        for name in ["learning_curves", "confusion_matrices", "calibration", "safety", "uncertainty", "feature_importance", "discrimination"]:
            source = directory / "figures" / (name+".png"); target = asset_dir / source.name
            if source.exists():
                if not target.exists(): shutil.copyfile(source, target)
                images.append(str(target))
            else: images.append(None)
        cal = pd.DataFrame([{ "Stage": label, **{k: round(value[k], 5) for k in ["brier", "nll", "ece"]}} for label,value in [("Uncalibrated",metrics["uncalibrated"]),("Sigmoid calibrated",m)]])
        cp = metrics["conformal"]
        conformal = pd.DataFrame([{ "Nominal coverage (%)": round(cp["nominal_coverage"]*100,2), "Observed coverage (%)": round(cp["empirical_coverage"]*100,2), "Mean set size": round(cp["mean_set_size"],3), "Empty sets": cp["empty_sets"]}])
        robustness = pd.DataFrame([{ "Stress test": r["experiment"], "n": r["n"], "Coverage (%)": round(r["selective"]["coverage"]*100,2), "Accepted malignant": r["selective"]["accepted_malignant"]} for r in json.loads((directory / "robustness.json").read_text())])
        ablations = pd.DataFrame([{ "Stage": name, "Coverage (%)": round(r["selective"]["coverage"]*100,2), "Accepted accuracy (%)": None if r["selective"]["accepted_accuracy"] is None else round(100*r["selective"]["accepted_accuracy"],2)} for name,r in json.loads((directory / "ablation.json").read_text()).items()])
        policy = f"Deployed threshold: **{engine.policy.max_p_malignant:.2f}**. Educational changes below do not alter this policy."
        false_negatives = pd.read_csv(directory / "false_negatives.csv")
        false_negatives = false_negatives[["row_index", "p_malignant", "decision"]].rename(
            columns={"row_index": "Source row", "p_malignant": "Malignant probability", "decision": "Gate decision"})
        false_negatives["Malignant probability"] = false_negatives["Malignant probability"].map(percent)
        return [title, stats, selected_info(manager.active.run_id)[1]] + images + [cal, conformal, robustness, ablations, false_negatives, policy, engine.metadata]

    def what_if(threshold, version):
        with manager.active.lock:
            engine = manager.active.require(version)
            rows = pd.read_json(manager.active.results / "threshold_sensitivity.json")
            row = rows.iloc[(rows.threshold-threshold).abs().argmin()]
            return f"**Educational threshold {row.threshold:.2f}:** coverage {percent(row.coverage)}, review/reject rate {percent(row.review_rate)}, accepted malignant cases {int(row.accepted_malignant)}.\n\n**Deployed threshold remains {engine.policy.max_p_malignant:.2f}.** Descriptive exploration of cached test predictions, never policy tuning."

    initial_choices = choices()
    initial_selected = manager.active.run_id or (initial_choices[0][1] if initial_choices else None)
    initial_version = manager.active.engine.metadata["version"] if manager.active.engine else ""
    initial_examples = studio.examples() if manager.active.engine else pd.DataFrame()
    names = initial_examples.example_name.tolist() if len(initial_examples) else []
    defaults = initial_examples.iloc[0][features].tolist() if len(initial_examples) else [None]*30
    theme = gr.themes.Soft(primary_hue="teal", neutral_hue="slate", font=["Segoe UI", "Arial", "sans-serif"])
    initial = result_values()
    initial_validation, initial_preview, _ = validate("Built-in WDBC", None, "quick")
    with gr.Blocks(theme=theme, css=CSS, title="SENTINEL-BC · Training Studio", analytics_enabled=False) as demo:
        gr.HTML('<div id="hero"><h1>SENTINEL-BC</h1><p>Your data. A real training run. A clear route from prediction to review.</p></div>')
        gr.HTML('<div id="research">Academic research only. Uses 30 extracted nucleus measurements, not images. No patient records or clinical clearance.</div>')
        active_banner = gr.Markdown(active_text(), elem_id="active")
        active_version, prediction_state, job_state = gr.State(initial_version), gr.State(), gr.State("")
        with gr.Tab("Train"):
            gr.Markdown("## Train your own experiment\nChoose data, check its labels, then train. The historical benchmark stays untouched.")
            with gr.Row():
                with gr.Column(scale=3):
                    dataset_choice = gr.Radio(["Built-in WDBC", "Upload labeled CSV"], value="Built-in WDBC", label="1 · Choose dataset")
                    training_upload = gr.File(label="Compatible labeled CSV · up to 10 MiB", file_types=[".csv"], type="filepath", visible=False)
                    validation = gr.Markdown(initial_validation)
                    with gr.Accordion("Preview and CSV templates", open=False):
                        preview = gr.Dataframe(initial_preview, interactive=False, label="First five validated rows")
                        gr.Markdown("Exactly 30 named WDBC measurements plus label, diagnosis or target. Use 0/B/BENIGN and 1/M/MALIGNANT. Recognized id/sample_id columns are excluded. Multiple target columns must agree. At least 350 rows and 70 of each class; role-specific checks may require more.")
                        with gr.Row():
                            gr.File(str(ASSETS / "training-template.csv"), label="Training CSV template")
                            gr.File(str(ASSETS / "prediction-template.csv"), label="Prediction CSV template")
                        gr.File(str(ASSETS / "wdbc-labeled-example.csv"), label="Labeled public WDBC example · repeated data, not external validation")
                with gr.Column(scale=2):
                    training_mode = gr.Radio([("Quick Training","quick"),("Full Training","full")], value="quick", label="2 · Choose training mode")
                    gr.Markdown("**Quick:** logistic regression, SVM and one MLP, with fewer epochs.\n\n**Full:** all six candidates, three neural members and complete evaluation. Both retain all five data roles.")
                    with gr.Accordion("Advanced training settings", open=False):
                        seed = gr.Number(value=config["seed"], label="Development random seed", precision=0)
                        epochs = gr.Number(value=config["quick"]["epochs"], label="Maximum neural epochs · early stopping still applies", precision=0)
                        gr.Markdown("Historical WDBC test rows stay locked even if the development seed changes. Other settings use configs/default.yaml.")
                    train_button = gr.Button("Train Models", variant="primary", size="lg")
                    cancel_button = gr.Button("Cancel training", interactive=False)
            training_status = gr.Markdown("Ready when you are. Training starts only when you click Train Models.")
            with gr.Accordion("Live training activity", open=True):
                live_logs = gr.Textbox(label="Actual phase and recent activity", lines=4, interactive=False)
            gr.Markdown("### 3 · Choose a completed experiment")
            with gr.Row():
                runs = gr.Dropdown(initial_choices, value=initial_selected, label="Completed runs · benchmark and your experiments", scale=4, allow_custom_value=True)
                refresh_runs = gr.Button("Refresh runs", scale=1)
                use_model = gr.Button("Use This Trained Model", variant="primary", interactive=bool(initial_selected), scale=2)
            selected_summary = gr.Markdown(selected_info(initial_selected)[0] if initial_selected else "Your completed training runs will appear here.")
            with gr.Accordion("Selected experiment comparison", open=False):
                selected_table = gr.Dataframe(selected_info(initial_selected)[1] if initial_selected else pd.DataFrame(), interactive=False)
        with gr.Tab("Predict"):
            gr.Markdown("## Make a prediction\nUse a labeled development example or upload an unlabeled CSV. Preserve the source measurement definitions and units.")
            prediction_source = gr.Radio(["Example","CSV","Manual"], value="Example", label="Prediction source")
            example = gr.Dropdown(names, value=names[0] if names else None, label="Labeled sample case", allow_custom_value=True)
            prediction_upload = gr.File(label="Prediction CSV · 30 features, no target · 100 rows / 1 MiB",file_types=[".csv"],type="filepath",visible=False)
            queue_batch = gr.Checkbox(value=False,label="Also add batch referrals to Review (optional)",visible=False)
            manual_inputs = []
            with gr.Accordion("Advanced: Enter 30 Measurements Manually", open=False):
                with gr.Row():
                    for start,label in [(0,"Means"),(10,"Standard errors"),(20,"Worst values")]:
                        with gr.Column():
                            gr.Markdown(f"### {label}")
                            for i in range(start,start+10): manual_inputs.append(gr.Number(value=defaults[i],label=features[i]))
            predict_button = gr.Button("Predict",variant="primary",size="lg")
            decision = gr.Markdown("Choose an example and click Predict." if initial_version else "No active model. Complete training in Train, then activate it.",elem_id="decision")
            referral = gr.Markdown()
            batch_table = gr.Dataframe(label="Batch predictions",interactive=False,visible=False)
            with gr.Accordion("Prediction details and explanation",open=False):
                prediction_detail = gr.JSON(label="Calibrated probabilities, entropy, neural dispersion and prediction set")
                explain_button = gr.Button("Explain this prediction")
                explanation = gr.Dataframe(label="Local SHAP contributions to malignant probability",interactive=False)
                gr.Markdown("Positive contributions move probability toward malignant relative to the background. Correlated inputs share attribution; importance is not causation.")
                report = gr.File(label="Download prediction report")
        with gr.Tab("Review"):
            gr.Markdown("## Human review\nSingle-case referrals appear automatically. Select a case, inspect its original prediction and record your review.")
            with gr.Row():
                queue_filter = gr.Dropdown(["ALL","PENDING","REVIEWED"],value="ALL",label="Show cases")
                refresh_queue = gr.Button("Refresh cases")
            queue_table = gr.Dataframe(store.queue_view(),label="Referred cases",interactive=False)
            saved_cases = store.queue_view().to_dict("records")
            case = gr.Dropdown(label="Select a case",choices=[(f"{r['Case'][-6:]} · {r['Prediction'].title()} · {r['Status']}",r["Case"]) for r in saved_cases],value=None,allow_custom_value=True)
            case_summary = gr.Markdown("Select a case from the table or list.")
            with gr.Row():
                reviewer = gr.Dropdown(["BENIGN","MALIGNANT","FURTHER_REVIEW"],value="FURTHER_REVIEW",label="Reviewer decision")
                notes = gr.Textbox(label="Optional teaching notes · no personal data",lines=2,max_length=2000)
            record_button = gr.Button("Record Review",variant="primary")
            review_feedback = gr.Markdown()
            with gr.Accordion("Original model output and audit history",open=False):
                case_detail = gr.JSON(label="Immutable prediction, model version and review audit")
            with gr.Row():
                export_button = gr.Button("Export anonymous decisions")
                export = gr.File(label="Export excludes notes and measurements")
        with gr.Tab("Results"):
            result_title = gr.Markdown(initial[0]); metrics_html = gr.HTML(initial[1])
            comparison = gr.Dataframe(initial[2],label="Model comparison · same locked test cases",interactive=False)
            with gr.Row():
                learning = gr.Image(initial[3],label="ANN training curves",interactive=False,height=310)
                confusion = gr.Image(initial[4],label="Confusion matrices",interactive=False,height=310)
            with gr.Row():
                reliability = gr.Image(initial[5],label="Reliability diagram and entropy",interactive=False,height=310)
                risk = gr.Image(initial[6],label="Risk, coverage and review workload",interactive=False,height=310)
            with gr.Accordion("Advanced reliability evidence",open=False):
                gr.Markdown("Calibration can worsen finite-sample scores. Conformal coverage is marginal under exchangeability; zero observed accepted malignancies does not establish zero future risk.")
                cal_table = gr.Dataframe(initial[10],label="Calibration before / after",interactive=False)
                uncertainty_plot = gr.Image(initial[7],label="Confidence, predictive entropy and neural disagreement",interactive=False)
                conformal_table = gr.Dataframe(initial[11],label="MAPIE prediction-set coverage",interactive=False)
                robust_table = gr.Dataframe(initial[12],label="Controlled development-data stress tests",interactive=False)
                ablation_table = gr.Dataframe(initial[13],label="Detailed ablations",interactive=False)
                importance = gr.Image(initial[8],label="Feature importance",interactive=False)
                discrimination = gr.Image(initial[9],label="ROC and precision–recall curves",interactive=False)
                false_negatives = gr.Dataframe(initial[14],label="Ordinary classifier false negatives",interactive=False)
                policy_description = gr.Markdown(initial[15])
                threshold = gr.Slider(.01,.49,value=.02,step=.01,label="Educational what-if threshold · frozen policy stays unchanged")
                threshold_result = gr.Markdown()
                with gr.Accordion("Model metadata and reproducibility",open=False):
                    metadata = gr.JSON(initial[16],label="Active run metadata")
            with gr.Accordion("Full methodology and references",open=False):
                gr.Markdown((ROOT / "docs/methodology.md").read_text(encoding="utf-8"))

        result_outputs = [result_title, metrics_html, comparison, learning, confusion, reliability, risk,
                          uncertainty_plot, importance, discrimination, cal_table, conformal_table,
                          robust_table, ablation_table, false_negatives, policy_description, metadata]

        def active_view():
            engine = manager.active.engine
            samples = studio.examples() if engine is not None else pd.DataFrame()
            options = samples.example_name.tolist() if len(samples) else []
            values = samples.iloc[0][features].tolist() if len(samples) else [None]*30
            message = ("Choose a case and click Predict." if engine is not None
                       else "No active model. Complete training in Train, then activate it.")
            return [active_text(), engine.metadata["version"] if engine is not None else "",
                    gr.update(choices=options, value=options[0] if options else None),
                    message, "", None, None, pd.DataFrame(), None,
                    gr.update(value=pd.DataFrame(), visible=False), ""] + values + result_values()

        def refresh_active():
            with manager.active.lock:
                return active_view()

        def activate(run_id):
            try:
                with manager.active.lock:
                    manager.activate(run_id)
                    return active_view()
            except (ValueError,OSError,KeyError) as exc: raise gr.Error(str(exc)) from exc

        def predict(source,upload,selected_example,version,batch_opt_in,*values):
            try:
                records,state,target = studio.predict(source,upload,selected_example,values,version,batch_opt_in)
                if source == "CSV":
                    summary = f"### {len(records)} predictions complete\nBatch insertion into Review was {'enabled' if batch_opt_in else 'not requested'}."
                    table = pd.DataFrame([{ "Row":i+1,"Prediction":r.get("predicted_class","Unavailable"),"Malignant probability (%)":None if r["p_malignant"] is None else round(r["p_malignant"]*100,2),"Decision":r["decision"],"Review case":r.get("review_case_id") or ""} for i,r in enumerate(records)])
                    assigned = ""
                else:
                    r = records[0]
                    gate = {"LOW_RISK_ESTIMATE":"Low-risk research estimate","HUMAN_REVIEW_REQUIRED":"Human review required","OUT_OF_SCOPE_INPUT":"Out-of-scope input"}[r["decision"]]
                    summary = f"### {r.get('predicted_class','Prediction unavailable').title()} · {percent(r['p_malignant'])} malignant probability\n**{gate}**\n\n{reason_text(r)}"
                    assigned = f"Automatically added to Review: **{r['review_case_id']}**. Repeated predictions reuse this case." if r.get("review_case_id") else ""
                    table = pd.DataFrame()
                queue,dropdown = queue_values()
                return summary,assigned,records if source=="CSV" else records[0],pd.DataFrame(),state,target,gr.update(value=table,visible=source=="CSV"),queue,dropdown
            except (ValueError,OSError,TypeError) as exc: raise gr.Error(str(exc)) from exc

        def load_example(name,version):
            if name is None:
                return [None]*30
            try:
                with manager.active.lock:
                    manager.active.require(version)
                    frame = studio.examples()
                    return frame.loc[frame.example_name==name,features].iloc[0].tolist()
            except (ValueError,IndexError) as exc: raise gr.Error(str(exc)) from exc

        def explain_case(state):
            try: return studio.explain(state)
            except ValueError as exc: raise gr.Error(str(exc)) from exc

        def table_case(frame, event: gr.SelectData):
            index = event.index[0]
            return frame.iloc[index]["Case"] if 0 <= index < len(frame) else gr.skip()

        dataset_choice.change(lambda choice:gr.update(visible=choice=="Upload labeled CSV"),dataset_choice,training_upload).then(validate,[dataset_choice,training_upload,training_mode],[validation,preview,train_button])
        training_upload.change(validate,[dataset_choice,training_upload,training_mode],[validation,preview,train_button],api_name="validate_dataset")
        training_mode.change(lambda mode:config[mode]["epochs"],training_mode,epochs)
        train_button.click(start_training,[dataset_choice,training_upload,training_mode,seed,epochs],[job_state,training_status,live_logs,train_button,cancel_button],api_name="start_training",concurrency_id="training_control")
        timer = gr.Timer(1)
        timer.tick(poll,job_state,[job_state,training_status,live_logs,train_button,runs,use_model,cancel_button],api_name=False,show_progress="hidden")
        cancel_button.click(lambda run_id:manager.cancel(run_id) if run_id else "No training job is running.",job_state,training_status,api_name="cancel_training",queue=False)
        refresh_runs.click(lambda:gr.update(choices=choices()),outputs=runs,api_name="refresh_runs")
        runs.change(selected_info,runs,[selected_summary,selected_table],api_name="selected_run")
        runs.change(lambda run_id:gr.update(interactive=bool(run_id)),runs,use_model,api_name=False)
        activation_outputs = [active_banner,active_version,example,decision,referral,prediction_detail,report,explanation,prediction_state,batch_table,threshold_result]+manual_inputs+result_outputs
        demo.load(refresh_active, outputs=activation_outputs, api_name=False, show_progress="hidden")
        use_model.click(activate,runs,activation_outputs,api_name="activate_run",concurrency_id="inference",concurrency_limit=1)
        prediction_source.change(lambda source:(gr.update(visible=source=="Example"),gr.update(visible=source=="CSV"),gr.update(visible=source=="CSV")),prediction_source,[example,prediction_upload,queue_batch])
        example.change(load_example,[example,active_version],manual_inputs,api_name="load_example",concurrency_id="inference")
        predict_button.click(predict,[prediction_source,prediction_upload,example,active_version,queue_batch]+manual_inputs,[decision,referral,prediction_detail,explanation,prediction_state,report,batch_table,queue_table,case],api_name="predict_case",concurrency_id="inference",concurrency_limit=1)
        explain_button.click(explain_case,prediction_state,explanation,api_name="explain_case",concurrency_id="inference")
        refresh_queue.click(queue_values,queue_filter,[queue_table,case],api_name="refresh_queue")
        queue_filter.change(queue_values,queue_filter,[queue_table,case])
        queue_table.select(table_case,queue_table,case,api_name=False)
        case.change(inspect_case,case,[case_summary,case_detail,reviewer,notes],api_name="inspect_case")
        record_button.click(record_review,[case,reviewer,notes,queue_filter],[review_feedback,queue_table,case,case_summary,case_detail],api_name="record_review")
        export_button.click(lambda:store.export(EXPORTS/(uuid.uuid4().hex+".csv")),outputs=export,api_name="export_reviews")
        threshold.change(what_if,[threshold,active_version],threshold_result,api_name="what_if",concurrency_id="inference")
    demo.studio = studio
    demo.studio_poll = poll
    return demo

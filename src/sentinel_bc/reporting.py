"""One locked evaluation, saved predictions, and figures reproducible without refitting."""
from dataclasses import replace
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
from .config import write_json
from .inference import Engine
from .evaluation import classification_metrics, selective_metrics, risk_coverage
from .safety_gate import LOW, REVIEW, apply_policy
from .explainability import explain, global_permutation, local_table


def bootstrap_auc(y, p):
    rng = np.random.default_rng(42)
    a, b = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    values = []
    for _ in range(1000):
        idx = np.r_[rng.choice(a, len(a)), rng.choice(b, len(b))]
        values.append(roc_auc_score(y[idx], p[idx]))
    return np.quantile(values, [.025, .975]).tolist()


def evaluate_locked_run(artifacts, results, split, x, y):
    if (results / "metrics.json").exists():
        raise FileExistsError("Locked evaluation already exists; use cached results")
    engine = Engine(artifacts)
    idx = split["test"]
    xt, yt = x.iloc[idx], y[idx]
    # Engine upload limits do not constrain the evaluator: use fixed chunks.
    started = time.perf_counter()
    predictions = sum([engine.predict(xt.iloc[i:i+100]) for i in range(0, len(xt), 100)], [])
    inference_ms = (time.perf_counter()-started)*1000/len(yt)
    p = np.array([r["p_malignant"] for r in predictions])
    raw = engine.bundle["base"].predict_proba(xt)[:, 1]
    comparison, probabilities = {}, {}
    for name, model in engine.bundle["candidates"].items():
        started = time.perf_counter()
        prob = model.predict_proba(xt)[:, 1]
        comparison[name] = classification_metrics(yt, prob) | {"inference_ms_per_case": (time.perf_counter()-started)*1000/len(yt)}
        probabilities[name] = prob
    metric = classification_metrics(yt, p)
    metric["roc_auc_ci95_bootstrap"] = bootstrap_auc(yt, p)
    metric["inference_ms_per_case_with_safety"] = inference_ms
    sets = np.array([[name in r["prediction_set"] for name in ["BENIGN", "MALIGNANT"]] for r in predictions])
    conformal = {"nominal_coverage": engine.metadata["config"]["conformal_confidence"],
                 "empirical_coverage": float(sets[np.arange(len(yt)), yt].mean()),
                 "coverage_ci95": __import__("sentinel_bc.evaluation", fromlist=["wilson"]).wilson(int(sets[np.arange(len(yt)), yt].sum()), len(yt)),
                 "mean_set_size": float(sets.sum(axis=1).mean()),
                 "empty_sets": int(sum(sets.sum(axis=1)==0)), "both_class_sets": int(sum(sets.sum(axis=1)==2)),
                 "benign_coverage": float(sets[yt==0, 0].mean()), "malignant_coverage": float(sets[yt==1, 1].mean())}
    decisions = [(r["decision"], r["reasons"]) for r in predictions]
    selective = selective_metrics(yt, decisions)
    uncertainty = {k: np.array([r["uncertainty"][k] for r in predictions]) for k in predictions[0]["uncertainty"]}
    valid = np.ones(len(yt), bool)
    ood = np.array([r["ood"] for r in predictions])
    uncertainty_accept = (p < engine.policy.max_p_malignant) & (uncertainty["entropy"] <= engine.policy.max_entropy) & (uncertainty["disagreement"] <= engine.policy.max_disagreement) & (uncertainty["mc_std"] <= engine.policy.max_mc_std) & engine.policy.feasible
    ablations = {
        "base_classifier": {"ordinary": classification_metrics(yt, raw), "selective": selective_metrics(yt, [(LOW if v < .5 else REVIEW, []) for v in raw])},
        "calibrated_classifier": {"ordinary": metric, "selective": selective_metrics(yt, [(LOW if v < .5 else REVIEW, []) for v in p])},
        "calibrated_with_uncertainty": {"selective": selective_metrics(yt, [(LOW if v else REVIEW, []) for v in uncertainty_accept])},
        "full_safety_gate": {"selective": selective},
    }
    sweep = []
    for threshold in np.linspace(.01, .49, 49):
        policy = replace(engine.policy, max_p_malignant=float(threshold))
        ds = apply_policy(p, uncertainty, sets, valid, ood, policy)
        sweep.append({"threshold": float(threshold)} | selective_metrics(yt, ds))
    # Stress tests and feature importance use policy development data, never tune test.
    xp, yp = x.iloc[split["policy"]], y[split["policy"]]
    stress = robustness(engine, xp, yp)
    demo_predictions = engine.predict(xp)
    low = next((i for i, r in enumerate(demo_predictions) if r["decision"] == LOW), None)
    high = int(np.argmax([r["p_malignant"] for r in demo_predictions]))
    ambiguous = int(np.argmax([r["uncertainty"]["entropy"] for r in demo_predictions]))
    demo_rows = []
    for name, i in [("Low-risk example", low), ("Malignant referral example", high), ("Most uncertain development example", ambiguous)]:
        if i is None:
            continue
        source_row = int(split["policy"][i])
        label = "MALIGNANT" if yp[i] else "BENIGN"
        demo_rows.append({"example_name": f"{name} · {label} · WDBC row {source_row}", "source_row": source_row,
                          "source_label": label, "data_role": "policy_development", **xp.iloc[i].to_dict()})
    pd.DataFrame(demo_rows).to_csv(results / "demo_cases.csv", index=False)
    global_importance = global_permutation(engine, xp, yp)
    global_importance.to_csv(results / "permutation_importance.csv", index=False)
    sv = explain(engine, xp.iloc[:8])
    shap_global = pd.DataFrame({"measurement": engine.features, "mean_abs_shap": np.abs(sv.values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    shap_global.to_csv(results / "shap_importance.csv", index=False)
    local_table(engine, xp.iloc[:1]).to_csv(results / "local_explanation.csv", index=False)
    table = pd.DataFrame({"row_index": idx, "y_malignant": yt, "p_uncalibrated": raw, "p_malignant": p,
                          "decision": [r["decision"] for r in predictions], "ood": ood,
                          "set_benign": sets[:, 0], "set_malignant": sets[:, 1], **uncertainty,
                          **{f"p_{name}": v for name, v in probabilities.items()}})
    table.to_csv(results / "test_predictions.csv", index=False)
    table.loc[(p < .5) & (yt == 1)].to_csv(results / "false_negatives.csv", index=False)
    write_json(results / "risk_coverage.json", risk_coverage(yt, p))
    write_json(results / "threshold_sensitivity.json", sweep)
    write_json(results / "robustness.json", stress)
    write_json(results / "ablation.json", ablations)
    write_json(results / "metrics.json", {"selected_model": engine.metadata["selected_model"], "version": engine.metadata["version"],
        "test": metric, "uncalibrated": classification_metrics(yt, raw), "selective": selective,
        "conformal": conformal, "models": comparison, "policy": engine.bundle["policy"],
        "protocol": "Single locked stratified test evaluation after CV selection, calibration, conformalization and policy freezing."})
    draw_figures(results)


def robustness(engine, x, y):
    rng = np.random.default_rng(42)
    scale = engine.bundle["background"].std().to_numpy()
    variants = {"unmodified_policy_data": x.copy(),
                "noise_005_train_std": x + rng.normal(0, .05, x.shape) * scale,
                "noise_020_train_std": x + rng.normal(0, .20, x.shape) * scale,
                "radius_plus_10pct": x.assign(**{"mean radius": x["mean radius"]*1.1}),
                "missing_texture": x.assign(**{"mean texture": np.nan}),
                "implausible_10x": x*10,
                "negative_radius": x.assign(**{"mean radius": -1})}
    rows = []
    for name, frame in variants.items():
        predictions = engine.predict(frame)
        usable = np.array([r["p_malignant"] is not None for r in predictions])
        ds = [(r["decision"], r["reasons"]) for r in predictions]
        row = {"experiment": name, "n": len(y), "valid_probability_rows": int(usable.sum()),
               "ood_fraction": float(np.mean([r["ood"] for r in predictions])), "selective": selective_metrics(y, ds)}
        if usable.any():
            row["classifier"] = classification_metrics(y[usable], np.array([r["p_malignant"] for r in predictions if r["p_malignant"] is not None]))
        rows.append(row)
    return rows


def draw_figures(results):
    """Regenerate plots from saved evaluation output without touching the models."""
    import json
    results = __import__("pathlib").Path(results)
    d = pd.read_csv(results / "test_predictions.csv")
    metrics = json.loads((results / "metrics.json").read_text())
    directory = results / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 125, "axes.titleweight": "bold"})
    y, p = d.y_malignant.to_numpy(), d.p_malignant.to_numpy()
    def save(fig, name):
        fig.tight_layout(); fig.savefig(directory / (name+".png"), bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in metrics["models"]:
        prob = d["p_"+name]
        fpr, tpr, _ = roc_curve(y, prob)
        axes[0].plot(fpr, tpr, label=name)
        precision, recall, _ = precision_recall_curve(y, prob)
        axes[1].plot(recall, precision, label=name)
    axes[0].set(xlabel="False-positive rate", ylabel="Malignant recall", title="Locked test ROC curves")
    axes[1].set(xlabel="Malignant recall", ylabel="Precision", title="Precision–recall curves")
    for ax in axes: ax.legend(fontsize=8)
    save(fig, "discrimination")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for column, name in [("p_uncalibrated", "Before calibration"), ("p_malignant", "Sigmoid calibration")]:
        observed, predicted = calibration_curve(y, d[column], n_bins=7, strategy="uniform")
        axes[0].plot(predicted, observed, "o-", label=name)
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="Mean malignant probability", ylabel="Observed malignant fraction", title="Reliability (114 test cases)")
    axes[0].legend(fontsize=8)
    axes[1].hist([d.entropy[y==0], d.entropy[y==1]], label=["Benign", "Malignant"], bins=12)
    axes[1].set(xlabel="Predictive entropy (bits)", ylabel="Count", title="Calibrated uncertainty")
    axes[1].legend()
    save(fig, "calibration")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    correct = (p >= .5) == y
    axes[0].scatter(d.confidence, correct.astype(int), alpha=.45)
    axes[0].set(xlabel="Calibrated confidence", ylabel="Correct (1) / error (0)", title="Confidence and error")
    axes[1].scatter(d.disagreement, d.mc_std, c=y, cmap="coolwarm", alpha=.7)
    axes[1].set(xlabel="Deep ensemble std", ylabel="MC Dropout std", title="Neural uncertainty signals")
    save(fig, "uncertainty")
    curve = pd.read_json(results / "risk_coverage.json")
    sweep = pd.read_json(results / "threshold_sensitivity.json")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(curve.coverage, curve.risk)
    axes[0].set(xlabel="Automated coverage", ylabel="Classifier error among accepted", title="Confidence-ranked risk")
    axes[1].plot(sweep.threshold, sweep.coverage, label="Coverage")
    axes[1].plot(sweep.threshold, sweep.review_rate, label="Review / reject")
    axes[1].axvline(metrics["policy"]["max_p_malignant"], color="gray", linestyle="--")
    axes[1].set(xlabel="Maximum malignant probability", ylabel="Fraction", title="Gate sensitivity (descriptive)")
    axes[1].legend()
    axes[2].plot(sweep.coverage, sweep.review_rate)
    axes[2].set(xlabel="Coverage", ylabel="Review / reject rate", title="Workload trade-off")
    save(fig, "safety")
    items = list(metrics["models"].items())
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, (name, m) in zip(axes.flat, items):
        ax.imshow(m["confusion_matrix"], cmap="Blues")
        for i in range(2):
            for j in range(2): ax.text(j, i, str(m["confusion_matrix"][i][j]), ha="center", va="center")
        ax.set(title=name, xlabel="Predicted", ylabel="Actual", xticks=[0, 1], yticks=[0, 1], xticklabels=["B", "M"], yticklabels=["B", "M"])
    for ax in list(axes.flat)[len(items):]: ax.axis("off")
    save(fig, "confusion_matrices")
    histories = json.loads((results / "training_history.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for seed, history in histories.items():
        axes[0].plot(history["loss"], label=f"seed {seed} fit")
        axes[0].plot(history["val_loss"], linestyle="--", label=f"seed {seed} validation")
        axes[1].plot(history["val_accuracy"], label=f"seed {seed}")
    axes[0].set(title="Neural training", xlabel="Epoch", ylabel="Regularized binary cross-entropy")
    axes[1].set(title="Internal validation", xlabel="Epoch", ylabel="Accuracy")
    for ax in axes: ax.legend(fontsize=7)
    save(fig, "learning_curves")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, filename, column, title in [(axes[0], "shap_importance.csv", "mean_abs_shap", "SHAP (8 development cases)"),
                                        (axes[1], "permutation_importance.csv", "mean_accuracy_drop", "Permutation (68 development cases)")]:
        imp = pd.read_csv(results / filename).head(10).iloc[::-1]
        ax.barh(imp.measurement, imp[column], color="#15857f")
        ax.set(title=title, xlabel=column.replace("_", " "))
    save(fig, "feature_importance")

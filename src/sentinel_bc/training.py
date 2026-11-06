"""Bounded development experiments, followed by a single locked test evaluation."""
import hashlib
import importlib.metadata
import logging
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from .config import ROOT, load_config, write_json
from .data import load_data, partitions, describe_data
from .models import classical_candidates, train_network, NeuralPredictor, tensorflow
from .calibration import calibrate
from .conformal import conformalize, prediction_sets
from .preprocessing import InputScreen
from .uncertainty import signals
from .safety_gate import select_policy
from .evaluation import classification_metrics

log = logging.getLogger(__name__)


def cv_summary(name, params, rows, seconds):
    keys = ["accuracy", "malignant_recall", "f1", "roc_auc", "pr_auc", "brier", "nll"]
    out = {"model": name, "parameters": params, "seconds": seconds, "folds": len(rows)}
    for k in keys:
        out[k + "_mean"] = float(np.mean([r[k] for r in rows]))
        out[k + "_std"] = float(np.std([r[k] for r in rows], ddof=1))
    out["selection_score"] = (out["accuracy_mean"] + out["malignant_recall_mean"]) / 2
    out["fold_metrics"] = rows
    return out


def train(mode="full", output_root=ROOT):
    config = load_config()
    settings = config[mode]
    artifacts = Path(output_root) / "artifacts" / mode
    results = Path(output_root) / "results" / mode
    # An existing locked experiment is never silently replaced or evaluated again.
    if (artifacts / "manifest.json").exists() or (results / "metrics.json").exists():
        raise FileExistsError(f"{mode} already exists. Use --output-root for an independent experiment.")
    artifacts.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    x, y = load_data()
    split = partitions(y, config)
    fit = split["fit"]
    xf, yf = x.iloc[fit], y[fit]
    write_json(results / "data_audit.json", describe_data(xf, yf) | {"full_data_sha256": describe_data(x, y)["sha256"],
        "full_class_counts": {"benign": int(sum(y==0)), "malignant": int(sum(y==1))},
        "note": "Descriptive distributions and correlation analysis use fitting rows only."})
    write_json(results / "partitions.json", {k: {"indices": v.tolist(), "n": len(v), "malignant": int(sum(y[v]))} for k, v in split.items()})
    folds = list(StratifiedKFold(settings["folds"], shuffle=True, random_state=config["seed"]).split(xf, yf))
    summaries, best_classical = [], {}
    for name, params, candidate in classical_candidates(config["seed"], mode):
        started, rows = time.perf_counter(), []
        for a, b in folds:
            model = clone(candidate).fit(xf.iloc[a], yf[a])
            rows.append(classification_metrics(yf[b], model.predict_proba(xf.iloc[b])[:, 1]))
        summary = cv_summary(name, params, rows, time.perf_counter()-started)
        summaries.append(summary)
        if name not in best_classical or (summary["selection_score"], -summary["nll_mean"]) > best_classical[name][0]:
            best_classical[name] = ((summary["selection_score"], -summary["nll_mean"]), candidate)
        log.info("cv model=%s parameters=%s accuracy=%.4f recall=%.4f", name, params, summary["accuracy_mean"], summary["malignant_recall_mean"])
    # Neural validation uses only a subset of each CV training fold.
    started, mlp_rows, ensemble_rows = time.perf_counter(), [], []
    seeds = [config["seed"] + 100 * i for i in range(settings["ensemble_size"])]
    for fold, (a, b) in enumerate(folds):
        members = []
        for seed in seeds:
            prep, network, _ = train_network(xf.iloc[a], yf[a], seed, settings["epochs"], settings["patience"])
            members.append((prep, network))
        mlp_rows.append(classification_metrics(yf[b], NeuralPredictor(members[:1]).predict_proba(xf.iloc[b])[:, 1]))
        ensemble_rows.append(classification_metrics(yf[b], NeuralPredictor(members).predict_proba(xf.iloc[b])[:, 1]))
        log.info("neural_cv fold=%d/%d complete", fold+1, len(folds))
    seconds = time.perf_counter()-started
    summaries.append(cv_summary("mlp", {"seeds": seeds[:1]}, mlp_rows, seconds))
    if mode == "full":
        summaries.append(cv_summary("deep_ensemble", {"seeds": seeds}, ensemble_rows, seconds))
    chosen = max(summaries, key=lambda r: (r["selection_score"], -r["nll_mean"]))
    write_json(results / "cross_validation.json", summaries)
    log.info("selection frozen model=%s rule=mean(accuracy,malignant_recall),nll_tiebreak", chosen["model"])
    candidates, timings = {}, {}
    for name, (_, candidate) in best_classical.items():
        started = time.perf_counter()
        candidates[name] = clone(candidate).fit(xf, yf)
        timings[name] = time.perf_counter()-started
    members, histories = [], {}
    started = time.perf_counter()
    for seed in seeds:
        prep, network, history = train_network(xf, yf, seed, settings["epochs"], settings["patience"])
        members.append((prep, network))
        histories[str(seed)] = history
        network.save(artifacts / f"mlp_{seed}.keras")
    neural = NeuralPredictor(members)
    candidates["mlp"] = NeuralPredictor(members[:1])
    timings["mlp"] = (time.perf_counter()-started)/len(members)
    if mode == "full":
        candidates["deep_ensemble"] = neural
        timings["deep_ensemble"] = time.perf_counter()-started
    base = candidates[chosen["model"]]
    cal = split["calibration"]
    calibrated = calibrate(base, x.iloc[cal], y[cal])
    conf = split["conformal"]
    cp = conformalize(calibrated, x.iloc[conf], y[conf], config["conformal_confidence"])
    screen = InputScreen().fit(xf, config["ood_quantile"])
    pi = split["policy"]
    xp = x.iloc[pi]
    p = calibrated.predict_proba(xp)[:, 1]
    u = signals(p, neural, xp, config["mc_passes"])
    sets = prediction_sets(cp, xp)
    valid, ood, _ = screen.inspect(xp)
    policy, sensitivity = select_policy(y[pi], p, u, sets, valid, ood, config["safety"])
    write_json(results / "policy_selection.json", {"policy": asdict(policy), "sweep": sensitivity,
        "objective": "Maximize coverage with zero observed accepted malignant policy cases and >=20% coverage; tie chooses smaller threshold.",
        "n": len(pi), "warning": "Empirical policy selection does not establish zero real-world risk."})
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + mode
    bundle = {"base": base, "calibrated": calibrated, "conformal": cp, "neural": neural,
              "screen": screen, "policy": asdict(policy), "features": list(x.columns), "version": version,
              "candidates": candidates, "background": xf.sample(24, random_state=config["seed"])}
    joblib.dump(bundle, artifacts / "bundle.joblib", compress=3)
    runtime = {p: importlib.metadata.version(p) for p in ["numpy", "scikit-learn", "tensorflow", "keras", "mapie", "xgboost", "joblib"]}
    metadata = {"version": version, "mode": mode, "features": list(x.columns), "config": config,
                "class_mapping": {"0": "BENIGN", "1": "MALIGNANT"}, "selected_model": chosen["model"],
                "selection_rule": "Maximize CV mean of accuracy and malignant recall, break ties with smaller NLL.",
                "data_sha256": describe_data(x, y)["sha256"], "python": platform.python_version(),
                "runtime_versions": runtime, "seeds": seeds, "training_seconds": timings,
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts.iterdir() if p.is_file()}}
    write_json(artifacts / "manifest.json", metadata)
    write_json(results / "training_history.json", histories)
    write_json(results / "experiment.json", metadata)
    # Everything above is now frozen. This is the only final-test prediction pass.
    from .reporting import evaluate_locked_run
    evaluate_locked_run(artifacts, results, split, x, y)
    log.info("completed mode=%s artifacts=%s", mode, artifacts)
    return artifacts


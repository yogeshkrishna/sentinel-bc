"""Metrics keep ordinary classification separate from selective acceptance."""
import numpy as np
from scipy.stats import norm
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             brier_score_loss, log_loss)
from .safety_gate import LOW


def wilson(successes, n, level=.95):
    if n == 0:
        return [None, None]
    z = norm.ppf(1 - (1 - level) / 2)
    p = successes / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return [float(center - half), float(center + half)]


def ece(y, p, bins=10):
    """Equal-width bins of malignant probability, weighted by bin frequency."""
    assignments = np.minimum((np.asarray(p) * bins).astype(int), bins - 1)
    return float(sum(np.mean(assignments == b) * abs(np.mean(p[assignments == b]) - np.mean(y[assignments == b]))
                     for b in range(bins) if np.any(assignments == b)))


def classification_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n": len(y), "positive_class": "MALIGNANT=1", "correct": int(sum(pred == y)),
            "accuracy": float(accuracy_score(y, pred)), "accuracy_ci95": wilson(sum(pred == y), len(y)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "malignant_recall": float(recall_score(y, pred, zero_division=0)),
            "recall_ci95": wilson(tp, tp + fn), "specificity": float(tn/(tn+fp)) if tn+fp else None,
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "pr_auc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "nll": float(log_loss(y, np.column_stack([1-p, p]), labels=[0, 1])), "ece": ece(y, p),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def selective_metrics(y, decisions):
    accept = np.asarray([d[0] == LOW for d in decisions])
    n, accepted = len(y), int(accept.sum())
    missed = int(sum(np.asarray(y)[accept] == 1))
    malignant = int(sum(np.asarray(y) == 1))
    return {"n": n, "accepted": accepted, "coverage": float(accepted/n), "review_rate": float(1-accepted/n),
            "accepted_accuracy": float(1-missed/accepted) if accepted else None,
            "accepted_accuracy_ci95": wilson(accepted-missed, accepted), "accepted_malignant": missed,
            "accepted_case_false_negative_rate": float(missed/accepted) if accepted else None,
            "malignant_escape_rate": float(missed/malignant) if malignant else None,
            "malignant_referral_rate": float(1-missed/malignant) if malignant else None,
            "referred_malignant": malignant-missed, "referred_benign": int(sum((np.asarray(y)==0) & ~accept)),
            "out_of_scope": sum(d[0] == "OUT_OF_SCOPE_INPUT" for d in decisions)}


def risk_coverage(y, p):
    order = np.argsort(np.maximum(p, 1-p))[::-1]
    errors = ((p >= .5).astype(int) != y)[order]
    return [{"coverage": float((i+1)/len(y)), "review_rate": float(1-(i+1)/len(y)),
             "risk": float(errors[:i+1].mean())} for i in range(len(y))]


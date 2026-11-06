"""Original conservative referral policy, independent of model implementation."""
from dataclasses import dataclass, asdict
import numpy as np

LOW = "LOW_RISK_ESTIMATE"
REVIEW = "HUMAN_REVIEW_REQUIRED"
OUT = "OUT_OF_SCOPE_INPUT"


@dataclass(frozen=True)
class SafetyPolicy:
    max_p_malignant: float = 0.0
    max_entropy: float = .65
    max_disagreement: float = .15
    max_mc_std: float = .15
    feasible: bool = False

    def __post_init__(self):
        if not 0 <= self.max_p_malignant <= .5:
            raise ValueError("Low-risk probability limit must be in [0, .5]")
        if not 0 <= self.max_entropy <= 1 or not 0 <= self.max_disagreement <= .5 or not 0 <= self.max_mc_std <= .5:
            raise ValueError("Uncertainty limits are outside their allowed ranges")


def decide(p, uncertainty: dict, prediction_set, valid: bool, ood: bool, policy: SafetyPolicy):
    values = [p] + [uncertainty.get(k, np.nan) for k in ("entropy", "disagreement", "mc_std")]
    if not valid or not np.isfinite(values).all() or not 0 <= p <= 1:
        return OUT, ["INVALID_INPUT_OR_SIGNAL"]
    if ood:
        return OUT, ["FEATURE_SPACE_OOD"]
    reasons = []
    if not policy.feasible:
        reasons.append("NO_FEASIBLE_DEVELOPMENT_POLICY")
    if p >= policy.max_p_malignant:
        reasons.append("MALIGNANT_PROBABILITY")
    for key, limit in [("entropy", policy.max_entropy), ("disagreement", policy.max_disagreement), ("mc_std", policy.max_mc_std)]:
        if uncertainty[key] > limit:
            reasons.append("HIGH_" + key.upper())
    if list(prediction_set) != [True, False]:
        reasons.append("CONFORMAL_SET_NOT_BENIGN_SINGLETON")
    return (REVIEW, reasons) if reasons else (LOW, ["SUPPORTED_LOW_RISK_RESEARCH_ESTIMATE"])


def apply_policy(p, u, sets, valid, ood, policy):
    return [decide(float(p[i]), {k: float(v[i]) for k, v in u.items()}, sets[i], bool(valid[i]), bool(ood[i]), policy)
            for i in range(len(p))]


def select_policy(y, p, u, sets, valid, ood, settings):
    """Maximize acceptance with zero observed accepted malignancies and >=20% coverage.

    This is an empirical constraint on a small development sample, not a risk bound.
    Other uncertainty limits are prespecified in config, avoiding a large search.
    """
    rows, candidates = [], []
    for threshold in np.linspace(.01, .49, 49):
        policy = SafetyPolicy(float(threshold), settings["max_entropy"], settings["max_disagreement"], settings["max_mc_std"], True)
        decisions = apply_policy(p, u, sets, valid, ood, policy)
        accept = np.array([d[0] == LOW for d in decisions])
        missed = int(sum(y[accept] == 1))
        coverage = float(accept.mean())
        feasible = missed <= settings["max_missed_malignant"] and coverage >= settings["min_coverage"]
        rows.append({"threshold": float(threshold), "n": len(y), "accepted": int(accept.sum()),
                     "coverage": coverage, "review_rate": 1 - coverage, "accepted_malignant": missed, "feasible": feasible})
        if feasible:
            candidates.append((int(accept.sum()), -float(threshold), policy))
    policy = max(candidates, key=lambda item: item[:2])[2] if candidates else SafetyPolicy()
    return policy, rows


"""Signals describe model behavior and do not guarantee individual reliability."""
import numpy as np


def entropy(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def signals(p, neural, x, passes=20):
    members = neural.member_probabilities(x)
    mc = neural.mc_probabilities(x, passes)
    return {"entropy": entropy(p), "confidence": np.maximum(p, 1 - p),
            "disagreement": members.std(axis=0), "mc_mean": mc.mean(axis=0), "mc_std": mc.std(axis=0)}


"""Training-only transforms and deliberately simple feature-space screening."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.covariance import LedoitWolf


def transformer():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler())


class InputScreen:
    def fit(self, x, quantile=.995):
        self.transform = transformer().fit(x)
        z = self.transform.transform(x)
        self.covariance = LedoitWolf().fit(z)
        self.distance_limit = float(np.quantile(self.covariance.mahalanobis(z), quantile) * 1.5)
        self.upper = np.asarray(x).max(axis=0) * 2
        return self

    def inspect(self, x):
        a = np.asarray(x, dtype=float)
        valid = np.isfinite(a).all(axis=1) & (a >= 0).all(axis=1)
        clean = np.clip(np.nan_to_num(a, nan=0, posinf=0, neginf=0), 0, self.upper * 100)
        distance = self.covariance.mahalanobis(self.transform.transform(pd.DataFrame(clean, columns=x.columns)))
        ood = (distance > self.distance_limit) | (a > self.upper).any(axis=1)
        return valid, ood, distance


def validate_frame(x: pd.DataFrame, features: list[str], max_rows=100) -> pd.DataFrame:
    if x.columns.duplicated().any() or set(x.columns) != set(features):
        raise ValueError("CSV must contain exactly the 30 named WDBC feature columns; no IDs or diagnosis column")
    if not 1 <= len(x) <= max_rows:
        raise ValueError(f"Provide between 1 and {max_rows} rows")
    try:
        return x.loc[:, features].astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Measurements must be numeric; missing or nonfinite values are out of scope") from exc


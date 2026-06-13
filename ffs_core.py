"""
ffs_core.py — Shared helpers for greedy forward feature selection per
                model family.

Used by run_pipeline.py (global FFS) and lithology_analysis.py (per-class
FFS). FFS uses a fast, fixed configuration per family — the heavy
hyperparameter tuning happens in the best-model comparison step, not in the
FFS loop. The returned per-step delta-R^2 values can be aggregated across
targets and model families to build the composite ranking (Eq. 1 in the
paper).
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from xgboost import XGBRegressor

_GAM_AVAILABLE = True  # spline+ridge replacement is always available


class _PLSWrapper(BaseEstimator, RegressorMixin):
    """sklearn-compatible PLS wrapper that returns 1-D predictions.

    Uses sklearn-style clone semantics: fitted attributes end with `_`.
    """

    def __init__(self, n_components=3):
        self.n_components = n_components

    def fit(self, X, y):
        X = np.asarray(X)
        nc = min(self.n_components, X.shape[1], max(1, X.shape[0] - 1))
        self.pls_ = PLSRegression(n_components=max(1, nc))
        self.pls_.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.pls_.predict(np.asarray(X)).ravel()


class _GAMWrapper(BaseEstimator, RegressorMixin):
    """GAM via per-feature spline basis + Ridge. Scales linearly in n."""

    def __init__(self, alpha=1.0, n_knots=5, degree=3):
        self.alpha = alpha
        self.n_knots = n_knots
        self.degree = degree

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.spline_ = SplineTransformer(
            n_knots=self.n_knots, degree=self.degree,
            knots="quantile", include_bias=False,
        )
        Z = self.spline_.fit_transform(X)
        self.ridge_ = Ridge(alpha=self.alpha)
        self.ridge_.fit(Z, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        Z = self.spline_.transform(np.asarray(X, dtype=float))
        return self.ridge_.predict(Z)


class _SVRWrapper(BaseEstimator, RegressorMixin):
    """Nystroem-approximated RBF kernel + Ridge. Linear scaling in n."""

    def __init__(self, alpha=1.0, gamma=0.1, n_components=200, random_state=42):
        self.alpha = alpha
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        nc = min(self.n_components, X.shape[0])
        self.nystroem_ = Nystroem(kernel="rbf", gamma=self.gamma,
                                  n_components=nc, random_state=self.random_state)
        Z = self.nystroem_.fit_transform(X)
        self.ridge_ = Ridge(alpha=self.alpha)
        self.ridge_.fit(Z, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        Z = self.nystroem_.transform(np.asarray(X, dtype=float))
        return self.ridge_.predict(Z)


# Names of model families supported by ffs_model().
FFS_FAMILIES_DEFAULT = [
    "Ridge", "PLS", "GAM", "ExtraTrees", "RF",
    "XGBoost", "SVR", "KNN", "MLP",
]


def ffs_model(name: str):
    """Return a fast, fixed-config base estimator for the FFS loop."""
    if name == "Ridge":
        return Ridge(alpha=1.0)
    if name == "PLS":
        return _PLSWrapper(n_components=3)
    if name == "GAM":
        return _GAMWrapper(alpha=1.0, n_knots=5)
    if name == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=50, n_jobs=-1, random_state=42)
    if name == "RF":
        return RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
    if name == "XGBoost":
        return XGBRegressor(n_estimators=80, max_depth=5, learning_rate=0.1,
                            n_jobs=-1, random_state=42, verbosity=0,
                            objective="reg:squarederror")
    if name == "SVR":
        return _SVRWrapper(alpha=1.0, gamma=0.1, n_components=200)
    if name == "KNN":
        return KNeighborsRegressor(n_neighbors=5, weights="distance")
    if name == "MLP":
        return MLPRegressor(hidden_layer_sizes=(64,), alpha=0.01,
                            max_iter=200, random_state=42)
    raise KeyError(name)


def _cap(X_df: pd.DataFrame, y: np.ndarray, cap: int | None, seed: int = 42):
    if cap is not None and len(y) > cap:
        idx = np.random.default_rng(seed).choice(len(y), cap, replace=False)
        idx.sort()
        return X_df.iloc[idx].reset_index(drop=True), y[idx]
    return X_df.reset_index(drop=True), y


def cv_r2(X_df: pd.DataFrame, y: np.ndarray, feats: list,
          model_name: str, cap: int | None, n_splits: int = 3,
          seed: int = 42) -> float:
    """Mean CV R^2 for the given model family on the supplied features."""
    base = ffs_model(model_name)
    Xc, yc = _cap(X_df[feats], y, cap, seed)
    if len(yc) < n_splits + 1:
        return float("nan")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, te in kf.split(Xc):
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("mdl", deepcopy(base)),
        ])
        try:
            pipe.fit(Xc.iloc[tr], yc[tr])
            scores.append(float(r2_score(yc[te], pipe.predict(Xc.iloc[te]))))
        except Exception:
            scores.append(float("nan"))
    return float(np.nanmean(scores))


def greedy_ffs(X_df: pd.DataFrame, y: np.ndarray, candidates: list,
               model_name: str, cap: int | None = None,
               delta_stop: float = 0.005, n_splits: int = 3,
               seed: int = 42, verbose: bool = True) -> list:
    """Greedy FFS for a single (target, model_name) pair.

    Returns a list of dicts, one per selected step, with keys:
      step, feature, r2_cumulative, r2_delta.
    """
    selected: list = []
    remaining = list(candidates)
    steps = []
    baseline = 0.0
    for step in range(1, len(candidates) + 1):
        best_feat, best_r2 = None, -np.inf
        for feat in remaining:
            trial = selected + [feat]
            r2 = cv_r2(X_df, y, trial, model_name, cap, n_splits, seed)
            if r2 > best_r2:
                best_r2, best_feat = r2, feat
        if best_feat is None:
            break
        delta = best_r2 - baseline
        selected.append(best_feat)
        remaining.remove(best_feat)
        steps.append({
            "step": step,
            "feature": best_feat,
            "r2_cumulative": round(best_r2, 4),
            "r2_delta": round(delta, 4),
        })
        if verbose:
            print(f"    step {step:2d}: +{best_feat} -> "
                  f"R2={best_r2:.4f} (delta={delta:+.4f})")
        baseline = best_r2
        if step >= 3 and delta < delta_stop:
            if verbose:
                print(f"    [FFS] early stop: delta<{delta_stop}")
            break
    return steps


# No per-family sample caps inside FFS: GAM and SVR are now linear-scaling
# (spline+ridge and Nystroem+ridge respectively); all families run on the
# full per-source-balanced data set.
DEFAULT_FFS_CAPS: dict[str, int] = {}


def cap_for(model_name: str, override: dict | None = None) -> int | None:
    """No sample cap inside FFS by default. Returning None disables capping."""
    if override and model_name in override:
        return override[model_name]
    return None

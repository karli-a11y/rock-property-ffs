"""recursive_ffs.py — Iterated-exclusion forward feature selection.

Motivation
----------
Greedy FFS is path-dependent: once predictor A is chosen, every predictor
correlated with A loses its incremental Δ-R^2 budget, so its composite-score
contribution is suppressed regardless of its standalone informativeness.

This script removes the path dependence by repeating the *entire* FFS pipeline
(all 14 targets × all 9 model families, with the family-specific FFS
configuration) iteratively. After each full FFS round the global #1 predictor
is removed from the candidate pool, and the full FFS is re-run. The
procedure stops when fewer than two candidates remain.

Outputs
-------
results/tables/feature_relevance_recursive.csv with columns
    iteration, excluded_so_far, feature, label, S_iter, rank_iter
plus a derived aggregated ranking
results/tables/feature_relevance_recursive_mean.csv with columns
    feature, label, S_mean (mean S across all iterations where the feature
    is still in the pool), n_iterations_present, S_iter1, S_max.

The script is independent of run_pipeline.py: it reads the master CSV,
re-uses MODEL_FAMILIES, but redefines the FFS helpers in-file to keep the
two scripts decoupled.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge as _Ridge
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).parent
OUT_DIR = BASE / os.environ.get("OUT_DIR_NAME", "results")
OUT_TAB = OUT_DIR / "tables"
OUT_TAB.mkdir(parents=True, exist_ok=True)
MASTER_CSV = BASE / "data" / "master_table.csv.gz"

from variables import NUMERIC_FEATURES, FEATURE_LABELS, HARMONIZATION_RISK
from target_registry import TargetRegistry
from master_split import apply_source_split

EXCLUDE_SOURCES = {"NorwayNPD"}
LOG_TARGETS = {"permeability_m2"}
FFS_N_SPLITS = 3
FFS_DELTA_STOP = 0.005


# ── Wrappers (mirroring run_pipeline) ─────────────────────────────────────────

class PLSWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=3):
        self.n_components = n_components
    def fit(self, X, y):
        X = np.asarray(X)
        nc = min(self.n_components, X.shape[1], max(1, X.shape[0] - 1))
        self.pls_ = PLSRegression(n_components=max(1, nc)).fit(X, y)
        return self
    def predict(self, X):
        return self.pls_.predict(np.asarray(X)).ravel()


class GAMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, n_knots=5):
        self.alpha = alpha
        self.n_knots = n_knots
    def fit(self, X, y):
        X = np.asarray(X)
        self.spl_ = SplineTransformer(n_knots=self.n_knots, degree=3,
                                       knots="uniform").fit(X)
        Z = self.spl_.transform(X)
        self.rdg_ = _Ridge(alpha=self.alpha).fit(Z, y)
        return self
    def predict(self, X):
        return self.rdg_.predict(self.spl_.transform(np.asarray(X)))


class SVRWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, gamma=0.1, n_components=200):
        self.alpha = alpha; self.gamma = gamma; self.n_components = n_components
    def fit(self, X, y):
        X = np.asarray(X)
        self.nys_ = Nystroem(gamma=self.gamma,
                              n_components=min(self.n_components, X.shape[0]),
                              random_state=42).fit(X)
        Z = self.nys_.transform(X)
        self.rdg_ = _Ridge(alpha=self.alpha).fit(Z, y)
        return self
    def predict(self, X):
        return self.rdg_.predict(self.nys_.transform(np.asarray(X)))


def _ffs_model(name: str):
    if name == "Ridge":      return Ridge(alpha=1.0)
    if name == "PLS":        return PLSWrapper(n_components=3)
    if name == "GAM":        return GAMWrapper(alpha=1.0, n_knots=5)
    if name == "ExtraTrees": return ExtraTreesRegressor(n_estimators=50, n_jobs=-1, random_state=42)
    if name == "RF":         return RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
    if name == "XGBoost":    return XGBRegressor(n_estimators=80, max_depth=5, learning_rate=0.1,
                                                  n_jobs=-1, random_state=42, verbosity=0,
                                                  objective="reg:squarederror")
    if name == "SVR":        return SVRWrapper(alpha=1.0, gamma=0.1, n_components=200)
    if name == "KNN":        return KNeighborsRegressor(n_neighbors=5, weights="distance")
    if name == "MLP":        return MLPRegressor(hidden_layer_sizes=(64,), alpha=0.01,
                                                  max_iter=200, random_state=42)
    raise KeyError(name)


MODEL_NAMES = ["Ridge", "PLS", "GAM", "ExtraTrees", "RF", "XGBoost", "SVR", "KNN", "MLP"]


def _ffs_r2(X_df: pd.DataFrame, y: np.ndarray, feats: list,
             model_name: str) -> float:
    base = _ffs_model(model_name)
    kf = KFold(n_splits=FFS_N_SPLITS, shuffle=True, random_state=42)
    scores = []
    for tr, te in kf.split(X_df):
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("mdl", deepcopy(base)),
        ])
        try:
            pipe.fit(X_df.iloc[tr][feats], y[tr])
            scores.append(float(r2_score(y[te], pipe.predict(X_df.iloc[te][feats]))))
        except Exception:
            scores.append(float("nan"))
    return float(np.nanmean(scores))


def forward_feature_selection(X_df: pd.DataFrame, y: np.ndarray,
                               candidates: list, model_name: str) -> list:
    selected, remaining = [], list(candidates)
    steps = []
    baseline = 0.0
    for step in range(1, len(candidates) + 1):
        best_feat, best_r2 = None, -np.inf
        for feat in remaining:
            trial = selected + [feat]
            r2 = _ffs_r2(X_df, y, trial, model_name)
            if r2 > best_r2:
                best_r2, best_feat = r2, feat
        if best_feat is None:
            break
        delta = best_r2 - baseline
        selected.append(best_feat)
        remaining.remove(best_feat)
        steps.append({"step": step, "feature": best_feat,
                       "r2_cumulative": best_r2, "r2_delta": delta})
        baseline = best_r2
        if step >= 3 and delta < FFS_DELTA_STOP:
            break
    return steps


# ── Aggregation Eq. (1) ───────────────────────────────────────────────────────

def aggregate_S(ffs_rows: list, all_features: list) -> pd.DataFrame:
    """Sum Δ-R^2 across (target, model) and normalise so Σ S_j = 1."""
    by_feat: dict[str, float] = {f: 0.0 for f in all_features}
    for row in ffs_rows:
        d = max(float(row.get("r2_delta", 0.0)), 0.0)
        by_feat[row["feature"]] = by_feat.get(row["feature"], 0.0) + d
    s = pd.Series(by_feat, name="S")
    total = s.sum()
    if total > 0:
        s = s / total
    out = (pd.DataFrame({"feature": s.index, "S": s.values})
             .merge(pd.DataFrame({"feature": list(FEATURE_LABELS.keys()),
                                   "label": list(FEATURE_LABELS.values())}),
                    on="feature", how="left")
             .sort_values("S", ascending=False)
             .reset_index(drop=True))
    out["rank"] = np.arange(1, len(out) + 1)
    return out


# ── Main driver ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading master table …")
    master = pd.read_csv(MASTER_CSV, low_memory=False)
    master = master[~master["source_db"].isin(EXCLUDE_SOURCES)].reset_index(drop=True)
    master = apply_source_split(master)
    print(f"  {len(master):,} rows  ({master['source_db'].nunique()} sub-sources)")

    registry = TargetRegistry(master, BASE / "config" / "targets.yaml")
    analyzed_targets = registry.get_all_analyzed()
    print(f"  Analyzed targets: {len(analyzed_targets)}")

    # Build candidate pool: all qualified features minus permeability target log-transform handled inside
    # We use the union of every per-target predictor set, which is the standardised pool.
    pool = list(set().union(*[set(registry.build_target_subset(master, t)[1])
                              for t in analyzed_targets]))
    pool.sort()
    print(f"  Initial pool: {len(pool)} features")

    all_rows = []
    excluded: list[str] = []
    iter_idx = 0

    while len(pool) >= 2:
        iter_idx += 1
        t0 = time.time()
        print(f"\n{'='*70}")
        print(f"ITERATION {iter_idx}  |  pool size = {len(pool)}  |  "
              f"excluded so far: {excluded}")
        print('='*70)

        ffs_rows = []
        for target in analyzed_targets:
            sub, _orig_preds = registry.build_target_subset(master, target)
            if sub.empty:
                continue
            # Restrict to the iteration's candidate pool (and intersect with the
            # target's admissible predictor set, which respects circularity rules).
            preds = [p for p in _orig_preds if p in pool]
            if len(preds) < 2:
                continue
            y = sub[target].values
            X = sub[preds].copy()
            if target in LOG_TARGETS:
                pos = y > 0
                X, y = X[pos].reset_index(drop=True), np.log10(y[pos])
            for fam in MODEL_NAMES:
                try:
                    steps = forward_feature_selection(X, y, preds, fam)
                except Exception as exc:
                    print(f"  [fail] {fam}/{target}: {exc}")
                    continue
                for row in steps:
                    row["target"] = target
                    row["model"]  = fam
                    ffs_rows.extend([row])

        S_df = aggregate_S(ffs_rows, pool)
        S_df["iteration"] = iter_idx
        S_df["excluded_so_far"] = ",".join(excluded) if excluded else "(none)"
        all_rows.append(S_df)

        winner = str(S_df.iloc[0]["feature"])
        winner_S = float(S_df.iloc[0]["S"])
        dt = time.time() - t0
        print(f"\n  → winner #{iter_idx}: {winner}  (S = {winner_S:.4f})  "
              f"[iter took {dt/60:.1f} min]")
        excluded.append(winner)
        pool = [p for p in pool if p != winner]

        # Save running results so a crash doesn't lose progress
        pd.concat(all_rows, ignore_index=True).to_csv(
            OUT_TAB / "feature_relevance_recursive.csv", index=False)

    # Derived aggregate: mean S per feature across iterations
    full = pd.concat(all_rows, ignore_index=True)
    agg = (full.groupby("feature")
                .agg(S_mean=("S", "mean"),
                     S_max=("S", "max"),
                     S_iter1=("S", lambda s: float(s.iloc[0])),
                     n_iterations_present=("S", "size"))
                .reset_index()
                .merge(pd.DataFrame({"feature": list(FEATURE_LABELS.keys()),
                                      "label": list(FEATURE_LABELS.values())}),
                       on="feature", how="left")
                .sort_values("S_mean", ascending=False))
    agg.to_csv(OUT_TAB / "feature_relevance_recursive_mean.csv", index=False)
    print("\nDone. Wrote feature_relevance_recursive[.csv|_mean.csv]")


if __name__ == "__main__":
    main()

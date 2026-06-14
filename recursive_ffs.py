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


# ── Per-target iterated-exclusion driver ──────────────────────────────────────

def run_pertarget_recursive(master: pd.DataFrame, analyzed_targets: list,
                             registry, single_target: str | None = None,
                             verbose: bool = False) -> pd.DataFrame:
    """Run iterated-exclusion FFS **independently for each (target, family)**.

    For one (target, family) pair the recursion is:
      iteration 1: full FFS over the target's admissible predictor set,
                   recording every step's Δ-R^2; the step-1 winner is the
                   pair's own iteration winner.
      iteration 2: remove that winner from this target's pool and re-run FFS.
      ... until fewer than two candidates remain.

    Each target removes *its own* winners (not a global one), so the
    substitution structure is resolved target by target. Returns a long-form
    DataFrame with columns: target, model, iteration, step, feature, r2_delta.
    """
    targets = [single_target] if single_target else analyzed_targets
    records: list[dict] = []
    for target in targets:
        sub, orig_preds = registry.build_target_subset(master, target)
        if sub.empty or len(orig_preds) < 2:
            continue
        y_full = sub[target].values
        X_full = sub[orig_preds].copy()
        if target in LOG_TARGETS:
            pos = y_full > 0
            X_full = X_full[pos].reset_index(drop=True)
            y_full = np.log10(y_full[pos])
        t0 = time.time()
        for fam in MODEL_NAMES:
            tpool = list(orig_preds)
            it = 0
            while len(tpool) >= 2:
                it += 1
                try:
                    steps = forward_feature_selection(X_full[tpool], y_full,
                                                       tpool, fam)
                except Exception as exc:
                    if verbose:
                        print(f"    [fail] {fam}/{target} it{it}: {exc}")
                    break
                if not steps:
                    break
                for s in steps:
                    records.append({
                        "target":    target,
                        "model":     fam,
                        "iteration": it,
                        "step":      s["step"],
                        "feature":   s["feature"],
                        "r2_delta":  float(s["r2_delta"]),
                    })
                winner = steps[0]["feature"]          # this pair's own winner
                tpool = [p for p in tpool if p != winner]
                if verbose:
                    print(f"    {target:20s} {fam:10s} it{it:2d}: "
                          f"remove {winner}  (pool now {len(tpool)})")
        if verbose or not single_target:
            print(f"  [done] {target:20s} "
                  f"[{(time.time()-t0)/60:.1f} min]")
    return pd.DataFrame(records)


def summarise(records: pd.DataFrame):
    """Build the two output tables from the long-form step records.

    Ranking (S_mean column): each predictor's total positive Δ-R^2 summed over
    *all* loops (targets, families, exclusion iterations, steps), normalised by
    the grand total so Σ_j S_j = 1. This is the parameter-ranking contribution.

    Per-iteration shares (S_iter, and the derived S_iter1 / S_max): within each
    exclusion-depth i the positive Δ-R^2 is aggregated across (target, family)
    and normalised so Σ_j S_j^(i) = 1, giving the substitution heatmap, the
    greedy depth-1 score S_iter1, and the latent maximum S_max.
    """
    rec = records.copy()
    rec["pos"] = rec["r2_delta"].clip(lower=0.0)

    # ranking: sum over all loops / grand total
    grand = rec["pos"].sum()
    rank = rec.groupby("feature")["pos"].sum()
    rank = rank / grand if grand > 0 else rank

    # per-depth normalised shares
    per_iter = rec.groupby(["iteration", "feature"], as_index=False)["pos"].sum()
    per_iter["S_iter"] = per_iter.groupby("iteration")["pos"].transform(
        lambda s: s / s.sum() if s.sum() > 0 else 0.0)

    long = per_iter[["iteration", "feature", "S_iter"]].copy()
    long["label"] = long["feature"].map(FEATURE_LABELS)
    long = long.sort_values(["iteration", "S_iter"], ascending=[True, False])

    s_iter1 = per_iter[per_iter["iteration"] == 1].set_index("feature")["S_iter"]
    s_max = per_iter.groupby("feature")["S_iter"].max()
    n_present = per_iter.groupby("feature")["iteration"].nunique()

    summary = pd.DataFrame({"feature": rank.index, "S_mean": rank.values})
    summary["S_max"] = summary["feature"].map(s_max).fillna(0.0)
    summary["S_iter1"] = summary["feature"].map(s_iter1).fillna(0.0)
    summary["n_iterations_present"] = (summary["feature"].map(n_present)
                                       .fillna(0).astype(int))
    summary["label"] = summary["feature"].map(FEATURE_LABELS)
    summary = summary.sort_values("S_mean", ascending=False).reset_index(drop=True)
    return long, summary


# ── Main driver ───────────────────────────────────────────────────────────────

def main() -> None:
    import os as _os
    test_target = _os.environ.get("TEST_TARGET")

    print("Loading master table …")
    master = pd.read_csv(MASTER_CSV, low_memory=False)
    master = master[~master["source_db"].isin(EXCLUDE_SOURCES)].reset_index(drop=True)
    master = apply_source_split(master)
    print(f"  {len(master):,} rows  ({master['source_db'].nunique()} sub-sources)")

    registry = TargetRegistry(master, BASE / "config" / "targets.yaml")
    analyzed_targets = registry.get_all_analyzed()
    print(f"  Analyzed targets: {len(analyzed_targets)}")

    if test_target:
        print(f"\n=== TEST MODE: single target '{test_target}' ===")
        rec = run_pertarget_recursive(master, analyzed_targets, registry,
                                      single_target=test_target, verbose=True)
        long, summary = summarise(rec)
        print(f"\n  records: {len(rec)} rows, "
              f"{rec['iteration'].max()} exclusion iterations")
        print("\n  Ranking contribution (S_mean) for this single target:")
        print(summary[["feature", "S_mean", "S_iter1", "S_max",
                       "n_iterations_present"]].to_string(index=False))
        return

    t0 = time.time()
    parts: list = []
    for ti, tgt in enumerate(analyzed_targets, 1):
        print(f"\n[{ti}/{len(analyzed_targets)}] target {tgt} …", flush=True)
        r = run_pertarget_recursive(master, analyzed_targets, registry,
                                    single_target=tgt)
        if not r.empty:
            parts.append(r)
            # checkpoint after every target so a long run survives a crash
            pd.concat(parts, ignore_index=True).to_csv(
                OUT_TAB / "feature_relevance_recursive_steps.csv", index=False)
            print(f"  [checkpoint] {sum(len(p) for p in parts)} step rows saved",
                  flush=True)
    rec = pd.concat(parts, ignore_index=True)
    long, summary = summarise(rec)
    long.to_csv(OUT_TAB / "feature_relevance_recursive.csv", index=False)
    summary.to_csv(OUT_TAB / "feature_relevance_recursive_mean.csv", index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. "
          f"Wrote feature_relevance_recursive[_steps|.csv|_mean.csv]")
    print("\nFull ranking:")
    print(summary[["feature", "S_mean", "S_iter1", "S_max"]].to_string(index=False))


if __name__ == "__main__":
    main()

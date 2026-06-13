"""
loso_validation.py — Systematic Leave-One-Source-Out validation.

For each of the nine source databases:
  hold the source out, refit the canonical best-model family per target
  on the remaining eight sources, predict on the held-out source, and
  record the cross-validated R^2 on the training pool, the hold-out
  R^2, and the transfer gap delta_R^2 = ho_r2 - cv_r2_train.

Only (source, target) pairs with at least 30 samples of the target in
the held-out source are evaluated.

Output: results/tables/loso_validation.csv
"""
from __future__ import annotations

import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

import os
BASE = Path(__file__).parent
MASTER_CSV = BASE / "data" / "master_table.csv.gz"
OUT_DIR = BASE / os.environ.get("OUT_DIR_NAME", "results") / "tables"

from variables import FEATURE_LABELS
from target_registry import TargetRegistry

EXCLUDE_FROM_MAIN = {"NorwayNPD"}

PER_SOURCE_CAP = 8_000
LOG_TARGETS = {"permeability_m2"}
SEED = 42
MIN_HOLDOUT_N = 30


def make_model(family: str):
    if family == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=150, max_depth=20,
                                    min_samples_leaf=1,
                                    random_state=SEED, n_jobs=-1)
    if family == "RF":
        return RandomForestRegressor(n_estimators=150, max_depth=20,
                                      min_samples_leaf=1,
                                      random_state=SEED, n_jobs=-1)
    if family == "XGBoost":
        return XGBRegressor(n_estimators=150, max_depth=5,
                             learning_rate=0.1, subsample=0.8,
                             objective="reg:squarederror",
                             random_state=SEED, n_jobs=-1, verbosity=0)
    return ExtraTreesRegressor(n_estimators=150, random_state=SEED, n_jobs=-1)


def evaluate_source(master: pd.DataFrame, holdout_source: str,
                    bm_global: pd.DataFrame, registry_full) -> list:
    train_master = master[master["source_db"] != holdout_source].reset_index(drop=True)
    holdout      = master[master["source_db"] == holdout_source].reset_index(drop=True)
    if len(holdout) < MIN_HOLDOUT_N:
        return []

    registry_train = TargetRegistry(train_master, BASE / "config" / "targets.yaml")
    analyzed = registry_train.get_all_analyzed()

    rows = []
    for target in analyzed:
        if target not in holdout.columns:
            continue
        ho = holdout.dropna(subset=[target]).copy()
        if len(ho) < MIN_HOLDOUT_N:
            continue
        ho_y = ho[target].values
        if target in LOG_TARGETS:
            mask = ho_y > 0
            ho = ho[mask].reset_index(drop=True)
            ho_y = np.log10(ho[target].values)
            if len(ho_y) < MIN_HOLDOUT_N:
                continue

        # Training subset
        sub_tr, preds = registry_train.build_target_subset(train_master, target)
        if sub_tr.empty or len(preds) < 2:
            continue
        df_tr = sub_tr.reset_index(drop=True)
        X_tr = df_tr[preds].copy()
        y_tr = df_tr[target].values
        if target in LOG_TARGETS:
            mask = y_tr > 0
            df_tr = df_tr[mask].reset_index(drop=True)
            X_tr = X_tr[mask].reset_index(drop=True)
            y_tr = np.log10(y_tr[mask])

        # Per-source cap
        rng = np.random.default_rng(99)
        keep = []
        for _, grp in df_tr.iloc[:len(y_tr)].groupby("source_db"):
            idx = grp.index.tolist()
            if len(idx) > PER_SOURCE_CAP:
                idx = rng.choice(idx, PER_SOURCE_CAP, replace=False).tolist()
            keep.extend(idx)
        keep = sorted([k for k in keep if k < len(y_tr)])
        X_tr = X_tr.iloc[keep].reset_index(drop=True)
        y_tr = y_tr[np.array(keep)]
        if len(y_tr) < 30:
            continue

        # Common predictors
        common_preds = [p for p in preds if p in ho.columns]
        if len(common_preds) < 2:
            continue
        X_tr_c = X_tr[common_preds].copy()
        ho_X = ho[common_preds].copy()

        # Best-model family
        bm_row = bm_global[bm_global["target"] == target]
        family = str(bm_row["best_model"].values[0]) if not bm_row.empty else "ExtraTrees"

        # 5-fold CV R^2 on training
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_scores = []
        for tr_i, te_i in kf.split(X_tr_c):
            p = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc",  StandardScaler()),
                ("mdl", make_model(family)),
            ])
            try:
                p.fit(X_tr_c.iloc[tr_i], y_tr[tr_i])
                cv_scores.append(r2_score(y_tr[te_i], p.predict(X_tr_c.iloc[te_i])))
            except Exception:
                cv_scores.append(float("nan"))
        cv_r2 = float(np.nanmean(cv_scores))

        # Refit and predict on hold-out
        try:
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc",  StandardScaler()),
                ("mdl", make_model(family)),
            ])
            pipe.fit(X_tr_c, y_tr)
            ho_pred = pipe.predict(ho_X)
            ho_r2 = float(r2_score(ho_y, ho_pred))
        except Exception:
            ho_r2 = float("nan")

        delta = ho_r2 - cv_r2
        rows.append({
            "source_held_out": holdout_source,
            "target":          target,
            "label":           FEATURE_LABELS.get(target, target),
            "best_model":      family,
            "n_train":         int(len(y_tr)),
            "n_holdout":       int(len(ho_y)),
            "n_predictors":    len(common_preds),
            "cv_r2_train":     round(cv_r2, 4),
            "holdout_r2":      round(ho_r2, 4),
            "delta_r2":        round(delta, 4),
        })
        print(f"  {holdout_source:20s} {target:25s} CV={cv_r2:+.3f}  HO={ho_r2:+.3f}  d={delta:+.3f}  (n_ho={len(ho_y)})")
    return rows


def main():
    print("Loading master table ...")
    master = pd.read_csv(MASTER_CSV, low_memory=False)
    master = master[~master["source_db"].isin(EXCLUDE_FROM_MAIN)].reset_index(drop=True)

    # Apply compendium split BEFORE LOSO: each primary_reference within a
    # compendium (P3, AMPEDEK, GlobalDB, MidGerman, HeapViolay, ESReviews,
    # USGS_WUS, USGS_MojaveREE, USGS_GreatBasinPlutons, Cornwall) becomes
    # its own sub-source. Otherwise LOSO on AMPEDEK = single hold-out for
    # 26 distinct lab campaigns, which is methodologically wrong.
    from master_split import apply_source_split
    master = apply_source_split(master)
    print(f"After compendium split: {master['source_db'].nunique()} sub-sources")

    # Use only sub-sources with enough samples to provide a meaningful hold-out
    src_counts = master["source_db"].value_counts()
    sources = sorted(src_counts[src_counts >= 30].index)
    print(f"Sources to LOSO over ({len(sources)} with n>=30): {sources}")

    bm_global = pd.read_csv(OUT_DIR / "best_model_global.csv")
    registry_full = TargetRegistry(master, BASE / "config" / "targets.yaml")

    all_rows = []
    for src in sources:
        print(f"\n=== Holding out: {src} (n={int((master['source_db']==src).sum()):,}) ===")
        all_rows.extend(evaluate_source(master, src, bm_global, registry_full))

    out = pd.DataFrame(all_rows)
    out.to_csv(OUT_DIR / "loso_validation.csv", index=False)

    print()
    print(f"Saved {OUT_DIR / 'loso_validation.csv'} ({len(out)} rows)")

    # Summary: delta_r2 distribution per target
    print()
    print("Per-target delta R^2 summary across hold-out sources:")
    if not out.empty:
        summary = (out.groupby("label")["delta_r2"]
                    .agg(["count", "median", "min", "max"])
                    .sort_values("median", ascending=False))
        print(summary.to_string())


if __name__ == "__main__":
    main()

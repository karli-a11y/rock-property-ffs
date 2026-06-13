"""
holdout_validation.py — External hold-out validation by source.

Strategy: hold out one whole source database from training, fit the
canonical best-model family per target on the remainder, then predict
on the held-out source. Report:

  - in-sample 5-fold CV R^2 (training-only)
  - out-of-sample R^2 on the held-out source
  - delta R^2 (transfer gap)

We choose Mielke2017 as the held-out source: it is mid-sized (1430 rows
in the master) and contributes to several thermal targets. Holding it
out leaves the other thermal sources (Weydt2020, AMPEDEK, GlobalDB,
P^3, Valgarður) to anchor the model.
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
HOLDOUT_SOURCE = "Mielke2017"

PER_SOURCE_CAP = 8_000
LOG_TARGETS = {"permeability_m2"}
SEED = 42


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


def main():
    print("Loading master table …")
    master = pd.read_csv(MASTER_CSV, low_memory=False)
    master = master[~master["source_db"].isin(EXCLUDE_FROM_MAIN)].reset_index(drop=True)

    train_master = master[master["source_db"] != HOLDOUT_SOURCE].reset_index(drop=True)
    holdout      = master[master["source_db"] == HOLDOUT_SOURCE].reset_index(drop=True)
    print(f"  Train: {len(train_master):,} rows; "
          f"Hold-out ({HOLDOUT_SOURCE}): {len(holdout):,} rows")

    registry_train = TargetRegistry(train_master, BASE / "config" / "targets.yaml")
    analyzed = registry_train.get_all_analyzed()

    bm_global = pd.read_csv(OUT_DIR / "best_model_global.csv")

    rows = []
    for target in analyzed:
        # In-sample subset (after capping) on the training master only
        sub_tr, preds = registry_train.build_target_subset(train_master, target)
        if sub_tr.empty or len(preds) < 2:
            continue

        # Fetch canonical best model for this target
        bm_row = bm_global[bm_global["target"] == target]
        if bm_row.empty:
            continue
        family = str(bm_row["best_model"].values[0])

        # Prepare training X, y
        df_tr = sub_tr.reset_index(drop=True)
        X_tr = df_tr[preds].copy()
        y_tr = df_tr[target].values

        if target in LOG_TARGETS:
            mask = y_tr > 0
            X_tr, y_tr = X_tr[mask].reset_index(drop=True), np.log10(y_tr[mask])

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

        # Hold-out X, y for this target
        if target not in holdout.columns:
            continue
        ho = holdout.dropna(subset=[target])
        if ho.empty or len(ho) < 30:
            continue

        # Predictors must exist in hold-out frame
        common_preds = [p for p in preds if p in ho.columns]
        if len(common_preds) < 2:
            continue
        ho_X = ho[common_preds].copy()
        ho_y = ho[target].values
        if target in LOG_TARGETS:
            mask = ho_y > 0
            ho_X, ho_y = ho_X[mask].reset_index(drop=True), np.log10(ho_y[mask])
        if len(ho_y) < 30:
            continue

        # Refit on train using common predictors
        X_tr_c = X_tr[common_preds].copy()

        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("mdl", make_model(family)),
        ])

        # In-sample 5-fold CV R^2 on training data
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_scores = []
        for tr_i, te_i in kf.split(X_tr_c):
            p = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc",  StandardScaler()),
                ("mdl", make_model(family)),
            ])
            p.fit(X_tr_c.iloc[tr_i], y_tr[tr_i])
            cv_scores.append(r2_score(y_tr[te_i], p.predict(X_tr_c.iloc[te_i])))
        cv_r2 = float(np.mean(cv_scores))

        # Refit on full train; predict on hold-out
        pipe.fit(X_tr_c, y_tr)
        ho_pred = pipe.predict(ho_X)
        ho_r2 = float(r2_score(ho_y, ho_pred))

        delta = ho_r2 - cv_r2
        rows.append({
            "target":        target,
            "label":         FEATURE_LABELS.get(target, target),
            "best_model":    family,
            "n_train":       int(len(y_tr)),
            "n_holdout":     int(len(ho_y)),
            "n_predictors":  len(common_preds),
            "cv_r2_train":   round(cv_r2, 4),
            "holdout_r2":    round(ho_r2, 4),
            "delta_r2":      round(delta, 4),
        })
        print(f"  {target:25s} CV-R2={cv_r2:+.3f}  HO-R2={ho_r2:+.3f}  "
              f"delta={delta:+.3f}  (n_ho={len(ho_y)})")

    out = pd.DataFrame(rows).sort_values("holdout_r2", ascending=False).reset_index(drop=True)
    out.to_csv(OUT_DIR / "holdout_validation.csv", index=False)
    print()
    print(f"Saved {OUT_DIR / 'holdout_validation.csv'} ({len(out)} targets)")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

"""
minimal_set_check.py — Non-destructive minimal-set check (CG review, comment 44).

Question: if only the cheap non-destructive backbone (porosity, bulk density,
grain density, V_P, V_S, thermal conductivity) were measured, how much of the
full-predictor best-model ceiling per target is retained?

For each analyzed target: restrict predictors to the non-destructive set
(minus the target itself), fit the canonical best-model family with the
canonical preprocessing (median imputation + scaling), and record the 5-fold
cross-validated R^2. Compare against the full-predictor ceiling in
best_model_global.csv.

Output: results/tables/minimal_set_check.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
MASTER_CSV = BASE / "data" / "master_table.csv.gz"
OUT_CSV = BASE / "results" / "tables" / "minimal_set_check.csv"

from variables import FEATURE_LABELS

EXCLUDE_FROM_MAIN = {"NorwayNPD"}
PER_SOURCE_CAP = 8_000
LOG_TARGETS = {"permeability_m2"}
SEED = 42

NONDESTRUCTIVE = [
    "porosity_pct",
    "bulk_density_gcm3",
    "grain_density_gcm3",
    "vp_dry_ms",
    "vs_dry_ms",
    "tc_dry_WmK",
]


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
    if family == "KNN":
        return KNeighborsRegressor(n_neighbors=5, weights="distance")
    return ExtraTreesRegressor(n_estimators=150, random_state=SEED, n_jobs=-1)


def main():
    print("Loading master table ...")
    master = pd.read_csv(MASTER_CSV, low_memory=False)
    master = master[~master["source_db"].isin(EXCLUDE_FROM_MAIN)].reset_index(drop=True)

    from master_split import apply_source_split
    master = apply_source_split(master)

    bm = pd.read_csv(BASE / "results" / "tables" / "best_model_global.csv")

    rows = []
    for _, bm_row in bm.iterrows():
        target = bm_row["target"]
        family = str(bm_row["best_model"])
        full_r2 = float(bm_row["best_r2"])

        preds = [p for p in NONDESTRUCTIVE if p != target and p in master.columns]
        df = master.dropna(subset=[target]).reset_index(drop=True)
        # require at least one minimal-set predictor actually measured
        df = df[df[preds].notna().any(axis=1)].reset_index(drop=True)
        y = df[target].values
        if target in LOG_TARGETS:
            mask = y > 0
            df = df[mask].reset_index(drop=True)
            y = np.log10(df[target].values)
        if len(y) < 30:
            continue

        # per-source cap as in the canonical pipeline
        rng = np.random.default_rng(99)
        keep = []
        for _, grp in df.groupby("source_db"):
            idx = grp.index.tolist()
            if len(idx) > PER_SOURCE_CAP:
                idx = rng.choice(idx, PER_SOURCE_CAP, replace=False).tolist()
            keep.extend(idx)
        keep = sorted(keep)
        X = df[preds].iloc[keep].reset_index(drop=True)
        y = y[np.array(keep)]

        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        scores = []
        for tr_i, te_i in kf.split(X):
            p = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("mdl", make_model(family)),
            ])
            try:
                p.fit(X.iloc[tr_i], y[tr_i])
                scores.append(r2_score(y[te_i], p.predict(X.iloc[te_i])))
            except Exception:
                scores.append(float("nan"))
        cv_r2 = float(np.nanmean(scores))

        rows.append({
            "target": target,
            "label": FEATURE_LABELS.get(target, target),
            "best_model": family,
            "n": int(len(y)),
            "n_predictors": len(preds),
            "minimal_set_r2": round(cv_r2, 4),
            "full_set_r2": round(full_r2, 4),
            "retained_share": round(cv_r2 / full_r2, 4) if full_r2 > 0 else np.nan,
        })
        print(f"{target:22s} {family:10s} minimal={cv_r2:+.3f}  full={full_r2:+.3f}  "
              f"share={cv_r2 / full_r2 if full_r2 > 0 else float('nan'):.2f}  (n={len(y):,})")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV} ({len(out)} rows)")
    ok = out[~out["target"].isin(NONDESTRUCTIVE)]
    print(f"\nNon-member targets: median retained share = {ok['retained_share'].median():.2f}")
    print(out.sort_values("retained_share", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()

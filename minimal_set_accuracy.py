"""
minimal_set_accuracy.py — Accuracy with the minimal measurement set.

Re-runs the tuned best-model comparison of run_pipeline.py (Global row) and
lithology_analysis.py (lithology classes) with the predictor pool restricted
to the seven properties that density, TCS and ultrasound measure directly
(phi, rho_b, rho_g, lambda, alpha, V_P, V_S). Rows, folds, model families,
hyperparameter grids and sample caps are those of the two scripts, so every
cell can be set against the full-predictor heatmap. The Poisson ratio is not
used as a predictor: its dynamic value follows from V_P and V_S, and its
static value needs the uniaxial test.

Finished cells are appended to the output file and skipped on a restart.
    python minimal_set_accuracy.py                    # all cells
    python minimal_set_accuracy.py --scope Magmatic   # one class (or Global)

Output: results/tables/minimal_set_accuracy.csv
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from pygam import LinearGAM
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.svm import SVR
from xgboost import XGBRegressor

from lithology_mapper import LithologyMapper
from master_split import apply_source_split
from target_registry import TargetRegistry

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
MASTER_CSV = BASE / "data" / "master_table.csv.gz"
OUT_DIR = BASE / os.environ.get("OUT_DIR_NAME", "results")
OUT_CSV = OUT_DIR / "tables" / "minimal_set_accuracy.csv"
FULL_CSV = OUT_DIR / "lithology" / "tables" / "best_model_by_lithology.csv"

MINIMAL_SET = ["porosity_pct", "bulk_density_gcm3", "grain_density_gcm3",
               "tc_dry_WmK", "td_dry_1e6m2s", "vp_dry_ms", "vs_dry_ms"]
EXCLUDE_FROM_MAIN = {"NorwayNPD"}
CLASS_MERGE = {"Plutonic": "Magmatic", "Volcanic": "Magmatic", "Pyroclastic": "Magmatic"}
SKIP_CLASSES = {"Unclassified", "Unconsolidated", "Evaporite"}
LOG_TARGETS = {"permeability_m2"}

N_ITER_GLOBAL = 15                      # run_pipeline.py
N_ITER_CLASS = 12                       # lithology_analysis.py
MAX_SAMPLES = 50_000
CLASS_SAMPLE_CAP = {"SVR": 5_000, "GAM": 8_000, "MLP": 8_000}


class PLSWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        X = np.asarray(X)
        nc = min(self.n_components, X.shape[1], max(1, X.shape[0] - 1))
        self.pls_ = PLSRegression(n_components=max(1, nc)).fit(X, y)
        return self

    def predict(self, X):
        return self.pls_.predict(np.asarray(X)).ravel()


class GAMWrapper(BaseEstimator, RegressorMixin):
    """Spline basis per feature + ridge (GAM of run_pipeline.py)."""
    def __init__(self, alpha=1.0, n_knots=5, degree=3):
        self.alpha = alpha
        self.n_knots = n_knots
        self.degree = degree

    def fit(self, X, y):
        self.spline_ = SplineTransformer(n_knots=self.n_knots, degree=self.degree,
                                         knots="quantile", include_bias=False)
        Z = self.spline_.fit_transform(np.asarray(X, dtype=float))
        self.ridge_ = Ridge(alpha=self.alpha).fit(Z, y)
        return self

    def predict(self, X):
        return self.ridge_.predict(self.spline_.transform(np.asarray(X, dtype=float)))


class PyGAMWrapper(BaseEstimator, RegressorMixin):
    """pygam LinearGAM (GAM of lithology_analysis.py)."""
    def __init__(self, lam=0.6):
        self.lam = lam

    def fit(self, X, y):
        self.gam_ = LinearGAM(lam=self.lam).fit(np.asarray(X), y)
        return self

    def predict(self, X):
        return self.gam_.predict(np.asarray(X))


class SVRWrapper(BaseEstimator, RegressorMixin):
    """Nystroem RBF kernel ridge (SVR of run_pipeline.py)."""
    def __init__(self, alpha=1.0, gamma=0.1, n_components=200, random_state=42):
        self.alpha = alpha
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.nystroem_ = Nystroem(kernel="rbf", gamma=self.gamma,
                                  n_components=min(self.n_components, X.shape[0]),
                                  random_state=self.random_state)
        self.ridge_ = Ridge(alpha=self.alpha).fit(self.nystroem_.fit_transform(X), y)
        return self

    def predict(self, X):
        return self.ridge_.predict(self.nystroem_.transform(np.asarray(X, dtype=float)))


def model_families(scope, n_jobs):
    """The nine families and search grids, as in run_pipeline.py / lithology_analysis.py."""
    trees = {"mdl__n_estimators": [80, 150], "mdl__max_depth": [None, 10, 20],
             "mdl__min_samples_leaf": [1, 3, 5]}
    if scope == "Global":
        gam = (GAMWrapper(), {"mdl__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                              "mdl__n_knots": [4, 5, 7]})
        svr = (SVRWrapper(), {"mdl__alpha": [0.01, 0.1, 1.0, 10.0],
                              "mdl__gamma": [0.01, 0.05, 0.1, 0.5],
                              "mdl__n_components": [100, 200, 400]})
    else:
        gam = (PyGAMWrapper(), {"mdl__lam": [0.001, 0.01, 0.1, 1.0, 10.0]})
        svr = (SVR(kernel="rbf"), {"mdl__C": [0.1, 1.0, 10.0, 100.0],
                                   "mdl__epsilon": [0.01, 0.1, 0.5]})
    return {
        "Ridge": (Ridge(), {"mdl__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}),
        "PLS": (PLSWrapper(), {"mdl__n_components": [1, 2, 3, 5, 8]}),
        "GAM": gam,
        "ExtraTrees": (ExtraTreesRegressor(random_state=42, n_jobs=n_jobs), trees),
        "RF": (RandomForestRegressor(random_state=42, n_jobs=n_jobs), trees),
        "XGBoost": (XGBRegressor(random_state=42, n_jobs=n_jobs, verbosity=0,
                                 objective="reg:squarederror"),
                    {"mdl__n_estimators": [80, 150], "mdl__learning_rate": [0.05, 0.1, 0.2],
                     "mdl__max_depth": [3, 5, 7], "mdl__subsample": [0.8, 1.0]}),
        "SVR": svr,
        "KNN": (KNeighborsRegressor(), {"mdl__n_neighbors": [3, 5, 10, 20],
                                        "mdl__weights": ["uniform", "distance"]}),
        "MLP": (MLPRegressor(random_state=42, max_iter=500),
                {"mdl__hidden_layer_sizes": [(100,), (100, 50), (200, 100)],
                 "mdl__alpha": [0.0001, 0.001, 0.01],
                 "mdl__activation": ["relu", "tanh"],
                 "mdl__learning_rate_init": [0.0001, 0.001, 0.01]}),
    }


def tuned_fit(model, grid, X, y, n_iter, n_jobs):
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("mdl", deepcopy(model))])
    search = RandomizedSearchCV(pipe, grid, n_iter=min(n_iter, 1 + len(grid)), cv=3,
                                scoring="r2", random_state=42, n_jobs=n_jobs, refit=True)
    return search.fit(X, y).best_estimator_


def cv_score(model, grid, X, y, n_splits, n_iter, n_jobs, min_test=1):
    scores = []
    for tr, te in KFold(n_splits, shuffle=True, random_state=42).split(X):
        if len(te) < min_test:
            continue
        try:
            fitted = tuned_fit(model, grid, X.iloc[tr], y[tr], n_iter, n_jobs)
            scores.append(r2_score(y[te], fitted.predict(X.iloc[te])))
        except Exception:
            scores.append(np.nan)
    return float(np.nanmean(scores)) if scores else -np.inf


def global_cell(master, registry, target, n_jobs):
    """STEP 5 of run_pipeline.py for one target."""
    sub, preds = registry.build_target_subset(master, target)
    preds = [p for p in preds if p in MINIMAL_SET]
    if sub.empty or len(preds) < TargetRegistry.MIN_PREDICTORS:
        return None
    df = sub.reset_index(drop=True)
    X, y = df[preds].copy(), df[target].to_numpy()
    if target in LOG_TARGETS:
        mask = y > 0
        X, y = X[mask].reset_index(drop=True), np.log10(y[mask])
    fam = {name: cv_score(m, g, X, y, 5, N_ITER_GLOBAL, n_jobs)
           for name, (m, g) in model_families("Global", n_jobs).items()}
    return len(y), preds, fam


def class_cell(master_cls, min_n, registry, cls_name, target, n_jobs):
    """STEP 5 of lithology_analysis.py for one class and target."""
    in_cls = master_cls[master_cls["lithology_class"] == cls_name]
    sub, preds = registry.build_target_subset(in_cls, target)
    if sub.empty or len(sub) < min_n or len(preds) < 2:
        return None
    preds = [p for p in preds if p in MINIMAL_SET]
    if len(preds) < 2:
        return None
    X, y = sub[preds].copy(), sub[target].to_numpy()
    if target in LOG_TARGETS:
        pos = y > 0
        X, y = X[pos], np.log10(y[pos])
    if len(y) > MAX_SAMPLES:
        idx = np.sort(np.random.default_rng(0).choice(len(y), MAX_SAMPLES, replace=False))
        X, y = X.iloc[idx], y[idx]
    if len(y) < min_n:
        return None
    fam = {}
    for name, (m, g) in model_families(cls_name, n_jobs).items():
        Xm, ym = X, y
        cap = CLASS_SAMPLE_CAP.get(name, MAX_SAMPLES)
        if len(y) > cap:
            idx = np.sort(np.random.default_rng(1).choice(len(y), cap, replace=False))
            Xm, ym = X.iloc[idx], y[idx]
        fam[name] = cv_score(m, g, Xm, ym, min(5, len(ym) // 6 + 1), N_ITER_CLASS,
                             n_jobs, min_test=3)
    return len(y), preds, fam


def load_data():
    master = pd.read_csv(MASTER_CSV, low_memory=False)
    master = master[~master["source_db"].isin(EXCLUDE_FROM_MAIN)].reset_index(drop=True)
    master = apply_source_split(master)
    mapper = LithologyMapper()
    master = mapper.apply(master)
    master["lithology_class"] = master["lithology_class"].replace(CLASS_MERGE)
    classes = [c for c in dict.fromkeys(CLASS_MERGE.get(c, c) for c in mapper.classes)
               if c not in SKIP_CLASSES]
    registry = TargetRegistry(master, BASE / "config" / "targets.yaml")
    return master, mapper, registry, classes


def add_full_ceiling(out):
    if not FULL_CSV.exists():
        return out
    full = pd.read_csv(FULL_CSV)[["target", "lithology", "best_model", "best_r2"]]
    full = full.rename(columns={"best_model": "full_model", "best_r2": "full_r2"})
    out = out.drop(columns=["full_model", "full_r2", "retained_share"], errors="ignore")
    out = out.merge(full, on=["target", "lithology"], how="left")
    out["retained_share"] = (out["best_r2"] / out["full_r2"]).round(3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="", help="Global or one lithology class")
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()

    print("Loading master table ...")
    master, mapper, registry, classes = load_data()
    master_cls = master[master["lithology_class"] != "Unclassified"].copy()
    scopes = ["Global"] + classes
    if args.scope:
        scopes = [args.scope]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        done = set(zip(prev["target"], prev["lithology"]))

    for scope in scopes:
        for target in registry.get_all_analyzed():
            if (target, scope) in done:
                continue
            if scope == "Global":
                res = global_cell(master, registry, target, args.n_jobs)
            else:
                res = class_cell(master_cls, mapper.min_samples_per_class, registry,
                                 scope, target, args.n_jobs)
            if res is None:
                continue
            n, preds, fam = res
            valid = {k: v for k, v in fam.items() if np.isfinite(v)}
            if not valid:
                continue
            best = max(valid, key=valid.get)
            row = {"target": target, "lithology": scope, "n": n,
                   "n_predictors": len(preds), "predictors": ";".join(preds),
                   "best_model": best, "best_r2": round(fam[best], 3),
                   "family_r2": json.dumps({k: round(v, 4) for k, v in fam.items()})}
            pd.DataFrame([row]).to_csv(OUT_CSV, mode="a", index=False,
                                       header=not OUT_CSV.exists())
            print(f"  {scope:22s} {target:20s} {best:10s} R2={fam[best]:.3f}  (n={n:,})")

    out = add_full_ceiling(pd.read_csv(OUT_CSV))
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV} ({len(out)} cells)")
    if "retained_share" in out:
        print(f"Median share of the full-predictor R2: {out['retained_share'].median():.2f}")


if __name__ == "__main__":
    main()

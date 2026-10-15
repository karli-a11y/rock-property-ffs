"""
minimal_set_search.py — Minimal set of laboratory measurement procedures.

The 17 targets are grouped by the laboratory procedure that measures them
(density, thermal conductivity scanner, ultrasound, kappameter, calorimeter,
uniaxial, Brazilian and triaxial test, permeameter). Every combination of
the nine procedures is scored by the mean over all 17 properties of
  1     if the property is measured, or follows exactly from measured ones
        (nu from V_P and V_S, E_dyn from V_P, V_S and rho_b,
        c_p = lambda / (rho_b * alpha)); the uniaxial test gives the static nu,
  R^2   otherwise: 5-fold CV R^2 (floored at 0) of predicting it from the
        measured properties with its best global model family
        (results/tables/best_model_global.csv) and fixed hyperparameters.
Rows and folds are fixed per property, so all combinations are compared on
the same samples. The minimal set is the combination at the knee of score
versus effort n/9 (largest score - n/9).

The leading combinations are scored again with CV seeds 7 and 13 and with
the Poisson ratio removed from the predictors.

Output: results/tables/minimal_set_search.csv
        results/tables/minimal_set_search_checks.csv
"""
from __future__ import annotations

import os

# one thread per model, the process pool does the parallel work
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import itertools
import warnings
from concurrent.futures import ProcessPoolExecutor
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

from master_split import apply_source_split

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
MASTER_CSV = BASE / "data" / "master_table.csv.gz"
OUT_DIR = BASE / os.environ.get("OUT_DIR_NAME", "results") / "tables"
N_WORKERS = int(os.environ.get("N_WORKERS", 8))

EXCLUDE_FROM_MAIN = {"NorwayNPD"}
PER_SOURCE_CAP = 8_000
LOG_TARGETS = {"permeability_m2"}
SEED = 42
CHECK_SEEDS = (7, 13)

PROCEDURES = {
    "density":        ["porosity_pct", "bulk_density_gcm3", "grain_density_gcm3"],
    "TCS":            ["tc_dry_WmK", "td_dry_1e6m2s"],
    "ultrasound":     ["vp_dry_ms", "vs_dry_ms", "poisson_ratio"],
    "kappameter":     ["mag_susc_1e3si"],
    "calorimeter":    ["cp_JkgK"],
    "uniaxial test":  ["ucs_MPa", "E_static_GPa"],
    "Brazilian test": ["tensile_MPa"],
    "triaxial test":  ["cohesion_MPa", "friction_angle_deg"],
    "permeameter":    ["permeability_m2"],
}
MEASURED = [p for props in PROCEDURES.values() for p in props]
TARGETS = MEASURED + ["E_dyn_GPa"]
EDYN_INPUTS = {"vp_dry_ms", "vs_dry_ms", "bulk_density_gcm3"}


def make_model(family: str):
    if family == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=150, max_depth=20,
                                   min_samples_leaf=1,
                                   random_state=SEED, n_jobs=1)
    if family == "RF":
        return RandomForestRegressor(n_estimators=150, max_depth=20,
                                     min_samples_leaf=1,
                                     random_state=SEED, n_jobs=1)
    if family == "XGBoost":
        return XGBRegressor(n_estimators=150, max_depth=5,
                            learning_rate=0.1, subsample=0.8,
                            objective="reg:squarederror",
                            random_state=SEED, n_jobs=1, verbosity=0)
    if family == "KNN":
        return KNeighborsRegressor(n_neighbors=5, weights="distance")
    return ExtraTreesRegressor(n_estimators=150, random_state=SEED, n_jobs=1)


def delivered(combo) -> set:
    """Properties a combination of procedures measures or computes exactly."""
    props = {p for proc in combo for p in PROCEDURES[proc]}
    if "uniaxial test" in combo:
        props.add("poisson_ratio")              # static value
    if "density" in combo and "ultrasound" in combo:
        props.add("E_dyn_GPa")
    if "density" in combo and "TCS" in combo:
        props.add("cp_JkgK")                    # c_p = lambda / (rho_b * alpha)
    return props


def predictors(target, combo, drop_nu=False) -> tuple:
    cols = [p for proc in combo for p in PROCEDURES[proc]]
    if target == "E_dyn_GPa":                   # circularity rule
        cols = [c for c in cols if c not in EDYN_INPUTS]
    if drop_nu:
        cols = [c for c in cols if c != "poisson_ratio"]
    return tuple(sorted(cols))


def load_data():
    master = pd.read_csv(MASTER_CSV, low_memory=False)
    master = master[~master["source_db"].isin(EXCLUDE_FROM_MAIN)].reset_index(drop=True)
    master = apply_source_split(master)
    bm = pd.read_csv(BASE / "results" / "tables" / "best_model_global.csv").set_index("target")

    data = {}
    for target in TARGETS:
        others = [c for c in MEASURED if c != target]
        df = master[master[target].notna() & master[others].notna().any(axis=1)]
        df = df.reset_index(drop=True)
        if target in LOG_TARGETS:
            df = df[df[target] > 0].reset_index(drop=True)

        rng = np.random.default_rng(99)
        keep = []
        for _, grp in df.groupby("source_db"):
            idx = grp.index.tolist()
            if len(idx) > PER_SOURCE_CAP:
                idx = rng.choice(idx, PER_SOURCE_CAP, replace=False).tolist()
            keep.extend(idx)
        df = df.loc[sorted(keep)].reset_index(drop=True)

        y = df[target].to_numpy(dtype=float)
        if target in LOG_TARGETS:
            y = np.log10(y)
        X = df[others].copy()
        if "permeability_m2" in X:
            X["permeability_m2"] = np.log10(X["permeability_m2"].where(X["permeability_m2"] > 0))
        data[target] = (str(bm.loc[target, "best_model"]), X, y)
        print(f"  {target:22s} {data[target][0]:10s} n={len(y):,}")
    return data


_DATA = {}


def _init_worker(data):
    _DATA.update(data)


def cv_r2(task):
    target, cols, seed = task
    family, X, y = _DATA[target]
    X = X[list(cols)]
    if not cols or X.notna().to_numpy().sum() == 0:
        return task, 0.0
    scores = []
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("mdl", make_model(family))])
        try:
            pipe.fit(X.iloc[tr], y[tr])
            scores.append(r2_score(y[te], pipe.predict(X.iloc[te])))
        except Exception:
            scores.append(np.nan)
    return task, float(np.nanmean(scores))


def score(combo, r2, seed=SEED, drop_nu=False) -> dict:
    got = delivered(combo)
    vals = {t: 1.0 if t in got else r2[(t, predictors(t, combo, drop_nu), seed)]
            for t in TARGETS}
    return {"combination": " + ".join(combo), "n_procedures": len(combo),
            "n_delivered": len(got),
            "score": round(float(np.mean([max(0.0, v) for v in vals.values()])), 4),
            **{f"r2_{t}": round(v, 4) for t, v in vals.items()}}


def tasks_for(combos, seed=SEED, drop_nu=False):
    return {(t, predictors(t, c, drop_nu), seed)
            for c in combos for t in TARGETS if t not in delivered(c)}


def run(pool, tasks, r2):
    todo = sorted(set(tasks) - set(r2), key=lambda tk: -len(_DATA[tk[0]][2]))
    print(f"  {len(todo)} CV runs ...")
    for i, (task, val) in enumerate(pool.map(cv_r2, todo, chunksize=4), 1):
        r2[task] = val
        if i % 500 == 0:
            print(f"    {i}/{len(todo)}")


def knee(table: pd.DataFrame) -> tuple:
    best = table.sort_values("score", ascending=False).groupby("n_procedures").head(1)
    best = best.sort_values("n_procedures")
    gap = best["score"].to_numpy() - best["n_procedures"].to_numpy() / len(PROCEDURES)
    return best, int(best["n_procedures"].iloc[int(np.argmax(gap))])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading master table ...")
    data = load_data()
    _DATA.update(data)
    combos = [c for k in range(1, len(PROCEDURES) + 1)
              for c in itertools.combinations(PROCEDURES, k)]

    r2 = {}
    with ProcessPoolExecutor(N_WORKERS, initializer=_init_worker, initargs=(data,)) as pool:
        print(f"Scoring all {len(combos)} combinations (seed {SEED})")
        run(pool, tasks_for(combos), r2)
        table = pd.DataFrame([score(c, r2) for c in combos])
        table = table.sort_values(["n_procedures", "score"], ascending=[True, False])
        table.to_csv(OUT_DIR / "minimal_set_search.csv", index=False)

        best, k = knee(table)
        print(best[["n_procedures", "combination", "score"]].to_string(index=False))
        print(f"Largest gap between score and effort at {k} procedures")

        # leading combinations: top 3 per size up to five procedures, best one above
        lead = [tuple(s.split(" + ")) for n, grp in table.groupby("n_procedures")
                for s in grp["combination"].head(3 if n <= 5 else 1)]
        print("Checks: CV seeds 7 and 13, Poisson ratio not used as predictor")
        for seed in CHECK_SEEDS:
            run(pool, tasks_for(lead, seed), r2)
        run(pool, tasks_for(lead, drop_nu=True), r2)

    rows = [{"variant": f"seed {SEED}", **score(c, r2)} for c in lead]
    rows += [{"variant": f"seed {s}", **score(c, r2, s)} for s in CHECK_SEEDS for c in lead]
    rows += [{"variant": "nu not a predictor", **score(c, r2, drop_nu=True)} for c in lead]
    checks = pd.DataFrame(rows)
    checks.to_csv(OUT_DIR / "minimal_set_search_checks.csv", index=False)
    for variant, grp in checks.groupby("variant", sort=False):
        b, k = knee(grp)
        print(f"  {variant:20s} knee at {k}: {b['combination'].iloc[k - 1]} "
              f"({b['score'].iloc[k - 1]:.3f})")
    print(f"\nSaved {OUT_DIR / 'minimal_set_search.csv'} and minimal_set_search_checks.csv")


if __name__ == "__main__":
    main()

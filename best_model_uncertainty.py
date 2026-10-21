"""
best_model_uncertainty.py — Spread and prediction error of the best-model cells.

For every cell of the best-model heatmap (all predictors) and of the minimal-set
check (only the seven properties of density, TCS and ultrasound), the nine
families are re-run with the tuned routine of run_pipeline.py (Global row) and
lithology_analysis.py (lithology classes): same rows, folds, sample caps and
hyperparameter search. The R2 of every outer fold and the out-of-fold
predictions are kept. For the winning family the output gives the standard
deviation of R2 over the folds, the RMSE, and the half-width of the 90 %
prediction interval (the 90 % quantile of the absolute out-of-fold errors), in
the unit of the target (log10 units for permeability), together with the
runner-up family and its R2.

Finished cells are appended to the output file and skipped on a restart.
    python best_model_uncertainty.py                  # all cells
    python best_model_uncertainty.py --scope Global   # one class (or Global)

Output: results/tables/best_model_uncertainty.csv
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from minimal_set_accuracy import (CLASS_SAMPLE_CAP, LOG_TARGETS, MAX_SAMPLES, MINIMAL_SET,
                                  N_ITER_CLASS, N_ITER_GLOBAL, OUT_DIR, load_data,
                                  model_families, tuned_fit)

OUT_CSV = OUT_DIR / "tables" / "best_model_uncertainty.csv"
POOLS = ["full", "minimal set"]
NAMES = {
    "porosity_pct": "Porosity", "bulk_density_gcm3": "Bulk density",
    "grain_density_gcm3": "Grain density", "vp_dry_ms": "P-wave velocity",
    "vs_dry_ms": "S-wave velocity", "tc_dry_WmK": "Thermal conductivity",
    "td_dry_1e6m2s": "Thermal diffusivity", "cp_JkgK": "Specific heat capacity",
    "poisson_ratio": "Poisson ratio", "E_dyn_GPa": "Dynamic Young's modulus",
    "E_static_GPa": "Static Young's modulus", "ucs_MPa": "UCS",
    "tensile_MPa": "Tensile strength", "cohesion_MPa": "Cohesion",
    "friction_angle_deg": "Friction angle", "permeability_m2": "Permeability",
    "mag_susc_1e3si": "Magnetic susceptibility",
}
UNITS = {
    "porosity_pct": "%", "bulk_density_gcm3": "g/cm3", "grain_density_gcm3": "g/cm3",
    "vp_dry_ms": "m/s", "vs_dry_ms": "m/s", "tc_dry_WmK": "W/(m K)",
    "td_dry_1e6m2s": "1e-6 m2/s", "cp_JkgK": "J/(kg K)", "poisson_ratio": "-",
    "E_dyn_GPa": "GPa", "E_static_GPa": "GPa", "ucs_MPa": "MPa", "tensile_MPa": "MPa",
    "cohesion_MPa": "MPa", "friction_angle_deg": "deg", "permeability_m2": "log10(m2)",
    "mag_susc_1e3si": "1e-3 SI",
}


def cell_data(master, master_cls, min_n, registry, scope, target, pool):
    """Rows and predictors of one cell, as in STEP 5 of the two scripts."""
    if scope == "Global":
        sub, preds = registry.build_target_subset(master, target)
    else:
        sub, preds = registry.build_target_subset(
            master_cls[master_cls["lithology_class"] == scope], target)
        if sub.empty or len(sub) < min_n or len(preds) < 2:
            return None
    if pool == "minimal set":
        preds = [p for p in preds if p in MINIMAL_SET]
    if sub.empty or len(preds) < 2:
        return None
    X, y = sub[preds].copy(), sub[target].to_numpy()
    if target in LOG_TARGETS:
        pos = y > 0
        X, y = X[pos], np.log10(y[pos])
    if scope != "Global":
        if len(y) > MAX_SAMPLES:
            idx = np.sort(np.random.default_rng(0).choice(len(y), MAX_SAMPLES, replace=False))
            X, y = X.iloc[idx], y[idx]
        if len(y) < min_n:
            return None
    return X, y


def family_cv(model, grid, X, y, n_splits, n_iter, n_jobs, min_test):
    """Outer CV with tuning inside each training fold; fold R2 and out-of-fold predictions."""
    pred = np.full(len(y), np.nan)
    scores = []
    for tr, te in KFold(n_splits, shuffle=True, random_state=42).split(X):
        if len(te) < min_test:
            continue
        try:
            fitted = tuned_fit(model, grid, X.iloc[tr], y[tr], n_iter, n_jobs)
            pred[te] = fitted.predict(X.iloc[te])
            scores.append(r2_score(y[te], pred[te]))
        except Exception:
            scores.append(np.nan)
    return np.array(scores), pred


def run_cell(X, y, scope, n_jobs):
    res = {}
    for name, (model, grid) in model_families(scope, n_jobs).items():
        Xm, ym = X, y
        if scope != "Global":
            cap = CLASS_SAMPLE_CAP.get(name, MAX_SAMPLES)
            if len(y) > cap:
                idx = np.sort(np.random.default_rng(1).choice(len(y), cap, replace=False))
                Xm, ym = X.iloc[idx], y[idx]
        if scope == "Global":
            scores, pred = family_cv(model, grid, Xm, ym, 5, N_ITER_GLOBAL, n_jobs, 1)
        else:
            scores, pred = family_cv(model, grid, Xm, ym, min(5, len(ym) // 6 + 1),
                                     N_ITER_CLASS, n_jobs, 3)
        if len(scores):
            res[name] = (scores, pred, ym)
    return res


def summarise(res):
    means = {k: float(np.nanmean(s)) for k, (s, _, _) in res.items()}
    best = max(means, key=means.get)
    rest = {k: v for k, v in means.items() if k != best}
    runner = max(rest, key=rest.get)
    scores, pred, y = res[best]
    ok = ~np.isnan(pred)
    err = np.abs(y[ok] - pred[ok])
    rank = int(np.ceil(0.9 * (len(err) + 1)))
    return {"family": best, "n": int(ok.sum()), "r2": round(means[best], 3),
            "r2_sd": round(float(np.nanstd(scores, ddof=1)), 4),
            "runner_up": runner, "runner_up_r2": round(rest[runner], 3),
            "rmse": float(f"{np.sqrt(np.mean(err ** 2)):.4g}"),
            "pi90": float(f"{np.sort(err)[rank - 1]:.4g}") if rank <= len(err) else np.inf,
            "family_r2": json.dumps({k: round(v, 4) for k, v in means.items()})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="", help="Global or one lithology class")
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()

    print("Loading master table ...")
    master, mapper, registry, classes = load_data()
    master_cls = master[master["lithology_class"] != "Unclassified"].copy()
    scopes = [args.scope] if args.scope else ["Global"] + classes

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        done = set(zip(prev["predictor_set"], prev["lithology"], prev["target"]))

    for pool in POOLS:
        for scope in scopes:
            for target in registry.get_all_analyzed():
                if (pool, scope, NAMES[target]) in done:
                    continue
                data = cell_data(master, master_cls, mapper.min_samples_per_class,
                                 registry, scope, target, pool)
                if data is None:
                    continue
                row = {"predictor_set": pool, "lithology": scope, "target": NAMES[target],
                       "unit": UNITS[target], **summarise(run_cell(*data, scope, args.n_jobs))}
                pd.DataFrame([row]).to_csv(OUT_CSV, mode="a", index=False,
                                           header=not OUT_CSV.exists())
                print(f"  {pool:11s} {scope:22s} {target:20s} {row['family']:10s} "
                      f"R2={row['r2']:.3f} sd={row['r2_sd']:.3f} pi90={row['pi90']:.4g}")

    out = pd.read_csv(OUT_CSV)
    print(f"\nSaved {OUT_CSV} ({len(out)} cells)")
    for pool, g in out.groupby("predictor_set"):
        ties = (g["r2"] - g["runner_up_r2"] < g["r2_sd"]).sum()
        print(f"  {pool}: runner-up within one fold SD of the winner in {ties} of {len(g)} cells")


if __name__ == "__main__":
    main()

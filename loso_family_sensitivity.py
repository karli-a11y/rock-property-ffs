"""
loso_family_sensitivity.py — how much the leave-one-source-out transfer gap
depends on the model family, for the targets whose winning family is a near
tie (runner-up within the fold-to-fold spread of best_model_uncertainty.py).

Re-runs loso_validation.evaluate_source() unchanged, only with the family of
selected targets overridden, and only for those targets.

Output: results/tables/loso_family_sensitivity.csv  (one row per held-out source,
        target and family; the canonical family is read from loso_validation.csv)
"""
from __future__ import annotations

import pandas as pd

import loso_validation as lv
from master_split import apply_source_split

RUNS = [  # (target, family) beside the canonical winner
    ("ucs_MPa", "ExtraTrees"), ("ucs_MPa", "RF"), ("poisson_ratio", "RF"),
    ("grain_density_gcm3", "RF"), ("tc_dry_WmK", "RF"), ("permeability_m2", "RF"),
]


def main():
    master = pd.read_csv(lv.MASTER_CSV, low_memory=False)
    master = master[~master["source_db"].isin(lv.EXCLUDE_FROM_MAIN)].reset_index(drop=True)
    master = apply_source_split(master)
    counts = master["source_db"].value_counts()
    sources = sorted(counts[counts >= 30].index)
    bm = pd.read_csv(lv.OUT_DIR / "best_model_global.csv")
    orig = lv.TargetRegistry.get_all_analyzed

    rows = []
    for target, family in RUNS:
        lv.TargetRegistry.get_all_analyzed = lambda self, t=target: [x for x in orig(self) if x == t]
        b = bm.copy()
        b.loc[b["target"] == target, "best_model"] = family
        for src in sources:
            rows.extend(lv.evaluate_source(master, src, b, None))
    lv.TargetRegistry.get_all_analyzed = orig

    out = pd.DataFrame(rows)
    canon = pd.read_csv(lv.OUT_DIR / "loso_validation.csv")
    canon = canon[canon["target"].isin({t for t, _ in RUNS})]
    out = pd.concat([canon, out], ignore_index=True)
    out.to_csv(lv.OUT_DIR / "loso_family_sensitivity.csv", index=False)
    summ = out.groupby(["target", "best_model"]).agg(
        n_sources=("delta_r2", "size"), median_gap=("delta_r2", "median"),
        median_holdout=("holdout_r2", "median"), median_cv=("cv_r2_train", "median"))
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()

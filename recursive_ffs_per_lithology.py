"""recursive_ffs_per_lithology.py — per-top_class iterated-exclusion FFS.

Same algorithm as recursive_ffs.py but stratified by `top_class`:
runs the full iterated-exclusion FFS pipeline independently for each
of the 5 paper lithology classes (Magmatic, Clastic sedimentary,
Carbonate sedimentary, Metamorphic, Evaporite).

Output
------
results/tables/feature_relevance_recursive_per_lithology.csv
    columns: top_class, iteration, excluded_so_far, feature, label,
             S_iter, rank_iter

results/tables/feature_relevance_recursive_per_lithology_mean.csv
    columns: top_class, feature, label, S_mean, n_iterations_present,
             S_iter1, S_max
"""
from __future__ import annotations

import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

# Re-use everything from recursive_ffs.py
sys.path.insert(0, str(Path(__file__).parent))
from recursive_ffs import (
    MASTER_CSV, EXCLUDE_SOURCES, BASE, OUT_TAB, MODEL_NAMES, LOG_TARGETS,
    FEATURE_LABELS, forward_feature_selection, aggregate_S,
)
from target_registry import TargetRegistry
from master_split import apply_source_split


PAPER_CLASSES = ['Magmatic', 'Clastic sedimentary', 'Carbonate sedimentary',
                 'Metamorphic', 'Evaporite']

MIN_SAMPLES_PER_TARGET = 30   # skip lith × target if fewer
MIN_PREDICTORS = 2


def run_recursive_ffs(master: pd.DataFrame, top_class: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run iterated-exclusion FFS on rows where top_class==<top_class>."""
    sub_master = master[master['top_class'] == top_class].copy()
    if len(sub_master) < 50:
        print(f"  [skip] {top_class}: only {len(sub_master)} rows total")
        return pd.DataFrame(), pd.DataFrame()

    print(f"\n{'#'*70}\n# TOP_CLASS: {top_class}  ({len(sub_master):,} rows)\n{'#'*70}")

    registry = TargetRegistry(sub_master, BASE / "config" / "targets.yaml")
    analyzed = registry.get_all_analyzed()
    print(f"  Analyzed targets in this class: {len(analyzed)} — {analyzed}")
    if not analyzed:
        return pd.DataFrame(), pd.DataFrame()

    pool = sorted(set().union(*[set(registry.build_target_subset(sub_master, t)[1])
                                 for t in analyzed]))
    print(f"  Initial pool size: {len(pool)} features")

    all_rows = []
    excluded: list[str] = []
    iter_idx = 0

    while len(pool) >= 2:
        iter_idx += 1
        t0 = time.time()
        print(f"\n  ITERATION {iter_idx} | pool={len(pool)} | excluded={excluded[-3:] if excluded else '(none)'}")

        ffs_rows = []
        for target in analyzed:
            sub, _orig_preds = registry.build_target_subset(sub_master, target)
            if sub.empty or len(sub) < MIN_SAMPLES_PER_TARGET:
                continue
            preds = [p for p in _orig_preds if p in pool]
            if len(preds) < MIN_PREDICTORS:
                continue
            y = sub[target].values
            X = sub[preds].copy()
            if target in LOG_TARGETS:
                pos = y > 0
                X, y = X[pos].reset_index(drop=True), np.log10(y[pos])
            if len(y) < MIN_SAMPLES_PER_TARGET:
                continue
            for fam in MODEL_NAMES:
                try:
                    steps = forward_feature_selection(X, y, preds, fam)
                except Exception as exc:
                    continue
                for row in steps:
                    row["target"] = target
                    row["model"] = fam
                    ffs_rows.extend([row])

        if not ffs_rows:
            print(f"  [skip] {top_class} iter {iter_idx}: no rows after FFS")
            break

        S_df = aggregate_S(ffs_rows, pool)
        S_df["iteration"] = iter_idx
        S_df["excluded_so_far"] = ",".join(excluded) if excluded else "(none)"
        S_df["top_class"] = top_class
        all_rows.append(S_df)

        winner = str(S_df.iloc[0]["feature"])
        winner_S = float(S_df.iloc[0]["S"])
        dt = time.time() - t0
        print(f"  → winner #{iter_idx}: {winner} (S={winner_S:.4f}) [{dt/60:.1f}m]")
        excluded.append(winner)
        pool = [p for p in pool if p != winner]

    if not all_rows:
        return pd.DataFrame(), pd.DataFrame()

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
    agg["top_class"] = top_class
    return full, agg


def main() -> None:
    print("Loading master table …")
    master = pd.read_csv(MASTER_CSV, low_memory=False)
    master = master[~master["source_db"].isin(EXCLUDE_SOURCES)].reset_index(drop=True)
    master = apply_source_split(master)
    print(f"  {len(master):,} rows, {master['source_db'].nunique()} sub-sources")

    if 'top_class' not in master.columns:
        raise KeyError("Master missing 'top_class' column — run classify_lithology.py first")

    all_full, all_agg = [], []
    for cls in PAPER_CLASSES:
        full, agg = run_recursive_ffs(master, cls)
        if not full.empty:
            all_full.append(full)
            all_agg.append(agg)

    if all_full:
        out_full = pd.concat(all_full, ignore_index=True)
        out_agg  = pd.concat(all_agg, ignore_index=True)
        fpath = OUT_TAB / "feature_relevance_recursive_per_lithology.csv"
        out_full.to_csv(fpath, index=False)
        gpath = OUT_TAB / "feature_relevance_recursive_per_lithology_mean.csv"
        out_agg.to_csv(gpath, index=False)
        print(f"\nDone. Wrote {fpath.name} and {gpath.name}")
        print(f"  Total iteration rows: {len(out_full):,}")
        print(f"  Classes processed: {out_full['top_class'].nunique()}")


if __name__ == "__main__":
    main()

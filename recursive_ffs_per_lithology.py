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

# Re-use the per-target recursion from recursive_ffs.py
sys.path.insert(0, str(Path(__file__).parent))
from recursive_ffs import (
    MASTER_CSV, EXCLUDE_SOURCES, BASE, OUT_TAB,
    run_pertarget_recursive, summarise,
)
from target_registry import TargetRegistry
from master_split import apply_source_split


PAPER_CLASSES = ['Magmatic', 'Clastic sedimentary', 'Carbonate sedimentary',
                 'Metamorphic', 'Evaporite']


def run_recursive_ffs(master: pd.DataFrame, top_class: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-target iterated-exclusion FFS on rows where top_class==<top_class>.

    Same per-(target, family) recursion as the global run, but restricted to
    this lithology class. Each target removes its own winners.
    """
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

    rec = run_pertarget_recursive(sub_master, analyzed, registry)
    if rec.empty:
        return pd.DataFrame(), pd.DataFrame()

    long, summary = summarise(rec)
    long["top_class"] = top_class
    summary["top_class"] = top_class
    return long, summary


def main() -> None:
    print("Loading master table …")
    master = pd.read_csv(MASTER_CSV, low_memory=False)
    master = master[~master["source_db"].isin(EXCLUDE_SOURCES)].reset_index(drop=True)
    master = apply_source_split(master)
    print(f"  {len(master):,} rows, {master['source_db'].nunique()} sub-sources")

    if 'top_class' not in master.columns:
        raise KeyError("Master missing 'top_class' column — run classify_lithology.py first")

    fpath = OUT_TAB / "feature_relevance_recursive_per_lithology.csv"
    gpath = OUT_TAB / "feature_relevance_recursive_per_lithology_mean.csv"
    all_full, all_agg = [], []
    for cls in PAPER_CLASSES:
        full, agg = run_recursive_ffs(master, cls)
        if not full.empty:
            all_full.append(full)
            all_agg.append(agg)
            # checkpoint after every class
            pd.concat(all_full, ignore_index=True).to_csv(fpath, index=False)
            pd.concat(all_agg, ignore_index=True).to_csv(gpath, index=False)
            print(f"  [checkpoint] {cls} written", flush=True)

    if all_full:
        out_full = pd.concat(all_full, ignore_index=True)
        print(f"\nDone. Wrote {fpath.name} and {gpath.name}")
        print(f"  Total iteration rows: {len(out_full):,}")
        print(f"  Classes processed: {out_full['top_class'].nunique()}")


if __name__ == "__main__":
    main()

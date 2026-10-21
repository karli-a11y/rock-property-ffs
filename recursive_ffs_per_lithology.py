"""recursive_ffs_per_lithology.py — per-class iterated-exclusion FFS.

Same algorithm as recursive_ffs.py but stratified by lithology class:
runs the full iterated-exclusion FFS pipeline independently for each
of the 5 paper lithology classes (Magmatic, Clastic sedimentary,
Carbonate sedimentary, Metamorphic, Evaporite). The classes are those of
lithology_mapper.py with the plutonic, volcanic and pyroclastic classes
merged into Magmatic, exactly as in lithology_analysis.py, so the per-class
rankings and the best-model heatmap use the same rows.

Output (--out-suffix is appended to each name, default none)
------
results/tables/feature_relevance_recursive_per_lithology.csv
    columns: top_class, iteration, excluded_so_far, feature, label,
             S_iter, rank_iter

results/tables/feature_relevance_recursive_per_lithology_mean.csv
    columns: top_class, feature, label, S_mean, n_iterations_present,
             S_iter1, S_max

results/tables/feature_relevance_recursive_per_lithology_steps.csv
    step-level records (top_class, target, model, iteration, step,
    feature, r2_delta), the input of the per-class bootstrap
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Re-use the per-target recursion from recursive_ffs.py
sys.path.insert(0, str(Path(__file__).parent))
from recursive_ffs import (
    MASTER_CSV, EXCLUDE_SOURCES, BASE, OUT_TAB,
    run_pertarget_recursive, summarise,
)
from target_registry import TargetRegistry
from master_split import apply_source_split
from lithology_mapper import LithologyMapper


PAPER_CLASSES = ['Magmatic', 'Clastic sedimentary', 'Carbonate sedimentary',
                 'Metamorphic', 'Evaporite']
CLASS_MERGE = {"Plutonic": "Magmatic", "Volcanic": "Magmatic", "Pyroclastic": "Magmatic"}


def run_recursive_ffs(master: pd.DataFrame, cls: str):
    """Per-target iterated-exclusion FFS on the rows of one lithology class.

    Same per-(target, family) recursion as the global run, but restricted to
    this lithology class. Each target removes its own winners.
    """
    sub_master = master[master['lithology_class'] == cls].copy()
    if len(sub_master) < 50:
        print(f"  [skip] {cls}: only {len(sub_master)} rows total")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print(f"\n{'#'*70}\n# CLASS: {cls}  ({len(sub_master):,} rows)\n{'#'*70}")

    registry = TargetRegistry(sub_master, BASE / "config" / "targets.yaml")
    analyzed = registry.get_all_analyzed()
    print(f"  Analyzed targets in this class: {len(analyzed)} — {analyzed}")
    if not analyzed:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    rec = run_pertarget_recursive(sub_master, analyzed, registry)
    if rec.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    long, summary = summarise(rec)
    for df in (rec, long, summary):
        df["top_class"] = cls
    return long, summary, rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-suffix", default="", help="appended to every output file name")
    sfx = ap.parse_args().out_suffix

    print("Loading master table …")
    master = pd.read_csv(MASTER_CSV, low_memory=False)
    master = master[~master["source_db"].isin(EXCLUDE_SOURCES)].reset_index(drop=True)
    master = apply_source_split(master)
    master = LithologyMapper().apply(master)
    master["lithology_class"] = master["lithology_class"].replace(CLASS_MERGE)
    print(f"  {len(master):,} rows, {master['source_db'].nunique()} sub-sources")
    print(f"  classes: {master['lithology_class'].value_counts().to_dict()}")

    fpath = OUT_TAB / f"feature_relevance_recursive_per_lithology{sfx}.csv"
    gpath = OUT_TAB / f"feature_relevance_recursive_per_lithology_mean{sfx}.csv"
    spath = OUT_TAB / f"feature_relevance_recursive_per_lithology_steps{sfx}.csv"
    all_full, all_agg, all_steps = [], [], []
    for cls in PAPER_CLASSES:
        full, agg, steps = run_recursive_ffs(master, cls)
        if not full.empty:
            all_full.append(full)
            all_agg.append(agg)
            all_steps.append(steps)
            # checkpoint after every class
            pd.concat(all_full, ignore_index=True).to_csv(fpath, index=False)
            pd.concat(all_agg, ignore_index=True).to_csv(gpath, index=False)
            pd.concat(all_steps, ignore_index=True).to_csv(spath, index=False)
            print(f"  [checkpoint] {cls} written", flush=True)

    if all_full:
        out_full = pd.concat(all_full, ignore_index=True)
        print(f"\nDone. Wrote {fpath.name}, {gpath.name} and {spath.name}")
        print(f"  Total iteration rows: {len(out_full):,}")
        print(f"  Classes processed: {out_full['top_class'].nunique()}")


if __name__ == "__main__":
    main()

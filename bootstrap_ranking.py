"""
bootstrap_ranking.py — Bootstrap confidence intervals on the global
                        per-target iterated-exclusion parameter ranking.

For B bootstrap resamples of the (target, model) units, we re-aggregate
the relevance ranking S_tilde_j (the realised Δ-R² summed over all
exclusion iterations and selection steps, normalised by the grand total)
and record the per-feature score. We then report mean, 5th and 95th
percentile per feature, plus the rank distribution (median rank, IQR).

This is computationally cheap because it operates on the already-cached
feature_relevance_recursive_steps.csv: each bootstrap resample is just a
re-aggregation across resampled (target, model) groups, not a re-fit.

With --per-lithology the same is done within each lithology class, from
feature_relevance_recursive_per_lithology_steps.csv
(recursive_ffs_per_lithology.py), and written to
feature_relevance_bootstrap_per_lithology.csv with a top_class column.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
import os
_V14 = BASE / os.environ.get("OUT_DIR_NAME", "results")
STEPS_CSV = _V14 / "tables" / "feature_relevance_recursive_steps.csv"
OUT_CSV = _V14 / "tables" / "feature_relevance_bootstrap.csv"
LITH_STEPS_CSV = _V14 / "tables" / "feature_relevance_recursive_per_lithology_steps.csv"
LITH_OUT_CSV = _V14 / "tables" / "feature_relevance_bootstrap_per_lithology.csv"

from variables import FEATURE_LABELS, HARMONIZATION_RISK


def aggregate_to_S(steps: pd.DataFrame) -> pd.Series:
    """Sum positive Δ-R² per feature over all loops, normalise → Σ S_j = 1."""
    pos = steps["r2_delta"].clip(lower=0.0)
    s = pos.groupby(steps["feature"]).sum()
    total = s.sum()
    return s / total if total > 0 else s


def bootstrap(steps: pd.DataFrame, B: int, seed: int) -> pd.DataFrame:
    pairs = steps[["target", "model"]].drop_duplicates().reset_index(drop=True)
    n_pairs = len(pairs)

    print(f"Loaded {len(steps)} step rows across {n_pairs} (target, model) pairs.")
    print(f"Running B={B} bootstrap resamples …")

    rng = np.random.default_rng(seed)
    all_features = sorted(steps["feature"].unique())
    boot_S = np.zeros((B, len(all_features)))
    boot_rank = np.zeros((B, len(all_features)), dtype=int)

    feat_idx = {f: i for i, f in enumerate(all_features)}

    for b in range(B):
        idx = rng.integers(0, n_pairs, size=n_pairs)
        sampled_pairs = pairs.iloc[idx]
        # Merge to keep all rows of each sampled (target, model) pair
        merged = sampled_pairs.merge(steps, on=["target", "model"], how="left")
        s_vec = aggregate_to_S(merged)
        for f in all_features:
            v = float(s_vec.get(f, 0.0))
            boot_S[b, feat_idx[f]] = v
        # ranks: 1 = highest
        order = np.argsort(-boot_S[b])  # descending
        ranks = np.empty(len(all_features), dtype=int)
        ranks[order] = np.arange(1, len(all_features) + 1)
        boot_rank[b] = ranks

        if (b + 1) % 100 == 0:
            print(f"  {b+1}/{B}")

    # Compute statistics
    rows = []
    for f in all_features:
        i = feat_idx[f]
        S_samples = boot_S[:, i]
        r_samples = boot_rank[:, i]
        rows.append({
            "feature":           f,
            "label":             FEATURE_LABELS.get(f, f),
            "S_mean":            float(np.mean(S_samples)),
            "S_p05":             float(np.percentile(S_samples, 5)),
            "S_p95":             float(np.percentile(S_samples, 95)),
            "rank_median":       int(np.median(r_samples)),
            "rank_p05":          int(np.percentile(r_samples, 5)),
            "rank_p95":          int(np.percentile(r_samples, 95)),
            "harmonization_risk": HARMONIZATION_RISK.get(f, "moderate"),
        })

    return (pd.DataFrame(rows)
            .sort_values("S_mean", ascending=False)
            .reset_index(drop=True))


def main(B: int = 500, seed: int = 42, per_lithology: bool = False):
    if not per_lithology:
        out_df = bootstrap(pd.read_csv(STEPS_CSV), B, seed)
        out_csv = OUT_CSV
    else:
        steps = pd.read_csv(LITH_STEPS_CSV)
        parts = []
        for cls, g in steps.groupby("top_class", sort=False):
            print(f"\n=== {cls}")
            parts.append(bootstrap(g, B, seed).assign(top_class=cls))
        out_df = pd.concat(parts, ignore_index=True)
        out_csv = LITH_OUT_CSV
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv} ({len(out_df)} rows).")
    print()
    try:
        print(out_df.head(12).to_string(index=False))
    except UnicodeEncodeError:
        print(out_df.head(12).to_string(index=False).encode("ascii", "replace").decode())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lithology", action="store_true")
    main(B=500, per_lithology=ap.parse_args().per_lithology)

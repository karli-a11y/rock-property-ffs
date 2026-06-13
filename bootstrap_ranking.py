"""
bootstrap_ranking.py — Bootstrap confidence intervals on the global
                        FFS-aggregated parameter ranking.

For B bootstrap resamples of the (target, model) FFS rows, we re-aggregate
Eq. 1 of the paper and record the per-feature normalised score S_j. We
then report mean, 5th and 95th percentile per feature, plus the rank
distribution (median rank, IQR).

This is computationally cheap because it operates on the already-cached
ffs_selection_order.csv: each bootstrap resample is just a re-aggregation
across resampled (target, model) groups, not a re-fit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
import os
_V14 = BASE / os.environ.get("OUT_DIR_NAME", "results")
FFS_CSV = _V14 / "tables" / "ffs_selection_order.csv"
OUT_CSV = _V14 / "tables" / "feature_relevance_bootstrap.csv"

from variables import FEATURE_LABELS, HARMONIZATION_RISK


def aggregate_to_S(ffs: pd.DataFrame) -> pd.Series:
    """Aggregate Δ-R² across (target, model) and normalise → Σ S_j = 1."""
    agg = {}
    for _, r in ffs.iterrows():
        d = float(r["r2_delta"])
        if not np.isfinite(d):
            d = float(r["r2_cumulative"])
        d = max(d, 0.0)
        agg[r["feature"]] = agg.get(r["feature"], 0.0) + d
    s = pd.Series(agg)
    total = s.sum()
    return s / total if total > 0 else s


def main(B: int = 500, seed: int = 42):
    ffs = pd.read_csv(FFS_CSV)
    pairs = ffs[["target", "model"]].drop_duplicates().reset_index(drop=True)
    n_pairs = len(pairs)

    print(f"Loaded {len(ffs)} FFS rows across {n_pairs} (target, model) pairs.")
    print(f"Running B={B} bootstrap resamples …")

    rng = np.random.default_rng(seed)
    all_features = sorted(ffs["feature"].unique())
    boot_S = np.zeros((B, len(all_features)))
    boot_rank = np.zeros((B, len(all_features)), dtype=int)

    feat_idx = {f: i for i, f in enumerate(all_features)}

    for b in range(B):
        idx = rng.integers(0, n_pairs, size=n_pairs)
        sampled_pairs = pairs.iloc[idx]
        # Merge to keep all rows of each sampled (target, model) pair
        merged = sampled_pairs.merge(ffs, on=["target", "model"], how="left")
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

    out_df = (pd.DataFrame(rows)
              .sort_values("S_mean", ascending=False)
              .reset_index(drop=True))
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV} ({len(out_df)} rows).")
    print()
    try:
        print(out_df.head(12).to_string(index=False))
    except UnicodeEncodeError:
        print(out_df.head(12).to_string(index=False).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main(B=500)

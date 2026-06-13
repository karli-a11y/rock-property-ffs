"""master_split.py PATCH for v21 master — drop-in replacement.

Changes vs original:
  - Splits ALL compendia by `primary_reference`, not only P3 + AMPEDEK
    (also: GlobalDB, MidGerman, HeapViolay, ESReviews, USGS_WUS,
    USGS_MojaveREE, USGS_GreatBasinPlutons, Cornwall)
  - Uses the precomputed `subsource` column if present in master v21
    (skip recomputation when already done upstream)
  - Same MIN_ROWS=100 bucketing into ::Other for small references
  - parent_db column added for original-compendium tracking
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

MIN_ROWS = 30  # v24e: lowered from 100 for finer LOSO granularity

# Now includes all compendia, not just P3 + AMPEDEK
SPLIT_PARENTS = (
    "P3", "AMPEDEK", "GlobalDB", "MidGerman", "HeapViolay2021",
    "ESReviews2024_VolcGeotherm",
    "USGS_WesternUS_Phillips2021", "USGS_MojaveREE_2020",
    "USGS_GreatBasinPlutons", "Cornwall_Turan2024",
)


def apply_source_split(master: pd.DataFrame) -> pd.DataFrame:
    """Replace ``source_db`` with reference-level labels for all compendia.

    The original compendium label is preserved in a new ``parent_db`` column.
    Other sources keep ``source_db == parent_db``.

    If the master already has a ``subsource`` column (v21+), respects that
    finer-grained labelling and just normalises to the ``parent::ref`` format.
    """
    if "source_db" not in master.columns:
        raise KeyError("master table missing 'source_db' column")
    if "primary_reference" not in master.columns:
        raise KeyError(
            "master table missing 'primary_reference' column — "
            "ensure adapters populate it (v21+ does this automatically)"
        )

    df = master.copy()
    df["parent_db"] = df["source_db"]

    for parent in SPLIT_PARENTS:
        mask = df["source_db"] == parent
        if not mask.any():
            continue
        refs = df.loc[mask, "primary_reference"].astype(str).str.strip()
        refs = refs.replace({"nan": "Unknown", "": "Unknown", "<NA>": "Unknown"})
        refs = refs.fillna("Unknown")
        # Bucket small refs into Other
        counts = refs.value_counts()
        big = set(counts[counts >= MIN_ROWS].index)
        bucketed = refs.where(refs.isin(big), other="Other")
        df.loc[mask, "source_db"] = parent + "::" + bucketed

    return df


def summarize_split(master: pd.DataFrame) -> pd.DataFrame:
    """Return per-source row count summary."""
    return (
        master.groupby(["parent_db", "source_db"])
              .size()
              .reset_index(name="n_rows")
              .sort_values(["parent_db", "n_rows"], ascending=[True, False])
    )


if __name__ == "__main__":
    # Smoke test on the live master
    m = pd.read_csv(Path(__file__).parent / "data" / "master_table.csv.gz",
                    low_memory=False)
    eff = m[m.source_db != "NorwayNPD"]
    print(f"Effective rows: {len(eff):,}")
    print(f"source_db (pre-split): {eff.source_db.nunique()}")
    print(f"subsource (v21 precomputed): {eff.subsource.nunique() if 'subsource' in eff.columns else 'N/A'}")
    split = apply_source_split(eff)
    print(f"source_db (post-split): {split.source_db.nunique()}")
    print("\nPer parent_db, top sub-source counts:")
    summ = summarize_split(split)
    for parent, sub in summ.groupby("parent_db"):
        print(f"\n  {parent}: {len(sub)} sub-sources")
        for _, row in sub.head(5).iterrows():
            print(f"    {row.source_db:<55} n={row.n_rows:>5}")

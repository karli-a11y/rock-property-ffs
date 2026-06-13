"""
lithology_mapper.py — Map raw lithology labels to standardised classes and groups.

Reads config/lithology.yaml and applies regex-based mapping rules to the raw
``lithology`` column of the harmonised master table.  Produces two new columns:

    lithology_class   (8 broad classes + Unclassified)
    lithology_group   (~30 finer groups)

Usage
-----
    from lithology_mapper import LithologyMapper

    mapper = LithologyMapper()                   # loads config/lithology.yaml
    df = mapper.apply(master)                    # adds two columns in-place & returns df
    report = mapper.coverage_report(df)          # DataFrame with mapping statistics
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml


class LithologyMapper:
    """Rule-based lithology classifier."""

    def __init__(self, config_path: Optional[Path] = None):
        cfg_path = config_path or Path(__file__).parent / "config" / "lithology.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

        self._classes: List[str] = cfg["classes"]
        self._groups: Dict[str, List[str]] = cfg["groups"]
        self._analysis_cfg: dict = cfg.get("analysis", {})

        # Pre-compile mapping rules
        self._rules: List[Tuple[re.Pattern, str, str]] = []
        for rule in cfg["mapping_rules"]:
            pat = re.compile(rule["pattern"], re.IGNORECASE)
            self._rules.append((pat, rule["class"], rule["group"]))

    # ── Core mapping ──────────────────────────────────────────────────────────

    def classify(self, raw_label: str) -> Tuple[str, str]:
        """Return (lithology_class, lithology_group) for a single raw label."""
        if not isinstance(raw_label, str) or not raw_label.strip():
            return ("Unclassified", "Unclassified")

        label = raw_label.strip()
        for pat, cls, grp in self._rules:
            if pat.search(label):
                return (cls, grp)
        return ("Unclassified", "Unclassified")

    def apply(self, df: pd.DataFrame, col: str = "lithology") -> pd.DataFrame:
        """Add ``lithology_class`` and ``lithology_group`` columns to *df* (in-place).

        Returns the modified DataFrame for chaining.
        """
        results = df[col].apply(
            lambda x: self.classify(x) if pd.notna(x) else ("Unclassified", "Unclassified")
        )
        df["lithology_class"] = results.apply(lambda t: t[0])
        df["lithology_group"] = results.apply(lambda t: t[1])
        return df

    # ── Reporting ─────────────────────────────────────────────────────────────

    def coverage_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-class and per-group sample counts and coverage percentages."""
        total = len(df)
        rows = []
        for cls in self._classes:
            mask = df["lithology_class"] == cls
            n_cls = int(mask.sum())
            for grp in self._groups.get(cls, [cls]):
                n_grp = int((df["lithology_group"] == grp).sum())
                rows.append({
                    "lithology_class": cls,
                    "lithology_group": grp,
                    "n_samples": n_grp,
                    "pct_total": round(100 * n_grp / total, 2) if total else 0,
                    "n_class_total": n_cls,
                })
        return pd.DataFrame(rows)

    def unmapped_labels(self, df: pd.DataFrame, col: str = "lithology") -> pd.DataFrame:
        """Return raw labels that fell into 'Unclassified', sorted by frequency."""
        mask = df["lithology_class"] == "Unclassified"
        unc = df.loc[mask, col].dropna()
        counts = unc.value_counts().reset_index()
        counts.columns = ["raw_lithology", "count"]
        return counts

    def database_lithology_matrix(self, df: pd.DataFrame,
                                   db_col: str = "source_db") -> pd.DataFrame:
        """Cross-tabulation of source_db × lithology_class (counts)."""
        return pd.crosstab(df[db_col], df["lithology_class"], margins=True)

    def database_lithology_proportions(self, df: pd.DataFrame,
                                        db_col: str = "source_db") -> pd.DataFrame:
        """Cross-tabulation of source_db × lithology_class (row-normalised %)."""
        ct = pd.crosstab(df[db_col], df["lithology_class"])
        return ct.div(ct.sum(axis=1), axis=0).round(4) * 100

    # ── Config accessors ──────────────────────────────────────────────────────

    @property
    def min_samples_per_class(self) -> int:
        return int(self._analysis_cfg.get("min_samples_per_class", 30))

    @property
    def min_samples_per_group(self) -> int:
        return int(self._analysis_cfg.get("min_samples_per_group", 20))

    @property
    def min_classes_for_lolo(self) -> int:
        return int(self._analysis_cfg.get("min_classes_for_lolo", 3))

    @property
    def classes(self) -> List[str]:
        return list(self._classes)

    @property
    def groups(self) -> Dict[str, List[str]]:
        return dict(self._groups)

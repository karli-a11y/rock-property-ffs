"""
target_registry.py — Unified target registry for v14 pipeline.

Single source of truth: loads config/targets.yaml and qualifies each variable
against the actual master table using four objective rules.

Rules
-----
R1  n_valid  >= min_samples (default 30)
R2  n_usable_predictors >= 2  (after circularity filter)
R3  n_lodo_groups >= 2, each with >= 5 samples  → else partial
R4  harmonization_risk != 'fatal'
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

# Import stable constants from v9 (unchanged)
from variables import NUMERIC_FEATURES, FEATURE_LABELS


class TargetRegistry:
    """Qualify all rock-property variables as prediction targets."""

    MIN_SAMPLES: int = 30
    MIN_PREDICTORS: int = 2
    MIN_LODO_GROUPS: int = 2
    MIN_LODO_GROUP_SIZE: int = 5

    def __init__(self, master: pd.DataFrame, targets_yaml: Path):
        self._master = master
        cfg = yaml.safe_load(Path(targets_yaml).read_text(encoding="utf-8"))

        # Flatten main + secondary into one ordered dict
        self._meta: Dict[str, dict] = {}
        for t in cfg["targets"].get("main", []):
            self._meta[t["name"]] = t
        for t in cfg["targets"].get("secondary", []):
            if isinstance(t, str):
                self._meta[t] = {"name": t}
            else:
                self._meta[t["name"]] = t

        self._circularity: Dict[str, List[str]] = cfg.get("circularity_rules", {})

        self._qualified: List[str] = []
        self._partial: List[str] = []
        self._excluded: List[dict] = []
        self._qualify()

    # ── Qualification ──────────────────────────────────────────────────────────

    def _qualify(self) -> None:
        for tname, tmeta in self._meta.items():
            # R4: fatal harmonization risk
            if tmeta.get("harmonization_risk") == "fatal":
                self._excluded.append({
                    "target": tname,
                    "reason": "R4: fatal harmonization risk",
                    "n_valid": 0, "n_predictors": 0,
                })
                continue

            # R1a: column must exist in master table
            if tname not in self._master.columns:
                self._excluded.append({
                    "target": tname,
                    "reason": "R1: column not in master table",
                    "n_valid": 0, "n_predictors": 0,
                })
                continue

            sub = self._master[self._master[tname].notna()].copy()
            n_valid = len(sub)
            min_s = int(tmeta.get("min_samples", self.MIN_SAMPLES))

            # R1b: minimum sample count
            if n_valid < min_s:
                self._excluded.append({
                    "target": tname,
                    "reason": f"R1: n_valid={n_valid} < {min_s}",
                    "n_valid": n_valid, "n_predictors": 0,
                })
                continue

            # R2: usable predictors after circularity filter
            preds = self._usable_predictors(tname, sub, min_s)
            if len(preds) < self.MIN_PREDICTORS:
                self._excluded.append({
                    "target": tname,
                    "reason": f"R2: only {len(preds)} usable predictor(s)",
                    "n_valid": n_valid, "n_predictors": len(preds),
                })
                continue

            # R3: enough source groups for LODO
            lodo_ok = self._check_lodo(sub)
            if lodo_ok:
                self._qualified.append(tname)
            else:
                self._partial.append(tname)

    def _usable_predictors(self, target: str, sub: pd.DataFrame,
                            min_valid: int) -> List[str]:
        circular = self._circularity.get(target, [])
        return [
            f for f in NUMERIC_FEATURES
            if f != target
            and f not in circular
            and f in sub.columns
            and int(sub[f].notna().sum()) >= min_valid
        ]

    def _check_lodo(self, sub: pd.DataFrame) -> bool:
        if "source_db" not in sub.columns:
            return False
        counts = sub.groupby("source_db").size()
        return int((counts >= self.MIN_LODO_GROUP_SIZE).sum()) >= self.MIN_LODO_GROUPS

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_qualified(self) -> List[str]:
        """Targets that pass all four rules (full analysis including LODO)."""
        return list(self._qualified)

    def get_partial(self) -> List[str]:
        """Targets that pass R1–R2–R4 but not R3 (analysis without LODO)."""
        return list(self._partial)

    def get_excluded(self) -> List[dict]:
        """Excluded targets with reason dicts."""
        return list(self._excluded)

    def get_all_analyzed(self) -> List[str]:
        """All targets that enter the analysis (qualified + partial)."""
        return self._qualified + self._partial

    def get_display_label(self, target: str) -> str:
        """Human-readable label for manuscript use."""
        meta = self._meta.get(target, {})
        return meta.get("label") or FEATURE_LABELS.get(target, target)

    def get_unit(self, target: str) -> str:
        meta = self._meta.get(target, {})
        return meta.get("unit", "")

    def get_harmonization_risk(self, target: str) -> str:
        meta = self._meta.get(target, {})
        return meta.get("harmonization_risk", "unknown")

    def get_circularity(self, target: str) -> List[str]:
        return list(self._circularity.get(target, []))

    def lodo_possible(self, target: str) -> bool:
        return target in self._qualified

    def build_target_subset(self, master: pd.DataFrame, target: str,
                             min_valid: int = 30) -> Tuple[pd.DataFrame, List[str]]:
        """Build target-specific modelling subset.

        Replicates v9 _build_target_subset logic using registry circularity rules.
        """
        if target not in master.columns:
            return pd.DataFrame(), []
        sub = master[master[target].notna()].copy()
        preds = self._usable_predictors(target, sub, min_valid)
        if len(preds) < self.MIN_PREDICTORS or len(sub) < min_valid:
            return pd.DataFrame(), []
        return sub, preds

    def get_presentation_subset(self, n: int = 8) -> List[str]:
        """Auto-select top-N qualified targets by data density (n_valid × n_preds).

        Used for dense figures only — not a scientific privileging.
        """
        scored = []
        for t in self._qualified:
            sub = self._master[self._master[t].notna()]
            preds = self._usable_predictors(t, sub, self.MIN_SAMPLES)
            scored.append((t, len(sub) * len(preds)))
        scored.sort(key=lambda x: -x[1])
        return [t for t, _ in scored[:n]]

    def qualification_report(self) -> pd.DataFrame:
        """DataFrame suitable for manuscript Table Q (qualified/partial/excluded)."""
        rows = []

        for t in self._qualified:
            sub = self._master[self._master[t].notna()]
            preds = self._usable_predictors(t, sub, self.MIN_SAMPLES)
            grp_counts = (sub.groupby("source_db").size()
                          if "source_db" in sub.columns else pd.Series([], dtype=int))
            rows.append({
                "target":        t,
                "label":         self.get_display_label(t),
                "status":        "qualified",
                "reason":        "",
                "n_valid":       len(sub),
                "n_predictors":  len(preds),
                "n_lodo_groups": int((grp_counts >= self.MIN_LODO_GROUP_SIZE).sum()),
                "harm_risk":     self.get_harmonization_risk(t),
            })

        for t in self._partial:
            sub = self._master[self._master[t].notna()]
            preds = self._usable_predictors(t, sub, self.MIN_SAMPLES)
            rows.append({
                "target":        t,
                "label":         self.get_display_label(t),
                "status":        "partial",
                "reason":        "R3: insufficient LODO groups",
                "n_valid":       len(sub),
                "n_predictors":  len(preds),
                "n_lodo_groups": 0,
                "harm_risk":     self.get_harmonization_risk(t),
            })

        for ex in self._excluded:
            rows.append({
                "target":        ex["target"],
                "label":         self.get_display_label(ex["target"]),
                "status":        "excluded",
                "reason":        ex["reason"],
                "n_valid":       ex.get("n_valid", 0),
                "n_predictors":  ex.get("n_predictors", 0),
                "n_lodo_groups": 0,
                "harm_risk":     self.get_harmonization_risk(ex["target"]),
            })

        return pd.DataFrame(rows)

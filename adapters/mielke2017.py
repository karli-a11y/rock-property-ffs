"""adapters/mielke2017.py — Mielke et al. (2017) thermal conductivity and Vp dataset (PANGAEA 874146).

1430 oven-dry rock samples: tc dry/sat, Vp dry/sat, porosity, rock type, region.
Header occupies first 42 lines; data starts at line 43 (0-indexed row 42).
"""

import numpy as np
import pandas as pd
from .base import BaseAdapter

_COL_MAP = {
    "Poros [% vol]":          ("porosity_pct",  1.0, "%",      "measured",  "moderate"),
    "k [W/m/K] (unsaturated)":("tc_dry_WmK",   1.0, "W/(m·K)", "dry",      "moderate"),
    "k [W/m/K] (saturated)":  ("tc_sat_WmK",   1.0, "W/(m·K)", "saturated","moderate"),
    "Vp [m/s] (unsaturated)": ("vp_dry_ms",    1.0, "m/s",    "dry",       "moderate"),
    "Vp [m/s] (saturated)":   ("vp_sat_ms",    1.0, "m/s",    "saturated", "moderate"),
}


class Mielke2017Adapter(BaseAdapter):
    """Mielke et al. (2017) thermal conductivity and Vp dataset — PANGAEA 874146."""

    def __init__(self, config, data_dir, column_mapping):
        super().__init__(config, data_dir, column_mapping)
        self.source_label = "Mielke2017"

    def load(self):
        if not self._file_exists():
            return pd.DataFrame(), pd.DataFrame()

        meta = self.config.get("meta_columns", {})
        df = pd.read_csv(
            self._file_path(),
            sep="\t",
            skiprows=self.config.get("skip_rows", 42),
            encoding="latin1",
        )

        mapped = {}
        log_rows = []
        for raw_col, (std_name, factor, unit, sem, risk) in _COL_MAP.items():
            if raw_col not in df.columns:
                continue
            series = pd.to_numeric(df[raw_col], errors="coerce") * factor
            if std_name in mapped:
                mapped[std_name] = mapped[std_name].combine_first(series)
            else:
                mapped[std_name] = series
            log_rows.append({
                "source": self.source_label,
                "original_col": raw_col,
                "standardised_col": std_name,
                "unit_factor": factor,
                "standardised_unit": unit,
                "semantic_class": sem,
                "risk_class": risk,
            })

        if not mapped:
            print(f"  [warn] {self.source_label}: no columns mapped")
            return pd.DataFrame(), pd.DataFrame()

        out = pd.DataFrame(mapped, index=df.index)
        out["source_db"] = self.source_label
        out["raw_row_index"] = df.index

        out = self.add_meta(
            out, df,
            id_col=meta.get("sample_id"),
            litho_col=meta.get("lithology"),
            country_col=meta.get("country"),
            region_col=meta.get("region"),
        )

        log = pd.DataFrame(log_rows)
        print(f"  {self.source_label}: {len(out)} samples, {out.shape[1]} columns")
        return out, log

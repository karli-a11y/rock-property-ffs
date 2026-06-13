"""adapters/geneva_basin.py — Geneva Basin CERN geomechanical dataset (Zenodo 4725585)."""

import numpy as np
import pandas as pd
from .base import BaseAdapter

_COL_MAP = {
    "ultrasonic_Vp_m_per_s": ("vp_dry_ms",   1.0, "m/s",   "dry",         "moderate"),
    "ultrasonic_Vs_m_per_s": ("vs_dry_ms",   1.0, "m/s",   "dry",         "moderate"),
    "UCS_MPa":               ("ucs_MPa",     1.0, "MPa",   "destructive", "high"),
    "BRA_tensile_MPa":       ("tensile_MPa", 1.0, "MPa",   "destructive", "high"),
}


class GenevaBasinAdapter(BaseAdapter):
    """Geneva Basin / CERN geomechanical dataset (318 samples, Vp, Vs, UCS, tensile)."""

    def __init__(self, config, data_dir, column_mapping):
        super().__init__(config, data_dir, column_mapping)
        self.source_label = "GenevaBasin"

    def load(self):
        if not self._file_exists():
            return pd.DataFrame(), pd.DataFrame()

        meta = self.config.get("meta_columns", {})
        df = pd.read_excel(
            self._file_path(),
            sheet_name=self.config.get("sheet", 0),
            header=self.config.get("header_row", 0),
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
            depth_col=meta.get("depth"),
        )

        log = pd.DataFrame(log_rows)
        print(f"  {self.source_label}: {len(out)} samples, {out.shape[1]} columns")
        return out, log

"""adapters/norway_npd.py — Norwegian Petroleum Directorate PorPerm database (Zenodo 4419060).

Permeability unit: milliDarcy (mD). Conversion to m²: 1 mD = 9.869233e-16 m².
Porosity unit: integer percent (%).
Grain density unit: g/cm³.
"""

import numpy as np
import pandas as pd
from .base import BaseAdapter

_MD_TO_M2 = 9.869233e-16  # milliDarcy → m²

_COL_MAP = {
    "Klinkenberg corrected gas perm. Hor.": ("permeability_m2",   _MD_TO_M2,  "m²",    "intrinsic", "high"),
    "porosity best of available":           ("porosity_pct",      1.0,        "%",     "measured",  "moderate"),
    "gain density gr/cm3":                  ("grain_density_gcm3", 1.0,       "g/cm³", "grain",     "low"),
    "Measured Depth":                       ("_depth_raw",         1.0,       "m",     "meta",      "low"),
}


class NorwayNPDAdapter(BaseAdapter):
    """Norwegian Petroleum Directorate PorPerm database (387 k plug measurements, Norwegian shelf)."""

    def __init__(self, config, data_dir, column_mapping):
        super().__init__(config, data_dir, column_mapping)
        self.source_label = "NorwayNPD"

    def load(self):
        if not self._file_exists():
            return pd.DataFrame(), pd.DataFrame()

        meta = self.config.get("meta_columns", {})
        print(f"  {self.source_label}: loading large XLSX (~387k rows), please wait…")
        df = pd.read_excel(
            self._file_path(),
            sheet_name=self.config.get("sheet", "Combined all wells"),
        )

        mapped = {}
        log_rows = []
        for raw_col, (std_name, factor, unit, sem, risk) in _COL_MAP.items():
            if raw_col not in df.columns:
                continue
            series = pd.to_numeric(df[raw_col], errors="coerce") * factor
            if std_name.startswith("_"):
                mapped[std_name] = series
                continue
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

        # Keep only rows with at least one property value
        prop_cols = [c for c in mapped if not c.startswith("_")]
        if not prop_cols:
            print(f"  [warn] {self.source_label}: no property columns mapped")
            return pd.DataFrame(), pd.DataFrame()

        out = pd.DataFrame({k: v for k, v in mapped.items() if not k.startswith("_")}, index=df.index)
        mask = out[prop_cols].notna().any(axis=1)
        out = out[mask].copy()
        df_sub = df[mask].copy()

        out["source_db"] = self.source_label
        out["raw_row_index"] = df_sub.index

        depth_raw = mapped.get("_depth_raw")

        out = out.reset_index(drop=True)
        df_sub = df_sub.reset_index(drop=True)

        out["sample_id"] = df_sub.get("Plug or sample number", pd.Series(np.nan, index=df_sub.index))
        out["lithology"]  = df_sub.get("main lithology",       pd.Series(np.nan, index=df_sub.index))
        out["rock_group"] = df_sub.get("Main Lithology Origin",pd.Series(np.nan, index=df_sub.index))
        out["country"]    = "Norway"
        out["region"]     = df_sub.get("Well Name",            pd.Series(np.nan, index=df_sub.index))
        if depth_raw is not None:
            out["depth_m"] = pd.to_numeric(depth_raw[mask].reset_index(drop=True), errors="coerce")
        else:
            out["depth_m"] = np.nan

        log = pd.DataFrame(log_rows)
        print(f"  {self.source_label}: {len(out)} samples, {out.shape[1]} columns")
        return out, log

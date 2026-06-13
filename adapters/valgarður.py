"""adapters/valgarður.py — Valgarður Icelandic rock petrophysics database.

DOI: 10.5281/zenodo.6980231  (Scott et al. 2022)
Sheet: "Petrophysical properties", header=[0,1,2], 1,163 rows, 47 columns.

Key column indices (0-based after reading with header=[0,1,2]):
  6   Region / location
  14  Lithology (text)
  18  Grain density (g/cm³)
  20  Total porosity (fraction → ×100 → porosity_pct)
  22  Connected porosity (fraction → ×100, combine_first with col 20)
  24  Intrinsic permeability (m²)
  28  Gas apparent permeability (m², combine_first with col 24)
  37  Vp dry (km/s → ×1000 → m/s)
  38  Vp saturated (km/s → ×1000)
  39  Vs dry (km/s → ×1000)
  40  Vs saturated (km/s → ×1000)
  41  Poisson ratio (static)
  42  UCS (MPa)
  43  Young's modulus static (GPa)
  45  Tensile strength (MPa)
  46  Thermal conductivity dry (W/(m·K))
"""

import io
import zipfile

import numpy as np
import pandas as pd

from .base import BaseAdapter

_COL_IDX = {
    18: ("grain_density_gcm3", 1.0),
    20: ("porosity_pct",       100.0),
    37: ("vp_dry_ms",          1000.0),
    38: ("vp_sat_ms",          1000.0),
    39: ("vs_dry_ms",          1000.0),
    40: ("vs_sat_ms",          1000.0),
    41: ("poisson_ratio",      1.0),
    42: ("ucs_MPa",            1.0),
    43: ("E_static_GPa",       1.0),
    45: ("tensile_MPa",        1.0),
    46: ("tc_dry_WmK",         1.0),
}

_SEMANTIC = {
    "grain_density_gcm3": ("grain",       "low"),
    "porosity_pct":       ("measured",    "moderate"),
    "vp_dry_ms":          ("dry",         "moderate"),
    "vp_sat_ms":          ("saturated",   "moderate"),
    "vs_dry_ms":          ("dry",         "moderate"),
    "vs_sat_ms":          ("saturated",   "moderate"),
    "poisson_ratio":      ("static",      "high"),
    "ucs_MPa":            ("destructive", "high"),
    "E_static_GPa":       ("static",      "high"),
    "tensile_MPa":        ("destructive", "high"),
    "tc_dry_WmK":         ("dry",         "moderate"),
    "permeability_m2":    ("intrinsic",   "high"),
}


class ValgarðurAdapter(BaseAdapter):
    """Valgarður petrophysical database — Iceland (Scott et al. 2022)."""

    def __init__(self, config, data_dir, column_mapping):
        super().__init__(config, data_dir, column_mapping)
        self.source_label = "Valgarður"

    def load(self):
        if not self._file_exists():
            return pd.DataFrame(), pd.DataFrame()

        zip_path = self._file_path()
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                excel_files = [n for n in zf.namelist()
                               if n.lower().endswith((".xlsx", ".xls"))
                               and not n.startswith("__MACOSX")]
                if not excel_files:
                    print(f"  [warn] {self.source_label}: no Excel file in zip")
                    return pd.DataFrame(), pd.DataFrame()
                # Prefer the main database file
                target = next((f for f in excel_files
                               if "valgardur_database" in f.lower().replace("ð", "d")), excel_files[0])
                raw_bytes = zf.read(target)
        except Exception as e:
            print(f"  [error] {self.source_label}: cannot open zip: {e}")
            return pd.DataFrame(), pd.DataFrame()

        print(f"  {self.source_label}: loading Excel…")
        try:
            df = pd.read_excel(
                io.BytesIO(raw_bytes),
                sheet_name="Petrophysical properties",
                header=[0, 1, 2],
            )
        except Exception as e:
            print(f"  [error] {self.source_label}: cannot parse Excel: {e}")
            return pd.DataFrame(), pd.DataFrame()

        mapped = {}
        log_rows = []

        for idx, (std_name, factor) in _COL_IDX.items():
            if idx >= df.shape[1]:
                continue
            series = pd.to_numeric(df.iloc[:, idx], errors="coerce") * factor
            mapped[std_name] = series
            sem, risk = _SEMANTIC.get(std_name, ("measured", "moderate"))
            log_rows.append({
                "source":            self.source_label,
                "original_col":      str(df.columns[idx]),
                "standardised_col":  std_name,
                "unit_factor":       factor,
                "standardised_unit": "?",
                "semantic_class":    sem,
                "risk_class":        risk,
            })

        # Connected porosity (col 22, fraction) fills missing total porosity
        if 22 < df.shape[1]:
            conn = pd.to_numeric(df.iloc[:, 22], errors="coerce") * 100.0
            if "porosity_pct" in mapped:
                mapped["porosity_pct"] = mapped["porosity_pct"].combine_first(conn)
            else:
                mapped["porosity_pct"] = conn

        # Permeability: intrinsic (col 24) preferred, gas apparent (col 28) fallback
        perm_int = pd.to_numeric(df.iloc[:, 24], errors="coerce") if 24 < df.shape[1] else None
        perm_gas = pd.to_numeric(df.iloc[:, 28], errors="coerce") if 28 < df.shape[1] else None
        if perm_int is not None and perm_gas is not None:
            mapped["permeability_m2"] = perm_int.combine_first(perm_gas)
        elif perm_int is not None:
            mapped["permeability_m2"] = perm_int
        elif perm_gas is not None:
            mapped["permeability_m2"] = perm_gas
        if "permeability_m2" in mapped:
            log_rows.append({
                "source": self.source_label, "original_col": "col_24+28",
                "standardised_col": "permeability_m2", "unit_factor": 1.0,
                "standardised_unit": "?", "semantic_class": "intrinsic",
                "risk_class": "high",
            })

        if not mapped:
            print(f"  [warn] {self.source_label}: no columns mapped")
            return pd.DataFrame(), pd.DataFrame()

        out = pd.DataFrame(mapped)
        out = out[out.notna().any(axis=1)].copy()
        out["source_db"]     = self.source_label
        out["raw_row_index"] = out.index

        # Meta columns
        if df.shape[1] > 14:
            out["lithology"] = df.iloc[out["raw_row_index"], 14].values
        if df.shape[1] > 6:
            out["region"] = df.iloc[out["raw_row_index"], 6].values
        out["country"] = "Iceland"

        out = out.reset_index(drop=True)
        log = pd.DataFrame(log_rows).drop_duplicates(subset=["standardised_col"])
        print(f"  {self.source_label}: {len(out)} samples, {out.shape[1]} columns")
        return out, log

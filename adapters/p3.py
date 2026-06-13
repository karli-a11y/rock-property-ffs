"""adapters/p3.py — Adapter for P3 PetroPhysical Property Database (GFZ Potsdam).

DOI: 10.5880/GFZ.4.8.2019.P3  (Bär, Reinsch, Bott 2019)
Published: https://essd.copernicus.org/articles/12/2485/2020/

Structure: Two-level header (rows 2+3), data from row 4.
75,573 samples, wide format.

Key column indices (0-based after reading with header=[2,3]):
  0   Sample ID
  7   Country
  8   State/Region
  29  Petrographic term (text lithology)
  56  Grain Density  value (kg/m³)  → ÷1000 → g/cm³
  63  Bulk Density   value (kg/m³)  → ÷1000 → g/cm³
  70  Total Porosity value (%)
  77  Effective Porosity value (%)   → combine_first with col 70
  84  Apparent Permeability (m²)
  91  Intrinsic Permeability (m²)    → combine_first with col 84
  98  Bulk Thermal Conductivity (W/(m·K))
  114 Specific Heat Capacity (J/(kg·K))
  128 Thermal Diffusivity (m²/s)     → ×1e6 → 1e-6 m²/s
  143 P-wave velocity (m/s)
  151 S-Wave velocity (m/s)
  159 Young's Modulus Dynamic (GPa)
  166 Young's Modulus Static (GPa)
  194 Cohesion (MPa)
  222 Poisson Ratio dynamic
  229 Poisson Ratio static           → combine_first with col 222
  236 UCS (MPa)
  244 Tensile Strength (MPa)
"""

import io
import zipfile

import numpy as np
import pandas as pd

from .base import BaseAdapter

_COL_IDX = {
    56:  ("grain_density_gcm3",  1e-3),
    63:  ("bulk_density_gcm3",   1e-3),
    70:  ("porosity_pct",        1.0),
    98:  ("tc_dry_WmK",          1.0),
    114: ("cp_JkgK",             1.0),
    128: ("td_dry_1e6m2s",       1e6),
    143: ("vp_dry_ms",           1.0),
    151: ("vs_dry_ms",           1.0),
    159: ("E_dyn_GPa",           1.0),
    166: ("E_static_GPa",        1.0),
    194: ("cohesion_MPa",        1.0),
    236: ("ucs_MPa",             1.0),
    244: ("tensile_MPa",         1.0),
}

_SEMANTIC = {
    "grain_density_gcm3": ("grain",       "low"),
    "bulk_density_gcm3":  ("dry",         "low"),
    "porosity_pct":       ("measured",    "moderate"),
    "tc_dry_WmK":         ("dry",         "moderate"),
    "cp_JkgK":            ("measured",    "low"),
    "td_dry_1e6m2s":      ("dry",         "moderate"),
    "vp_dry_ms":          ("dry",         "moderate"),
    "vs_dry_ms":          ("dry",         "moderate"),
    "E_dyn_GPa":          ("dynamic",     "moderate"),
    "E_static_GPa":       ("static",      "high"),
    "cohesion_MPa":       ("destructive", "high"),
    "ucs_MPa":            ("destructive", "high"),
    "tensile_MPa":        ("destructive", "high"),
    "permeability_m2":    ("intrinsic",   "high"),
    "poisson_ratio":      ("static",      "high"),
}


class P3Adapter(BaseAdapter):
    """P3 PetroPhysical Property Database — GFZ Potsdam (Bär et al. 2019)."""

    def __init__(self, config, data_dir, column_mapping):
        super().__init__(config, data_dir, column_mapping)
        self.source_label = "P3"

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
                    print(f"  [warn] {self.source_label}: no Excel in zip")
                    return pd.DataFrame(), pd.DataFrame()
                raw_bytes = zf.read(excel_files[0])
        except Exception as e:
            print(f"  [error] {self.source_label}: cannot open zip: {e}")
            return pd.DataFrame(), pd.DataFrame()

        print(f"  {self.source_label}: loading Excel (~75k rows)…")
        df = pd.read_excel(io.BytesIO(raw_bytes), header=[2, 3])

        mapped = {}
        log_rows = []

        for idx, (std_name, factor) in _COL_IDX.items():
            if idx >= df.shape[1]:
                continue
            series = pd.to_numeric(df.iloc[:, idx], errors="coerce") * factor
            if std_name in mapped:
                mapped[std_name] = mapped[std_name].combine_first(series)
            else:
                mapped[std_name] = series
            sem, risk = _SEMANTIC.get(std_name, ("measured", "moderate"))
            log_rows.append({
                "source":           self.source_label,
                "original_col":     str(df.columns[idx]),
                "standardised_col": std_name,
                "unit_factor":      factor,
                "standardised_unit": "?",
                "semantic_class":   sem,
                "risk_class":       risk,
            })

        # Porosity: effective (col 77) fills missing total (col 70)
        if 77 < df.shape[1]:
            eff = pd.to_numeric(df.iloc[:, 77], errors="coerce")
            if "porosity_pct" in mapped:
                mapped["porosity_pct"] = mapped["porosity_pct"].combine_first(eff)
            else:
                mapped["porosity_pct"] = eff

        # Permeability: intrinsic (col 91) preferred, apparent (col 84) as fallback
        perm_app = pd.to_numeric(df.iloc[:, 84], errors="coerce") if 84 < df.shape[1] else None
        perm_int = pd.to_numeric(df.iloc[:, 91], errors="coerce") if 91 < df.shape[1] else None
        if perm_int is not None and perm_app is not None:
            mapped["permeability_m2"] = perm_int.combine_first(perm_app)
        elif perm_int is not None:
            mapped["permeability_m2"] = perm_int
        elif perm_app is not None:
            mapped["permeability_m2"] = perm_app
        if "permeability_m2" in mapped:
            log_rows.append({"source": self.source_label, "original_col": "col_91+84",
                             "standardised_col": "permeability_m2", "unit_factor": 1.0,
                             "standardised_unit": "?", "semantic_class": "intrinsic",
                             "risk_class": "high"})

        # Poisson: static (col 229) preferred, dynamic (col 222) as fallback
        pois_stat = pd.to_numeric(df.iloc[:, 229], errors="coerce") if 229 < df.shape[1] else None
        pois_dyn  = pd.to_numeric(df.iloc[:, 222], errors="coerce") if 222 < df.shape[1] else None
        if pois_stat is not None and pois_dyn is not None:
            mapped["poisson_ratio"] = pois_stat.combine_first(pois_dyn)
        elif pois_stat is not None:
            mapped["poisson_ratio"] = pois_stat
        elif pois_dyn is not None:
            mapped["poisson_ratio"] = pois_dyn
        if "poisson_ratio" in mapped:
            log_rows.append({"source": self.source_label, "original_col": "col_229+222",
                             "standardised_col": "poisson_ratio", "unit_factor": 1.0,
                             "standardised_unit": "?", "semantic_class": "static",
                             "risk_class": "high"})

        if not mapped:
            print(f"  [warn] {self.source_label}: no columns mapped")
            return pd.DataFrame(), pd.DataFrame()

        out = pd.DataFrame(mapped)

        # Keep only rows with at least one valid measurement
        out = out[out.notna().any(axis=1)].copy()
        out["source_db"]     = self.source_label
        out["raw_row_index"] = out.index

        # Meta columns
        if df.shape[1] > 0:
            out["sample_id"] = df.iloc[out["raw_row_index"], 0].values
        if df.shape[1] > 29:
            out["lithology"] = df.iloc[out["raw_row_index"], 29].values
        if df.shape[1] > 7:
            out["country"] = df.iloc[out["raw_row_index"], 7].values
        if df.shape[1] > 8:
            out["region"] = df.iloc[out["raw_row_index"], 8].values

        out = out.reset_index(drop=True)
        log = pd.DataFrame(log_rows).drop_duplicates(subset=["standardised_col"])
        print(f"  {self.source_label}: {len(out)} samples, {out.shape[1]} columns")
        return out, log

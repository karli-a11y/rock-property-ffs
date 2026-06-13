"""adapters/central_apennines.py — Central Apennines Italy rock mechanics database.

Structure (sheet="Parameters", header=None):
  Row 0: empty
  Row 1: section labels
  Row 2: sub-section labels (Specimen, Country, …, Lithology, Physical …, Ultrasonic, Brazilian, UCS, Triaxial)
  Row 3: column symbols (γ, n, VP, Ed, ν, σt, σc, E, G, ν, c, ϕ, …, Is50)
  Row 4: units (kN/m³, %, m/s, GPa, –, MPa, MPa, GPa, GPa, –, MPa, °, …, MPa)
  Row 5+: data (90 specimens)

Column index→property mapping:
  1  Specimen (sample ID)
  2  Country
  3  Region
  7  Lithology
  8  γ  unit weight kN/m³  → bulk_density_gcm3  (÷ 9.81)
  9  n  porosity %          → porosity_pct
  10 VP m/s                 → vp_dry_ms
  11 Ed dynamic E (GPa)     → E_dyn_GPa
  12 ν  dynamic Poisson     → poisson_ratio (fallback)
  13 σt tensile MPa         → tensile_MPa
  14 σc UCS MPa             → ucs_MPa
  15 E  static E (GPa)      → E_static_GPa
  17 ν  static Poisson      → poisson_ratio (primary)
  18 c  cohesion MPa        → cohesion_MPa
  19 ϕ  friction angle °    → friction_angle_deg
"""

import pandas as pd
from .base import BaseAdapter

_COL_IDX = {
    8:  ("bulk_density_gcm3",  1.0 / 9.81),   # kN/m³ → g/cm³
    9:  ("porosity_pct",       1.0),
    10: ("vp_dry_ms",          1.0),
    11: ("E_dyn_GPa",          1.0),
    13: ("tensile_MPa",        1.0),
    14: ("ucs_MPa",            1.0),
    15: ("E_static_GPa",       1.0),
    18: ("cohesion_MPa",       1.0),
    19: ("friction_angle_deg", 1.0),
}
_POISSON_DYN_IDX  = 12   # fallback
_POISSON_STAT_IDX = 17   # primary


class CentralApenninesAdapter(BaseAdapter):
    """Central Apennines Italy rock mechanics database (90 specimens)."""

    def __init__(self, config, data_dir, column_mapping):
        super().__init__(config, data_dir, column_mapping)
        self.source_label = "CentralApennines"

    def load(self):
        if not self._file_exists():
            return pd.DataFrame(), pd.DataFrame()

        raw = pd.read_excel(
            self._file_path(),
            sheet_name=self.config.get("sheet", "Parameters"),
            header=None,
            skiprows=5,
            na_values=["-", ""],
        ).dropna(how="all")

        if raw.empty:
            print(f"  [warn] {self.source_label}: no data rows found")
            return pd.DataFrame(), pd.DataFrame()

        mapped = {}
        log_rows = []

        for idx, (std_name, factor) in _COL_IDX.items():
            if idx >= raw.shape[1]:
                continue
            series = pd.to_numeric(raw.iloc[:, idx], errors="coerce") * factor
            mapped[std_name] = series

            log_rows.append({
                "source":            self.source_label,
                "original_col":      f"col_{idx}",
                "standardised_col":  std_name,
                "unit_factor":       factor,
                "standardised_unit": "?",
                "semantic_class":    "measured",
                "risk_class":        "moderate",
            })

        # Poisson's ratio: prefer static (col 17), fill with dynamic (col 12)
        stat = pd.to_numeric(raw.iloc[:, _POISSON_STAT_IDX], errors="coerce") if _POISSON_STAT_IDX < raw.shape[1] else None
        dyn  = pd.to_numeric(raw.iloc[:, _POISSON_DYN_IDX],  errors="coerce") if _POISSON_DYN_IDX  < raw.shape[1] else None
        if stat is not None and dyn is not None:
            mapped["poisson_ratio"] = stat.combine_first(dyn)
        elif stat is not None:
            mapped["poisson_ratio"] = stat
        elif dyn is not None:
            mapped["poisson_ratio"] = dyn
        if "poisson_ratio" in mapped:
            log_rows.append({"source": self.source_label, "original_col": "col_17+12",
                             "standardised_col": "poisson_ratio", "unit_factor": 1.0,
                             "standardised_unit": "?", "semantic_class": "measured", "risk_class": "moderate"})

        out = pd.DataFrame(mapped)
        out["source_db"]      = self.source_label
        out["raw_row_index"]  = raw.index

        meta = self.config.get("meta_columns", {})
        id_col     = meta.get("sample_id")
        litho_col  = meta.get("lithology")
        country_col = meta.get("country")
        region_col  = meta.get("region")

        if id_col is not None and isinstance(id_col, int) and id_col < raw.shape[1]:
            out["sample_id"] = raw.iloc[:, id_col].values
        if litho_col is not None and isinstance(litho_col, int) and litho_col < raw.shape[1]:
            out["lithology"] = raw.iloc[:, litho_col].values
        if country_col is not None and isinstance(country_col, int) and country_col < raw.shape[1]:
            out["country"] = raw.iloc[:, country_col].values
        if region_col is not None and isinstance(region_col, int) and region_col < raw.shape[1]:
            out["region"] = raw.iloc[:, region_col].values

        log = pd.DataFrame(log_rows).drop_duplicates(subset=["standardised_col"])
        print(f"  {self.source_label}: {len(out)} samples, {out.shape[1]} columns")
        return out, log

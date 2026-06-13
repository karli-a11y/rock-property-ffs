"""adapters/midgerman.py — Mid-German Crystalline Rise petrophysical database.

Column names in this CSV use CP437-style mojibake when read as latin1:
  0xFC (ü) represents ³   (cube/g/cm³)
  0xFD (ý) represents ²   (square/m²)
  0xFA (ú) represents ·   (middle dot in J/(kg·K))
  0xF8 (ø) represents °   (degree, angle of friction)
  0x3F (?) represents ·   (in W/(m·K))
Column keys below use the mojibake characters as they actually appear.
"""

import numpy as np
import pandas as pd
from .base import BaseAdapter

# Exact column matches (as they appear when read with encoding='latin1')
_EXACT = {
    "Bulk Density [g/cm\xfc]":            ("bulk_density_gcm3",  1.0),
    "Grain Density [g/cm\xfc]":           ("grain_density_gcm3", 1.0),
    "Porosity [%]":                        ("porosity_pct",       1.0),
    "Thermal Conductivity, Sample Average [W/(m?K)]": ("tc_dry_WmK", 1.0),
    "Thermal Diffusivity, Sample Average [1E-6 m\xfd/s]": ("td_dry_1e6m2s", 1.0),
    "Specific Heat Capacity [J/(kg\xfaK)]": ("cp_JkgK",           1.0),
    "Permeability, apparent [m\xfd]":      ("permeability_m2",    1.0),
    "Permeability, intrinsic [m\xfd]":     ("permeability_m2",    1.0),
    "Compressive Wave Velocity , Sample Average [m/s]": ("vp_dry_ms", 1.0),
    "Shear Wave Velocity, Sample Average [m/s]": ("vs_dry_ms",    1.0),
    "Unconfined Compressive Strength [MN/m\xfd]": ("ucs_MPa",     1.0),
    "Tensile Strength [MPa]":              ("tensile_MPa",        1.0),
    "Cohesion [MPa]":                      ("cohesion_MPa",       1.0),
    "Angle of Friction [\xf8]":            ("friction_angle_deg", 1.0),
    "Static Poisson's Ratio [-]":          ("poisson_ratio",      1.0),
}

# Prefix matches for multiline / complex column names
_PREFIX = {
    "Dynamic Young's Modulus [GN/m":   ("E_dyn_GPa",    1.0),
    "Static Young's Modulus [GN/m":    ("E_static_GPa", 1.0),
    "Dynamic Poisson's Ratio [-]":     ("poisson_ratio", 1.0),
}

_SEMANTIC = {
    "bulk_density_gcm3":  ("dry",         "low"),
    "grain_density_gcm3": ("grain",       "low"),
    "porosity_pct":       ("measured",    "moderate"),
    "tc_dry_WmK":         ("dry",         "moderate"),
    "td_dry_1e6m2s":      ("dry",         "moderate"),
    "cp_JkgK":            ("measured",    "low"),
    "permeability_m2":    ("intrinsic",   "high"),
    "vp_dry_ms":          ("dry",         "moderate"),
    "vs_dry_ms":          ("dry",         "moderate"),
    "ucs_MPa":            ("destructive", "high"),
    "tensile_MPa":        ("destructive", "high"),
    "cohesion_MPa":       ("destructive", "high"),
    "friction_angle_deg": ("destructive", "high"),
    "E_dyn_GPa":          ("dynamic",     "moderate"),
    "E_static_GPa":       ("static",      "high"),
    "poisson_ratio":      ("static",      "high"),
}


class MidGermanAdapter(BaseAdapter):
    """Mid-German Crystalline Rise petrophysical database (TU Darmstadt / TUdatalib)."""

    def __init__(self, config, data_dir, column_mapping):
        super().__init__(config, data_dir, column_mapping)
        self.source_label = "MidGerman"

    def load(self):
        if not self._file_exists():
            return pd.DataFrame(), pd.DataFrame()

        meta = self.config.get("meta_columns", {})
        df = pd.read_csv(
            self._file_path(),
            sep=";",
            header=self.config.get("header_row", 2),
            encoding="latin1",
            low_memory=False,
        )

        mapped = {}
        log_rows = []

        for raw_col in df.columns:
            std_name, factor = None, None

            if raw_col in _EXACT:
                std_name, factor = _EXACT[raw_col]
            else:
                for prefix, (sn, fc) in _PREFIX.items():
                    if raw_col.startswith(prefix):
                        std_name, factor = sn, fc
                        break

            if std_name is None:
                continue

            series = pd.to_numeric(df[raw_col], errors="coerce") * factor
            if std_name in mapped:
                mapped[std_name] = mapped[std_name].combine_first(series)
            else:
                mapped[std_name] = series

            sem, risk = _SEMANTIC.get(std_name, ("unknown", "unknown"))
            log_rows.append({
                "source": self.source_label,
                "original_col": raw_col,
                "standardised_col": std_name,
                "unit_factor": factor,
                "standardised_unit": "?",
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

        log = pd.DataFrame(log_rows).drop_duplicates(subset=["standardised_col"])
        print(f"  {self.source_label}: {len(out)} samples, {out.shape[1]} columns")
        return out, log

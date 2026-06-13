"""adapters/global_db.py — Adapter for Global Petrophysical Database V5 (long format)."""

import numpy as np
import pandas as pd
from .base import BaseAdapter


# Long-format property name → (standardised_col, unit_factor)
LONG_FORMAT_MAP = {
    "Porosity - effective (%)":            ("porosity_pct",            1.0),
    "Porosity (%)":                        ("porosity_pct",            1.0),
    "Porosity - total (%)":                ("porosity_pct",            1.0),
    "Density - bulk (kg/m3)":              ("bulk_density_gcm3",       1e-3),
    "Density - bulk dry (kg/m3)":          ("bulk_density_gcm3",       1e-3),
    "Density - bulk wet (kg/m3)":          ("bulk_density_gcm3_wet",   1e-3),
    "Density - grain (kg/m3)":             ("grain_density_gcm3",      1e-3),
    "Density - matrix (kg/m3)":            ("grain_density_gcm3",      1e-3),
    "Permeability - intrinsic (m2)":       ("permeability_m2",         1.0),
    "Permeability - apparent (m2)":        ("permeability_m2",         1.0),
    "Thermal conductivity - bulk (w/m/k)": ("tc_dry_WmK",              1.0),
    "Thermal conductivity - bulk dry (w/m/k)": ("tc_dry_WmK",          1.0),
    "Thermal conductivity - bulk wet (w/m/k)": ("tc_sat_WmK",          1.0),
    "Heat capacity - specific (j/kg/k)":   ("cp_JkgK",                 1.0),
    "Heat capacity - volumetric (j/m3/k)": ("cv_jm3k",                 1.0),
    "P-wave velocity, dry (km s-1)":       ("vp_dry_ms",             1000.0),
    "P-wave velocity, saturated (km s-1)": ("vp_sat_ms",             1000.0),
    "S-wave velocity, dry (km s-1)":       ("vs_dry_ms",             1000.0),
    "S-wave velocity, saturated (km s-1)": ("vs_sat_ms",             1000.0),
    "Young modulus coef. (gpa)":           ("E_static_GPa",            1.0),
    "Uniaxial compressive strength (mpa)": ("ucs_MPa",                 1.0),
    "Tensile strength (mpa)":              ("tensile_MPa",             1.0),
    "Poisson's ratio (-)":                 ("poisson_ratio",           1.0),
    "Bulk resistivity (?m)":               ("resistivity_dry_Ohm",     1.0),
    "Total porosity\n(%)":                 ("porosity_pct",            1.0),
    "Connected porosity (-)":              ("porosity_connected_pct", 100.0),
}


class GlobalDBAdapter(BaseAdapter):
    """Global Petrophysical Database V5 — long format, pivot to wide."""

    def __init__(self, config, data_dir, column_mapping):
        super().__init__(config, data_dir, column_mapping)
        self.source_label = "GlobalDB"

    def load(self):
        if not self._file_exists():
            return pd.DataFrame(), pd.DataFrame()

        lf_cfg = self.config.get("long_format", {})
        prop_col = lf_cfg.get("property_col", "Property and units")
        val_col  = lf_cfg.get("value_col", "Value")
        meta = self.config.get("meta_columns", {})

        df = pd.read_excel(
            self._file_path(),
            sheet_name=self.config.get("sheet", "Data"),
        )

        if prop_col not in df.columns or val_col not in df.columns:
            print(f"  [warn] {self.source_label}: expected long-format columns not found")
            return pd.DataFrame(), pd.DataFrame()

        # Meta columns present in source
        meta_col_names = [v for v in meta.values() if v in df.columns]
        sample_id_col  = meta.get("sample_id", "Sample ID")

        # Map property names to standard
        df["_std_col"] = df[prop_col].map(LONG_FORMAT_MAP).apply(
            lambda x: x[0] if isinstance(x, tuple) else np.nan
        )
        df["_factor"] = df[prop_col].map(LONG_FORMAT_MAP).apply(
            lambda x: x[1] if isinstance(x, tuple) else np.nan
        )

        mapped_rows = df.dropna(subset=["_std_col"]).copy()
        mapped_rows["_value_std"] = (
            pd.to_numeric(mapped_rows[val_col], errors="coerce")
            * mapped_rows["_factor"]
        )

        # Build mapping log
        mapping_log = (
            mapped_rows[["_std_col"]]
            .drop_duplicates()
            .rename(columns={"_std_col": "standardised_col"})
            .assign(source=self.source_label,
                    original_col=lambda d: d["standardised_col"].map(
                        {v[0]: k for k, v in LONG_FORMAT_MAP.items()}
                    ),
                    unit_factor=lambda d: d["standardised_col"].map(
                        {v[0]: v[1] for k, v in LONG_FORMAT_MAP.items()}
                    ),
                    standardised_unit="various",
                    semantic_class="long_format_pivot",
                    risk_class="moderate")
        )

        # Pivot: one row per sample
        if sample_id_col in df.columns:
            pivot_meta = df[[sample_id_col] + [c for c in meta_col_names
                                                if c != sample_id_col]].drop_duplicates(
                subset=[sample_id_col]
            )
            wide = mapped_rows.pivot_table(
                index=sample_id_col,
                columns="_std_col",
                values="_value_std",
                aggfunc="mean",
            ).reset_index()
            wide = wide.merge(pivot_meta, on=sample_id_col, how="left")
        else:
            wide = mapped_rows.pivot_table(
                columns="_std_col",
                values="_value_std",
                aggfunc="mean",
            ).reset_index()

        wide["source_db"] = self.source_label
        wide["raw_row_index"] = wide.index

        # Attach metadata
        wide["sample_id"] = wide.get(sample_id_col, pd.Series(np.nan, index=wide.index))
        wide["lithology"]  = wide.get(meta.get("lithology", ""), pd.Series(np.nan, index=wide.index))
        wide["rock_group"] = wide.get(meta.get("rock_group", ""), pd.Series(np.nan, index=wide.index))
        wide["country"]    = wide.get(meta.get("country", ""), pd.Series(np.nan, index=wide.index))
        wide["region"]     = wide.get(meta.get("region", ""), pd.Series(np.nan, index=wide.index))
        wide["depth_m"]    = np.nan

        print(f"  {self.source_label}: {len(wide)} samples, {wide.shape[1]} columns")
        return wide, mapping_log

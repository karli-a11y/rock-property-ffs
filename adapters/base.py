"""
adapters/base.py — Abstract base adapter for database ingestion (P1.1)

All source adapters must subclass BaseAdapter and implement load().
The returned DataFrame always has:
  - standardised feature columns (from COLUMN_MAPPING)
  - source_db column (string identifier)
  - metadata columns: sample_id, lithology, rock_group, country, region, depth_m
  - raw_row_index (original row position for provenance linking)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


class BaseAdapter(ABC):
    """Abstract base class for source database adapters."""

    def __init__(self, config: dict, data_dir: Path, column_mapping: dict):
        """
        Parameters
        ----------
        config : dict
            Database-specific config block from config/databases.yaml
        data_dir : Path
            Root directory containing raw database files
        column_mapping : dict
            Mapping from raw column names → (std_name, factor, unit, semantic, risk)
        """
        self.config = config
        self.data_dir = data_dir
        self.column_mapping = column_mapping
        self.source_label = None  # set by subclass

    @abstractmethod
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and harmonise the source database.

        Returns
        -------
        harmonised : pd.DataFrame
            Standardised feature table with source_db and metadata columns.
        mapping_log : pd.DataFrame
            One row per raw→standard column mapping applied, with columns:
            [source, original_col, standardised_col, unit_factor, standardised_unit,
             semantic_class, risk_class]
        """
        raise NotImplementedError

    def apply_mapping(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply column_mapping to a wide-format raw DataFrame.
        Handles duplicate mapped targets by taking the first non-null value.
        Returns (harmonised_df, mapping_log_df).
        """
        mapped = {}
        mapping_log = []

        for raw_col in df_raw.columns:
            key = str(raw_col).strip().lower()
            if key in self.column_mapping:
                entry = self.column_mapping[key]
                std_name, factor = entry[0], entry[1]
                unit = entry[2] if len(entry) > 2 else "?"
                semantic = entry[3] if len(entry) > 3 else "unknown"
                risk = entry[4] if len(entry) > 4 else "unknown"

                series = pd.to_numeric(df_raw[raw_col], errors="coerce") * factor
                if std_name in mapped:
                    mapped[std_name] = mapped[std_name].combine_first(series)
                else:
                    mapped[std_name] = series

                mapping_log.append({
                    "source": self.source_label,
                    "original_col": raw_col,
                    "standardised_col": std_name,
                    "unit_factor": factor,
                    "standardised_unit": unit,
                    "semantic_class": semantic,
                    "risk_class": risk,
                })

        if not mapped:
            return pd.DataFrame(), pd.DataFrame()

        out = pd.DataFrame(mapped, index=df_raw.index)
        out["source_db"] = self.source_label
        out["raw_row_index"] = df_raw.index
        log_df = pd.DataFrame(mapping_log)
        return out, log_df

    def add_meta(self, out: pd.DataFrame, df_raw: pd.DataFrame,
                 id_col=None, litho_col=None, rock_col=None,
                 country_col=None, region_col=None, depth_col=None) -> pd.DataFrame:
        """Attach metadata columns to the mapped frame."""
        def _get(col):
            if col and col in df_raw.columns:
                return df_raw[col].reset_index(drop=True)
            return pd.Series(np.nan, index=range(len(df_raw)))

        out = out.reset_index(drop=True)
        out["sample_id"] = _get(id_col)
        out["lithology"]  = _get(litho_col)
        out["rock_group"] = _get(rock_col)
        out["country"]    = _get(country_col)
        out["region"]     = _get(region_col)
        out["depth_m"]    = pd.to_numeric(_get(depth_col), errors="coerce")
        return out

    def _file_path(self) -> Path:
        """Resolve the database file path."""
        return self.data_dir / self.config["file"]

    def _file_exists(self) -> bool:
        path = self._file_path()
        if not path.exists():
            print(f"  [skip] {self.source_label}: file not found ({path.name})")
            return False
        return True

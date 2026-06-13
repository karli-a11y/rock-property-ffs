"""adapters/ampedek.py — Adapter for AMPEDEK database."""

import pandas as pd
from .base import BaseAdapter


class AMPEDEKAdapter(BaseAdapter):
    """AMPEDEK multi-source petrophysical database."""

    def __init__(self, config, data_dir, column_mapping):
        super().__init__(config, data_dir, column_mapping)
        self.source_label = "AMPEDEK"

    def load(self):
        if not self._file_exists():
            return pd.DataFrame(), pd.DataFrame()

        meta = self.config.get("meta_columns", {})
        df = pd.read_excel(
            self._file_path(),
            sheet_name=self.config.get("sheet", "MetaData_RockProperties"),
            header=self.config.get("header_row", 3),
        )

        out, log = self.apply_mapping(df)
        if out.empty:
            print(f"  [warn] {self.source_label}: no columns mapped")
            return pd.DataFrame(), pd.DataFrame()

        out = self.add_meta(
            out, df,
            id_col=meta.get("sample_id"),
            litho_col=meta.get("lithology"),
            rock_col=meta.get("rock_group"),
            country_col=meta.get("country"),
            region_col=meta.get("region"),
            depth_col=meta.get("depth"),
        )
        print(f"  {self.source_label}: {len(out)} samples, {out.shape[1]} columns")
        return out, log

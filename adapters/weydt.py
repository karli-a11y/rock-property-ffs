"""adapters/weydt.py — Adapter for Weydt et al. (2020) database."""

import pandas as pd
from .base import BaseAdapter


class WeydtAdapter(BaseAdapter):
    """Weydt et al. (2020) petrophysical and mechanical rock property database."""

    def __init__(self, config, data_dir, column_mapping):
        super().__init__(config, data_dir, column_mapping)
        self.source_label = "Weydt2020"

    def load(self):
        if not self._file_exists():
            return pd.DataFrame(), pd.DataFrame()

        meta = self.config.get("meta_columns", {})
        df = pd.read_excel(
            self._file_path(),
            sheet_name=self.config.get("sheet", "Petrophysical rock properties"),
            header=self.config.get("header_row", 5),
        )

        out, log = self.apply_mapping(df)
        if out.empty:
            print(f"  [warn] {self.source_label}: no columns mapped")
            return pd.DataFrame(), pd.DataFrame()

        out = self.add_meta(
            out, df,
            id_col=meta.get("sample_id"),
            litho_col=meta.get("lithology"),
            region_col=meta.get("region"),
            depth_col=meta.get("depth"),
        )
        print(f"  {self.source_label}: {len(out)} samples, {out.shape[1]} columns")
        return out, log

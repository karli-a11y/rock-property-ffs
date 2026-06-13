"""adapters/gemex.py — GEMex Mexico petrophysical database (Weydt et al. 2020, Los Humeros / Acoculco).

Column names in this CSV use CP437-style mojibake when read as latin1:
  0xFC (ü) represents ³   (cube superscript)
  0xFD (ý) represents ²   (square superscript)
  0x3F (?) represents φ   (Greek phi, for friction angle)
These are normalized before column mapping.
"""

import pandas as pd
from .base import BaseAdapter

_MOJIBAKE = str.maketrans({
    '\xfc': '³',   # ü → ³
    '\xfd': '²',   # ý → ²
})


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.translate(_MOJIBAKE) for c in df.columns]
    return df


class GEMexAdapter(BaseAdapter):
    """Weydt et al. (2020) GEMex Mexico petrophysical and mechanical rock property database."""

    def __init__(self, config, data_dir, column_mapping):
        super().__init__(config, data_dir, column_mapping)
        self.source_label = "GEMex2020"

    def load(self):
        if not self._file_exists():
            return pd.DataFrame(), pd.DataFrame()

        meta = self.config.get("meta_columns", {})
        df = pd.read_csv(
            self._file_path(),
            sep=";",
            header=self.config.get("header_row", 5),
            encoding="latin1",
            low_memory=False,
        )
        df = _norm_cols(df)

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

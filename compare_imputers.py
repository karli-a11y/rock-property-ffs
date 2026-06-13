"""Generate a side-by-side comparison of best-model R^2 under KNN vs median
imputation. Reads results/tables/best_model_global.csv and
results_median/tables/best_model_global.csv and writes a delta table."""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent
# results now contains median (primary) results; results_knn contains the KNN sensitivity run
MED = pd.read_csv(BASE / "results"     / "tables" / "best_model_global.csv")
KNN = pd.read_csv(BASE / "results_knn" / "tables" / "best_model_global.csv")

key = "target"
score = "best_r2" if "best_r2" in KNN.columns else "r2"

K = KNN[[key, score]].rename(columns={score: "r2_knn"})
M = MED[[key, score]].rename(columns={score: "r2_median"})
df = K.merge(M, on=key)
df["delta_knn_minus_median"] = df["r2_knn"] - df["r2_median"]
df = df.sort_values("delta_knn_minus_median", ascending=False)

out_csv = BASE / "results" / "tables" / "imputer_sensitivity.csv"
df.to_csv(out_csv, index=False)
print(df.to_string(index=False))
print(f"\nsaved {out_csv}")
print(f"\nKNN better: {(df['delta_knn_minus_median'] > 0).sum()} / {len(df)}")
print(f"|delta| > 0.02: {(df['delta_knn_minus_median'].abs() > 0.02).sum()}")

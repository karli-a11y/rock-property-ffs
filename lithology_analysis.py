"""
lithology_analysis.py — Lithology-aware analysis pipeline (geoscience revision).

Changes vs. previous version:
  - LOLO validation removed (not reported in paper)
  - Cross-database within-lithology analysis removed
  - Feature importance extended to ALL targets × ALL lithology classes
  - Per-lithology feature ranking multipanel figure added
  - Best-model-per-(target, lithology) analysis with hyperparameter tuning added
  - Best-model heatmap figure and LaTeX table generated

Run:  python lithology_analysis.py
"""
from __future__ import annotations

import shutil
import warnings
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from pygam import LinearGAM as _LinearGAM
from sklearn.base import BaseEstimator, RegressorMixin


class _PLSWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=3):
        self.n_components = n_components
    def fit(self, X, y):
        X = np.asarray(X)
        nc = min(self.n_components, X.shape[1], max(1, X.shape[0] - 1))
        self.pls_ = PLSRegression(n_components=max(1, nc))
        self.pls_.fit(X, y)
        return self
    def predict(self, X):
        return self.pls_.predict(np.asarray(X)).ravel()


class _GAMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, lam=0.6):
        self.lam = lam
    def fit(self, X, y):
        X = np.asarray(X)
        self.gam_ = _LinearGAM(lam=self.lam).fit(X, y)
        return self
    def predict(self, X):
        return self.gam_.predict(np.asarray(X))

warnings.filterwarnings("ignore")

BASE       = Path(__file__).parent
MASTER_CSV = BASE / "data" / "master_table.csv.gz"
import os
_V14 = BASE / os.environ.get("OUT_DIR_NAME", "results")
OUT_DIR    = _V14 / "lithology"
FIG_DIR    = OUT_DIR / "figures"
TAB_DIR    = OUT_DIR / "tables"
MS_DIR     = BASE / "manuscript"
MS_TAB     = MS_DIR / "paper_tables"

for d in [OUT_DIR, FIG_DIR, TAB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

from variables import NUMERIC_FEATURES, FEATURE_LABELS, HARMONIZATION_RISK
from target_registry import TargetRegistry
from lithology_mapper import LithologyMapper

# ── Configuration ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
})

CLASS_COLOURS = {
    "Magmatic":              "#4477AA",
    "Metamorphic":           "#228833",
    "Clastic sedimentary":   "#CCBB44",
    "Carbonate sedimentary": "#66CCEE",
    "Evaporite":             "#AA3377",
    # Legacy fine-class colours (kept so any pre-merge plots still work)
    "Plutonic":              "#4477AA",
    "Volcanic":              "#EE6677",
    "Pyroclastic":           "#EE8866",
    "Unconsolidated":        "#BBBBBB",
    "Unclassified":          "#DDDDDD",
}

RISK_COL = {
    "low":      "#2ca02c",
    "moderate": "#ff7f0e",
    "high":     "#d62728",
    "unknown":  "#aaaaaa",
}

# Model families — identical set to run_pipeline.py (9 families)
MODEL_FAMILIES = {
    "Ridge":      (Ridge(), {"mdl__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}),
    "PLS":        (_PLSWrapper(), {"mdl__n_components": [1, 2, 3, 5, 8]}),
    "GAM":        (_GAMWrapper(), {"mdl__lam": [0.001, 0.01, 0.1, 1.0, 10.0]}),
    "ExtraTrees": (ExtraTreesRegressor(random_state=42, n_jobs=-1), {
        "mdl__n_estimators": [80, 150],
        "mdl__max_depth":    [None, 10, 20],
        "mdl__min_samples_leaf": [1, 3, 5],
    }),
    "RF":         (RandomForestRegressor(random_state=42, n_jobs=-1), {
        "mdl__n_estimators": [80, 150],
        "mdl__max_depth":    [None, 10, 20],
        "mdl__min_samples_leaf": [1, 3, 5],
    }),
    "XGBoost":    (XGBRegressor(random_state=42, n_jobs=-1, verbosity=0,
                                objective="reg:squarederror"), {
        "mdl__n_estimators":  [80, 150],
        "mdl__learning_rate": [0.05, 0.1, 0.2],
        "mdl__max_depth":     [3, 5, 7],
        "mdl__subsample":     [0.8, 1.0],
    }),
    "SVR":        (SVR(kernel="rbf"), {
        "mdl__C":       [0.1, 1.0, 10.0, 100.0],
        "mdl__epsilon": [0.01, 0.1, 0.5],
    }),
    "KNN":        (KNeighborsRegressor(), {
        "mdl__n_neighbors": [3, 5, 10, 20],
        "mdl__weights":     ["uniform", "distance"],
    }),
    "MLP":        (MLPRegressor(random_state=42, max_iter=500), {
        "mdl__hidden_layer_sizes": [(100,), (100, 50), (200, 100)],
        "mdl__alpha":              [0.0001, 0.001, 0.01],
        "mdl__activation":         ["relu", "tanh"],
        "mdl__learning_rate_init": [0.0001, 0.001, 0.01],
    }),
}

N_ITER_HP   = 12
LOG_TARGETS = {"permeability_m2"}
MAX_SAMPLES = 50_000
MODEL_SAMPLE_CAP = {"SVR": 5_000, "GAM": 8_000, "MLP": 8_000}


def _pipe(model):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("mdl", deepcopy(model)),
    ])


def _best_pipe(base_model, param_grid, X_tr, y_tr, cv=3):
    pipe = _pipe(base_model)
    if not param_grid:
        pipe.fit(X_tr, y_tr)
        return pipe
    search = RandomizedSearchCV(
        pipe, param_grid, n_iter=min(N_ITER_HP, 1 + len(param_grid)),
        cv=cv, scoring="r2", random_state=42, n_jobs=-1, refit=True,
    )
    search.fit(X_tr, y_tr)
    return search.best_estimator_


def savefig(fig, name, ms_name=None):
    p = FIG_DIR / f"{name}.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {p.name}")
    if ms_name and MS_DIR.exists():
        dest = MS_DIR / ms_name
        if MS_DIR.exists():
            shutil.copy2(p, dest)
            print(f"  [->ms] {dest.name}")


def cramers_v(confusion_matrix: np.ndarray) -> float:
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    r, k = confusion_matrix.shape
    phi2 = chi2 / n
    phi2_corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    k_corr = k - (k - 1) ** 2 / (n - 1)
    r_corr = r - (r - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("LITHOLOGY-AWARE ANALYSIS PIPELINE")
print("=" * 70)

print("\nLoading master table …")
master = pd.read_csv(MASTER_CSV, low_memory=False)
master = master[master["source_db"] != "NorwayNPD"].reset_index(drop=True)
from master_split import apply_source_split
master = apply_source_split(master)
print(f"  {master.shape[0]:,} rows × {master.shape[1]} cols "
      f"(NorwayNPD excluded; {master['source_db'].nunique()} sub-sources after split)")

print("Applying lithology classification …")
mapper = LithologyMapper()
master = mapper.apply(master)

# Coarser five-class scheme used for the stratified analysis:
#   Plutonic + Volcanic + Pyroclastic  -> Magmatic
#   Metamorphic                        -> Metamorphic
#   Clastic sedimentary                -> Clastic sedimentary
#   Carbonate sedimentary              -> Carbonate sedimentary
#   Evaporite                          -> Evaporite (small; kept separate)
# Evaporite and Unconsolidated are dropped from the stratified analysis (2026-05-20)
# because they qualify only for porosity / density targets and leave the
# mechanical columns of the heatmap empty. The four major classes (Magmatic,
# Clastic sedimentary, Carbonate sedimentary, Metamorphic) carry the full mechanical
# story; Evaporite + Unconsolidated counts are still reported in Data.
CLASS_MERGE = {
    "Plutonic":    "Magmatic",
    "Volcanic":    "Magmatic",
    "Pyroclastic": "Magmatic",
}
master["lithology_class_fine"] = master["lithology_class"]
master["lithology_class"] = master["lithology_class"].replace(CLASS_MERGE)

# Patch the mapper's class list so downstream code that iterates
# `mapper.classes` sees the merged scheme.
_FINE_TO_COARSE = {c: CLASS_MERGE.get(c, c) for c in mapper.classes}
mapper._classes = [c for c in dict.fromkeys(_FINE_TO_COARSE.values())]

classified = (master["lithology_class"] != "Unclassified").sum()
print(f"  Classified: {classified:,} / {len(master):,} ({100*classified/len(master):.1f}%)")
print(f"  Merged class scheme: {mapper.classes}")

registry = TargetRegistry(master, BASE / "config" / "targets.yaml")
qualified = registry.get_qualified()
analyzed  = registry.get_all_analyzed()

print(f"\nTargets: {len(analyzed)} analyzed, {len(qualified)} qualified")

RF = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: LITHOLOGY COVERAGE & MAPPING
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== STEP 1: Lithology mapping report ===")

cov = mapper.coverage_report(master)
cov.to_csv(TAB_DIR / "lithology_coverage.csv", index=False)

db_lith_mat = mapper.database_lithology_matrix(master)
db_lith_pct = mapper.database_lithology_proportions(master)
db_lith_mat.to_csv(TAB_DIR / "database_lithology_matrix.csv")
db_lith_pct.to_csv(TAB_DIR / "database_lithology_proportions.csv")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: CONFOUNDING ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== STEP 2: Database × lithology confounding ===")

mask = master["lithology_class"] != "Unclassified"
ct = pd.crosstab(master.loc[mask, "source_db"], master.loc[mask, "lithology_class"])
chi2, p_chi2, dof, _ = chi2_contingency(ct.values)
cv = cramers_v(ct.values)

print(f"  Cramér's V: {cv:.3f}  ({'strong' if cv > 0.3 else 'moderate' if cv > 0.15 else 'weak'})")

# Fig L1: Database × lithology composition
db_order = master["source_db"].value_counts().index.tolist()
pct = db_lith_pct.drop(columns=["Unclassified"], errors="ignore")
pct = pct.reindex([d for d in db_order if d in pct.index])

fig, ax = plt.subplots(figsize=(10, 5))
bottom = np.zeros(len(pct))
for cls in [c for c in CLASS_COLOURS if c in pct.columns and c != "Unclassified"]:
    vals = pct[cls].values if cls in pct.columns else np.zeros(len(pct))
    ax.barh(range(len(pct)), vals, left=bottom, color=CLASS_COLOURS[cls], label=cls,
            edgecolor="white", linewidth=0.5)
    bottom += vals
ax.set_yticks(range(len(pct)))
ax.set_yticklabels(pct.index)
ax.set_xlabel("Proportion (%)")
ax.set_title(f"Lithological composition by source database  (Cramér's V = {cv:.2f})")
ax.legend(loc="lower right", fontsize=7, ncol=2, framealpha=0.9)
ax.set_xlim(0, 105)
fig.tight_layout()
savefig(fig, "Fig_L1_database_lithology_composition", "Fig_L1_database_lithology_composition.png")

# Fig L2: Database × lithology heatmap
fig, ax = plt.subplots(figsize=(10, 4.5))
hm_data = db_lith_pct.drop(columns=["Unclassified"], errors="ignore")
hm_data = hm_data.reindex([d for d in db_order if d in hm_data.index])
im = ax.imshow(hm_data.values, aspect="auto", cmap=plt.cm.YlOrRd, vmin=0, vmax=70)
ax.set_xticks(range(hm_data.shape[1]))
ax.set_xticklabels(hm_data.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(hm_data.shape[0]))
ax.set_yticklabels(hm_data.index, fontsize=9)
for i in range(hm_data.shape[0]):
    for j in range(hm_data.shape[1]):
        val = hm_data.values[i, j]
        if val > 0.5:
            color = "white" if val > 35 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color=color)
plt.colorbar(im, ax=ax, label="Proportion (%)", shrink=0.8)
ax.set_title("Database × lithology class composition (%)")
fig.tight_layout()
savefig(fig, "Fig_L2_database_lithology_heatmap", "Fig_L2_database_lithology_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: LITHOLOGY-STRATIFIED MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== STEP 3: Lithology-stratified performance ===")

master_cls = master[master["lithology_class"] != "Unclassified"].copy()

_strat_csv = TAB_DIR / "lithology_stratified_performance.csv"
if _strat_csv.exists():
    print("  Loading cached results.")
    strat_df = pd.read_csv(_strat_csv)
else:
    strat_rows = []
    for target in analyzed:
        sub, preds = registry.build_target_subset(master_cls, target)
        if sub.empty or len(preds) < 2:
            continue

        for cls_name in mapper.classes:
            if cls_name in ("Unclassified", "Unconsolidated", "Evaporite"):
                continue
            cls_mask = sub["lithology_class"] == cls_name
            n_cls = int(cls_mask.sum())
            if n_cls < mapper.min_samples_per_class:
                continue

            X_cls = sub.loc[cls_mask, preds].copy()
            y_cls = sub.loc[cls_mask, target].values
            if target in LOG_TARGETS:
                pos = y_cls > 0
                X_cls, y_cls = X_cls[pos], y_cls[pos]
                y_cls = np.log10(y_cls)
                if len(y_cls) < mapper.min_samples_per_class:
                    continue
            if len(y_cls) > MAX_SAMPLES:
                idx = np.random.default_rng(0).choice(len(y_cls), MAX_SAMPLES, replace=False)
                idx.sort()
                X_cls, y_cls = X_cls.iloc[idx], y_cls[idx]

            r2s = []
            for seed in [42, 7]:
                kf = KFold(n_splits=min(5, len(y_cls) // 6 + 1), shuffle=True, random_state=seed)
                for tr, te in kf.split(X_cls):
                    if len(te) < 3:
                        continue
                    pipe = _pipe(deepcopy(RF))
                    try:
                        pipe.fit(X_cls.iloc[tr], y_cls[tr])
                        r2s.append(float(r2_score(y_cls[te], pipe.predict(X_cls.iloc[te]))))
                    except Exception:
                        pass

            if r2s:
                strat_rows.append({
                    "target":          target,
                    "lithology_class": cls_name,
                    "n_samples":       n_cls,
                    "r2_mean":         round(float(np.nanmean(r2s)), 3),
                    "r2_std":          round(float(np.nanstd(r2s)), 3),
                })

    strat_df = pd.DataFrame(strat_rows)
    strat_df.to_csv(_strat_csv, index=False)
    print(f"  Saved ({len(strat_df)} rows)")

# Fig L4: Lithology-stratified performance heatmap
if not strat_df.empty:
    pivot = strat_df.pivot_table(index="target", columns="lithology_class",
                                  values="r2_mean", aggfunc="first")
    pivot = pivot.reindex(pivot.mean(axis=1).sort_values(ascending=False).index)
    y_labels = [registry.get_display_label(t) for t in pivot.index]

    fig, ax = plt.subplots(figsize=(10, max(4, len(y_labels) * 0.45)))
    im = ax.imshow(pivot.values, aspect="auto", cmap=plt.cm.RdYlGn, vmin=-0.5, vmax=1.0)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(y_labels, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.2 or val > 0.8 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5, color=color)
    plt.colorbar(im, ax=ax, label="$R^2$ (Random CV within class)", shrink=0.8)
    ax.set_title("Predictive performance by lithology class (Random Forest, 5-fold CV)")
    fig.tight_layout()
    savefig(fig, "Fig_L4_lithology_stratified_heatmap", "Fig_L4_lithology_stratified_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: FEATURE IMPORTANCE BY LITHOLOGY — ALL TARGETS, ALL CLASSES
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== STEP 4: Feature importance by lithology (all targets × all classes) ===")

_fimp_csv = TAB_DIR / "feature_importance_by_lithology.csv"
if _fimp_csv.exists():
    print("  Loading cached results.")
    fimp_df = pd.read_csv(_fimp_csv)
else:
    fimp_rows = []
    for target in analyzed:
        sub, preds = registry.build_target_subset(master_cls, target)
        if sub.empty or len(preds) < 2:
            continue

        for cls_name in mapper.classes:
            if cls_name in ("Unclassified", "Unconsolidated", "Evaporite"):
                continue
            cls_mask = sub["lithology_class"] == cls_name
            n_cls = int(cls_mask.sum())
            if n_cls < mapper.min_samples_per_class:
                continue

            X_cls = sub.loc[cls_mask, preds].copy()
            y_cls = sub.loc[cls_mask, target].values
            if target in LOG_TARGETS:
                pos = y_cls > 0
                X_cls, y_cls = X_cls[pos], y_cls[pos]
                y_cls = np.log10(y_cls)
                if len(y_cls) < mapper.min_samples_per_class:
                    continue
            if len(y_cls) > MAX_SAMPLES:
                idx = np.random.default_rng(0).choice(len(y_cls), MAX_SAMPLES, replace=False)
                idx.sort()
                X_cls, y_cls = X_cls.iloc[idx], y_cls[idx]

            pipe = _pipe(deepcopy(RF))
            try:
                pipe.fit(X_cls, y_cls)
                imp = pipe.named_steps["mdl"].feature_importances_
                for i, feat in enumerate(preds):
                    fimp_rows.append({
                        "target":          target,
                        "lithology_class": cls_name,
                        "feature":         feat,
                        "feature_label":   FEATURE_LABELS.get(feat, feat),
                        "importance":      round(float(imp[i]), 4),
                    })
            except Exception:
                pass

    fimp_df = pd.DataFrame(fimp_rows)
    fimp_df.to_csv(_fimp_csv, index=False)
    print(f"  Saved ({len(fimp_df)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4b: PER-LITHOLOGY GREEDY FFS ACROSS MODEL FAMILIES
# ══════════════════════════════════════════════════════════════════════════════
#
# For every (lithology_class, target, model_family) triple, run greedy FFS
# with the same fast configurations used in run_pipeline.py (see ffs_core.py).
# Aggregate delta-R^2 across (target, model) within each class to produce the
# class-specific composite ranking used in Fig_L5.

print("\n=== STEP 4b: Per-lithology FFS (all families × all targets × all classes) ===")

from ffs_core import greedy_ffs, cap_for, FFS_FAMILIES_DEFAULT

_ffs_lith_csv = TAB_DIR / "ffs_by_lithology.csv"
_rel_lith_csv = TAB_DIR / "feature_relevance_by_lithology.csv"

# Smaller per-model caps for the per-class FFS (classes are smaller anyway):
LITH_FFS_CAP_OVERRIDE = {
    "Ridge": 4000, "PLS": 4000, "ExtraTrees": 4000, "RF": 4000,
    "XGBoost": 4000, "KNN": 4000,
    "GAM": 3000, "MLP": 3000, "SVR": 1500,
}

if _ffs_lith_csv.exists() and _rel_lith_csv.exists():
    print("  Loading cached per-lithology FFS results.")
    ffs_lith_df = pd.read_csv(_ffs_lith_csv)
    rel_lith_df = pd.read_csv(_rel_lith_csv)
else:
    rows = []
    for target in analyzed:
        sub_t, preds = registry.build_target_subset(master_cls, target)
        if sub_t.empty or len(preds) < 2:
            continue
        for cls_name in mapper.classes:
            if cls_name in ("Unclassified", "Unconsolidated", "Evaporite"):
                continue
            cls_mask = sub_t["lithology_class"] == cls_name
            n_cls = int(cls_mask.sum())
            if n_cls < mapper.min_samples_per_class:
                continue
            X_full = sub_t.loc[cls_mask, preds].copy().reset_index(drop=True)
            y_full = sub_t.loc[cls_mask, target].values
            if target in LOG_TARGETS:
                pos = y_full > 0
                X_full = X_full[pos].reset_index(drop=True)
                y_full = np.log10(y_full[pos])
                if len(y_full) < mapper.min_samples_per_class:
                    continue
            for fam in FFS_FAMILIES_DEFAULT:
                cap = cap_for(fam, LITH_FFS_CAP_OVERRIDE)
                print(f"  FFS [{fam:10s}] {cls_name:24s} {target:18s} "
                      f"n={len(y_full):,}, {len(preds)} cand")
                try:
                    steps = greedy_ffs(X_full, y_full, preds, fam,
                                        cap=cap, verbose=False)
                except Exception as exc:
                    print(f"    [FFS] {fam} failed ({cls_name}, {target}): {exc}")
                    continue
                for s in steps:
                    rows.append({
                        **s,
                        "feature_label":   FEATURE_LABELS.get(s["feature"],
                                                              s["feature"]),
                        "target":          target,
                        "model":           fam,
                        "lithology_class": cls_name,
                    })

    ffs_lith_df = pd.DataFrame(rows)
    ffs_lith_df.to_csv(_ffs_lith_csv, index=False)
    print(f"  Saved ffs_by_lithology.csv ({len(ffs_lith_df)} rows)")

    # Composite score per (class, feature): normalised cumulative delta-R^2
    rel_rows = []
    if not ffs_lith_df.empty:
        for cls_name in ffs_lith_df["lithology_class"].unique():
            sub_c = ffs_lith_df[ffs_lith_df["lithology_class"] == cls_name]
            agg = {}
            for _, r in sub_c.iterrows():
                d = float(r["r2_delta"])
                if not np.isfinite(d):
                    d = float(r["r2_cumulative"])
                d = max(d, 0.0)
                agg[r["feature"]] = agg.get(r["feature"], 0.0) + d
            tot = sum(agg.values()) or 1.0
            for feat, score in agg.items():
                rel_rows.append({
                    "lithology_class":    cls_name,
                    "feature":            feat,
                    "label":              FEATURE_LABELS.get(feat, feat),
                    "composite_score":    round(score / tot, 6),
                    "harmonization_risk": HARMONIZATION_RISK.get(feat, "moderate"),
                })
    rel_lith_df = pd.DataFrame(rel_rows)
    rel_lith_df.to_csv(_rel_lith_csv, index=False)
    print(f"  Saved feature_relevance_by_lithology.csv ({len(rel_lith_df)} rows)")


# ── Per-lithology feature ranking: aggregate across all targets ────────────────
# For each lithology class: compute mean importance of each feature across all targets

print("\n  Generating Fig_L5: Per-lithology feature rankings (multipanel) …")

if not fimp_df.empty:
    # Prefer classes that have FFS rows (new FFS-based composite); fall back
    # to classes present in the legacy RF-MDI table.
    if not rel_lith_df.empty:
        present_classes = set(rel_lith_df["lithology_class"].unique())
    else:
        present_classes = set(fimp_df["lithology_class"].unique())
    valid_classes = [c for c in mapper.classes
                     if c not in ("Unclassified", "Unconsolidated", "Evaporite")
                     and c in present_classes]

    # Global ranking (from v14_runner.py output) for comparison
    _rel_csv_global = _V14 / "tables" / "feature_relevance.csv"
    rel_global = pd.read_csv(_rel_csv_global) if _rel_csv_global.exists() else pd.DataFrame()

    n_classes = len(valid_classes)
    ncols = 2
    # ceiling division so that we always have enough panels (Global + classes)
    nrows = (n_classes + 1 + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 7, nrows * 4.2),
                              sharey=False)
    axes_flat = np.array(axes).flatten()
    TOP_N = 12  # top features per panel

    panel_idx = 0

    # Panel 0: Global ranking
    if not rel_global.empty and panel_idx < len(axes_flat):
        ax = axes_flat[panel_idx]
        top = rel_global.head(TOP_N)
        colors = [RISK_COL.get(r, RISK_COL["unknown"]) for r in top["harmonization_risk"]]
        ax.barh(top["label"][::-1], top["composite_score"][::-1],
                color=colors[::-1], edgecolor="white", lw=0.4)
        ax.set_xlabel("Composite importance")
        ax.set_title("Global (all lithologies)", fontsize=10, fontweight="bold",
                     color="#333333")
        ax.axvline(0, color="#555555", lw=0.6)
        patches = [mpatches.Patch(color=RISK_COL[r], label=r.capitalize())
                   for r in ["low", "moderate", "high"]]
        ax.legend(handles=patches, fontsize=6.5, loc="lower right")
        panel_idx += 1

    for cls_name in valid_classes:
        if panel_idx >= len(axes_flat):
            break
        ax = axes_flat[panel_idx]

        # Per-class composite ranking: prefer FFS-aggregated delta-R^2 when
        # available, fall back to mean RF MDI otherwise.
        cls_rel = (rel_lith_df[rel_lith_df["lithology_class"] == cls_name]
                   if not rel_lith_df.empty else pd.DataFrame())
        if not cls_rel.empty:
            cls_rel = cls_rel.sort_values("composite_score", ascending=False)
            top_feats  = cls_rel.head(TOP_N).set_index("feature")["composite_score"]
            feat_labels = cls_rel.head(TOP_N)["label"].tolist()
            xlabel = "Composite FFS importance"
        else:
            cls_fimp = fimp_df[fimp_df["lithology_class"] == cls_name]
            feat_mean = (cls_fimp.groupby("feature")["importance"]
                         .mean().sort_values(ascending=False))
            feat_lbl  = {row["feature"]: row["feature_label"]
                         for _, row in cls_fimp.drop_duplicates("feature").iterrows()}
            top_feats   = feat_mean.head(TOP_N)
            feat_labels = [feat_lbl.get(f, f) for f in top_feats.index]
            xlabel = r"Composite FFS $\Delta R^2$"
        feat_risk = {f: HARMONIZATION_RISK.get(f, "moderate") for f in top_feats.index}
        colors = [RISK_COL.get(feat_risk.get(f, "moderate"), RISK_COL["unknown"])
                  for f in top_feats.index]

        ax.barh(feat_labels[::-1], top_feats.values[::-1],
                color=colors[::-1], edgecolor="white", lw=0.4)
        ax.set_xlabel(xlabel)
        ax.set_title(cls_name, fontsize=10, fontweight="bold",
                     color=CLASS_COLOURS.get(cls_name, "#333333"))
        ax.axvline(0, color="#555555", lw=0.6)
        n_samples = int(fimp_df[fimp_df["lithology_class"] == cls_name]["target"].nunique())
        ax.text(0.97, 0.04, f"n targets: {n_samples}", transform=ax.transAxes,
                fontsize=7, ha="right", color="#666666")
        panel_idx += 1

    # Hide unused axes
    for ax in axes_flat[panel_idx:]:
        ax.set_visible(False)

    # Shared legend for risk colours
    risk_patches = [mpatches.Patch(color=RISK_COL[r], label=f"{r.capitalize()} harm. risk")
                    for r in ["low", "moderate", "high"]]
    fig.legend(handles=risk_patches, loc="lower center", ncol=3,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Feature relevance ranking: global and per lithology class\n"
                 r"(FFS $\Delta R^2$ aggregated across nine model families and all analyzed targets; "
                 "colour = harmonisation risk)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    savefig(fig, "Fig_L5_per_lithology_feature_rankings",
            "Fig_L5_per_lithology_feature_rankings.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: BEST-MODEL HEATMAP (target × lithology, with HP tuning)
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== STEP 5: Best-model heatmap (all target × lithology combinations) ===")

_bm_csv = TAB_DIR / "best_model_by_lithology.csv"
if _bm_csv.exists():
    print("  Loading cached results.")
    bm_df = pd.read_csv(_bm_csv)
else:
    bm_rows = []

    # Targets to analyze (all analyzed)
    targets_bm = analyzed

    # Scopes: Global (already computed in v14_runner) + each lithology class
    # Load global results if available
    _bm_global_csv = _V14 / "tables" / "best_model_global.csv"
    if _bm_global_csv.exists():
        bm_global = pd.read_csv(_bm_global_csv)
        bm_rows.extend(bm_global.to_dict("records"))
        print(f"  Loaded {len(bm_global)} global best-model rows")
    else:
        print("  Warning: global best-model CSV not found; run v14_runner.py first")

    # Per-lithology class analysis
    for cls_name in mapper.classes:
        if cls_name in ("Unclassified", "Unconsolidated", "Evaporite"):
            continue

        cls_mask_master = master_cls["lithology_class"] == cls_name
        n_cls_total = int(cls_mask_master.sum())
        if n_cls_total < 50:
            print(f"  [skip] {cls_name}: only {n_cls_total} classified samples")
            continue

        print(f"\n  Lithology: {cls_name} (n={n_cls_total:,})")

        for target in targets_bm:
            sub, preds = registry.build_target_subset(
                master_cls[cls_mask_master], target)
            if sub.empty or len(sub) < mapper.min_samples_per_class or len(preds) < 2:
                continue

            X = sub[preds].copy()
            y = sub[target].values
            if target in LOG_TARGETS:
                pos = y > 0
                X, y = X[pos], y[pos]
                y = np.log10(y)
            if len(y) > MAX_SAMPLES:
                idx = np.random.default_rng(0).choice(len(y), MAX_SAMPLES, replace=False)
                idx.sort()
                X, y = X.iloc[idx], y[idx]
            n = len(y)
            if n < mapper.min_samples_per_class:
                continue

            best_r2, best_model = -np.inf, "N/A"
            kf = KFold(n_splits=min(5, n // 6 + 1), shuffle=True, random_state=42)

            for fname, (mdl, param_grid) in MODEL_FAMILIES.items():
                mdl_cap = MODEL_SAMPLE_CAP.get(fname, MAX_SAMPLES)
                if len(y) > mdl_cap:
                    _idx = np.random.default_rng(1).choice(len(y), mdl_cap, replace=False)
                    _idx.sort()
                    Xm, ym = X.iloc[_idx], y[_idx]
                else:
                    Xm, ym = X, y
                kf_m = KFold(n_splits=min(5, len(ym) // 6 + 1), shuffle=True, random_state=42)
                r2s = []
                for tr, te in kf_m.split(Xm):
                    if len(te) < 3:
                        continue
                    try:
                        fitted = _best_pipe(mdl, param_grid, Xm.iloc[tr], ym[tr], cv=3)
                        r2s.append(float(r2_score(ym[te], fitted.predict(Xm.iloc[te]))))
                    except Exception:
                        r2s.append(float("nan"))
                mean_r2 = float(np.nanmean(r2s)) if r2s else -np.inf
                if mean_r2 > best_r2:
                    best_r2, best_model = mean_r2, fname

            if best_r2 > -np.inf:
                bm_rows.append({
                    "target":     target,
                    "lithology":  cls_name,
                    "best_model": best_model,
                    "best_r2":    round(best_r2, 3),
                    "n_samples":  n,
                })
                print(f"    {target}: {best_model} R²={best_r2:.3f} (n={n})")

    bm_df = pd.DataFrame(bm_rows)
    bm_df.to_csv(_bm_csv, index=False)
    print(f"\n  Saved best_model_by_lithology.csv ({len(bm_df)} rows)")


# ── Fig: Best-model heatmap ───────────────────────────────────────────────────

print("\n  Generating best-model heatmap figure …")

if not bm_df.empty:
    # Build pivot tables for R² values and model names
    r2_pivot  = bm_df.pivot_table(index="lithology", columns="target",
                                   values="best_r2",    aggfunc="first")
    mdl_pivot = bm_df.pivot_table(index="lithology", columns="target",
                                   values="best_model", aggfunc="first")

    # Row order: Global first, then lithology classes sorted by mean R²
    row_order = ["Global"] + [
        r for r in r2_pivot.index if r != "Global"
    ]
    row_order = [r for r in row_order if r in r2_pivot.index]
    r2_pivot  = r2_pivot.reindex(row_order)
    mdl_pivot = mdl_pivot.reindex(row_order)

    # Column order: by mean R² across lithologies (descending)
    col_order = r2_pivot.mean(axis=0, skipna=True).sort_values(ascending=False).index.tolist()
    r2_pivot  = r2_pivot[col_order]
    mdl_pivot = mdl_pivot[col_order]

    # Pretty column labels
    col_labels = [registry.get_display_label(t).split("(")[0].strip() for t in col_order]

    # Shorten model names for annotation (kept compact to fit cells)
    MODEL_SHORT = {
        "Ridge": "Ridge", "PLS": "PLS", "GAM": "GAM",
        "ExtraTrees": "ET", "RF": "RF", "XGBoost": "XGB",
        "SVR": "SVR", "KNN": "KNN", "MLP": "MLP",
    }

    # Make cells comfortably square-ish; minimum size large enough that
    # both R^2 value and model label remain readable
    fig_h = max(5.0, len(row_order) * 0.85)
    fig_w = max(13, len(col_order) * 1.05)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    r2_vals = r2_pivot.values.astype(float)
    im = ax.imshow(r2_vals, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=9)
    # Per-row sample-count annotations (max n across this row's targets)
    row_n = {}
    if "n_samples" in bm_df.columns:
        for r in row_order:
            sub_n = bm_df[bm_df["lithology"] == r]["n_samples"].dropna()
            if len(sub_n):
                row_n[r] = int(sub_n.max())
    yt_labels = [f"{r}  (n≤{row_n[r]:,})" if r in row_n else r
                 for r in row_order]
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(yt_labels, fontsize=10)

    # Separator between Global and per-lithology rows
    if "Global" in row_order and len(row_order) > 1:
        ax.axhline(0.5, color="black", lw=1.5, zorder=5)

    # Annotate cells: R² on top, model abbreviation below
    for i in range(len(row_order)):
        for j in range(len(col_order)):
            r2v  = r2_vals[i, j]
            mname = mdl_pivot.iloc[i, j] if not pd.isna(mdl_pivot.iloc[i, j]) else ""
            if np.isnan(r2v):
                continue
            txt_col = "white" if r2v > 0.75 or r2v < 0.15 else "black"
            mshort  = MODEL_SHORT.get(str(mname), str(mname)[:4])
            ax.text(j, i - 0.20, f"{r2v:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=txt_col)
            ax.text(j, i + 0.25, mshort,        ha="center", va="center",
                    fontsize=8,                  color=txt_col, style="italic")

    cbar = plt.colorbar(im, ax=ax, label="Best $R^2$ (5-fold CV, HP-tuned)", shrink=0.8)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    savefig(fig, "Fig_BM_best_model_heatmap", "Fig_BM_best_model_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: LaTeX TABLES
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== STEP 6: Generating LaTeX tables ===")


def _latex_escape(s: str) -> str:
    return (str(s)
            .replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            .replace("\u00b2", "$^2$").replace("\u00b3", "$^3$"))


# ── Table L1: Database × lithology matrix ────────────────────────────────────

def write_db_lith_table():
    ct_tbl = pd.crosstab(master["source_db"], master["lithology_class"])
    ct_tbl = ct_tbl.drop(columns=["Unclassified"], errors="ignore")
    db_order_tbl = master["source_db"].value_counts().index.tolist()
    ct_tbl = ct_tbl.reindex([d for d in db_order_tbl if d in ct_tbl.index])
    classes = list(ct_tbl.columns)

    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\caption{Database~$\times$~lithology class sample counts. "
        rf"Cram\'er's~$V = {cv:.2f}$ indicates "
        rf"{'strong' if cv > 0.3 else 'moderate'} confounding "
        r"between database identity and lithological composition.}",
        r"\label{tab:db-lithology-matrix}",
        r"\footnotesize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{l" + "r" * len(classes) + "r}",
        r"\toprule",
        "Database & " + " & ".join([_latex_escape(c) for c in classes]) + r" & Total \\",
        r"\midrule",
    ]
    for db in ct_tbl.index:
        row_vals = [str(int(ct_tbl.loc[db, c])) for c in classes]
        total    = int(ct_tbl.loc[db].sum())
        lines.append(f"{_latex_escape(db)} & " + " & ".join(row_vals) +
                     f" & {total:,}" + r" \\")
    totals = [str(int(ct_tbl[c].sum())) for c in classes]
    grand  = int(ct_tbl.values.sum())
    lines += [r"\midrule",
              r"\textbf{Total} & " + " & ".join(totals) + f" & {grand:,}" + r" \\",
              r"\bottomrule", r"\end{tabular}}", r"\end{table}"]

    path = TAB_DIR / "Table_L1_database_lithology_matrix.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    if MS_TAB.exists():
        shutil.copy2(path, MS_TAB / path.name)
    print(f"  Saved {path.name}")

write_db_lith_table()


# ── Table L3: Lithology-stratified performance ───────────────────────────────

def write_strat_table():
    if strat_df.empty:
        return
    pivot = strat_df.pivot_table(index="target", columns="lithology_class",
                                  values="r2_mean", aggfunc="first")
    classes = [c for c in pivot.columns if c != "Unclassified"]

    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\caption{Random Forest $R^2$ (5-fold CV) within individual lithology classes. "
        r"Empty cells indicate insufficient sample size ($n < 30$). "
        r"Within-class performance generally exceeds the pooled benchmark, "
        r"particularly for physically direct relationships.}",
        r"\label{tab:lithology-stratified}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "r" * len(classes) + "}",
        r"\toprule",
        "Target & " + " & ".join([_latex_escape(c) for c in classes]) + r" \\",
        r"\midrule",
    ]
    for target in pivot.index:
        tgt  = _latex_escape(registry.get_display_label(target))
        vals = []
        for cls_name in classes:
            v = pivot.loc[target, cls_name] if cls_name in pivot.columns else float("nan")
            vals.append(f"{v:.2f}" if not np.isnan(v) else "---")
        lines.append(f"{tgt} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    path = TAB_DIR / "Table_L3_lithology_stratified.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    if MS_TAB.exists():
        shutil.copy2(path, MS_TAB / path.name)
    print(f"  Saved {path.name}")

write_strat_table()


# ── Table: Best-model heatmap ─────────────────────────────────────────────────

def write_best_model_table():
    if bm_df.empty:
        return

    r2_pivot  = bm_df.pivot_table(index="lithology", columns="target",
                                   values="best_r2",    aggfunc="first")
    mdl_pivot = bm_df.pivot_table(index="lithology", columns="target",
                                   values="best_model", aggfunc="first")

    row_order = ["Global"] + [r for r in sorted(r2_pivot.index) if r != "Global"]
    row_order = [r for r in row_order if r in r2_pivot.index]
    r2_pivot  = r2_pivot.reindex(row_order)
    mdl_pivot = mdl_pivot.reindex(row_order)

    col_order = r2_pivot.mean(axis=0, skipna=True).sort_values(ascending=False).index.tolist()
    r2_pivot  = r2_pivot[col_order]
    mdl_pivot = mdl_pivot[col_order]

    col_labels = [registry.get_display_label(t).split("(")[0].strip() for t in col_order]
    MODEL_SHORT = {
        "Linear": "Lin", "Ridge": "Rdg", "Lasso": "Las", "ElasticNet": "EN",
        "ExtraTrees": "ET", "RF": "RF", "GBT": "GBT", "SVR": "SVR", "KNN": "KNN",
    }

    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\caption{Best predictive performance per target variable and lithology class. "
        r"Each cell reports the maximum mean cross-validated $R^2$ across all nine "
        r"model families (with hyperparameter tuning) and the corresponding model. "
        r"Empty cells indicate insufficient sample size ($n < 30$). "
        r"Row ``Global'' uses all lithologies pooled.}",
        r"\label{tab:best-model-heatmap}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{l" + "r" * len(col_order) + "}",
        r"\toprule",
        r"Lithology & " + " & ".join([_latex_escape(l) for l in col_labels]) + r" \\",
        r"\midrule",
    ]
    for row_name in row_order:
        if row_name == "Global":
            lines.append(r"\midrule")
        cells = []
        for t in col_order:
            r2v  = r2_pivot.loc[row_name, t] if t in r2_pivot.columns else float("nan")
            mname = mdl_pivot.loc[row_name, t] if t in mdl_pivot.columns else ""
            if pd.isna(r2v) or np.isnan(float(r2v) if not pd.isna(r2v) else float("nan")):
                cells.append("---")
            else:
                mshort = MODEL_SHORT.get(str(mname), str(mname)[:4])
                cells.append(rf"{float(r2v):.2f} \textit{{{mshort}}}")
        lines.append(f"{_latex_escape(row_name)} & " + " & ".join(cells) + r" \\")
        if row_name == "Global":
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\par\smallskip\footnotesize",
        r"Model abbreviations: Lin = Linear regression; Rdg = Ridge; Las = Lasso; "
        r"EN = Elastic Net; ET = Extra Trees; RF = Random Forest; GBT = Gradient Boosted Trees; "
        r"SVR = Support Vector Regression (RBF); KNN = $k$-Nearest Neighbours.",
        r"\end{table}",
    ]

    path = TAB_DIR / "table_best_model_heatmap.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    if MS_TAB.exists():
        shutil.copy2(path, MS_TAB / path.name)
    print(f"  Saved {path.name}")

write_best_model_table()


print("\n" + "=" * 70)
print("LITHOLOGY ANALYSIS COMPLETE")
print("=" * 70)
print(f"  Cramér's V: {cv:.3f}")
print(f"  Best-model rows: {len(bm_df)}")
print(f"  Stratified combos: {len(strat_df)}")

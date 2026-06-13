"""
run_pipeline.py — Main analysis pipeline.

Approach:
  - Merge of all source databases except NorwayNPD (excluded due to extreme size
    and limited target coverage that biased global statistics).
  - Greedy forward feature selection (FFS) per model family.
  - Aggregated, normalized delta-R^2 yields the parameter ranking.
  - Per-lithology stratified analysis is performed by lithology_analysis.py.

Model families:
  Ridge, PLS, GAM, ExtraTrees, RF, XGBoost, SVR, KNN, MLP

Outputs go to results/.  Figures are mirrored to manuscript/ for the manuscript.

Run:  python run_pipeline.py
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
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import os
from sklearn.impute import SimpleImputer, KNNImputer

IMPUTER_STRATEGY = os.environ.get("IMPUTER", "median").lower()
_OUT_DIR_NAME = os.environ.get("OUT_DIR_NAME", "results")

def _make_imputer():
    if IMPUTER_STRATEGY == "median":
        return SimpleImputer(strategy="median")
    return KNNImputer(n_neighbors=5)
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import SVR  # noqa: F401  (retained for reference; not used in main pipeline)
from xgboost import XGBRegressor


# ── Wrapper classes for non-standard sklearn estimators ───────────────────────

class PLSWrapper(BaseEstimator, RegressorMixin):
    """PLSRegression wrapper that returns 1-D predictions.
    Uses sklearn-style clone semantics: fitted attributes end with `_`."""
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        import numpy as np
        X = np.asarray(X)
        nc = min(self.n_components, X.shape[1], max(1, X.shape[0] - 1))
        self.pls_ = PLSRegression(n_components=max(1, nc))
        self.pls_.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        import numpy as np
        return self.pls_.predict(np.asarray(X)).ravel()


class GAMWrapper(BaseEstimator, RegressorMixin):
    """Generalised additive model via per-feature spline basis + Ridge.

    Mathematically a GAM with smoothness penalty: each feature is expanded
    into a cubic B-spline basis (no cross terms), and the basis coefficients
    are fitted with L2 regularisation. Scales linearly in the sample size,
    unlike the iterative back-fitting of pygam.
    """
    def __init__(self, alpha=1.0, n_knots=5, degree=3):
        self.alpha = alpha
        self.n_knots = n_knots
        self.degree = degree

    def fit(self, X, y):
        import numpy as np
        X = np.asarray(X, dtype=float)
        self.spline_ = SplineTransformer(
            n_knots=self.n_knots, degree=self.degree,
            knots="quantile", include_bias=False,
        )
        Z = self.spline_.fit_transform(X)
        self.ridge_ = Ridge(alpha=self.alpha)
        self.ridge_.fit(Z, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        import numpy as np
        Z = self.spline_.transform(np.asarray(X, dtype=float))
        return self.ridge_.predict(Z)


class SVRWrapper(BaseEstimator, RegressorMixin):
    """Kernel ridge regression with RBF kernel approximated by the Nystroem
    method. Functionally equivalent to RBF-kernel SVR / KRR but scales
    linearly in the sample size, so the full per-source-balanced data set
    can be used.
    """
    def __init__(self, alpha=1.0, gamma=0.1, n_components=200, random_state=42):
        self.alpha = alpha
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y):
        import numpy as np
        X = np.asarray(X, dtype=float)
        nc = min(self.n_components, X.shape[0])
        self.nystroem_ = Nystroem(
            kernel="rbf", gamma=self.gamma,
            n_components=nc, random_state=self.random_state,
        )
        Z = self.nystroem_.fit_transform(X)
        self.ridge_ = Ridge(alpha=self.alpha)
        self.ridge_.fit(Z, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        import numpy as np
        Z = self.nystroem_.transform(np.asarray(X, dtype=float))
        return self.ridge_.predict(Z)

warnings.filterwarnings("ignore")

BASE       = Path(__file__).parent
MASTER_CSV = BASE / "data" / "master_table.csv.gz"
OUT_DIR    = BASE / _OUT_DIR_NAME
MS_DIR     = BASE / "manuscript"
TAB_DIR    = MS_DIR / "paper_tables"
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

EXCLUDE_SOURCES = {"NorwayNPD"}   # exclude large NPD database

from variables import NUMERIC_FEATURES, FEATURE_LABELS, HARMONIZATION_RISK
from target_registry import TargetRegistry
from master_split import apply_source_split

# ── Load data & registry ──────────────────────────────────────────────────────

print("Loading master table …")
master = pd.read_csv(MASTER_CSV, low_memory=False)
print(f"  {master.shape[0]:,} rows × {master.shape[1]} cols (full)")
master = master[~master["source_db"].isin(EXCLUDE_SOURCES)].reset_index(drop=True)
print(f"  {master.shape[0]:,} rows × {master.shape[1]} cols (after excluding: {EXCLUDE_SOURCES})")
master = apply_source_split(master)
print(f"  source_db split into {master['source_db'].nunique()} sub-sources "
      f"(P3 and AMPEDEK split by primary reference; parent_db preserves the parent label)")

registry = TargetRegistry(master, BASE / "config" / "targets.yaml")
analyzed_targets  = registry.get_all_analyzed()
qualified_targets = registry.get_qualified()
partial_targets   = registry.get_partial()

print(f"\nTarget qualification summary:")
print(f"  Analyzed: {len(analyzed_targets)} — {analyzed_targets}")

qrpt = registry.qualification_report()
qrpt.to_csv(OUT_DIR / "tables" / "qualification_report.csv", index=False)

PRESENTATION_SUBSET = registry.get_presentation_subset(n=8)

# ── Model families with hyperparameter search spaces ─────────────────────────

MODEL_FAMILIES = {
    "Ridge":      (Ridge(), {"mdl__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}),
    "PLS":        (PLSWrapper(), {"mdl__n_components": [1, 2, 3, 5, 8]}),
    "GAM":        (GAMWrapper(), {
        "mdl__alpha":   [0.01, 0.1, 1.0, 10.0, 100.0],
        "mdl__n_knots": [4, 5, 7],
    }),
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
    "SVR":        (SVRWrapper(), {
        "mdl__alpha":        [0.01, 0.1, 1.0, 10.0],
        "mdl__gamma":        [0.01, 0.05, 0.1, 0.5],
        "mdl__n_components": [100, 200, 400],
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

N_SEEDS   = [42, 7]
N_ITER_HP = 15   # RandomizedSearchCV iterations for hyperparameter tuning

LOG_TARGETS  = {"permeability_m2"}   # targets modelled on log10 scale
MAX_SAMPLES  = None                   # no global cap
MAX_SAMPLES_PER_SOURCE = None         # per-source cap removed (2026-05-16): NorwayNPD is excluded
                                       # entirely and the 8000-row cap mostly hurt the larger merged
                                       # compendia (GlobalDB, P3) without correcting any imbalance
                                       # that 5-fold CV could not already absorb.
# All families scale linearly in sample size (Nystroem-RBF for SVR, spline+ridge for GAM).
MODEL_SAMPLE_CAP = {}


def _prepare_Xy(sub: pd.DataFrame, preds: list, target: str,
                model_name: str = "", rng: int = 0):
    """Return (X, y, n) with log10 for LOG_TARGETS, per-source and per-model caps."""
    df = sub.reset_index(drop=True)
    X = df[preds].copy()
    y = df[target].values

    if target in LOG_TARGETS:
        mask = y > 0
        df = df[mask].reset_index(drop=True)
        X = X[mask].reset_index(drop=True)
        y = np.log10(y[mask])

    # Per-source cap: prevent any single database from dominating.
    # Disabled when MAX_SAMPLES_PER_SOURCE is None (the current default).
    if MAX_SAMPLES_PER_SOURCE is not None and "source_db" in df.columns:
        rng_src = np.random.default_rng(rng + 99)
        keep = []
        for _, grp in df.groupby("source_db"):
            idx = grp.index.tolist()
            if len(idx) > MAX_SAMPLES_PER_SOURCE:
                idx = rng_src.choice(idx, MAX_SAMPLES_PER_SOURCE, replace=False).tolist()
            keep.extend(idx)
        keep = sorted(keep)
        X = X.iloc[keep].reset_index(drop=True)
        y = y[np.array(keep)]

    # Per-model cap (only SVR and GAM have caps; all others use full dataset)
    cap = MODEL_SAMPLE_CAP.get(model_name, None)
    if cap is not None and len(y) > cap:
        idx = np.random.default_rng(rng).choice(len(y), cap, replace=False)
        idx.sort()
        X, y = X.iloc[idx], y[idx]
    return X, y, len(y)


def _pipe(model):
    return Pipeline([
        ("imp",  _make_imputer()),
        ("sc",   StandardScaler()),
        ("mdl",  deepcopy(model)),
    ])


def _best_pipe(base_model, param_grid, X_tr, y_tr, cv=3):
    """Fit with RandomizedSearchCV inside training fold; return best estimator pipeline."""
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
    p = OUT_DIR / "figures" / f"{name}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {p.name}")
    if ms_name and MS_DIR.exists():
        dst = MS_DIR / ms_name
        try:
            shutil.copyfile(p, dst)
        except Exception as exc:
            print(f"  [fig] failed to mirror to {dst}: {exc}")


def _fmt_r2(v):
    try:
        f = float(v)
        return (rf"$\phantom{{-}}{f:.3f}$" if f >= 0 else rf"$-{abs(f):.3f}$")
    except Exception:
        return "---"


plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 9, "axes.labelsize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── STEP 1 — Fig01: Study design ─────────────────────────────────────────────

print("\n=== STEP 1: Study design figure ===")

def fig01_study_design():
    n_analyzed = len(analyzed_targets)
    n_qual     = len(qualified_targets)

    PHASE_FILL = {
        1: "#B8D0E8",
        2: "#9ECECE",
        3: "#F2DFA5",
        4: "#AACF97",
    }
    PHASE_EDGE = {
        1: "#4A7FA5",
        2: "#347A7A",
        3: "#A07830",
        4: "#3E7A2E",
    }
    PHASE_LABEL_COL = {
        1: "#2E5A7A",
        2: "#1F5A5A",
        3: "#7A5A18",
        4: "#2E5A20",
    }
    PHASE_NAMES = {
        1: "① Data & harmonisation",
        2: "② Target qualification",
        3: "③ Model benchmarking",
        4: "④ Relevance & lithology",
    }

    # Standardised predictor pool size after harmonisation (matches yaml)
    n_predictor_pool = 19  # saturated variants excluded
    # Count source databases dynamically from master (post NorwayNPD exclusion)
    n_source_dbs = master["source_db"].nunique()
    n_primary_refs = master["primary_reference"].nunique() if "primary_reference" in master.columns else None

    # Each box: (x_centre, title, description, phase)
    boxes = [
        (1.6,  f"{n_source_dbs} source\ndatabases",
                f"P³ ({master.loc[master.source_db=='P3','primary_reference'].nunique()} refs),\n"
                f"AMPEDEK ({master.loc[master.source_db=='AMPEDEK','primary_reference'].nunique()} refs),\n"
                f"GlobalDB ({master.loc[master.source_db=='GlobalDB','primary_reference'].nunique()} refs),\n"
                f"MidGerman, Weydt2020,\nUSGS, Cornwall,\nand {n_source_dbs - 7}+ smaller sources",
                1),
        (4.6,  "Harmonised\nmaster table",
                f"{len(master):,} rows.\n{n_primary_refs} primary\nreferences resolved.",
                1),
        (7.6,  "Predictor\nqualification",
                f"{n_predictor_pool} standardised\npredictor variables\nretained after data-quality\nscreening.",
                2),
        (10.6, "Target\nqualification",
                f"{n_qual} of {n_predictor_pool} variables\nqualify as targets\n(rules R1, R2, R3).",
                2),
        (13.6, "Per-family FFS\n& best-model HP tuning",
                "Greedy forward selection\nplus nested-CV best-model\ncomparison for every target\nwith each of nine families.",
                3),
        (16.6, "Aggregated ranking\n& best-model heatmap",
                "Global ranking,\nper-lithology stratification,\nbest $R^2$ per cell.",
                4),
    ]

    box_w = 2.65
    box_h = 2.20
    band_h = 0.50

    phase_spans = {}
    for i, (x, _t, _d, ph) in enumerate(boxes):
        xl = x - box_w / 2
        xr = x + box_w / 2
        if ph not in phase_spans:
            phase_spans[ph] = [xl, xr]
        else:
            phase_spans[ph][0] = min(phase_spans[ph][0], xl)
            phase_spans[ph][1] = max(phase_spans[ph][1], xr)

    fig_w = boxes[-1][0] + box_w / 2 + 0.5
    fig_h = 5.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    box_y = 1.5
    band_y_bot = box_y + box_h / 2 + 0.18

    for ph, (xl, xr) in phase_spans.items():
        band = mpatches.FancyBboxPatch(
            (xl, band_y_bot), xr - xl, band_h,
            boxstyle="round,pad=0.03", linewidth=0.9,
            edgecolor=PHASE_EDGE[ph], facecolor=PHASE_FILL[ph], alpha=0.55, zorder=1,
        )
        ax.add_patch(band)
        ax.text((xl + xr) / 2, band_y_bot + band_h / 2, PHASE_NAMES[ph],
                ha="center", va="center", fontsize=12, fontweight="bold",
                color=PHASE_LABEL_COL[ph])

    for x, title, desc, ph in boxes:
        rect = mpatches.FancyBboxPatch(
            (x - box_w / 2, box_y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.07", linewidth=1.5,
            edgecolor=PHASE_EDGE[ph], facecolor=PHASE_FILL[ph], zorder=2,
        )
        ax.add_patch(rect)
        # Title
        ax.text(x, box_y + box_h / 2 - 0.38, title, ha="center", va="center",
                fontsize=12, fontweight="bold", color="#111111",
                multialignment="center", linespacing=1.2)
        # Body
        ax.text(x, box_y - 0.22, desc, ha="center", va="center",
                fontsize=10, color="#222222",
                multialignment="center", linespacing=1.25)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0]   + box_w / 2
        x2 = boxes[i+1][0] - box_w / 2
        ax.annotate("", xy=(x2, box_y), xytext=(x1, box_y),
                    arrowprops=dict(arrowstyle="-|>", color="#444444",
                                     lw=1.8, mutation_scale=18),
                    zorder=3)

    ax.axhline(box_y - box_h / 2 - 0.06, color="#cccccc", lw=0.6,
               xmin=0.02, xmax=0.98, zorder=0)
    fig.tight_layout(pad=0.3)
    savefig(fig, "Fig01_study_design", "Fig01_study_design.png")

fig01_study_design()

# ── STEP 2 — Model-family comparison (with hyperparameter tuning) ─────────────

print("\n=== STEP 2: Model-family comparison (9 families, with HP tuning) ===")

_fam_csv = OUT_DIR / "tables" / "model_family_comparison.csv"
if _fam_csv.exists():
    print("  Loading cached results.")
    fam_df = pd.read_csv(_fam_csv)
else:
    fam_rows = []
    for target in analyzed_targets:
        sub, preds = registry.build_target_subset(master, target)
        if sub.empty:
            print(f"  [skip] {target}: no data")
            continue

        for fname, (mdl, param_grid) in MODEL_FAMILIES.items():
            X, y, n = _prepare_Xy(sub, preds, target, model_name=fname)
            if fname == list(MODEL_FAMILIES.keys())[0]:
                print(f"  {target}: n={n:,}, predictors={len(preds)}")
            r2s, top_feats = [], []
            for seed in N_SEEDS:
                kf = KFold(n_splits=5, shuffle=True, random_state=seed)
                for tr, te in kf.split(X):
                    try:
                        fitted = _best_pipe(mdl, param_grid, X.iloc[tr], y[tr], cv=3)
                        r2s.append(float(r2_score(y[te], fitted.predict(X.iloc[te]))))
                        m = fitted.named_steps["mdl"]
                        if hasattr(m, "feature_importances_"):
                            top_feats.append(preds[int(np.argmax(m.feature_importances_))])
                        elif hasattr(m, "coef_"):
                            c = m.coef_
                            top_feats.append(preds[int(np.argmax(np.abs(c)))])
                    except Exception:
                        r2s.append(float("nan"))

            top = max(set(top_feats), key=top_feats.count) if top_feats else "N/A"
            r2_mean = float(np.nanmean(r2s))
            r2_std  = float(np.nanstd(r2s))
            fam_rows.append({
                "target":        target,
                "model":         fname,
                "r2_mean":       round(r2_mean, 3),
                "r2_std":        round(r2_std,  3),
                "top_pred":      top,
                "top_pred_lbl":  FEATURE_LABELS.get(top, top),
            })
            print(f"    {fname}: R²={r2_mean:.3f} ± {r2_std:.3f}")

    fam_df = pd.DataFrame(fam_rows)
    fam_df.to_csv(_fam_csv, index=False)
    print("  Saved model_family_comparison.csv")

# ── STEP 3 — Random CV performance (no LODO) ─────────────────────────────────

print("\n=== STEP 3: Random CV performance ===")

_val_csv = OUT_DIR / "tables" / "validation_regime.csv"
if _val_csv.exists():
    print("  Loading cached results.")
    val_df = pd.read_csv(_val_csv)
else:
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    val_rows = []

    for target in analyzed_targets:
        sub, preds = registry.build_target_subset(master, target)
        if sub.empty:
            continue
        X, y, _ = _prepare_Xy(sub, preds, target, model_name="RF")

        r2_rand = []
        for seed in [42, 7, 13]:
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            for tr, te in kf.split(X):
                pipe = _pipe(deepcopy(rf_model))
                try:
                    pipe.fit(X.iloc[tr], y[tr])
                    r2_rand.append(float(r2_score(y[te], pipe.predict(X.iloc[te]))))
                except Exception:
                    r2_rand.append(float("nan"))

        r2_mean = float(np.nanmean(r2_rand))
        r2_p05  = float(np.nanpercentile([v for v in r2_rand if not np.isnan(v)] or [0], 5))
        r2_p95  = float(np.nanpercentile([v for v in r2_rand if not np.isnan(v)] or [0], 95))

        val_rows.append({
            "target":        target,
            "r2_random_cv":  round(r2_mean, 3),
            "r2_rand_p05":   round(r2_p05,  3),
            "r2_rand_p95":   round(r2_p95,  3),
        })
        print(f"  {target}: rand_cv={r2_mean:.3f} [{r2_p05:.3f}, {r2_p95:.3f}]")

    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(_val_csv, index=False)
    print("  Saved validation_regime.csv")

# ── STEP 4 — Feature relevance (placeholder; overwritten by FFS-based ranking in Step 6) ──

print("\n=== STEP 4: Feature relevance (FFS-based; computed after Step 6) ===")

_rel_csv = OUT_DIR / "tables" / "feature_relevance.csv"
if _rel_csv.exists():
    print("  Loading cached results.")
    rel_df = pd.read_csv(_rel_csv)
else:
    # Bootstrap with RF MDI — will be replaced after FFS (Step 6) completes
    imp_agg = {f: [] for f in NUMERIC_FEATURES}
    rf_fast = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1)

    for target in analyzed_targets:
        sub, preds = registry.build_target_subset(master, target)
        if sub.empty:
            continue
        X, y, _ = _prepare_Xy(sub, preds, target, model_name="RF")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for tr, te in kf.split(X):
            pipe = _pipe(deepcopy(rf_fast))
            try:
                pipe.fit(X.iloc[tr], y[tr])
                imp = pipe.named_steps["mdl"].feature_importances_
                for i, p in enumerate(preds):
                    imp_agg[p].append(float(imp[i]))
            except Exception:
                pass

    rel_rows = []
    for feat in NUMERIC_FEATURES:
        vals = imp_agg[feat]
        if vals:
            rel_rows.append({
                "feature":            feat,
                "label":              FEATURE_LABELS.get(feat, feat),
                "composite_score":    float(np.mean(vals)),
                "std_importance":     float(np.std(vals)),
                "harmonization_risk": HARMONIZATION_RISK.get(feat, "moderate"),
            })
    rel_df = pd.DataFrame(rel_rows).sort_values("composite_score", ascending=False)
    rel_df.to_csv(_rel_csv, index=False)
    print(f"  Bootstrap ranking (RF MDI) — will be replaced by FFS-based ranking in Step 6")

# ── STEP 5 — Global best-model per target (with HP tuning) ───────────────────

print("\n=== STEP 5: Global best-model per target (with HP tuning) ===")

_best_csv = OUT_DIR / "tables" / "best_model_global.csv"
if _best_csv.exists():
    print("  Loading cached results.")
    best_global_df = pd.read_csv(_best_csv)
else:
    best_rows = []
    for target in analyzed_targets:
        sub, preds = registry.build_target_subset(master, target)
        if sub.empty:
            continue

        best_r2, best_model = -np.inf, "N/A"
        for fname, (mdl, param_grid) in MODEL_FAMILIES.items():
            X, y, n_eff = _prepare_Xy(sub, preds, target, model_name=fname)
            if fname == list(MODEL_FAMILIES.keys())[0]:
                print(f"  {target}: n={n_eff:,}")
            r2s = []
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for tr, te in kf.split(X):
                try:
                    fitted = _best_pipe(mdl, param_grid, X.iloc[tr], y[tr], cv=3)
                    r2s.append(float(r2_score(y[te], fitted.predict(X.iloc[te]))))
                except Exception:
                    r2s.append(float("nan"))
            mean_r2 = float(np.nanmean(r2s))
            if mean_r2 > best_r2:
                best_r2, best_model = mean_r2, fname
            print(f"    {fname}: R²={mean_r2:.3f}")

        best_rows.append({
            "target":      target,
            "lithology":   "Global",
            "best_model":  best_model,
            "best_r2":     round(best_r2, 3),
        })
        print(f"  -> Best: {best_model}  R2={best_r2:.3f}")

    best_global_df = pd.DataFrame(best_rows)
    best_global_df.to_csv(_best_csv, index=False)
    print("  Saved best_model_global.csv")

# ── STEP 6 — True Forward Feature Selection ───────────────────────────────────

print("\n=== STEP 6: Forward Feature Selection (greedy, RF base model, all samples) ===")

_ffs_csv = OUT_DIR / "tables" / "ffs_selection_order.csv"


# ── FFS PER MODEL FAMILY ─────────────────────────────────────────────────────
#
# FFS uses a *fast, fixed* configuration per family (no inner HP search) so
# that running it across nine families remains tractable. Hyperparameter
# tuning still happens in the best-model comparison (Step 5). The resulting
# delta-R^2 values are aggregated over both targets and model families to
# build the composite ranking (Eq. 1 in the paper).

FFS_N_SPLITS = 3            # 3-fold CV inside FFS (vs. 5 in main analysis)
FFS_DELTA_STOP = 0.005      # early stop threshold

def _ffs_model(name: str):
    """Return a fast, fixed-config base estimator for the FFS loop."""
    if name == "Ridge":
        return Ridge(alpha=1.0)
    if name == "PLS":
        return PLSWrapper(n_components=3)
    if name == "GAM":
        return GAMWrapper(alpha=1.0, n_knots=5)
    if name == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=50, n_jobs=-1, random_state=42)
    if name == "RF":
        return RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
    if name == "XGBoost":
        return XGBRegressor(n_estimators=80, max_depth=5, learning_rate=0.1,
                            n_jobs=-1, random_state=42, verbosity=0,
                            objective="reg:squarederror")
    if name == "SVR":
        return SVRWrapper(alpha=1.0, gamma=0.1, n_components=200)
    if name == "KNN":
        return KNeighborsRegressor(n_neighbors=5, weights="distance")
    if name == "MLP":
        return MLPRegressor(hidden_layer_sizes=(64,), alpha=0.01,
                            max_iter=200, random_state=42)
    raise KeyError(name)


def _ffs_cap(X_df: pd.DataFrame, y: np.ndarray, model_name: str, seed: int = 42):
    """No sample cap inside FFS — every family runs on the full
    per-source-balanced dataset."""
    return X_df, y


def _ffs_r2_single(X_df: pd.DataFrame, y: np.ndarray, feats: list[str],
                   model_name: str, n_splits: int = FFS_N_SPLITS,
                   seed: int = 42) -> float:
    """Return mean CV R² for the given model family on `feats`."""
    base = _ffs_model(model_name)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    Xc, yc = _ffs_cap(X_df[feats], y, model_name, seed)
    for tr, te in kf.split(Xc):
        pipe = Pipeline([
            ("imp", _make_imputer()),
            ("sc",  StandardScaler()),
            ("mdl", deepcopy(base)),
        ])
        try:
            pipe.fit(Xc.iloc[tr], yc[tr])
            scores.append(float(r2_score(yc[te], pipe.predict(Xc.iloc[te]))))
        except Exception:
            scores.append(float("nan"))
    return float(np.nanmean(scores))


def forward_feature_selection(X_df: pd.DataFrame, y: np.ndarray,
                              candidates: list[str],
                              model_name: str) -> list[dict]:
    """Greedy FFS for one (target, model_name) pair."""
    selected: list[str] = []
    remaining = list(candidates)
    steps = []
    baseline = 0.0

    for step in range(1, len(candidates) + 1):
        best_feat, best_r2 = None, -np.inf
        for feat in remaining:
            trial = selected + [feat]
            r2 = _ffs_r2_single(X_df, y, trial, model_name)
            if r2 > best_r2:
                best_r2, best_feat = r2, feat
        if best_feat is None:
            break
        delta = best_r2 - baseline
        selected.append(best_feat)
        remaining.remove(best_feat)
        steps.append({
            "step":          step,
            "feature":       best_feat,
            "feature_label": FEATURE_LABELS.get(best_feat, best_feat),
            "r2_cumulative": round(best_r2, 4),
            "r2_delta":      round(delta, 4),
        })
        print(f"    step {step:2d}: +{best_feat} -> R2={best_r2:.4f} "
              f"(delta={delta:+.4f})")
        baseline = best_r2
        if step >= 3 and delta < FFS_DELTA_STOP:
            print(f"    [FFS] early stop: delta<{FFS_DELTA_STOP}")
            break
    return steps


if _ffs_csv.exists():
    print("  Loading cached FFS results.")
    ffs_df = pd.read_csv(_ffs_csv)
else:
    ffs_rows = []
    fams = list(MODEL_FAMILIES.keys())
    for target in analyzed_targets:
        sub, preds = registry.build_target_subset(master, target)
        if sub.empty:
            continue
        for fam in fams:
            X_raw, y, n = _prepare_Xy(sub, preds, target, model_name=fam)
            print(f"  FFS [{fam:10s}] {target}: n={len(y):,}, "
                  f"{len(preds)} candidates")
            try:
                steps = forward_feature_selection(X_raw, y, preds, fam)
            except Exception as exc:
                print(f"    [FFS] {fam} failed for {target}: {exc}")
                continue
            for row in steps:
                row["target"] = target
                row["model"]  = fam
            ffs_rows.extend(steps)

    ffs_df = pd.DataFrame(ffs_rows)
    ffs_df.to_csv(_ffs_csv, index=False)
    print(f"  Saved ffs_selection_order.csv ({len(ffs_df)} rows)")

# ── Feature ranking from FFS delta-R² (replaces RF MDI ranking) ──────────────

print("\n  [FFS] Computing feature ranking from accumulated delta-R² ...")
delta_agg: dict[str, float] = {}
for _, row in ffs_df.iterrows():
    feat  = row["feature"]
    delta = float(row["r2_delta"])
    if not np.isfinite(delta):
        delta = float(row["r2_cumulative"])  # fallback for legacy inf entries
    delta = max(delta, 0.0)
    delta_agg[feat] = delta_agg.get(feat, 0.0) + delta

total = sum(delta_agg.values()) or 1.0
rel_rows_ffs = []
for feat, score in delta_agg.items():
    rel_rows_ffs.append({
        "feature":            feat,
        "label":              FEATURE_LABELS.get(feat, feat),
        "composite_score":    round(score / total, 6),
        "std_importance":     0.0,
        "harmonization_risk": HARMONIZATION_RISK.get(feat, "moderate"),
    })
rel_df = (pd.DataFrame(rel_rows_ffs)
          .sort_values("composite_score", ascending=False)
          .reset_index(drop=True))
rel_df.to_csv(OUT_DIR / "tables" / "feature_relevance.csv", index=False)
print(f"  Feature ranking updated from FFS delta-R²; top-5: {rel_df['label'].head(5).tolist()}")

# ── FFS visualisation ─────────────────────────────────────────────────────────
#
# ffs_df now contains one row per (target, model, step). For the per-target
# curve figure and selection-order heatmap we pick, per target, the model
# family whose FFS path reaches the highest final cumulative R² — i.e. the
# canonical FFS path for that target. The aggregated relevance score
# (Eq. 1 in the paper) above already uses *all* (model, target) pairs.

if "model" in ffs_df.columns:
    _final_per_pair = (ffs_df.sort_values("step")
                       .groupby(["target", "model"])["r2_cumulative"]
                       .last()
                       .reset_index())
    _best_per_target = (_final_per_pair
                        .sort_values("r2_cumulative", ascending=False)
                        .groupby("target")
                        .first()
                        .reset_index()[["target", "model"]])
    _bm = dict(zip(_best_per_target["target"], _best_per_target["model"]))
    ffs_df_canonical = (ffs_df
                        .merge(_best_per_target, on=["target", "model"],
                               how="inner")
                        .reset_index(drop=True))
else:
    ffs_df_canonical = ffs_df
    _bm = {}


def fig_ffs_curves():
    """Panel of cumulative-R² curves with filled area, one subplot per target.

    Each panel shows the canonical FFS path (the model family whose path
    reaches the highest final cumulative R² for that target).
    """
    targets_ffs = ffs_df_canonical["target"].unique().tolist()
    n     = len(targets_ffs)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    # Distinct palette — one saturated colour per panel, cycling
    PALETTE = [
        "#2563A8", "#C03A2B", "#27A96A", "#8E44AD",
        "#D07F1A", "#1A8C9A", "#B5451B", "#2E7D32",
        "#6A3DBB", "#A0522D", "#1565C0", "#2D7D4F",
        "#7B3F00", "#2F5E8E", "#6D4C8E", "#3A6B35",
        "#B03060", "#1A5276",
    ]
    TGREY = "#444444"

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.6, nrows * 3.5),
                             sharey=False, sharex=False)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (ax, target) in enumerate(zip(axes_flat, targets_ffs)):
        col = PALETTE[idx % len(PALETTE)]
        sub = ffs_df_canonical[ffs_df_canonical["target"] == target].sort_values("step")
        if sub.empty:
            ax.set_visible(False)
            continue

        steps  = sub["step"].values.astype(int)
        r2s    = sub["r2_cumulative"].values.astype(float)
        labels = sub["feature_label"].values
        r2_mono = np.maximum.accumulate(r2s)
        max_step = int(max(steps))

        # Extend to step edges for fill_between (step-function look)
        x_ext = np.repeat(steps, 2)
        x_ext = np.concatenate([[steps[0]], x_ext, [max_step]])
        y_ext = np.repeat(r2_mono, 2)
        y_ext = np.concatenate([[0], y_ext, [r2_mono[-1]]])
        # Actually use smooth curve with fill under it
        ax.fill_between(steps, 0, r2_mono,
                        color=col, alpha=0.18, zorder=1,
                        interpolate=True)

        # Line
        ax.plot(steps, r2_mono, "-", color=col, lw=2.2, zorder=2,
                solid_capstyle="round")

        # Dots
        ax.scatter(steps, r2_mono, color=col, edgecolors="white",
                   s=65, lw=1.2, zorder=3)

        # Horizontal guide at final R²
        ax.axhline(r2_mono[-1], color=col, lw=0.7, ls="--",
                   alpha=0.4, zorder=0)

        # Numbered feature legend
        legend_lines = [f"{s}. {lbl}" for s, lbl in zip(steps, labels)]
        legend_text  = "\n".join(legend_lines)
        r2_end = r2_mono[-1]
        y_pos  = 0.04 if r2_end > 0.55 else 0.97
        va_pos = "bottom" if r2_end > 0.55 else "top"
        ax.text(0.97, y_pos, legend_text,
                transform=ax.transAxes,
                ha="right", va=va_pos,
                fontsize=5.8, color=TGREY, linespacing=1.5,
                family="monospace",
                bbox=dict(facecolor="white", alpha=0.80,
                          edgecolor=col, linewidth=0.6,
                          boxstyle="round,pad=0.35"))

        # Title coloured to match
        target_lbl = FEATURE_LABELS.get(target, target)
        ax.set_title(target_lbl, fontsize=8.2, fontweight="bold",
                     pad=5, color=col)

        # Axes formatting — y-label on every panel
        ax.set_ylim(0, 1.0)
        ax.set_xlim(0.5, max_step + 0.5)
        ax.set_xticks(steps)
        ax.set_xticklabels([str(s) for s in steps], fontsize=7.5)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.tick_params(axis="y", labelsize=7.5)
        ax.set_xlabel("Selection step", fontsize=8, color=TGREY)
        ax.set_ylabel("Cumulative CV $R^2$", fontsize=8, color=TGREY)

        ax.grid(axis="y", color="#e0e0e0", lw=0.7, zorder=0)
        ax.set_facecolor("#fafafa")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")

    for ax in axes_flat[len(targets_ffs):]:
        ax.set_visible(False)

    fig.tight_layout(h_pad=2.5, w_pad=2.0)
    savefig(fig, "Fig_FFS_curves", "Fig_FFS_curves.png")


SHORT_LABEL = {
    "porosity_pct":         r"$\phi$",
    "bulk_density_gcm3":    r"$\rho_b$",
    "grain_density_gcm3":   r"$\rho_g$",
    "permeability_m2":      r"$k$",
    "tc_dry_WmK":           r"$\lambda$",
    "td_dry_1e6m2s":        r"$\alpha$",
    "cp_JkgK":              r"$c_p$",
    "vp_dry_ms":            r"$V_p$",
    "vs_dry_ms":            r"$V_s$",
    "ucs_MPa":              r"UCS",
    "E_static_GPa":         r"$E_s$",
    "E_dyn_GPa":            r"$E_d$",
    "poisson_ratio":        r"$\nu$",
    "tensile_MPa":          r"$\sigma_t$",
    "friction_angle_deg":   r"$\varphi$",
    "cohesion_MPa":         r"$c$",
    "mag_susc_1e3si":       r"$\chi$",
    "resistivity_dry_Ohm":  r"$\rho_e$",
    "depth_m":              r"$z$",
}

LABEL_TO_SHORT = {FEATURE_LABELS[k]: v for k, v in SHORT_LABEL.items() if k in FEATURE_LABELS}


def fig_ffs_heatmap():
    """Heatmap: targets (rows) × selection step (cols), cell = feature label.

    Uses the canonical FFS path per target (same as fig_ffs_curves)."""
    targets_ffs = ffs_df_canonical["target"].unique().tolist()
    max_step = int(ffs_df_canonical["step"].max())

    # Build label matrix
    label_mat = pd.DataFrame("", index=targets_ffs, columns=range(1, max_step + 1))
    r2_mat    = pd.DataFrame(np.nan, index=targets_ffs, columns=range(1, max_step + 1))
    for _, row in ffs_df_canonical.iterrows():
        t, s = row["target"], int(row["step"])
        label_mat.loc[t, s] = LABEL_TO_SHORT.get(row["feature_label"], row["feature_label"])
        r2_mat.loc[t, s]    = row["r2_cumulative"]

    # Color by cumulative R² at each step
    fig_h = max(7, len(targets_ffs) * 0.55)
    fig_w = max(11, max_step * 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = cm.get_cmap("YlGn")
    for ri, target in enumerate(targets_ffs):
        for ci, step in enumerate(range(1, max_step + 1)):
            val = r2_mat.loc[target, step]
            lbl = label_mat.loc[target, step]
            color = cmap(float(val)) if not np.isnan(val) else "#f0f0f0"
            rect = plt.Rectangle([ci, ri], 1, 1, facecolor=color,
                                  edgecolor="white", linewidth=0.5)
            ax.add_patch(rect)
            if lbl:
                ax.text(ci + 0.5, ri + 0.5, lbl, ha="center", va="center",
                        fontsize=11, wrap=False)

    ax.set_xlim(0, max_step)
    ax.set_ylim(0, len(targets_ffs))
    ax.set_xticks([s - 0.5 for s in range(1, max_step + 1)])
    ax.set_xticklabels([str(s) for s in range(1, max_step + 1)], fontsize=9)
    ax.set_yticks([i + 0.5 for i in range(len(targets_ffs))])
    ax.set_yticklabels(
        [FEATURE_LABELS.get(t, t) for t in targets_ffs], fontsize=9
    )
    ax.set_xlabel("Selection step", fontsize=10)
    ax.set_title(
        "Forward Feature Selection — feature added per step\n"
        "(colour = cumulative CV R²; canonical model family per target)",
        fontsize=10)

    sm = cm.ScalarMappable(cmap=cmap,
                           norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Cumulative CV R²", shrink=0.6, pad=0.02)
    fig.tight_layout()
    savefig(fig, "Fig_FFS_heatmap", "Fig_FFS_heatmap.png")


if not ffs_df.empty:
    fig_ffs_curves()
    fig_ffs_heatmap()

print("\n=== v14_runner.py complete ===")
print(f"  Analyzed targets: {len(analyzed_targets)}")
print("  Next step: python lithology_analysis.py && python v14_figures_tables.py")

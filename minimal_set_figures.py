"""
minimal_set_figures.py — Figures and table of the minimal-set analysis.

Reads the outputs of minimal_set_search.py and minimal_set_accuracy.py and
draws the score of all procedure combinations (paper Fig. 10) and the
accuracy of the properties the minimal set does not deliver (Fig. 11), and
writes Supplementary Table SD as LaTeX.

Output: results/figures/Fig_minimal_set_search.png
        results/figures/Fig_BM_minimal_set.png
        results/tables/tableSG_minimal_set.tex
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from variables import FEATURE_LABELS

BASE = Path(__file__).parent
OUT_DIR = BASE / os.environ.get("OUT_DIR_NAME", "results")
TAB_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
MS_DIR = BASE / "manuscript"

MEASURED = ["porosity_pct", "bulk_density_gcm3", "grain_density_gcm3",
            "tc_dry_WmK", "td_dry_1e6m2s", "vp_dry_ms", "vs_dry_ms"]
COMPUTED = ["poisson_ratio", "E_dyn_GPa", "cp_JkgK"]
HEATMAP_COLS = ["E_static_GPa", "ucs_MPa", "cohesion_MPa", "permeability_m2",
                "tensile_MPa", "friction_angle_deg", "mag_susc_1e3si"]
ROWS = ["Global", "Carbonate sedimentary", "Clastic sedimentary", "Magmatic", "Metamorphic"]
SHORT = {"ExtraTrees": "ET", "XGBoost": "XGB"}


def name(col):
    """Display name with symbol, e.g. 'Porosity $\\phi$'."""
    return FEATURE_LABELS.get(col, col).split(" (")[0]


def symbol(col):
    m = re.search(r"\$[^$]+\$", name(col))
    return m.group(0) if m else name(col)


def save(fig, filename):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [fig] {path}")
    if MS_DIR.exists():
        shutil.copyfile(path, MS_DIR / filename)


def fig_search(search, checks):
    best = search.sort_values("score", ascending=False).groupby("n_procedures").head(1)
    best = best.sort_values("n_procedures")
    n = np.r_[0, best["n_procedures"].to_numpy()]
    s = np.r_[0.0, best["score"].to_numpy()]
    k = int(n[np.argmax(s - n / 9)])

    # range of the best combinations over the CV seeds
    seeds = checks[checks["variant"].str.startswith("seed")]
    rng = seeds.groupby("combination")["score"].agg(["min", "max"]).reindex(best["combination"])
    yerr = [s[1:] - rng["min"].to_numpy(), rng["max"].to_numpy() - s[1:]]

    blue, brown, grey, light = "#004E8A", "#A84B16", "#B7C7D8", "#E2C4B0"
    style = {"font.family": "Arial", "font.size": 12,
             "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(7.6, 4.3))
        jitter = np.random.default_rng(0).uniform(-0.16, 0.16, len(search))
        ax.scatter(search["n_procedures"] + jitter, search["score"], s=14, color=grey,
                   edgecolor="none", zorder=2,
                   label=f"all {len(search)} combinations, one dot each")
        ax.plot([0, 9], [0, 1], ls="--", color="#8a8a8a", lw=1.3, zorder=1,
                label="effort line ($n$/9)")
        for ni, si in zip(n[1:-1], s[1:-1]):
            if ni != k:
                ax.plot([ni, ni], [ni / 9, si], color=light, lw=2.2, zorder=3,
                        solid_capstyle="butt")
        ax.plot([k, k], [k / 9, s[k]], color=brown, lw=3.2, zorder=4, solid_capstyle="butt",
                label="gap score $-$ effort, largest = knee point")
        ax.errorbar(n[1:], s[1:], yerr=yerr, fmt="none", ecolor=blue, elinewidth=1.2,
                    capsize=3, zorder=5)
        ax.plot(n, s, color=blue, lw=2.2, marker="o", ms=6, zorder=5,
                label="best combination per size (CV seed range)")

        prev = []
        for row in best.itertuples():
            procs = row.combination.split(" + ")
            added = [p for p in procs if p not in prev][0]
            prev = procs
            label = added if row.n_procedures == 1 else "+ " + added
            if row.n_procedures <= k:
                ax.text(row.n_procedures - 0.12, row.score + 0.02, f"{label}  {row.score:.2f}",
                        color=brown, fontsize=11, fontweight="bold", va="bottom", ha="right")
            else:
                ax.text(row.n_procedures, row.score + 0.022, label, color="#444444",
                        fontsize=9.5, rotation=40, ha="left", va="bottom",
                        rotation_mode="anchor")
        ax.scatter([k], [s[k]], s=130, color=brown, edgecolor="white", lw=1.2, zorder=6)
        ax.set_xlim(-1.0, 9.6)
        ax.set_ylim(0, 1.12)
        ax.set_xticks(range(10))
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_xlabel("number of measurement procedures $n$")
        ax.set_ylabel("score = mean $R^2$ over 17 properties")
        ax.grid(axis="y", color="#E6E6E6", lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right", frameon=False, fontsize=10)
        fig.tight_layout()
    save(fig, "Fig_minimal_set_search.png")
    print(f"  knee at {k} procedures: {best['combination'].iloc[k - 1]} ({s[k]:.3f})")


def fig_accuracy(acc):
    rows = [r for r in ROWS if r in set(acc["lithology"])]

    def pivot(col):
        return acc.pivot_table(index="lithology", columns="target", values=col,
                               aggfunc="first").reindex(index=rows, columns=HEATMAP_COLS)

    r2, model, share = pivot("best_r2"), pivot("best_model"), pivot("retained_share")
    n_max = acc.groupby("lithology")["n"].max()

    with plt.rc_context({"font.size": 10, "xtick.labelsize": 8, "ytick.labelsize": 8}):
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        im = ax.imshow(r2.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(HEATMAP_COLS)))
        ax.set_xticklabels([symbol(c) for c in HEATMAP_COLS], rotation=40, ha="right",
                           fontsize=9)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r if r == "Global" else f"{r}  (n≤{n_max[r]:,})" for r in rows],
                           fontsize=10)
        ax.axhline(0.5, color="black", lw=1.5, zorder=5)
        for i in range(len(rows)):
            for j in range(len(HEATMAP_COLS)):
                v = r2.iloc[i, j]
                if not np.isfinite(v):
                    continue
                c = "white" if v > 0.75 or v < 0.15 else "black"
                ax.text(j, i - 0.27, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        fontweight="bold", color=c)
                if np.isfinite(share.iloc[i, j]):
                    ax.text(j, i + 0.02, f"{100 * share.iloc[i, j]:.0f} %", ha="center",
                            va="center", fontsize=8.5, color=c)
                m = str(model.iloc[i, j])
                ax.text(j, i + 0.30, SHORT.get(m, m), ha="center", va="center", fontsize=8,
                        color=c, style="italic")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        cb = plt.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label("Best $R^2$ (5-fold CV, HP-tuned), minimal-set predictors")
        cb.ax.tick_params(labelsize=8)
        fig.tight_layout()
    save(fig, "Fig_BM_minimal_set.png")


def table_sd(acc):
    g = acc[acc["lithology"] == "Global"].copy()
    # 0 = predicted, 1 = computed exactly, 2 = measured by the minimal set
    g["block"] = g["target"].map(lambda t: 2 if t in MEASURED else 1 if t in COMPUTED else 0)
    g = g.sort_values(["block", "best_r2"], ascending=[True, False])
    lines = []
    for block, part in g.groupby("block"):
        if block == 1:
            lines.append(r"\midrule")
        status = ["", "computed", "measured"][block]
        for r in part.itertuples():
            share = 100 * r.best_r2 / r.full_r2
            lines.append(f"{name(r.target)} & {status} & {r.best_model} & "
                         f"{r.n:,} & {r.best_r2:.2f} & {r.full_r2:.2f} & {share:.0f} \\\\"
                         .replace(",", "{,}"))
    caption = (
        r"Supplementary Table~SD. Accuracy with the minimal measurement set on the "
        r"global dataset. For each target the hyperparameter-tuned best-model comparison "
        r"of Figure~\ref{fig:best-model-heatmap} is re-run with the predictor pool "
        r"restricted to the seven properties that density, TCS, and ultrasound measure "
        r"($\phi$, $\rho_b$, $\rho_g$, $\lambda$, $\alpha$, $V_P$, $V_S$), minus the "
        r"target itself where it is one of them. Rows, folds, model families, and "
        r"hyperparameter search are those of the full run, so both $R^2$ columns refer "
        r"to the same samples. The upper block lists the seven properties the minimal "
        r"set does not deliver. For the lower block, which the minimal set measures or "
        r"computes exactly ($\nu$ from $V_P$ and $V_S$, $E_d$ from $V_P$, $V_S$, and "
        r"$\rho_b$, $c_p = \lambda/(\rho_b\,\alpha)$), the $R^2$ shows how well the other "
        r"predictors would replace it, with $E_d$ predicted from $\phi$, $\rho_g$, "
        r"$\lambda$, and $\alpha$ only (circularity rule). The share is the minimal-set "
        r"$R^2$ as a percentage of the full-set value."
    )
    tex = "\n".join([
        r"\begin{table}[tbp]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{tab:supp-minimal-set}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Target & Minimal set & Family & $n$ & $R^2$ minimal & $R^2$ full & Share (\%) \\",
        r"\midrule",
        *lines,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    out = TAB_DIR / "tableSG_minimal_set.tex"
    out.write_text(tex, encoding="utf-8")
    print(f"  [tab] {out}")
    if (MS_DIR / "paper_tables").exists():
        shutil.copyfile(out, MS_DIR / "paper_tables" / out.name)


def main():
    search = pd.read_csv(TAB_DIR / "minimal_set_search.csv")
    checks = pd.read_csv(TAB_DIR / "minimal_set_search_checks.csv")
    acc = pd.read_csv(TAB_DIR / "minimal_set_accuracy.csv")
    fig_search(search, checks)
    fig_accuracy(acc)
    table_sd(acc)


if __name__ == "__main__":
    main()

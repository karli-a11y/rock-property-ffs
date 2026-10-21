# Predictor Importance Across Lithologies — Analysis Code and Data

This repository contains the harmonised master table and the complete analysis
pipeline for the paper

> Holler, J. K., Seib, L., Pham, H., Sass, I.:
> *Predictor Importance Across Lithologies: A Multi-Model Forward Feature
> Selection Analysis of Rock-Property Databases.*
> Computers & Geosciences (under review).

The pipeline runs greedy forward feature selection (FFS) for 17 rock-property
targets with nine regression model families, aggregates the incremental R²
gains into a model-family-robust predictor ranking, repeats the analysis in an
iterated-exclusion variant and within five lithology classes, validates
out-of-source transfer with a leave-one-source-out (LOSO) experiment, and
searches all combinations of nine laboratory measurement procedures for the
minimal measurement set.

## Repository layout

```
├── data/
│   ├── master_table.csv.gz    # harmonised master table (read directly by all scripts)
│   └── SOURCES.md             # provenance of the 30 source databases
├── config/                    # variable, target, and lithology configuration
│   ├── targets.yaml           # target qualification rules and circularity exclusions
│   └── lithology.yaml         # regex rules of the lithology classifier
├── adapters/                  # per-database ingestion adapters (documentation of
│                              # how the original sources were harmonised; the
│                              # original raw files are NOT redistributed here)
├── variables.py               # the 19 standardised variables, labels, risk classes
├── ffs_core.py                # greedy FFS core routine
├── target_registry.py         # target qualification (rules R1–R3) and predictor pools
├── lithology_mapper.py        # regex-based lithology classifier
├── master_split.py            # compendium split into primary-reference sub-sources
├── paper_supplement/          # machine-readable data behind the supplementary tables
└── *.py                       # analysis stages, see below
```

## Setup

Python ≥ 3.10. Install the dependencies:

```bash
pip install -r requirements.txt
```

No further setup is needed. All scripts read `data/master_table.csv.gz`
directly (pandas decompresses on the fly) and write their outputs to
`results/`.

## Data

`data/master_table.csv.gz` is the full harmonised merge of all 30 source
databases (one row per specimen/measurement record). Relevant columns:

* the 19 standardised variables listed in `variables.py`,
* `source_db`, `parent_db`, `primary_reference`, `subsource` — row-level
  provenance,
* `top_class` — a lithology class from the original harmonisation. The
  analyses assign the five lithology classes of the paper with
  `lithology_mapper.py` instead.

The NorwayNPD porosity–permeability database is contained in the table but is
excluded from all analyses at load time (it would dominate the multi-target
merge with two-variable rows). After this exclusion and cross-compendium
deduplication the analyses operate on 28,604 effective rows from 267 primary
references. Corrections to the master table are listed at the end of
[`data/SOURCES.md`](data/SOURCES.md).

The original source databases are **not** redistributed in this repository.
Their provenance, licences, and DOIs are listed in
[`data/SOURCES.md`](data/SOURCES.md), and the `adapters/` directory documents
exactly how each source was mapped into the master table.

## Running the analysis

Run the stages in the order below. Stages 2–4 are compute-intensive (hours on
a desktop machine) because they re-run the full FFS pipeline many times.
Stage 8 takes about an hour (`N_WORKERS` sets the number of parallel
processes, default 8), stages 9 and 10 several hours each.

| # | Script | What it does | Main outputs (in `results/`) |
|---|--------|--------------|------------------------------|
| 1 | `python run_pipeline.py` | Target qualification, per-family greedy FFS, nested-CV model-family comparison, global best-model table | `tables/qualification_report.csv`, `tables/ffs_selection_order.csv`, `tables/model_family_comparison.csv`, `tables/best_model_global.csv`, `tables/feature_relevance.csv` |
| 2 | `python recursive_ffs.py` | Per-target iterated-exclusion FFS (each target removes its own leading predictor each round); gains summed over all loops give the relevance ranking S̃ⱼ | `tables/feature_relevance_recursive_steps.csv`, `tables/feature_relevance_recursive.csv`, `tables/feature_relevance_recursive_mean.csv` |
| 3 | `python recursive_ffs_per_lithology.py` | The same per-target iterated-exclusion within each of the five lithology classes of `lithology_mapper.py` | `tables/feature_relevance_recursive_per_lithology.csv`, `..._mean.csv`, `..._steps.csv` |
| 4 | `python lithology_analysis.py` | Per-class FFS, per-class best-model search, best-model heatmap | `lithology/tables/best_model_by_lithology.csv`, `lithology/tables/*.csv`, `lithology/figures/*.png` |
| 5 | `python loso_validation.py` | Leave-one-source-out transfer experiment over the 149 eligible sub-sources | `tables/loso_validation.csv` |
| 6 | `python holdout_validation.py` | Per-target hold-out check with the canonical best family | `tables/holdout_validation.csv` |
| 7 | `python bootstrap_ranking.py` and `python bootstrap_ranking.py --per-lithology` | Bootstrap stability intervals of the global ranking and of the five class rankings | `tables/feature_relevance_bootstrap.csv`, `tables/feature_relevance_bootstrap_per_lithology.csv` |
| 8 | `python minimal_set_search.py` | Scores all 511 combinations of the nine laboratory procedures over the 17 targets (measured or exactly computed = 1, otherwise CV R²) and finds the knee of score vs. effort; repeats the leading combinations with CV seeds 7 and 13 and without the Poisson ratio as predictor | `tables/minimal_set_search.csv`, `tables/minimal_set_search_checks.csv` |
| 9 | `python minimal_set_accuracy.py` | Tuned best-model comparison of stages 1 and 4 with only the seven properties of the minimal set as predictors, global and per lithology class | `tables/minimal_set_accuracy.csv` |
| 10 | `python best_model_uncertainty.py` | All nine families re-run in every best-model cell, with all predictors and with the minimal set, keeping fold R² and out-of-fold predictions: standard deviation of R² over the folds, RMSE, 90 % prediction interval, runner-up family | `tables/best_model_uncertainty.csv` |
| 11 | `python minimal_set_figures.py` | Figures and table of the minimal-set analysis | `figures/Fig_minimal_set_search.png`, `figures/Fig_BM_minimal_set.png`, `tables/tableSG_minimal_set.tex` |
| 12 | `python loso_family_sensitivity.py` | LOSO of stage 5 repeated with the runner-up family for the targets whose winning family is a near tie (UCS, Poisson ratio, grain density, thermal conductivity, permeability) | `tables/loso_family_sensitivity.csv` |

Optional imputation-sensitivity run (`paper_supplement/imputer_sensitivity.csv`): re-run
stage 1 with KNN imputation into a second output directory, then compare:

```bash
OUT_DIR_NAME=results_knn IMPUTER=knn python run_pipeline.py
python compare_imputers.py
```

(On Windows PowerShell: `$env:OUT_DIR_NAME="results_knn"; $env:IMPUTER="knn"; python run_pipeline.py`.)

All random seeds are fixed (seed 42 for primary runs, seeds 7 and 13 for
stability checks), so repeated runs reproduce the published numbers up to
hardware-level nondeterminism in the tree ensembles. The published numbers
were produced with Python 3.11, scikit-learn 1.8.0, XGBoost 2.1.1, pygam
0.12.0, pandas 2.3.1 and NumPy 2.2.6. Other library versions can shift single
cells slightly, mostly where XGBoost is the best family.

## Mapping outputs to the paper

| Paper item | Produced from |
|---|---|
| Table 1 (variables) | `variables.py`, `results/tables/qualification_report.csv` |
| Fig. 3 (FFS curves) | `results/tables/ffs_selection_order.csv` |
| Figs. 4–5 (relevance rankings, global + per class) | `feature_relevance_recursive_mean.csv`, `..._per_lithology_mean.csv` |
| Fig. 6 (iterated exclusion) | `feature_relevance_recursive.csv`, `..._per_lithology.csv` |
| Fig. 7 (cross-family rank spread) | `results/tables/ffs_selection_order.csv` |
| Fig. 8 (model-family comparison) | `results/tables/model_family_comparison.csv` |
| Fig. 9 / Table 3 (best-model heatmap, class priorities) | `best_model_global.csv`, `lithology/tables/best_model_by_lithology.csv`, `results/tables/best_model_uncertainty.csv` |
| Table 2, Figs. 10–11, Supplementary Table SD (minimal measurement set) | `results/tables/minimal_set_search*.csv`, `results/tables/minimal_set_accuracy.csv`, `results/tables/best_model_uncertainty.csv` |
| Fig. 12 (LOSO transfer gaps) | `results/tables/loso_validation.csv`, `results/tables/loso_family_sensitivity.csv` |
| Supplementary Tables SA–SD | stages 1–12 as listed above |

## License

Code: MIT License (see `LICENSE`). The master table aggregates published
open data; the per-source licences of the original databases are listed in
`data/SOURCES.md` and apply to the respective subsets.

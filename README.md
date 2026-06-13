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
iterated-exclusion variant and within five lithology classes, and validates
out-of-source transfer with a leave-one-source-out (LOSO) experiment.

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
* `top_class` — the lithology class assigned by `lithology_mapper.py`.

The NorwayNPD porosity–permeability database is contained in the table but is
excluded from all analyses at load time (it would dominate the multi-target
merge with two-variable rows). After this exclusion and cross-compendium
deduplication the analyses operate on 28,612 effective rows from 267 primary
references.

The original source databases are **not** redistributed in this repository.
Their provenance, licences, and DOIs are listed in
[`data/SOURCES.md`](data/SOURCES.md), and the `adapters/` directory documents
exactly how each source was mapped into the master table.

## Running the analysis

Run the stages in the order below. Stages 2–4 are compute-intensive (hours on
a desktop machine) because they re-run the full FFS pipeline many times.

| # | Script | What it does | Main outputs (in `results/`) |
|---|--------|--------------|------------------------------|
| 1 | `python run_pipeline.py` | Target qualification, per-family greedy FFS, nested-CV model-family comparison, global best-model table | `tables/qualification_report.csv`, `tables/ffs_selection_order.csv`, `tables/model_family_comparison.csv`, `tables/best_model_global.csv`, `tables/feature_relevance.csv` |
| 2 | `python recursive_ffs.py` | Iterated-exclusion FFS on the global pool (relevance ranking S̃ⱼ) | `tables/feature_relevance_recursive.csv`, `tables/feature_relevance_recursive_mean.csv` |
| 3 | `python recursive_ffs_per_lithology.py` | Iterated-exclusion FFS within each of the five lithology classes | `tables/feature_relevance_recursive_per_lithology.csv`, `..._mean.csv` |
| 4 | `python lithology_analysis.py` | Per-class FFS, per-class best-model search, best-model heatmap | `lithology/tables/best_model_by_lithology.csv`, `lithology/tables/*.csv`, `lithology/figures/*.png` |
| 5 | `python loso_validation.py` | Leave-one-source-out transfer experiment over the 149 eligible sub-sources | `tables/loso_validation.csv` |
| 6 | `python holdout_validation.py` | Per-target hold-out check with the canonical best family | `tables/holdout_validation.csv` |
| 7 | `python bootstrap_ranking.py` | Bootstrap stability intervals of the global ranking | `tables/feature_relevance_bootstrap.csv` |
| 8 | `python minimal_set_check.py` | Non-destructive minimal-set check (six-property backbone) | `tables/minimal_set_check.csv` |

Optional imputation-sensitivity run (paper Supplementary Table SF): re-run
stage 1 with KNN imputation into a second output directory, then compare:

```bash
OUT_DIR_NAME=results_knn IMPUTER=knn python run_pipeline.py
python compare_imputers.py
```

(On Windows PowerShell: `$env:OUT_DIR_NAME="results_knn"; $env:IMPUTER="knn"; python run_pipeline.py`.)

All random seeds are fixed (seed 42 for primary runs, seed 7 for
fold-stability checks), so repeated runs reproduce the published numbers up to
hardware-level nondeterminism in the tree ensembles.

## Mapping outputs to the paper

| Paper item | Produced from |
|---|---|
| Table 1 (variables) | `variables.py`, `results/tables/qualification_report.csv` |
| Fig. 3 (FFS curves) | `results/tables/ffs_selection_order.csv` |
| Figs. 4–5 (relevance rankings, global + per class) | `feature_relevance_recursive_mean.csv`, `..._per_lithology_mean.csv` |
| Fig. 6 (iterated exclusion) | `feature_relevance_recursive.csv`, `..._per_lithology.csv` |
| Fig. 7 (cross-family rank spread) | `results/tables/ffs_selection_order.csv` |
| Fig. 8 (model-family comparison) | `results/tables/model_family_comparison.csv` |
| Fig. 9 / Table 2 (best-model heatmap) | `best_model_global.csv`, `lithology/tables/best_model_by_lithology.csv` |
| Fig. 10 (LOSO transfer gaps) | `results/tables/loso_validation.csv` |
| Supplementary Tables SA–SG | stages 1–8 as listed above |

## License

Code: MIT License (see `LICENSE`). The master table aggregates published
open data; the per-source licences of the original databases are listed in
`data/SOURCES.md` and apply to the respective subsets.

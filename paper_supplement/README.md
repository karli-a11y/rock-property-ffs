# Paper supplement — machine-readable data

Data behind supplementary tables that are referenced in the paper
but not printed in the manuscript (kept compact for the journal).
Regenerate with `python _export_supp_data.py`.

- `source_lithology_counts.csv` — Per-source sample counts and lithology-class counts of the 30 analysed source databases (was Supplementary Table SA). (31 data rows)
- `model_family_performance.csv` — Model-family predictive performance (R^2 mean +/- SD) for all 17 targets and nine model families (was Supplementary Table SC). (17 data rows)
- `modal_predictor_by_target.csv` — Dominant predictor (modal top-1 across the nine families) and stability indicator for all 17 targets (was Supplementary Table SD). (17 data rows)
- `imputer_sensitivity.csv` — Imputation sensitivity: best-model CV R^2 under median vs KNN imputation (was Supplementary Table SH). (15 data rows)
- `physical_bounds.csv` — Physical-bound screening ranges applied during harmonisation (was Supplementary Table SJ). (13 data rows)
- `best_model_map.csv` — Best CV R^2 and winning model per target and lithology class (was Supplementary Table SK; the figure stays in the paper). (5 data rows)
- `ffs_selection_order.csv` — Forward-feature-selection entry order for all 17 targets (was Supplementary Tables SE1+SE2). (138 data rows)
- `SE_loso.csv` — full per-(held-out source, target) LOSO transferability (summary stays as Supplementary Table SC).

Minimal measurement set (written by `minimal_set_search.py` and `minimal_set_accuracy.py`):

- `minimal_set_search.csv` — Score of all 511 combinations of the nine laboratory procedures and the R^2 of every property in each combination (1 = measured or computed exactly), CV seed 42 (Fig. 10). (511 data rows)
- `minimal_set_search_checks.csv` — The leading combinations scored again with CV seeds 7 and 13 and without the Poisson ratio as predictor. (76 data rows)
- `minimal_set_accuracy.csv` — Tuned best-model R^2 with the seven minimal-set properties as the only predictors, per target and lithology class, with the full-predictor ceiling and the retained share (Fig. 11, Supplementary Table SD). (79 data rows)

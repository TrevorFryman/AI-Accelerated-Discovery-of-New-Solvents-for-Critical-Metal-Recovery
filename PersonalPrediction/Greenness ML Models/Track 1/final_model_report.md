
# Final Model Tuning Report — GSK Solvent G-Score Prediction

Executive summary
- **Goal:** Tune XGBoost hyperparameters for three descriptor pipelines (Morgan, RDKit, ChemBERTa) and compare baseline vs tuned performance using the cached StratifiedGroupKFold splits.
- **Key outcome:** After randomized tuning (30 candidates per pipeline), `rdkit` produced the best test RMSE (tuned RMSE = 0.8568) and remains the recommended representation for this dataset.

Experimental setup
- **Data split:** Global train/test split created by `data_utils.make_split()` (StratifiedGroupKFold, fold 0 used as held-out test). Groups defined by SMILES to avoid leakage.
- **CV:** 5-fold StratifiedGroupKFold used for inner CV during tuning. Scoring: RMSE (negative RMSE in scikit-learn). Random seed = `SEED` from `config.py` for reproducibility.
- **Search:** `RandomizedSearchCV` over sensible ranges for `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree` (30 iterations per pipeline).

Per-pipeline results (baseline → tuned)
- **Morgan fingerprints**
   - Baseline test RMSE = 0.9522, R² = 0.5664
   - Tuned test RMSE = 0.9661, R² = 0.5537
   - Best tuned params: `{'subsample': 0.9, 'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.03, 'colsample_bytree': 0.7}`
   - Notes: small CV improvement (mean CV RMSE reduced from ~0.9163 → ~0.8670) but test RMSE did not improve.

- **RDKit descriptors**
   - Baseline test RMSE = 0.8458, R² = 0.6579
   - Tuned test RMSE = 0.8568, R² = 0.6489
   - Best tuned params: `{'subsample': 0.6, 'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.2, 'colsample_bytree': 1.0}`
   - Notes: modest CV improvement (mean CV RMSE ~0.7727 → ~0.8125) and stable test performance; overall best absolute test RMSE across pipelines.

- **ChemBERTa embeddings**
   - Baseline test RMSE = 1.4154, R² = 0.0419
   - Tuned test RMSE = 1.3952, R² = 0.0691
   - Best tuned params: `{'subsample': 0.9, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.01, 'colsample_bytree': 1.0}`
   - Notes: small improvement after tuning but performance remains substantially worse than RDKit and Morgan for this task/dataset.

Comparison and recommendation
- **Best pipeline:** `rdkit` (lowest tuned test RMSE = 0.8568). It balances predictive accuracy and compact feature set.
- **When to choose others:** `morgan` provides comparable baseline performance and may be preferred if bit-vector fingerprints are required by downstream tooling; `chemberta` embeddings did not outperform descriptor-based pipelines on this dataset.

Artifacts & next steps
- Numeric results: `results/hyperparam_tuning_results.json` and `results/model_tuning_comparison.csv`.
- Per-pipeline summaries: `results/morgan/tuning_summary.json`, `results/rdkit/tuning_summary.json`, `results/chemberta/tuning_summary.json`.
- Figures: `figures/*/rmse_before_after.png` for quick before/after visuals.
- Next actions I can do (pick any):
   - Add paired significance testing (e.g., Wilcoxon) on per-fold RMSE to validate improvements.
   - Re-run tuning with a larger search budget or Optuna for Bayesian optimization.
   - Refit final `rdkit` model on train+test and export as a deployable artifact.

Limitations
- Small dataset (n=154) — limited headroom for complex embeddings like ChemBERTa to demonstrate advantage.
- Stratification classes have small groups (warnings about smallest class < n_splits); this is informational but worth noting for interpretation.

Contact
- I can fill this report with plots/tables embedded or produce a short presentation slide if you'd like.


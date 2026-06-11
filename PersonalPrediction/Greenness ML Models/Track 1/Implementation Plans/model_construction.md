# GSK Solvent G-Score — Implementation Plan (Updated)

## Phase 1 Status: COMPLETE ✓
All EDA scripts, figures, and EDA_Report.md are verified working.

---

## Phase 2: Feature Generation + Phase 3: ML Training

### Folder Reorganization

Move existing 7 EDA figures → `figures/EDA/`
Create model subdirectories: `figures/{morgan,rdkit,chemberta}/`
Create result subdirectories: `results/{EDA,morgan,rdkit,chemberta}/`

---

## Proposed Changes

### Shared Utilities

#### [NEW] `data_utils.py`
Guarantees **identical** train/test splits across all three pipelines:
- `load_dataset()` — loads and cleans the CSV
- `assign_groups()` — maps duplicate SMILES to the same group ID
- `make_split()` — StratifiedGroupKFold(n_splits=5, fold=0): ~80/20 split cached to `data/train_test_split.json`
- `get_cv_splitter()` — returns StratifiedGroupKFold for inner CV

#### [NEW] `plot_utils.py`
Shared dark-theme plotting utilities:
- `plot_parity()` — actual vs predicted (train + test overlay)
- `plot_residuals()` — residuals vs predicted + histogram
- `plot_feature_importance_xgb()` — XGBoost gain-based bar chart
- `plot_shap_importance()` — SHAP-based summary
- `plot_oof()` — out-of-fold predictions
- `plot_cv_fold_scores()` — per-fold metric bar chart
- `plot_learning_curves()` — sklearn learning_curve() with group-CV
- `plot_stability()` — boxplot across multiple seeds

---

### Feature Generators

#### [NEW] `generate_morgan_features.py`
- Morgan ECFP4 (radius=2, 2048 bits) via RDKit
- Output: `descriptors/morgan_fingerprints.npz`

#### [NEW] `generate_rdkit_features.py`
- All ~209 RDKit 2D descriptors (raw, no imputation)
- Output: `descriptors/rdkit_descriptors.npz`

#### [NEW] `generate_chemberta_features.py`
- ChemBERTa [CLS] token embeddings (768 dims)
- GPU inference, batch size 32
- Output: `descriptors/chemberta_embeddings.npz`

---

### Training Scripts

Each training script:
1. Loads features + identical cached split from `data_utils`
2. 5-fold stratified group CV with OOF predictions
3. Trains final model on full training set
4. Evaluates on held-out test set
5. Stability analysis (5 seeds: 42, 123, 456, 789, 1337)
6. Learning curves via `sklearn.model_selection.learning_curve`
7. Generates all 8 plots
8. Saves metrics JSON, predictions CSV, and model file

#### [NEW] `train_morgan_xgb.py`
- Features: 2048 binary Morgan bits
- Feature importance: XGBoost gain + SHAP
- Model saved: `models/morgan_xgb.json`

#### [NEW] `train_rdkit_xgb.py`
- Features: RDKit descriptors with preprocessing
  - Replace Inf → NaN
  - Median imputation (fit on train only)
  - VarianceThreshold (fit on train only)
  - StandardScaler (fit on train only)
- Preprocessors saved: `models/rdkit_preprocessor.joblib`
- Feature importance: XGBoost gain + SHAP (named descriptors)

#### [NEW] `train_chemberta_xgb.py`
- Features: 768-dim ChemBERTa embeddings
- StandardScaler (optional, fit on train only)
- Feature importance: SHAP only (dim_0..dim_767)

---

## Output Structure

```
figures/
  EDA/            ← existing 7 EDA figures (moved)
  morgan/         ← 8 plots
  rdkit/          ← 8 plots
  chemberta/      ← 8 plots

results/
  morgan/         metrics.json, cv_metrics.json, stability_metrics.json,
                  oof_predictions.csv, test_predictions.csv
  rdkit/          (same)
  chemberta/      (same)

models/
  morgan_xgb.json
  rdkit_xgb.json
  rdkit_preprocessor.joblib
  chemberta_xgb.json
  chemberta_scaler.joblib

data/
  train_test_split.json   ← cached split indices (shared across pipelines)

descriptors/
  morgan_fingerprints.npz
  rdkit_descriptors.npz
  chemberta_embeddings.npz
```

## Verification Plan
1. Run all three feature generators — check NPZ output sizes
2. Run all three training scripts — confirm zero errors, metrics logged
3. Verify `data/train_test_split.json` is identical across all runs
4. Inspect figures for correctness

# GSK Solvent G-Score Prediction Walkthrough

## Overview
This project implements a complete ML workflow for predicting the G-score of solvents using XGBoost regression. The dataset is the GSK solvent sustainability dataset, with `G-score` as the target variable.

Three independent pipelines were created and executed:

- **Pipeline A:** Morgan fingerprints
- **Pipeline B:** RDKit molecular descriptors
- **Pipeline C:** ChemBERTa embeddings

All pipelines use identical train/test splits, random seeds, and XGBoost hyperparameters.

## Data Splitting and Leakage Control
- The same global train/test split is shared across all pipelines using `data_utils.py`.
- Duplicate SMILES are assigned the same group ID and kept in the same train/test split to prevent leakage.
- Splits are stratified by `Classification` to keep solvent families represented in both train and test sets.
- The cached split file is `data/train_test_split.json`.

## Generated Artifacts
### EDA
- `results/EDA/EDA_Report.md`: dataset overview, missing value analysis, duplicate SMILES details, SMILES validation, and molecular summary.
- `figures/EDA/`: EDA figures including histogram, violin, boxplot, correlation heatmap, missing values, and class-level distributions.

### Descriptors
- `descriptors/morgan_fingerprints.npz`
- `descriptors/rdkit_descriptors.npz`
- `descriptors/chemberta_embeddings.npz`

### Models and Preprocessors
- `models/morgan_xgb.json`
- `models/rdkit_xgb.json`
- `models/rdkit_preprocessor.joblib`
- `models/chemberta_xgb.json`
- `models/chemberta_scaler.joblib`

### Results
- `results/morgan/`
- `results/rdkit/`
- `results/chemberta/`

Each pipeline folder contains:
- `metrics.json`
- `cv_metrics.json`
- `stability_metrics.json`
- `oof_predictions.csv`
- `test_predictions.csv`

### Comparison
- `results/model_comparison.md`
- `results/model_comparison.json`

## Model Performance Summary
| Pipeline | Features | Train RMSE | Train R² | CV RMSE | CV R² | Test RMSE | Test R² |
|---|---|---|---|---|---|---|---|
| Morgan | 2048 | 0.403 | 0.892 | 0.916 ± 0.115 | 0.399 ± 0.202 | 0.952 | 0.566 |
| RDKit | 217 | 0.057 | 0.998 | 0.784 ± 0.133 | 0.552 ± 0.200 | 0.800 | 0.694 |
| ChemBERTa | 768 | 0.056 | 0.998 | 1.177 ± 0.078 | 0.041 ± 0.157 | 1.415 | 0.042 |

## Key Findings
- **Best performing model:** `RDKit` descriptors achieved the strongest generalization performance on the test set with `RMSE = 0.800` and `R² = 0.694`.
- **Morgan fingerprints** also performed well and produced a strong baseline with `RMSE = 0.952` and `R² = 0.566` on test.
- **ChemBERTa embeddings** showed strong training fit but poorer test generalization (`RMSE = 1.415`, `R² = 0.042`), suggesting that this embedding representation may need additional tuning or larger data to improve generalization.

## Validation Outputs Produced
- 5-fold cross-validation metrics
- out-of-fold predictions
- stability analysis across multiple random seeds
- learning curves
- residual analysis
- parity plots
- feature importance plots
- SHAP explanations

## Scripts
All scripts are executable independently:
- `generate_morgan_features.py`
- `generate_rdkit_features.py`
- `generate_chemberta_features.py`
- `train_morgan_xgb.py`
- `train_rdkit_xgb.py`
- `train_chemberta_xgb.py`
- `run_all_pipelines.py`

## Conclusion
The project is complete and organized according to the task requirements. The pipeline outputs are fully structured in `results/` and `figures/` with EDA artifacts separated into `EDA` subfolders.

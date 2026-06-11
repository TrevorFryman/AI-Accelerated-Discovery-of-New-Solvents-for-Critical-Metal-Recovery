# Morgan Fingerprint Model Report

## 1. Model Architecture

- Algorithm: XGBoost Regressor (`xgboost.XGBRegressor`)
- Objective: `reg:squarederror`
- Input features: 2048 Morgan fingerprint bits (radius=2, nBits=2048)
- Target: G-score
- random_state: 42

## 2. Hyperparameters (Best Found via RandomizedSearchCV)

| Hyperparameter | Search Space | Best Value |
|---|---|---|
| n_estimators | [50, 100, 200, 300, 500] | 50 |
| max_depth | [2, 3, 4, 5, 6, 8] | 5 |
| learning_rate | [0.01, 0.02, 0.05, 0.1, 0.2, 0.3] | 0.2 |
| subsample | [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] | 0.5 |
| colsample_bytree | [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0] | 0.6 |
| min_child_weight | [1, 2, 3, 5, 7, 10] | 1 |

Search configuration: `RandomizedSearchCV` with `n_iter=100`, 5-fold CV (`KFold(n_splits=5, shuffle=True, random_state=42)`), scoring=`neg_root_mean_squared_error`, performed on the training split only (80% of data; the test set was untouched during tuning).

Best training-CV RMSE during search: 0.8014

## 3. Validation Metrics (80/20 Train-Test Split)

| Set | RMSE | MAE | R2 |
|---|---|---|---|
| Train | 0.2820 | 0.2311 | 0.9496 |
| Test | 0.7432 | 0.5697 | 0.6933 |

## 4. Cross-Validation Results (5-Fold OOF, Full Dataset, Best Hyperparameters)

| Metric | Value |
|---|---|
| RMSE | 0.8691 |
| MAE | 0.6443 |
| R2 | 0.5346 |

## 5. Fit Assessment

- Train R2: 0.9496
- 5-fold OOF R2: 0.5346
- Gap: 0.4150
- **Assessment: Overfitting**

## 6. Comparison to Baseline (Default Hyperparameters)

| Model | Test RMSE | Test MAE | Test R2 | OOF RMSE | OOF MAE | OOF R2 |
|---|---|---|---|---|---|---|
| Baseline (default params) | 0.9207 | 0.7488 | 0.5294 | 0.9830 | 0.7533 | 0.4047 |
| Optimized (RandomizedSearchCV) | 0.7432 | 0.5697 | 0.6933 | 0.8691 | 0.6443 | 0.5346 |

## 7. Strengths

- Hyperparameter tuning improved 5-fold OOF R2 from 0.4047 (baseline) to 0.5346, indicating better generalization than the default-parameter model.
- Morgan fingerprints require no manual descriptor engineering and capture local substructure information directly from SMILES.
- The pipeline is fully reproducible (fixed random_state=42 for splitting, cross-validation, and model fitting).

## 8. Weaknesses

- A train/CV R2 gap of 0.4150 remains, suggesting some residual overfitting despite tuning — likely driven by the high-dimensional (2048-bit), sparse feature space relative to the small sample size (154 molecules).
- Many of the 2048 fingerprint bits are constant (zero) across the dataset and contribute no information, increasing the effective dimensionality without benefit.
- The dataset is small (154 samples), so both the test-set metrics and the RandomizedSearchCV results carry meaningful variance; results should be interpreted with that uncertainty in mind.

## 9. Output Files

- Best model: `models/morgan_xgboost_model.pkl`
- Baseline plots: `results/Morgan/learning_curve.png`, `results/Morgan/parity_residual_plots.png`

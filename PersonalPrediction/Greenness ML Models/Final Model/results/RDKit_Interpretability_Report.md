# RDKit Descriptor Interpretability Report

## 1. Model Architecture

- Algorithm: XGBoost Regressor (`xgboost.XGBRegressor`)
- Objective: `reg:squarederror`
- Input features (7): MolWt, LogP, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds, RingCount
- Target: G-score
- random_state: 42

## 2. Hyperparameters (Best Found via RandomizedSearchCV)

| Hyperparameter | Search Space | Best Value |
|---|---|---|
| n_estimators | [50, 100, 200, 300, 500] | 500 |
| max_depth | [2, 3, 4, 5, 6, 8] | 3 |
| learning_rate | [0.01, 0.02, 0.05, 0.1, 0.2, 0.3] | 0.01 |
| subsample | [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] | 0.7 |
| colsample_bytree | [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0] | 0.8 |
| min_child_weight | [1, 2, 3, 5, 7, 10] | 1 |

Search configuration: `RandomizedSearchCV` with `n_iter=100`, 5-fold CV (`KFold(n_splits=5, shuffle=True, random_state=42)`), scoring=`neg_root_mean_squared_error`, performed on the training split only (80% of data; the test set was untouched during tuning).

Best training-CV RMSE during search: 0.9184

## 3. Baseline vs Optimized Comparison

| Model | Test RMSE | Test MAE | Test R2 | OOF RMSE | OOF MAE | OOF R2 |
|---|---|---|---|---|---|---|
| Baseline (default params) | 0.9805 | 0.7860 | 0.4663 | 0.9632 | 0.7529 | 0.4284 |
| Optimized (RandomizedSearchCV) | 0.9475 | 0.7329 | 0.5016 | 0.9331 | 0.7244 | 0.4636 |

## 4. Validation Metrics (80/20 Train-Test Split, Optimized Model)

| Set | RMSE | MAE | R2 |
|---|---|---|---|
| Train | 0.5304 | 0.4065 | 0.8217 |
| Test | 0.9475 | 0.7329 | 0.5016 |

## 5. Cross-Validation Results (5-Fold OOF, Full Dataset, Best Hyperparameters)

| Metric | Value |
|---|---|
| RMSE | 0.9331 |
| MAE | 0.7244 |
| R2 | 0.4636 |

## 6. Fit Assessment

- Train R2: 0.8217
- 5-fold OOF R2: 0.4636
- Gap: 0.3581
- **Assessment: Overfitting**

## 7. Feature Importance (XGBoost Gain-Based)

| Rank | Descriptor | Importance |
|---|---|---|
| 1 | TPSA | 0.2385 |
| 2 | NumHAcceptors | 0.1979 |
| 3 | NumHDonors | 0.1427 |
| 4 | NumRotatableBonds | 0.1267 |
| 5 | LogP | 0.1238 |
| 6 | MolWt | 0.0865 |
| 7 | RingCount | 0.0839 |

![Feature Importance](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/results/RDKit/feature_importance.png)

## 8. SHAP Analysis

SHAP (SHapley Additive exPlanations) values were computed using `shap.TreeExplainer` on the full dataset with the optimized model.

| Rank | Descriptor | Mean |SHAP value| |
|---|---|---|
| 1 | TPSA | 0.6069 |
| 2 | LogP | 0.2991 |
| 3 | RingCount | 0.1477 |
| 4 | MolWt | 0.1184 |
| 5 | NumHAcceptors | 0.1168 |
| 6 | NumRotatableBonds | 0.1074 |
| 7 | NumHDonors | 0.0524 |

![SHAP Summary](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/results/RDKit/shap_summary.png)

![SHAP Importance Bar](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Track%202/results/RDKit/shap_importance_bar.png)

## 9. Which Descriptors Most Strongly Influence G-score Predictions

Based on both the gain-based feature importance and the SHAP analysis, `TPSA` is the descriptor with the largest mean absolute SHAP value (and therefore the strongest influence on G-score predictions), followed by `LogP` and `RingCount`. This is broadly consistent with the EDA finding that TPSA showed the strongest linear correlation with G-score among the descriptors examined.

## 10. Output Files

- Best model: `models/rdkit_xgboost_model.pkl`
- Best hyperparameters: `models/rdkit_best_params.json`
- Baseline plots: `results/RDKit/learning_curve.png`, `results/RDKit/parity_residual_plots.png`
- Feature importance: `results/RDKit/feature_importance.png`
- SHAP plots: `results/RDKit/shap_summary.png`, `results/RDKit/shap_importance_bar.png`

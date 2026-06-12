# RDKit Descriptor Model - Regularization Check (Step 3)

Step 3 of the fine-tuning plan: extended the hyperparameter search space with explicit regularization terms (`reg_alpha`, `reg_lambda`, `gamma`), capped `max_depth` to [2, 3, 4], and biased `min_child_weight` toward higher values, using `RandomizedSearchCV` (n_iter=250) on the 13-descriptor feature set under the Step 1 `StratifiedGroupKFold` split (fold 0 = held-out test set). The search CV is a separate `StratifiedGroupKFold(5)` over the training portion only.

Feature columns (13): MolWt, LogP, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds, RingCount, NumAromaticRings, FractionCSP3, MolMR, HeavyAtomCount, NumAliphaticRings, BertzCT

## Hyperparameters

| Hyperparameter | Search Space | Best Value (Step 3) | Current Best |
|---|---|---|---|
| n_estimators | [50, 100, 200, 300, 500] | 500 | 500 |
| max_depth | [2, 3, 4] | 3 | 3 |
| learning_rate | [0.01, 0.02, 0.05, 0.1] | 0.05 | 0.01 |
| subsample | [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] | 0.8 | 0.7 |
| colsample_bytree | [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0] | 0.7 | 0.8 |
| min_child_weight | [1, 2, 3, 5, 7, 10, 15] | 2 | 1 |
| reg_alpha | [0, 0.01, 0.1, 0.5, 1.0, 2.0] | 0 | - |
| reg_lambda | [0.5, 1.0, 1.5, 2.0, 3.0, 5.0] | 3.0 | - |
| gamma | [0, 0.05, 0.1, 0.5, 1.0] | 0.1 | - |

Best training-CV RMSE during search: 0.8615

## Results Comparison

| Config | Set | RMSE | MAE | R2 |
|---|---|---|---|---|
| Current best (Step 2 params) | Train | 0.4198 | 0.3266 | 0.8893 |
| Current best (Step 2 params) | Test | 1.0625 | 0.8568 | 0.3105 |
| Current best (Step 2 params) | OOF | 0.8881 | 0.6778 | 0.5123 |
| Step 3 (regularized) | Train | 0.2227 | 0.1778 | 0.9689 |
| Step 3 (regularized) | Test | 1.0174 | 0.8124 | 0.3677 |
| Step 3 (regularized) | OOF | 0.8700 | 0.6684 | 0.5320 |

## Train-CV R2 Gap

- Current best gap: 0.3770
- Step 3 gap (regularized): 0.4369
- Change: +0.0599

## Step 3 Success Check

- Train R2 dropped vs current best: False (0.9689 vs 0.8893)
- OOF R2 stable or improved (>= best - 0.01): True (0.5320 vs 0.5123)
- Train-CV gap shrunk vs current best: False (0.4369 vs 0.3770)
- **Overall: FAIL**

Step 3 hyperparameters were **rejected**. `model/rdkit_xgboost_model.pkl` and `model/rdkit_best_params.json` remain at the current best (Step 2) configuration, refit on the Step 1 training split.

## Feature Importance (XGBoost Gain-Based, current model)

| Rank | Descriptor | Importance |
|---|---|---|
| 1 | TPSA | 0.1778 |
| 2 | NumHAcceptors | 0.1412 |
| 3 | NumRotatableBonds | 0.1069 |
| 4 | MolMR | 0.0820 |
| 5 | HeavyAtomCount | 0.0801 |
| 6 | NumHDonors | 0.0756 |
| 7 | LogP | 0.0732 |
| 8 | BertzCT | 0.0611 |
| 9 | NumAromaticRings | 0.0503 |
| 10 | MolWt | 0.0461 |
| 11 | FractionCSP3 | 0.0438 |
| 12 | RingCount | 0.0331 |
| 13 | NumAliphaticRings | 0.0287 |

![Feature Importance](RDKit/feature_importance.png)

## SHAP Ranking (Mean |SHAP value|, current model)

| Rank | Descriptor | Mean |SHAP value| |
|---|---|---|
| 1 | TPSA | 0.6143 |
| 2 | LogP | 0.2571 |
| 3 | MolMR | 0.2300 |
| 4 | MolWt | 0.1309 |
| 5 | BertzCT | 0.1063 |
| 6 | NumHAcceptors | 0.0858 |
| 7 | NumRotatableBonds | 0.0624 |
| 8 | NumHDonors | 0.0518 |
| 9 | FractionCSP3 | 0.0490 |
| 10 | RingCount | 0.0240 |
| 11 | HeavyAtomCount | 0.0229 |
| 12 | NumAromaticRings | 0.0098 |
| 13 | NumAliphaticRings | 0.0024 |

![SHAP Summary](RDKit/shap_summary.png)

![SHAP Importance Bar](RDKit/shap_importance_bar.png)

## Output Files

- Current model: `model/rdkit_xgboost_model.pkl`
- Current hyperparameters: `model/rdkit_best_params.json`

# RDKit Descriptor Model - Re-Tuning Report (Step 3)

Step 3 of the fine-tuning plan: re-ran `RandomizedSearchCV` (n_iter=100) on the expanded 13-descriptor feature set, using a `StratifiedGroupKFold` split (grouped by `solvent_SMILES`, stratified by `Classification`). Fold 0 is the held-out test set; the search CV was a separate `StratifiedGroupKFold(5)` over the training portion only (126 samples).

Feature columns (13): MolWt, LogP, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds, RingCount, NumAromaticRings, FractionCSP3, MolMR, HeavyAtomCount, NumAliphaticRings, BertzCT

## Result: Re-tuning REVERTED

`RandomizedSearchCV` selected hyperparameters that performed *worse* than the
Step 2 hyperparameters on every metric. sklearn warned that the least
populated class in the training split has only 4 members for
`StratifiedGroupKFold(n_splits=5)` — with 126 training samples spread across
11 solvent classes, the inner search CV folds are too small/noisy to give a
reliable hyperparameter ranking, and `RandomizedSearchCV` overfit to that
noisy signal.

| Hyperparameter | Search Space | Step 3 (rejected) | Step 2 (kept) |
|---|---|---|---|
| n_estimators | [50, 100, 200, 300, 500] | 50 | 500 |
| max_depth | [2, 3, 4, 5, 6, 8] | 4 | 3 |
| learning_rate | [0.01, 0.02, 0.05, 0.1, 0.2, 0.3] | 0.05 | 0.01 |
| subsample | [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] | 0.9 | 0.7 |
| colsample_bytree | [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0] | 1.0 | 0.8 |
| min_child_weight | [1, 2, 3, 5, 7, 10] | 2 | 1 |

Best training-CV RMSE during search: 0.8510 (but this did not translate to better held-out performance)

## Results Comparison

| Step | Set | RMSE | MAE | R2 |
|---|---|---|---|---|
| Step 2 (13 feat, kept params) | Train | 0.4198 | 0.3266 | 0.8893 |
| Step 2 (13 feat, kept params) | Test | 1.0625 | 0.8568 | 0.3105 |
| Step 2 (13 feat, kept params) | OOF | 0.8881 | 0.6778 | 0.5123 |
| Step 3 (13 feat, rejected params) | Train | 0.4452 | 0.3438 | 0.8756 |
| Step 3 (13 feat, rejected params) | Test | 1.2347 | 0.9649 | 0.0688 |
| Step 3 (13 feat, rejected params) | OOF | 0.9643 | 0.7257 | 0.4250 |

## Train-CV R2 Gap

- Step 2 gap (kept params): 0.3770
- Step 3 gap (rejected params): 0.4506
- Change: +0.0736 (worse)

## Decision

Step 3 hyperparameters were **rejected and reverted**. `model/rdkit_xgboost_model.pkl`
and `model/rdkit_best_params.json` remain at the Step 2 values
(`n_estimators=500, max_depth=3, learning_rate=0.01, subsample=0.7,
colsample_bytree=0.8, min_child_weight=1`), refit on the Step 1
`StratifiedGroupKFold` training split with the 13-feature descriptor set.
This is the current best model: **OOF R2=0.5123, Test R2=0.3105, Train-CV
gap=0.3770**.

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

- Best model: `model/rdkit_xgboost_model.pkl` (Step 2 params, 13 features)
- Best hyperparameters: `model/rdkit_best_params.json`

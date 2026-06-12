# RDKit Descriptor Model - Final Model Report (Step 5)

Step 5 of the fine-tuning plan: with the final hyperparameter set (`model/rdkit_best_params.json`, from Step 2/3) and the 13-descriptor feature set (Step 2) settled, the model was refit on the full 154-sample dataset for deployment/reporting, and all interpretability and diagnostic plots were regenerated against this final model. The Step 4 nested-CV scores remain the headline generalization metric for this model; the full-data fit and OOF metrics below are diagnostic only.

Feature columns (13): MolWt, LogP, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds, RingCount, NumAromaticRings, FractionCSP3, MolMR, HeavyAtomCount, NumAliphaticRings, BertzCT

Final hyperparameters: `{'subsample': 0.7, 'n_estimators': 500, 'min_child_weight': 1, 'max_depth': 3, 'learning_rate': 0.01, 'colsample_bytree': 0.8}`

## Headline Generalization Metric (Step 4, Nested CV)

| Metric | Mean | 95% CI Lower | 95% CI Upper |
|---|---|---|---|
| RMSE | 0.8721 | 0.7238 | 1.0203 |
| MAE | 0.6699 | 0.5292 | 0.8107 |
| R2 | 0.5050 | 0.3342 | 0.6757 |

## Diagnostic Metrics (Final Model, Not Generalization Estimates)

| Set | RMSE | MAE | R2 |
|---|---|---|---|
| Full-data fit (refit model) | 0.4790 | 0.3710 | 0.8581 |
| OOF (5-fold SGKF, final params) | 0.8881 | 0.6778 | 0.5123 |

## Feature Importance (XGBoost Gain-Based, Final Model)

| Rank | Descriptor | Importance |
|---|---|---|
| 1 | TPSA | 0.1547 |
| 2 | NumHAcceptors | 0.1260 |
| 3 | NumHDonors | 0.1025 |
| 4 | NumRotatableBonds | 0.0932 |
| 5 | MolMR | 0.0768 |
| 6 | FractionCSP3 | 0.0740 |
| 7 | HeavyAtomCount | 0.0694 |
| 8 | NumAromaticRings | 0.0672 |
| 9 | LogP | 0.0646 |
| 10 | BertzCT | 0.0617 |
| 11 | MolWt | 0.0446 |
| 12 | NumAliphaticRings | 0.0332 |
| 13 | RingCount | 0.0319 |

![Feature Importance](RDKit/feature_importance.png)

## SHAP Ranking (Mean |SHAP value|, Final Model)

| Rank | Descriptor | Mean |SHAP value| |
|---|---|---|
| 1 | TPSA | 0.5230 |
| 2 | MolMR | 0.2164 |
| 3 | LogP | 0.1902 |
| 4 | MolWt | 0.1153 |
| 5 | BertzCT | 0.1087 |
| 6 | NumHAcceptors | 0.0997 |
| 7 | FractionCSP3 | 0.0895 |
| 8 | NumHDonors | 0.0823 |
| 9 | NumRotatableBonds | 0.0617 |
| 10 | RingCount | 0.0311 |
| 11 | HeavyAtomCount | 0.0287 |
| 12 | NumAromaticRings | 0.0122 |
| 13 | NumAliphaticRings | 0.0063 |

![SHAP Summary](RDKit/shap_summary.png)

![SHAP Importance Bar](RDKit/shap_importance_bar.png)

## Step 5 Success Check

- TPSA feature-importance rank: 1 (of 13)
- TPSA SHAP rank: 1 (of 13)
- **TPSA remains a top-2 driver: True**

## Diagnostic Plots

- `results/RDKit/parity_residual_plots.png`
- `results/RDKit/learning_curve.png`

## Output Files

- Final model (refit on all 154 samples): `model/rdkit_xgboost_model.pkl`
- Final hyperparameters: `model/rdkit_best_params.json`

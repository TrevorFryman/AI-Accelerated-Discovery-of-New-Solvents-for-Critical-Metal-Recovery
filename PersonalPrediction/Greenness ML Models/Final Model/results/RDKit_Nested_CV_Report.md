# RDKit Descriptor Model - Nested Cross-Validation Report (Step 4)

Step 4 of the fine-tuning plan: nested cross-validation. An outer 5-fold `StratifiedGroupKFold` (grouped by `solvent_SMILES`, stratified by `Classification`, same as Step 1) provides 5 independent test folds. For each outer fold, hyperparameters are tuned via `RandomizedSearchCV` (n_iter=150) using an inner 4-fold `StratifiedGroupKFold` over that fold's training data only, then the outer fold's test score is computed from a model trained with those fold-specific best hyperparameters. The reported mean +/- 95% CI (t-distribution, df=4) is the headline performance estimate for this model.

Feature columns (13): MolWt, LogP, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds, RingCount, NumAromaticRings, FractionCSP3, MolMR, HeavyAtomCount, NumAliphaticRings, BertzCT

## Per-Fold Results

| Outer Fold | n_test | RMSE | MAE | R2 |
|---|---|---|---|---|
| 1 | 32 | 1.0173 | 0.8008 | 0.3679 |
| 2 | 32 | 0.9810 | 0.7582 | 0.4735 |
| 3 | 31 | 0.7533 | 0.6335 | 0.6994 |
| 4 | 32 | 0.8245 | 0.6438 | 0.3980 |
| 5 | 31 | 0.7843 | 0.5134 | 0.5863 |

## Per-Fold Best Hyperparameters

| Outer Fold | n_estimators | max_depth | learning_rate | subsample | colsample_bytree | min_child_weight | reg_alpha | reg_lambda | gamma |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 500 | 4 | 0.05 | 0.9 | 0.6 | 5 | 0.1 | 1.5 | 0 |
| 2 | 500 | 3 | 0.05 | 0.8 | 0.7 | 2 | 0 | 3.0 | 0.1 |
| 3 | 500 | 3 | 0.05 | 0.8 | 0.7 | 2 | 0 | 3.0 | 0.1 |
| 4 | 500 | 3 | 0.05 | 0.8 | 0.3 | 2 | 0.01 | 1.0 | 0.5 |
| 5 | 500 | 3 | 0.05 | 0.8 | 0.7 | 2 | 0 | 3.0 | 0.1 |

## Nested CV Summary (Mean +/- 95% CI, t-distribution df=4)

| Metric | Mean | 95% CI Lower | 95% CI Upper |
|---|---|---|---|
| RMSE | 0.8721 | 0.7238 | 1.0203 |
| MAE | 0.6699 | 0.5292 | 0.8107 |
| R2 | 0.5050 | 0.3342 | 0.6757 |

## Comparison to Single-Split Metrics (Step 1/2 Config)

| Metric | Single-Split Test | Single-Split OOF | Nested CV Mean |
|---|---|---|---|
| RMSE | 1.0625 | 0.8881 | 0.8721 |
| MAE | 0.8568 | 0.6778 | 0.6699 |
| R2 | 0.3105 | 0.5123 | 0.5050 |

## Success Check

- Single-split Test R2 (0.3105) within nested CV 95% CI [0.3342, 0.6757]: False
- Single-split OOF R2 (0.5123) within nested CV 95% CI [0.3342, 0.6757]: True

At least one single-split estimate falls outside the nested CV confidence interval, meaning the single 80/20-style split was not fully representative (high variance from the small held-out set, as expected with n~31 per fold). The nested CV mean +/- 95% CI should be trusted and reported instead of the single-split numbers as the headline metric for the RDKit descriptor model (R2 = 0.5050 [0.3342, 0.6757], RMSE = 0.8721 [0.7238, 1.0203]).
